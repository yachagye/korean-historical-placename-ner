# -*- coding: utf-8 -*-
"""
3번: NER 본학습 (XML 금라벨 데이터, Lightning.ai L40S 전용)

새니티(2번) 통과 판정 후의 본학습 스크립트.
파이프라인: 0번 변환 → 1번 검증 → 2번 새니티 → 3번 본학습

- 정수 라벨 + CrossEntropyLoss(ignore_index=-100)
- 개체 단위(span exact match) P/R/F1 평가, BIO 위반(B 없는 I) 집계
- decode_entities / span_prf 순수 함수 (단위 테스트 대상)

새니티 수치 근거:
- best checkpoint / EarlyStopping 기준: val_span_f1 (max)
  근거: 과제의 측정 단위는 개체. val_loss는 토큰 평균이라 O 98.4%의
  영향이 커서 수렴 후반 개체 성능과 갈라질 수 있음.
- 3 에폭 + EarlyStopping(patience=5)
- val_check_interval=0.25, 전체 val(29.7만 청크) 사용
- 운용 요소는 4번 유지: batch 80 + accum 2, bf16-mixed,
  ModelCheckpoint(save_last=True), TensorBoard/CSV 로거,
  오프셋 인덱싱 + 워커별 파일 핸들 재사용

GPU 지정: L40S
운용: 스튜디오 최상단에 data/, model/과 같은 층위로 train.py로 저장하여 실행

출력 (스튜디오 최상단 기준):
- model/checkpoints/best-epoch={E}-step={S}-val_span_f1={F1}.ckpt : best 모델 (val_span_f1 max 기준 1개)
- model/checkpoints/last.ckpt : 최신 상태 (재개용)
- model/logs/tensorboard/, model/logs/csv/ : train_loss, val_loss, val_span_f1/p/r, val_bio_violations, lr
- 콘솔: 검증마다 [VAL] 요약 (개체 단위 F1/P/R, BIO 위반, 토큰별 진단 — 2번 새니티와 동일 형식)

Lightning 터미널 프롬프트:
  python train.py --data_dir data
재개(별도 스크립트 불필요):
  python train.py --data_dir data --ckpt_path model/checkpoints/last.ckpt
"""

import json
import math
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

torch.set_float32_matmul_precision('high')

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

TAGS = ['O', 'B-LOC', 'I-LOC']
NUM_LABELS = 3
IGNORE_INDEX = -100


# === 개체 디코딩·지표 (2번과 동일한 순수 함수: torch 무관) ===

def decode_entities(label_seq):
    """정수 라벨 시퀀스에서 (시작, 끝) 개체 구간을 엄격 BIO로 복원.
    B 없이 출현한 I는 개체로 인정하지 않고 위반으로 집계.
    Returns: (entities, violation_count)
    """
    entities = []
    violations = 0
    start = None
    prev = 0
    for i, v in enumerate(label_seq):
        if v == 1:  # B
            if start is not None:
                entities.append((start, i))
            start = i
        elif v == 2:  # I
            if prev == 0:
                violations += 1  # B 없는 I
                start = None
        else:  # O
            if start is not None:
                entities.append((start, i))
                start = None
        prev = v
    if start is not None:
        entities.append((start, len(label_seq)))
    return entities, violations


def span_prf(pred_entities, gold_entities):
    """개체 완전 일치 기준 TP/FP/FN"""
    pred_set = set(pred_entities)
    gold_set = set(gold_entities)
    tp = len(pred_set & gold_set)
    return tp, len(pred_set) - tp, len(gold_set) - tp


def prf_score(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


class NERDataset(Dataset):
    """JSONL 기반 NER Dataset (오프셋 인덱싱 + 워커별 파일 핸들 재사용)"""

    def __init__(self, data_path, tokenizer, max_length: int = 256):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.line_offsets = []
        with open(self.data_path, 'rb') as f:
            offset = 0
            for line in tqdm(f, desc=f"Indexing {self.data_path.name}"):
                self.line_offsets.append(offset)
                offset = f.tell()
        print(f"[Dataset] {self.data_path.name}: {len(self.line_offsets):,} samples")

        # 워커별 lazy open용 핸들
        self._fp = None

        # 첫 샘플 형식 확인 (메인 프로세스에서는 로컬 open으로 검증)
        if self.line_offsets:
            with open(self.data_path, 'rb') as f:
                f.seek(self.line_offsets[0])
                s0 = json.loads(f.readline().decode('utf-8'))
            for k in ('c', 'l', 'n'):
                if k not in s0:
                    raise ValueError(f"데이터 형식 오류: '{k}' 키 없음. keys={list(s0.keys())}")

    def _get_fp(self):
        """DataLoader 멀티프로세싱에서 워커마다 파일 핸들 1개씩만 재사용"""
        if self._fp is None:
            self._fp = open(self.data_path, 'rb')
        return self._fp

    def __del__(self):
        try:
            if self._fp is not None:
                self._fp.close()
        except Exception:
            pass

    def _load_sample(self, idx: int) -> dict:
        fp = self._get_fp()
        fp.seek(self.line_offsets[idx])
        line = fp.readline()

        # 손상 라인 방어: 장시간 학습 중단 방지를 위해 무해화(n=0)
        try:
            return json.loads(line.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {'c': '', 'l': [], 'n': 0}

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        sample = self._load_sample(idx)
        text = sample.get('c', '')
        l_raw = sample.get('l', [])
        try:
            raw_length = int(sample.get('n', 0))
        except (TypeError, ValueError):
            raw_length = 0

        length = min(raw_length, len(text), len(l_raw))

        # [0]/[1]/[2] 단일 원소 리스트 → 정수
        labels_char = [lab[0] for lab in l_raw[:length]]

        enc = self.tokenizer(
            list(text[:length]),
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        word_ids = enc.word_ids(batch_index=0)

        # 라벨 정렬: 실제 문자 토큰만 정수 라벨, 나머지는 ignore
        aligned = [IGNORE_INDEX] * self.max_length
        for i, widx in enumerate(word_ids):
            if widx is not None and widx < length:
                aligned[i] = labels_char[widx]

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned, dtype=torch.long),
        }


class NERModel(pl.LightningModule):
    """SikuRoBERTa 기반 토큰 분류 모델 (개체 단위 평가)"""

    def __init__(
        self,
        model_name: str = 'SIKU-BERT/sikuroberta',
        num_labels: int = NUM_LABELS,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        dropout_rate: float = 0.1,
        total_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self._reset_val_metrics()

    def _reset_val_metrics(self):
        self.vm = {
            'losses': [],
            'span_tp': 0, 'span_fp': 0, 'span_fn': 0,
            'pred_violations': 0,
            'token_tp': np.zeros(NUM_LABELS),
            'token_fp': np.zeros(NUM_LABELS),
            'token_fn': np.zeros(NUM_LABELS),
        }

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.last_hidden_state))

    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits.view(-1, NUM_LABELS), batch['labels'].view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits.view(-1, NUM_LABELS), batch['labels'].view(-1))
        self.vm['losses'].append(float(loss.detach().cpu()))

        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        gold = batch['labels'].cpu().numpy()

        for b in range(gold.shape[0]):
            valid = gold[b] != IGNORE_INDEX
            p_seq = pred[b][valid].tolist()
            g_seq = gold[b][valid].tolist()

            p_ents, p_viol = decode_entities(p_seq)
            g_ents, _ = decode_entities(g_seq)
            tp, fp, fn = span_prf(p_ents, g_ents)
            self.vm['span_tp'] += tp
            self.vm['span_fp'] += fp
            self.vm['span_fn'] += fn
            self.vm['pred_violations'] += p_viol

            p_arr, g_arr = np.array(p_seq), np.array(g_seq)
            for i in range(NUM_LABELS):
                pi, gi = p_arr == i, g_arr == i
                self.vm['token_tp'][i] += (pi & gi).sum()
                self.vm['token_fp'][i] += (pi & ~gi).sum()
                self.vm['token_fn'][i] += (~pi & gi).sum()
        return loss

    def on_validation_epoch_end(self):
        avg_loss = float(np.mean(self.vm['losses'])) if self.vm['losses'] else 0.0
        p, r, f1 = prf_score(self.vm['span_tp'], self.vm['span_fp'], self.vm['span_fn'])

        print('\n' + '=' * 70)
        print(f"[VAL] step={self.global_step:,} / epoch={self.current_epoch}")
        print(f"- val_loss: {avg_loss:.6f}")
        print(f"- 개체 단위(span exact): F1={f1:.6f}, P={p:.6f}, R={r:.6f}")
        print(f"  (TP={self.vm['span_tp']:,}, FP={self.vm['span_fp']:,}, FN={self.vm['span_fn']:,})")
        print(f"- 예측 BIO 위반(B 없는 I): {self.vm['pred_violations']:,}건")
        print('  토큰별 진단:')
        for i, tag in enumerate(TAGS):
            tp_i, fp_i, fn_i = (self.vm['token_tp'][i],
                                self.vm['token_fp'][i], self.vm['token_fn'][i])
            pi, ri, fi = prf_score(tp_i, fp_i, fn_i)
            print(f"  {tag}: F1={fi:.6f}, P={pi:.6f}, R={ri:.6f} (gold {int(tp_i + fn_i):,})")
        if self.vm['token_tp'][1] + self.vm['token_fp'][1] == 0:
            print('  ⚠️ B-LOC 예측 0건 — 붕괴 신호')
        print('=' * 70 + '\n')

        # 체크포인트·조기 종료 모니터: val_span_f1 (max)
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_span_f1', f1, prog_bar=True)
        self.log('val_span_p', p)
        self.log('val_span_r', r)
        self.log('val_bio_violations', float(self.vm['pred_violations']))

        self._reset_val_metrics()

    def on_train_start(self):
        # eval-mode 경고 방지: 학습 시작 시 전체 모듈 train 모드 강제
        self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
        )
        if self.hparams.total_steps:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.total_steps * self.hparams.warmup_ratio),
                num_training_steps=int(self.hparams.total_steps),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
            }
        return optimizer


class NERDataModule(pl.LightningDataModule):
    """데이터 모듈"""

    def __init__(self, data_dir: str, tokenizer, batch_size: int = 80,
                 max_length: int = 256, num_workers: int = 12):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self._setup_done = False

    def setup(self, stage: Optional[str] = None):
        if self._setup_done:
            return
        if stage == 'fit' or stage is None:
            self.train_dataset = NERDataset(
                self.data_dir / 'train.jsonl', self.tokenizer, self.max_length)
            self.val_dataset = NERDataset(
                self.data_dir / 'val.jsonl', self.tokenizer, self.max_length)
            self._setup_done = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='train.jsonl/val.jsonl이 있는 폴더')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='재개할 체크포인트 경로 (예: model/checkpoints/last.ckpt)')
    args = parser.parse_args()

    config = {
        'model_name': 'SIKU-BERT/sikuroberta',
        'num_labels': NUM_LABELS,
        'max_length': 256,
        'batch_size': 80,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'warmup_ratio': 0.1,
        'dropout_rate': 0.1,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 2,
        'precision': 'bf16-mixed',
        'seed': 42,
        'num_workers': 12,
    }

    pl.seed_everything(config['seed'], workers=True)

    data_dir = Path(args.data_dir).expanduser()
    if not (data_dir / 'train.jsonl').exists() or not (data_dir / 'val.jsonl').exists():
        raise FileNotFoundError(f"train.jsonl/val.jsonl이 {data_dir}에 없습니다.")

    base_dir = Path.cwd()
    ckpt_dir = base_dir / 'model' / 'checkpoints'
    log_dir = base_dir / 'model' / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print('토크나이저 로딩(use_fast=True)...')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)

    print('데이터 모듈 준비...')
    data_module = NERDataModule(
        data_dir=str(data_dir),
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        num_workers=config['num_workers'],
    )

    data_module.setup('fit')
    actual_train_size = len(data_module.train_dataset)
    steps_per_epoch = math.ceil(actual_train_size / config['batch_size'])
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / config['accumulate_grad_batches'])
    total_steps = optimizer_steps_per_epoch * config['num_epochs']

    print('모델 초기화...')
    model = NERModel(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        dropout_rate=config['dropout_rate'],
        total_steps=total_steps,
    )

    # ✅ best 기준: val_span_f1 (max) — 측정 단위와 선택 기준의 일치
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename='best-epoch={epoch}-step={step}-val_span_f1={val_span_f1:.4f}',
            auto_insert_metric_name=False,
            monitor='val_span_f1',
            mode='max',
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor='val_span_f1',
            mode='max',
            patience=5,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    loggers = [
        TensorBoardLogger(save_dir=str(log_dir), name='tensorboard'),
        CSVLogger(save_dir=str(log_dir), name='csv'),
    ]

    use_gpu = torch.cuda.is_available()

    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        accelerator='gpu' if use_gpu else 'cpu',
        devices=1,
        precision=config['precision'] if use_gpu else '32-true',
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=50,
        val_check_interval=0.25,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # 2번 새니티와 동일 구성 (True는 CUBLAS_WORKSPACE_CONFIG 미설정 시 RuntimeError 위험)
        benchmark=False,
        num_sanity_val_steps=0,
    )

    print('\n' + '=' * 60)
    print('3번: NER 본학습 시작 (L40S)')
    print('모델: SikuRoBERTa (역사 지명 NER, XML 금라벨)')
    print(f'학습 데이터: {actual_train_size:,}개 청크')
    print(f'검증 데이터: {len(data_module.val_dataset):,}개 청크 (전체 사용)')
    print(f'배치: {config["batch_size"]} × accum {config["accumulate_grad_batches"]}'
          f' = 유효 {config["batch_size"] * config["accumulate_grad_batches"]}')
    print(f'에폭당 배치 스텝: {steps_per_epoch:,} / 총 Optimizer 스텝: {total_steps:,}')
    print(f'검증 주기: 에폭의 25% (에폭당 4회)')
    print(f'모니터: val_span_f1 (max), EarlyStopping patience=5')
    print(f'Precision: {config["precision"]}')
    print(f'체크포인트: {ckpt_dir}')
    print(f'로그: {log_dir}')
    if args.ckpt_path:
        print(f'[RESUME] 재개 체크포인트: {args.ckpt_path}')
    print('=' * 60 + '\n')

    trainer.fit(model, data_module, ckpt_path=args.ckpt_path or None)

    print('\n학습 완료!')
    print(f'best 체크포인트: {callbacks[0].best_model_path}')
    print(f'best val_span_f1: {callbacks[0].best_model_score}')


if __name__ == '__main__':
    main()