# -*- coding: utf-8 -*-
"""
2번: 새니티 NER 학습 (XML 금라벨 데이터, Lightning/서버 전용)

목적: 본학습 전 비용 낭비 방지를 위한 단시간 학습 점검
  1) B-LOC가 0으로 붕괴하지 않는지
  2) I-LOC가 B 없이 난발되지 않는지 (BIO 위반율 집계)
  3) 개체 단위(span exact match) F1이 학습 진행에 따라 상승하는지

- 평가 지표를 개체 단위 P/R/F1로 교체 (토큰별 통계는 진단용 유지)
  근거: 과제 목표가 지명 구간의 온전한 추출이므로 측정 단위는 개체.
  토큰 micro F1은 O 98.4%인 본 데이터에서 성능을 부풀림.
- 라벨 처리를 정수 + CrossEntropyLoss(ignore_index=-100)로 정리
  (데이터의 [0]/[1]/[2] 단일 원소 리스트 형식은 그대로 수용)

GPU 지정: L40S

data 폴더에 train.jsonl, val.jsonl 저장

Lightning 터미널 프롬프트:
python 2_NER_새니티학습.py --data_dir ~/data --max_steps 300
"""

import json
import argparse
from pathlib import Path

import numpy as np

TAGS = ['O', 'B-LOC', 'I-LOC']
NUM_LABELS = 3
MAX_LEN = 256
IGNORE_INDEX = -100


# === 개체 디코딩·지표 (순수 함수: torch 무관, 단위 테스트 대상) ===

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


# === 이하 학습 본체 (torch/Lightning) ===

def build_training():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import pytorch_lightning as pl
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    from tqdm import tqdm

    torch.set_float32_matmul_precision('high')

    class NERDataset(Dataset):
        """JSONL 기반 NER Dataset (오프셋 인덱싱 + word_ids 정렬)"""

        def __init__(self, jsonl_path, tokenizer, max_length=MAX_LEN):
            self.jsonl_path = Path(jsonl_path)
            self.tokenizer = tokenizer
            self.max_length = max_length

            self.line_offsets = []
            with open(self.jsonl_path, 'rb') as f:
                offset = 0
                for line in tqdm(f, desc=f"Indexing {self.jsonl_path.name}"):
                    self.line_offsets.append(offset)
                    offset = f.tell()
            print(f"[Dataset] {self.jsonl_path.name}: {len(self.line_offsets):,} samples")

            if self.line_offsets:
                s0 = self._load_sample(0)
                for k in ('c', 'l', 'n'):
                    if k not in s0:
                        raise ValueError(f"데이터 형식 오류: '{k}' 키 없음. keys={list(s0.keys())}")

        def _load_sample(self, idx):
            with open(self.jsonl_path, 'rb') as f:
                f.seek(self.line_offsets[idx])
                line = f.readline()
            return json.loads(line.decode('utf-8'))

        def __len__(self):
            return len(self.line_offsets)

        def __getitem__(self, idx):
            sample = self._load_sample(idx)
            text = sample['c']
            length = int(sample['n'])
            # [0]/[1]/[2] 단일 원소 리스트 → 정수
            labels_char = [lab[0] for lab in sample['l'][:length]]

            enc = self.tokenizer(
                list(text),
                is_split_into_words=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
            )
            word_ids = enc.word_ids(batch_index=0)

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
        def __init__(self, model_name="SIKU-BERT/sikuroberta",
                     learning_rate=2e-5, warmup_ratio=0.1,
                     dropout_rate=0.1, total_steps=1000):
            super().__init__()
            self.save_hyperparameters()
            self.bert = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_LABELS)
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
            self.log("train_loss", loss, on_step=True, prog_bar=True)
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

            print("\n" + "=" * 70)
            print("[SANITY] Validation summary")
            print(f"- val_loss: {avg_loss:.6f}")
            print(f"- 개체 단위(span exact): F1={f1:.6f}, P={p:.6f}, R={r:.6f}")
            print(f"  (TP={self.vm['span_tp']:,}, FP={self.vm['span_fp']:,}, FN={self.vm['span_fn']:,})")
            print(f"- 예측 BIO 위반(B 없는 I): {self.vm['pred_violations']:,}건")
            print("  토큰별 진단:")
            for i, tag in enumerate(TAGS):
                tp_i, fp_i, fn_i = (self.vm['token_tp'][i],
                                    self.vm['token_fp'][i], self.vm['token_fn'][i])
                pi, ri, fi = prf_score(tp_i, fp_i, fn_i)
                print(f"  {tag}: F1={fi:.6f}, P={pi:.6f}, R={ri:.6f} (gold {int(tp_i + fn_i):,})")
            if self.vm['token_tp'][1] + self.vm['token_fp'][1] == 0:
                print("  ⚠️ B-LOC 예측 0건 — 붕괴 신호")
            print("=" * 70 + "\n")

            self.log("val_loss", avg_loss, prog_bar=True)
            self.log("val_span_f1", f1, prog_bar=True)
            self._reset_val_metrics()

        def configure_optimizers(self):
            import torch as _t
            optimizer = _t.optim.AdamW(self.parameters(),
                                       lr=self.hparams.learning_rate, weight_decay=0.01)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.total_steps * self.hparams.warmup_ratio),
                num_training_steps=int(self.hparams.total_steps),
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    return NERDataset, NERModel, AutoTokenizer, DataLoader, pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="train.jsonl/val.jsonl이 있는 폴더")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_interval_steps", type=int, default=100)
    parser.add_argument("--limit_val_batches", type=float, default=0.02,
                        help="val 배치 사용 비율 (29.7만 청크이므로 새니티는 0.02 권장)")
    args = parser.parse_args()

    print("=" * 80)
    print("2번: NER 새니티 학습 (XML 금라벨, 개체 단위 F1)")
    print("=" * 80)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"train.jsonl/val.jsonl이 {data_dir}에 없습니다.")

    NERDataset, NERModel, AutoTokenizer, DataLoader, pl = build_training()

    tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta", use_fast=True)
    train_ds = NERDataset(train_path, tokenizer)
    val_ds = NERDataset(val_path, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = NERModel(total_steps=args.max_steps)

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        val_check_interval=args.val_interval_steps,
        limit_val_batches=args.limit_val_batches,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)
    print("\n✅ 새니티 학습 완료 — 위 Validation summary로 학습 신호를 판정하십시오.")


if __name__ == "__main__":
    main()