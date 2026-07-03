# 조선시대 역사 지명 개체명 인식 (Korean Historical Place-Name NER)

**[한국어](#한국어)** | **[English](#english)**

---

## 한국어

색인 XML 사료를 학습 정답(gold label)으로 삼아, 조선시대 한문 사료에서 **지명 개체**를 자동 추출하는 개체명 인식(NER) 파이프라인입니다. 국사편찬위원회 한국사데이터베이스의 색인 태그를 정답으로 사용하며, 사료별 색인 관례의 편차를 두 모델의 판정 분기로 전환한 **이질 앙상블(heterogeneous ensemble)** 구조로 재현과 정밀을 함께 확보합니다.

관련 논문: 양정현, 「색인 XML 사료를 활용한 조선시대 역사 지명 개체명 인식 모델의 구축」 (투고 중, 게재 확정 시 서지·DOI 갱신 예정).
선행 연구(표점 추론): [korean-classical-chinese-punctuation](https://github.com/yachagye/korean-classical-chinese-punctuation)

### 핵심 특징

- **색인 태그 = gold label**: 색인 XML의 전수 태깅을 정답으로 직접 학습. 태깅되지 않은 위치는 신뢰 가능한 부정 예시로 사용.
- **두 모델 구성**: 동일 아키텍처(SikuRoBERTa + 토큰 분류)에 학습 출처 구성만 달리하여 상보적 판정 양상을 확보.
- **개체 단위 평가**: O 라벨이 약 98.3%를 차지하므로 토큰 micro F1 대신 **개체 완전 일치(span exact match)** P/R/F1로 평가.
- **단계적 검수**: 자동 병합 → 수동 검수 → 지명 데이터베이스 활용 검수로 이어지는 재현 가능한 스크립트 체계.

### 성능 (자체 검증셋, span exact)

| 모델 | 학습 출처 | F1 | Precision | Recall | 성격 |
| --- | --- | --- | --- | --- | --- |
| 승정원일기·실록 모델 | 조선왕조실록·승정원일기·고순종실록 | **0.9049** | 0.8982 | 0.9118 | 고재현·저정밀 |
| 종합 모델 | 위 + 한국사료총서·고려사·고려사절요·삼국사기·삼국유사 | **0.8915** | 0.8803 | 0.9030 | 저재현·고정밀 |

> 두 모델의 검증셋은 구성이 다릅니다(승정원일기·실록 val은 순수 연대기, 종합 val은 7종 전체). 두 F1을 직접 비교하지 마십시오.

학습 자료 규모: 1,334개 파일(파싱 실패 0), 총 한자 293,652,373자, 지명 개체 2,075,132개, 고유 지명 94,846종.

### 모델 가중치

학습된 두 모델의 가중치는 Hugging Face에 공개합니다: [yachagye/korean-historical-placename-ner](https://huggingface.co/yachagye/korean-historical-placename-ner)
기반 모델: [SIKU-BERT/sikuroberta](https://huggingface.co/SIKU-BERT/sikuroberta) (108M params).

### 파이프라인

변환 → 검증 → 새니티 → 본학습 → 추론·병합. 학습(2·3단계)은 Lightning.ai L40S, 나머지는 로컬(Windows) 환경 기준입니다. 파이프라인 규격의 정본은 [`docs/파이프라인_지침.md`](docs/파이프라인_지침.md)이며, 아래는 요약입니다.

| 단계 | 스크립트 | 입력 → 출력 |
| --- | --- | --- |
| 0 변환 | `0_NER_학습데이터_변환_xml_jsonl.py` | 색인 XML → `train/val.jsonl`, 변환 통계 |
| 1 검증 | `1_NER_학습데이터_검증.py` | `train/val.jsonl` → 검증 보고 (형식·BIO·분할 누수·분포) |
| 2 새니티 | `2_NER_새니티학습_Lightning.py` | 단시간 학습으로 붕괴 여부 점검 |
| 3 본학습 | `3_NER_학습_Lightning.py` | `train/val.jsonl` → `best.ckpt`, 로그 |
| 4 추론·병합 | `4_NER_추론_txt.py` | 원문 txt + ckpt → 단독/통합본/검수통합본 |
| 부속 | `NER_model_loader.py` | ckpt → SikuRoBERTa + 분류기 복원 (추론 공통 로더) |
| 부속 | `xml_메타데이터_분석.py` | 색인 XML 전수 집계 (출처·장르별 태깅률·밀도) |
| 부속 | `지명DB_정제.py` | 지명DB → 정제본 CSV |
| 부속 | `지명DB_추론_검수.py` | `[]` 태깅 txt + 정제 지명DB → DB 활용 검수본 |

#### 데이터 형식 (JSONL)

문자 단위 BIO 라벨(0=O, 1=B-LOC, 2=I-LOC)로 변환합니다.

```json
{"c": "慶尙道晉州府在京南", "l": [[1],[2],[2],[1],[2],[2],[0],[0],[0]], "n": 9, "id": "...", "src": "...", "p": []}
```

- `c` 본문(무표점 한자열), `l` 문자 단위 라벨, `n` 길이, `id` 기사 ID, `src` 출처, `p` 인명 구간 메타필드(지명·인명 2-클래스 확장용으로 보존).
- 기사 ID 해시 분할로 train/val 분리(분할 누수 0 보장).

#### 이질 앙상블 병합

동일 원문을 두 모델로 각각 추론한 뒤 예측을 개체 수준에서 결합합니다.

1. **합집합**: 두 모델의 span을 모두 후보로 채택 → 종합 모델의 누락 복구(재현 확보).
2. **겹침 시 최장**: 서로 겹치는 span은 가장 넓은 것 하나만 유지(동일 길이면 종합 우선) → 경계 정밀 확보.

추론 span 내부 비한자(구두점·기호)는 경계 오류로 보고 연속 한자 구간으로 분할합니다(`金遷。○山` → `金遷`+`山`). 잔여 비지명 단편은 다운스트림 정제 대상입니다.

#### 단계적 검수

자동 병합(`_통합본`) → 겹침 사례 수동 선택(`_검수통합본`) → 지명DB 최장일치 보조 검수(`_DB검수`). 지명 데이터베이스는 학습·평가에는 사용하지 않으며, 검수 단계에서만 보조 자원으로 활용합니다(동형이의 위험 때문에 자동 삽입이 아닌 사람의 선택).

### 사용법

#### 설치

```bash
pip install torch pytorch-lightning transformers
```

#### 추론

```bash
python 4_NER_추론_txt.py
```

실행 후 모드(단독/병합/검수 병합)와 입력 방식(단일 파일/폴더 순회)을 선택하고, 두 모델의 체크포인트를 지정하면 통합본이 생성됩니다. 지명으로 판정된 개체는 대괄호 `[ ]`로, 결락 표지는 원문 그대로 `□`로 표시됩니다.

출력 예시(신증동국여지승람 죽산현):

```
[竹山縣]
東至[陰竹縣]界二十二里，南至[忠淸道][鎭川縣]界二十六里，西至[安城郡]界二十三里…
[天民川]。在縣東十里。源出[巾之]、[鼎陪]兩山，入[驪州][驪江]。
```

#### 학습 (재현)

```bash
# 0. 변환
python 0_NER_학습데이터_변환_xml_jsonl.py
# 1. 검증
python 1_NER_학습데이터_검증.py
# 2. 새니티 (L40S)
python 2_NER_새니티학습_Lightning.py --data_dir data --max_steps 300
# 3. 본학습 (L40S)
python 3_NER_학습_Lightning.py --data_dir data
```

주요 하이퍼파라미터: max_length 256(문자 단위), batch_size 80(유효 배치 160), lr 2e-5(AdamW), 3 epochs, bf16-mixed, seed 42. best/EarlyStopping 모니터는 **val_span_f1(max)**.

### 데이터 출처

학습 자료는 국사편찬위원회 한국사데이터베이스의 색인 XML 사료입니다. 원본 XML은 [공공데이터포털](https://www.data.go.kr/)에서 내려받을 수 있습니다. **본 저장소는 원본 사료를 재배포하지 않으며**, 변환·학습·추론 코드와 학습된 가중치만 공개합니다.

### 인용

```bibtex
@article{yang_placename_ner,
  author  = {양정현},
  title   = {색인 XML 사료를 활용한 조선시대 역사 지명 개체명 인식 모델의 구축},
  note    = {투고 중 — 게재 확정 시 서지·DOI 갱신 예정},
  year    = {2026}
}
```

관련 선행 논문:
- 양정현, 2025, 「딥러닝 기반 한국 고전한문 표점 추론 자동화 모델의 구축과 활용」, 『역사학연구』 100. DOI: 10.37924/JSSW.100.9
- 양정현, 2026, 「해석의 관습 — 한국 고전한문 전용 표점 추론 모델의 개선」, 『민족문화연구』 111, 7–29. DOI: 10.17948/kcs.2026..111.7

### 라이선스

**Apache-2.0.** 기반 모델 [SikuRoBERTa](https://huggingface.co/SIKU-BERT/sikuroberta)가 Apache-2.0으로 배포되므로, 파생 코드와 가중치도 동일 라이선스로 공개합니다. 기반 모델의 저작자 표시와 개조 내역은 [`NOTICE`](NOTICE)에 명시합니다. 학습에 사용한 색인 XML은 [공공데이터포털](https://www.data.go.kr/)의 이용 조건을 따르며, 본 저장소는 원본 사료를 재배포하지 않습니다.

---

## English

A named entity recognition (NER) pipeline that automatically extracts **place-name entities** from Classical Chinese (Literary Sinitic) historical sources of the Joseon dynasty, using indexed XML sources as gold labels. The index tags of the Korean History Database (National Institute of Korean History, 국사편찬위원회) are reused directly as gold labels without any additional manual annotation. A **heterogeneous ensemble** design turns the divergence of indexing conventions across sources into complementary model judgments, securing both recall and precision.

Paper: Yang Jung-Hyun, *Building a Named Entity Recognition Model for Historical Place Names of the Joseon Dynasty Using Indexed XML Sources* (under review; citation and DOI will be updated upon acceptance).
Prior work (punctuation restoration): [korean-classical-chinese-punctuation](https://github.com/yachagye/korean-classical-chinese-punctuation)

### Key features

- **Index tags as gold labels**: the exhaustive index tagging in the XML sources is learned directly as ground truth; untagged positions serve as reliable negative examples.
- **Two-model design**: identical architecture (SikuRoBERTa + token classification) trained on different source compositions, yielding complementary judgment patterns.
- **Entity-level evaluation**: since O labels account for ~98.3% of tokens, evaluation uses **span exact match** P/R/F1 rather than token-level micro F1, which would inflate scores.
- **Staged review**: a reproducible script-based workflow from automatic merging to manual review to gazetteer-assisted review.

### Performance (in-domain validation sets, span exact)

| Model | Training sources | F1 | Precision | Recall | Profile |
| --- | --- | --- | --- | --- | --- |
| Seungjeongwon-Sillok model | Veritable Records of Joseon (조선왕조실록) · Seungjeongwon ilgi (승정원일기) · Gojong/Sunjong Sillok | **0.9049** | 0.8982 | 0.9118 | high recall, lower precision |
| Comprehensive model | the above + Hanguk saryo chongseo, Goryeosa, Goryeosa jeoryo, Samguk sagi, Samguk yusa | **0.8915** | 0.8803 | 0.9030 | lower recall, high precision |

> The two validation sets differ in composition (the Seungjeongwon-Sillok val set contains only chronicles; the comprehensive val set spans all seven source groups). Do not compare the two F1 scores directly.

Training corpus: 1,334 files (0 parsing failures), 293,652,373 Chinese characters in total, 2,075,132 place-name entities, 94,846 unique place names.

### Model weights

Trained weights for both models are released on Hugging Face: [yachagye/korean-historical-placename-ner](https://huggingface.co/yachagye/korean-historical-placename-ner)
Base model: [SIKU-BERT/sikuroberta](https://huggingface.co/SIKU-BERT/sikuroberta) (108M params).

### Pipeline

Conversion → validation → sanity check → main training → inference & merging. Training (steps 2–3) runs on Lightning.ai L40S; the rest runs locally (Windows). The authoritative pipeline specification is [`docs/파이프라인_지침.md`](docs/파이프라인_지침.md) (in Korean); the table below is a summary.

| Step | Script | Input → Output |
| --- | --- | --- |
| 0 Conversion | `0_NER_학습데이터_변환_xml_jsonl.py` | indexed XML → `train/val.jsonl`, conversion stats |
| 1 Validation | `1_NER_학습데이터_검증.py` | `train/val.jsonl` → validation report (format, BIO, split leakage, distribution) |
| 2 Sanity | `2_NER_새니티학습_Lightning.py` | short training run to check for label collapse |
| 3 Training | `3_NER_학습_Lightning.py` | `train/val.jsonl` → `best.ckpt`, logs |
| 4 Inference & merge | `4_NER_추론_txt.py` | source txt + ckpt → single-model / merged / reviewed-merged outputs |
| Utility | `NER_model_loader.py` | ckpt → restores SikuRoBERTa + classifier (shared inference loader) |
| Utility | `xml_메타데이터_분석.py` | exhaustive XML metadata census (tagging rates and densities by source/genre) |
| Utility | `지명DB_정제.py` | gazetteer → cleaned CSV |
| Utility | `지명DB_추론_검수.py` | `[]`-tagged txt + cleaned gazetteer → gazetteer-assisted review output |

#### Data format (JSONL)

Sources are converted to character-level BIO labels (0=O, 1=B-LOC, 2=I-LOC).

```json
{"c": "慶尙道晉州府在京南", "l": [[1],[2],[2],[1],[2],[2],[0],[0],[0]], "n": 9, "id": "...", "src": "...", "p": []}
```

- `c` text (unpunctuated Chinese-character string), `l` character-level labels, `n` length, `id` article ID, `src` source, `p` person-name span metadata (preserved for a future two-class place/person extension).
- Train/val split by article-ID hashing (zero split leakage guaranteed).

#### Heterogeneous ensemble merging

The same text is inferred by both models, and predictions are combined at the entity level.

1. **Union**: spans from both models are all kept as candidates → recovers omissions of the comprehensive model (recall).
2. **Longest-on-overlap**: among overlapping spans, only the widest one is kept (ties go to the comprehensive model) → boundary precision.

Non-Chinese characters (punctuation, symbols) inside a predicted span are treated as boundary errors, and the span is split into contiguous Chinese-character segments (`金遷。○山` → `金遷` + `山`). Residual non-place fragments are left for downstream cleanup.

#### Staged review

Automatic merging (`_통합본`) → manual selection on overlapping cases (`_검수통합본`) → gazetteer longest-match assisted review (`_DB검수`). The gazetteer is never used in training or evaluation; it serves only as an auxiliary resource at the review stage (candidates are proposed, and a human selects — never auto-inserted, due to the risk of homographs).

### Usage

#### Installation

```bash
pip install torch pytorch-lightning transformers
```

#### Inference

```bash
python 4_NER_추론_txt.py
```

After launching, choose a mode (single model / merge / reviewed merge) and an input method (single file / folder traversal), then point to the two model checkpoints to produce the merged output. Detected place names are marked with brackets `[ ]`; lacuna markers remain as `□` in the original text.

Sample output (Sinjeung dongguk yeoji seungnam, Juksan-hyeon):

```
[竹山縣]
東至[陰竹縣]界二十二里，南至[忠淸道][鎭川縣]界二十六里，西至[安城郡]界二十三里…
[天民川]。在縣東十里。源出[巾之]、[鼎陪]兩山，入[驪州][驪江]。
```

#### Training (reproduction)

```bash
# 0. Conversion
python 0_NER_학습데이터_변환_xml_jsonl.py
# 1. Validation
python 1_NER_학습데이터_검증.py
# 2. Sanity (L40S)
python 2_NER_새니티학습_Lightning.py --data_dir data --max_steps 300
# 3. Training (L40S)
python 3_NER_학습_Lightning.py --data_dir data
```

Key hyperparameters: max_length 256 (character-level), batch_size 80 (effective batch 160), lr 2e-5 (AdamW), 3 epochs, bf16-mixed, seed 42. Checkpointing/EarlyStopping monitor: **val_span_f1 (max)**.

### Data sources

The training material consists of indexed XML sources from the Korean History Database, National Institute of Korean History. The original XML files are available from the [Korea Public Data Portal](https://www.data.go.kr/). **This repository does not redistribute the original sources**; only the conversion/training/inference code and trained weights are released.

### Citation

```bibtex
@article{yang_placename_ner,
  author  = {Yang, Jung-Hyun},
  title   = {색인 XML 사료를 활용한 조선시대 역사 지명 개체명 인식 모델의 구축},
  note    = {under review — citation and DOI will be updated upon acceptance},
  year    = {2026}
}
```

Related prior papers:
- Yang Jung-Hyun, 2025, "딥러닝 기반 한국 고전한문 표점 추론 자동화 모델의 구축과 활용" [Building and Applying a Deep-Learning-Based Automatic Punctuation Model for Korean Classical Chinese], *Yeoksahak yeongu* 100. DOI: 10.37924/JSSW.100.9
- Yang Jung-Hyun, 2026, "해석의 관습 — 한국 고전한문 전용 표점 추론 모델의 개선" [Conventions of Interpretation: Improving a Punctuation Model Dedicated to Korean Classical Chinese], *Minjok munhwa yeongu* 111, 7–29. DOI: 10.17948/kcs.2026..111.7

### License

**Apache-2.0.** The base model [SikuRoBERTa](https://huggingface.co/SIKU-BERT/sikuroberta) is distributed under Apache-2.0, and the derived code and weights are released under the same license. Attribution to the base model and a description of modifications are provided in [`NOTICE`](NOTICE). The indexed XML sources used for training are subject to the terms of the [Korea Public Data Portal](https://www.data.go.kr/); this repository does not redistribute the original sources.
