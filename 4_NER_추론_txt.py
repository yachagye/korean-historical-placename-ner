# -*- coding: utf-8 -*-
"""
한국 고전한문 지명 NER 추론 ([] 표시형) — 단독 / 이질 앙상블 병합 / 검수 병합 합본

모드(시작 시 콘솔 선택):
  [1] 단독 추론   : ckpt 1개 → 입력_추론.txt
  [2] 병합 추론   : ckpt 2개 → 입력_실록모델.txt / 입력_종합모델.txt / 입력_통합본.txt
  [3] 검수 병합   : ckpt 2개 → 입력_검수통합본.txt
                  겹치는 사례만 콘솔에서 수동 선택(겹침 없는 span은 자동 채택).

입력 방식(시작 시 콘솔 선택, 전 모드 공통):
  [1] 단일 파일
  [2] 폴더 순회(하위 폴더 재귀). 출력물(_추론/_실록모델/_종합모델/_통합본/_검수통합본)은
      입력에서 제외하여 재추론을 막는다. 폴더 모드에서는 모델을 1회만 올려
      전 파일을 처리한 뒤 해제한다(파일마다 재로딩 방지).

병합 규칙(자동, 모드 2):
1. 합집합        : 두 모델의 span을 모두 후보로 둔다.
2. 겹침 시 최장  : 서로 겹치는 span끼리 묶어 가장 넓은 span 하나만 남긴다
                (동일 길이면 종합 모델 우선). 분할 단계가 구두점 흡수를 이미 제거하므로,
                남는 경계 차이는 표제어를 더 온전히 담은 쪽(보통 실록_승정원일기)을 택해 재현을 확보한다.

검수 병합(모드 3):
- 겹치지 않는 span은 합집합으로 자동 채택.
- 겹치는 그룹은 두 모델의 태깅 줄을 문맥으로 제시하고, 그룹 내 후보 span을 번호로 나열해 사용자가 하나를 선택한다(1:1이면 1=실록/2=종합, 다중 겹침이면 후보가 더 나열됨).
- 번호 대신 한자를 직접 입력하면 그 자리 연속 한자 구간 안에서 span을 만들어, 두 모델이 조각내 후보에 없는 개체(于弗山壇 등)를 채택한다.
- 자동 규칙(merge_spans)을 사용자 선택으로 대체하는 트랙. 추론부는 모드 2와 동일.

전제(고정):
- ckpt 구조 동일 (bert + classifier, BIO 3태그), tokenizer/model_name은 ckpt에 존재.
- 추론 입력은 한자만 추출하여 투입(학습 분포 일치), [] 표시는 원문 좌표로 환원.
- 추론 span 내부 비한자는 경계 오류로 보고 연속 한자 구간으로 분할(상세는 split_noncjk).
- 학습/추론 토크나이즈 일관성: 문자 단위 + word_ids 정렬.
- 병합은 VRAM 절약을 위해 한 모델로 전 줄을 추론한 뒤 해제하고 다음 모델을 올린다.
"""

import gc
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog


# ---------------- UI ----------------
# 출력 접미사 (입력 필터·출력 명명에서 공통 사용)
OUT_SUFFIXES = ("_추론", "_실록모델", "_종합모델", "_통합본", "_검수통합본")


def pick_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path or ""


def pick_dir(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path or ""


def ask_mode():
    print("모드 선택:")
    print("  [1] 단독 추론 (ckpt 1개 → _추론.txt)")
    print("  [2] 병합 추론 (ckpt 2개 → _실록모델/_종합모델/_통합본.txt)")
    print("  [3] 검수 병합 (ckpt 2개, 겹침 사례를 수동 선택 → _검수통합본.txt)")
    while True:
        m = input("선택 (1/2/3): ").strip()
        if m in ("1", "2", "3"):
            return m
        print("1, 2, 또는 3을 입력하십시오.")


def ask_input_kind():
    print("입력 방식:")
    print("  [1] 단일 파일")
    print("  [2] 폴더 순회")
    while True:
        k = input("선택 (1/2): ").strip()
        if k in ("1", "2"):
            return k
        print("1 또는 2를 입력하십시오.")


def collect_inputs(kind):
    """입력 방식에 따라 처리할 txt 경로 리스트 반환. 출력물은 제외."""
    if kind == "1":
        p = pick_file("입력 TXT 선택", [("Text", "*.txt")])
        return [p] if p else []

    root = pick_dir("입력 폴더 선택 (하위 폴더 재귀)")
    if not root:
        return []
    files = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if not name.lower().endswith(".txt"):
                continue
            stem = os.path.splitext(name)[0]
            if stem.endswith(OUT_SUFFIXES):  # 출력물 재입력 방지
                continue
            files.append(os.path.join(dirpath, name))
    return sorted(files)


# ---------------- Load / Release ----------------
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    hparams = ckpt["hyper_parameters"]
    model_name = hparams["model_name"]
    dropout_rate = hparams.get("dropout_rate", 0.1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    bert = AutoModel.from_pretrained(model_name)
    classifier = nn.Linear(bert.config.hidden_size, 3)
    dropout = nn.Dropout(dropout_rate)

    state_dict = ckpt["state_dict"]
    bert_sd = {k[5:]: v for k, v in state_dict.items() if k.startswith("bert.")}
    cls_sd = {k[11:]: v for k, v in state_dict.items() if k.startswith("classifier.")}

    bert.load_state_dict(bert_sd, strict=True)
    classifier.load_state_dict(cls_sd, strict=True)

    bert.to(device).eval()
    classifier.to(device).eval()
    dropout.to(device).eval()

    return tokenizer, bert, classifier, dropout


def release_model(bert, classifier, dropout, device):
    del bert, classifier, dropout
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------- Core ----------------
def decide_max_len(tokenizer, model):
    cand = []
    if isinstance(getattr(tokenizer, "model_max_length", None), int):
        cand.append(tokenizer.model_max_length)
    if isinstance(getattr(model.config, "max_position_embeddings", None), int):
        cand.append(model.config.max_position_embeddings)

    cand = [c for c in cand if 16 <= c <= 4096]
    if not cand:
        raise RuntimeError("max_length 자동 결정 불가")

    return min(cand)


def decide_stride(max_len):
    return max(16, max_len // 4)


def extract_spans(text, tokenizer, bert, classifier, dropout, device, max_len, stride):
    """정제(한자만) 텍스트에서 (start, end) span 리스트 복원. 끝 위치는 exclusive."""
    chars = list(text)

    enc = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = bert(input_ids=input_ids, attention_mask=attn_mask)
        logits = classifier(dropout(out.last_hidden_state))
        preds = torch.argmax(logits, dim=-1).cpu()

    spans = []
    n = len(chars)

    for w in range(preds.size(0)):
        word_ids = enc.word_ids(batch_index=w)
        labels = [0] * n
        for ti, ci in enumerate(word_ids):
            if ci is not None and 0 <= ci < n:
                labels[ci] = preds[w, ti].item()

        i = 0
        while i < n:
            if labels[i] == 0:
                i += 1
                continue
            s = i
            i += 1
            while i < n and labels[i] == 2:
                i += 1
            spans.append((s, i))

    # 중첩 청크로 인한 중복 제거 + 겹침 시 긴 span 우선 (단일 모델 내부 정리)
    uniq = {(s, e): (s, e) for s, e in spans}
    spans = sorted(uniq.values(), key=lambda x: (x[0], -(x[1] - x[0])))
    kept = []
    cur = None
    for s, e in spans:
        if cur is None:
            cur = (s, e)
        elif s >= cur[1]:
            kept.append(cur)
            cur = (s, e)
        elif (e - s) > (cur[1] - cur[0]):
            cur = (s, e)
    if cur:
        kept.append(cur)

    return kept


# ---------------- Merge (자동, 모드 2) ----------------
def group_overlaps(spans):
    """겹치는 span끼리 묶어 그룹 리스트로 반환. 각 그룹은 [(s,e),...].
    시작 위치로 정렬 후 직전 그룹의 끝과 겹치면 같은 그룹.
    merge_spans(자동)와 review_merge_line(수동)이 공유."""
    cand = sorted(set(spans), key=lambda x: x[0])
    groups = []
    if not cand:
        return groups
    group = [cand[0]]
    group_end = cand[0][1]
    for s, e in cand[1:]:
        if s < group_end:  # 직전 그룹과 겹침
            group.append((s, e))
            group_end = max(group_end, e)
        else:
            groups.append(group)
            group = [(s, e)]
            group_end = e
    groups.append(group)
    return groups


def merge_spans(spans_sillok, spans_total):
    """합집합 + 겹침 시 더 넓은 span 채택(동일 길이면 종합 우선).
    두 모델 span을 한데 모아 겹치는 것끼리 묶어 각 그룹에서 가장 넓은 span만 남긴다."""
    total_set = set(spans_total)
    merged = []
    for group in group_overlaps(list(spans_sillok) + list(spans_total)):
        merged.append(_pick_widest(group, total_set))
    return sorted(merged, key=lambda x: x[0])


def _pick_widest(group, total_set):
    """그룹에서 가장 넓은 span. 동일 길이면 종합 모델 것 우선."""
    return max(group, key=lambda x: (x[1] - x[0], (x[0], x[1]) in total_set))


# ---------------- Review Merge (수동, 모드 3) ----------------
def ask_group_choice(ordered, line, gstart, gend):
    """번호(쉼표·공백 복수, 0=선택 안 함) 또는 한자 직접 입력을 받는다.
    한자 입력은 그룹 자리의 연속 한자 구간(구두점으로 끊긴 연속 한자) 안에서 찾아 span으로 만든다 — 두 모델이 조각낸 개체를 온전히 채택할 때 사용
    (예: 실록 于弗 + 종합 山壇 → 于弗山壇 입력). 번호 span이 서로 겹치거나 직접 입력이 범위에 없으면 거부하고 다시 묻는다."""
    while True:
        sel = input(f"    선택 (1-{len(ordered)}, 복수는 쉼표, 0=선택 안 함, 또는 한자 직접 입력): ").strip()
        if not sel:
            print("    입력하십시오.")
            continue

        # 한자 직접 입력: 숫자·구분자 외 글자가 있으면 직접 입력으로 간주.
        if any(not (c.isdigit() or c in " ,") for c in sel):
            text = sel.replace(",", "").replace(" ", "")
            cs, ce = cjk_chunk(line, gstart, gend)
            off = line[cs:ce].find(text)
            if off < 0:
                print("    이 자리의 연속 한자 구간에서 찾을 수 없습니다(구두점 너머는 묶을 수 없음). 다시 입력하십시오.")
                continue
            s = cs + off
            return [(s, s + len(text))]

        # 번호 선택
        nums = [t for t in sel.replace(",", " ").split() if t]
        if not all(t.isdigit() and 0 <= int(t) <= len(ordered) for t in nums):
            print(f"    0-{len(ordered)} 중에서 입력하십시오.")
            continue
        if "0" in nums:
            if len(set(nums)) > 1:
                print("    0(선택 안 함)은 단독으로만 입력하십시오.")
                continue
            return []
        picked = sorted((ordered[int(t) - 1] for t in dict.fromkeys(nums)),
                        key=lambda x: x[0])
        if any(picked[i][1] > picked[i + 1][0] for i in range(len(picked) - 1)):
            print("    서로 겹치는 후보는 함께 선택할 수 없습니다.")
            continue
        return picked


def review_merge_line(line, spans_s, spans_t, line_no):
    """겹치는 그룹만 사용자에게 문맥과 함께 제시하고 후보 중 하나를 선택받는다.
    겹침 없는 단독 span은 자동 채택. Returns: 선택된 span 리스트.
    line_no는 겹침이 있는 줄에서만 헤더로 출력(겹침 없으면 조용히 자동 채택)."""
    src = {}
    for sp in spans_t:
        src[sp] = "종합"
    for sp in spans_s:
        src[sp] = "실록·종합" if sp in spans_t else "실록"

    groups = group_overlaps(list(spans_s) + list(spans_t))
    header_done = False

    chosen = []
    for group in groups:
        if len(group) == 1:        # 겹침 아님 → 자동 채택
            chosen.append(group[0])
            continue

        if not header_done:        # 겹침 있는 줄에서만 두 모델 태깅 줄 제시
            print(f"\n[줄 {line_no}]")
            print(f"  실록: {bracketize(line, spans_s)}")
            print(f"  종합: {bracketize(line, spans_t)}")
            header_done = True

        ordered = sorted(group, key=lambda x: (x[0], -(x[1] - x[0])))
        gstart = min(s for s, _ in group)
        gend = max(e for _, e in group)
        for i, (s, e) in enumerate(ordered, 1):
            print(f"    {i}: [{line[s:e]}]  ({src.get((s, e), '?')})")
        chosen.extend(ask_group_choice(ordered, line, gstart, gend))

    # 직접 입력이 연속 한자 구간으로 확장돼 이미 채택된 span을 삼키는 경우, 피포함 span 제거
    chosen = [sp for sp in chosen
              if not any(o != sp and o[0] <= sp[0] and sp[1] <= o[1] for o in chosen)]
    return sorted(chosen, key=lambda x: x[0])


# ---------------- CJK Filter ----------------
def is_chinese_char(char):
    """한자 판별 (학습 파이프라인 0번과 동일 기준)"""
    code = ord(char)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or
            0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F or
            0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or
            0x2CEB0 <= code <= 0x2EBEF or 0x30000 <= code <= 0x3134F or
            0x31350 <= code <= 0x323AF or 0x2F800 <= code <= 0x2FA1F or
            0xF900 <= code <= 0xFADF or 0x2F00 <= code <= 0x2FDF or
            0x2E80 <= code <= 0x2EFF)


def cjk_chunk(text, gstart, gend):
    """gstart~gend가 속한 연속 한자 구간의 [s,e)로 좌우 확장(비한자에서 멈춤).
    직접 입력 시 구두점 너머는 묶지 않되, 같은 연속 한자 구간 안에서는 전후를 흡수한다."""
    s, e = gstart, gend
    while s > 0 and is_chinese_char(text[s - 1]):
        s -= 1
    while e < len(text) and is_chinese_char(text[e]):
        e += 1
    return s, e


def extract_cjk(text):
    """한자만 추출. Returns: (정제 텍스트, 정제→원문 위치 리스트)"""
    clean = []
    pos = []
    for i, ch in enumerate(text):
        if is_chinese_char(ch):
            clean.append(ch)
            pos.append(i)
    return ''.join(clean), pos


def split_noncjk(text, spans):
    """span 내부에 비한자(구두점·기호·공백 등)가 끼면 연속 한자 구간별로 분할.
    예: 金遷。○山 → [金遷], [山]. 지명 내부에는 비한자 정답이 없다는 전제(학습 분포).
    분할 결과 1글자 한자 구간도 그대로 유지(다운스트림 정규화에서 거름)."""
    out = []
    for s, e in spans:
        run_start = None
        for i in range(s, e):
            if is_chinese_char(text[i]):
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    out.append((run_start, i))
                    run_start = None
        if run_start is not None:
            out.append((run_start, e))
    return out


def bracketize(text, spans):
    out = text
    for s, e in sorted(spans, reverse=True):
        out = out[:s] + "[" + out[s:e] + "]" + out[e:]
    return out


# ---------------- Inference over all lines (one model) ----------------
def infer_all_lines(lines, tokenizer, bert, classifier, dropout, device, desc):
    """각 줄을 정제→추론→원문좌표 환원. Returns: [(원문줄, [원문좌표 span])]"""
    max_len = decide_max_len(tokenizer, bert)
    stride = decide_stride(max_len)

    results = []
    for line in tqdm(lines, desc=desc):
        s = line.rstrip("\n")
        if not s.strip():
            results.append((s, []))
            continue

        clean, pos = extract_cjk(s)
        if not clean:
            results.append((s, []))
            continue

        spans = extract_spans(clean, tokenizer, bert, classifier, dropout,
                              device, max_len, stride)
        spans = [(pos[a], pos[b - 1] + 1) for a, b in spans]  # 정제→원문 좌표
        spans = split_noncjk(s, spans)  # span 내부 비한자에서 분할
        results.append((s, spans))

    return results


def run_model_over_files(ckpt_path, file_lines, device, desc):
    """ckpt 1개를 올려 여러 파일을 모두 추론한 뒤 해제(파일마다 재로딩 방지).
    file_lines: {경로: [원문 줄,...]}.  Returns: {경로: [(원문줄, [span])]}"""
    tok, bert, clf, drop = load_model(ckpt_path, device)
    out = {}
    for path, lines in file_lines.items():
        name = os.path.basename(path)
        out[path] = infer_all_lines(lines, tok, bert, clf, drop, device,
                                    f"{desc} · {name}")
    release_model(bert, clf, drop, device)
    del tok
    return out


# ---------------- Main ----------------
def main():
    mode = ask_mode()
    kind = ask_input_kind()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = collect_inputs(kind)
    if not inputs:
        print("처리할 입력 TXT가 없습니다.")
        raise SystemExit
    print(f"입력 파일 {len(inputs)}개")

    # 파일별 원문 줄 적재 (한 번만 읽어 두 모델이 공유)
    file_lines = {}
    for p in inputs:
        with open(p, "r", encoding="utf-8") as f:
            file_lines[p] = f.readlines()

    if mode == "1":
        ckpt = pick_file("ckpt 선택 (.ckpt)", [("Checkpoint", "*.ckpt")])
        if not ckpt:
            raise SystemExit

        res = run_model_over_files(ckpt, file_lines, device, "단독 추론")
        for path, lines_spans in res.items():
            base, ext = os.path.splitext(path)
            rows = [bracketize(line, sp) for line, sp in lines_spans]
            with open(f"{base}_추론{ext}", "w", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
            print(f"[저장] {base}_추론{ext}")

    elif mode == "2":
        ckpt_sillok = pick_file("① 승정원일기_실록 모델 ckpt (고재현)",
                                [("Checkpoint", "*.ckpt")])
        if not ckpt_sillok:
            raise SystemExit
        ckpt_total = pick_file("② 종합 모델 ckpt (고정밀)",
                               [("Checkpoint", "*.ckpt")])
        if not ckpt_total:
            raise SystemExit

        # 모델별 1회 로딩으로 전 파일 추론 (VRAM 절약, 순차 교체)
        res_sillok = run_model_over_files(ckpt_sillok, file_lines, device,
                                          "실록모델(고재현)")
        res_total = run_model_over_files(ckpt_total, file_lines, device,
                                         "종합모델(고정밀)")

        for path in inputs:
            base, ext = os.path.splitext(path)
            out_s, out_t, out_m = [], [], []
            for (line, sp_s), (_, sp_t) in zip(res_sillok[path], res_total[path]):
                out_s.append(bracketize(line, sp_s))
                out_t.append(bracketize(line, sp_t))
                out_m.append(bracketize(line, merge_spans(sp_s, sp_t)))

            for suffix, rows in [("_실록모델", out_s), ("_종합모델", out_t),
                                 ("_통합본", out_m)]:
                with open(f"{base}{suffix}{ext}", "w", encoding="utf-8") as f:
                    f.write("\n".join(rows) + "\n")
            print(f"[저장] {os.path.basename(base)} → 실록모델/종합모델/통합본")

    else:  # mode == "3" : 검수 병합 (겹침 사례 수동 선택)
        ckpt_sillok = pick_file("① 승정원일기_실록 모델 ckpt (고재현)",
                                [("Checkpoint", "*.ckpt")])
        if not ckpt_sillok:
            raise SystemExit
        ckpt_total = pick_file("② 종합 모델 ckpt (고정밀)",
                               [("Checkpoint", "*.ckpt")])
        if not ckpt_total:
            raise SystemExit

        # 추론은 모드 2와 동일(모델 1회 로딩). 추론 완료 후 파일별 검수 선택.
        res_sillok = run_model_over_files(ckpt_sillok, file_lines, device,
                                          "실록모델(고재현)")
        res_total = run_model_over_files(ckpt_total, file_lines, device,
                                         "종합모델(고정밀)")

        for path in inputs:
            base, ext = os.path.splitext(path)
            print(f"\n=== 검수 병합: {os.path.basename(path)} "
                  f"(겹침 사례만 선택, 나머지는 자동 채택) ===")
            rows = []
            for idx, ((line, sp_s), (_, sp_t)) in enumerate(
                    zip(res_sillok[path], res_total[path]), 1):
                chosen = review_merge_line(line, sp_s, sp_t, idx)
                rows.append(bracketize(line, chosen))
            with open(f"{base}_검수통합본{ext}", "w", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
            print(f"[저장] {base}_검수통합본{ext}")


if __name__ == "__main__":
    main()