# -*- coding: utf-8 -*-
"""
지명DB 추론 검수 — DB 표제어와 모델 태깅이 어긋나는 자리를 건별로 선택

목적:
  이미 [] 태깅된 추론 출력에서, 정제 지명DB에 등재된 표면형을 본문에서 최장일치로 찾아 기존 태깅과 함께 검토한다. DB 매칭은 학습에서 배제했던 방식이며 여기서는 검수 보조로만 되살린다
  — 동형이의(지명자가 인명·일반어로 쓰인 자리) 때문에 자동 삽입이 아니라 사람이 선택한다.

분기(검수통합본 group_overlaps 구조 차용). 화면에서 모델 태깅은 [], DB 후보는 <>로 표시:
  - 기존 [] 단독(겹침 없음) 또는 []≡DB 완전 일치 → 통과(자동 유지), 묻지 않음.
  - DB 후보가 기존 []에 포함됨(모델이 더 길게 감쌈, 예: [<北倉>津]) → 모델 최장 자동 채택, 묻지 않음.
  - DB 후보 단독(모델 누락) → 선택지 제시(채택 / 0=선택 안 함).
  - 그 외 겹침(모델이 DB에 포함되거나 경계 교차) → 선택지 제시(단일 / 복수 / 0).
  선택은 서로 겹치지 않는 후보의 복수 선택 허용, 0은 그 그룹 전부 버림.
  선택지에서 번호 대신 한자를 직접 입력하면, 그 자리의 연속 한자 구간(구두점으로 끊긴 연속 한자) 안에서 위치를 찾아 span으로 채택한다
  — 후보가 조각나 정답이 선택지에 없을 때 사용(예: 모델 屹紅溫 + DB 屹紅·溫井 → 屹紅溫井 입력; [楊山]古城 → 楊山古城).
  직접 입력이 이미 채택된 좁은 span을 포함하면 좁은 쪽을 제거한다(마지막 선택 우선).

입력:
  - [] 태깅 txt (단일 파일/폴더 순회). 출력물(_DB검수)은 재입력 제외.
  - 정제 지명DB CSV (헤더 name,출전 — 지명DB_정제.py 출력). 표면형 2자 이상.
출력:
  - 선택분을 기존 span과 합쳐 [] 재출력 → 입력_DB검수.txt.
"""

import os
import csv
import tkinter as tk
from tkinter import filedialog

OUT_SUFFIX = "_DB검수"


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
    if kind == "1":
        p = pick_file("검수할 [] 태깅 TXT 선택", [("Text", "*.txt")])
        return [p] if p else []
    root = pick_dir("입력 폴더 선택 (하위 폴더 재귀)")
    if not root:
        return []
    files = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if not name.lower().endswith(".txt"):
                continue
            if os.path.splitext(name)[0].endswith(OUT_SUFFIX):
                continue
            files.append(os.path.join(dirpath, name))
    return sorted(files)


def load_db(csv_path):
    """정제 지명DB(name,출전) 적재. Returns: (name→출전 dict, 최대 표면형 길이)."""
    try:
        text = open(csv_path, encoding="utf-8-sig").read()
    except UnicodeDecodeError:
        text = open(csv_path, encoding="cp949").read()
    name_src = {}
    reader = csv.reader(text.splitlines())
    next(reader, None)  # 헤더 name,출전
    for row in reader:
        if not row or not row[0].strip():
            continue
        name_src[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
    maxlen = max((len(k) for k in name_src), default=0)
    return name_src, maxlen


def parse_bracketed(line):
    """[] 태깅 줄 → (clean 본문, [(s,e),...]). []는 마커로 간주해 제거."""
    clean = []
    spans = []
    start = None
    for ch in line:
        if ch == "[":
            start = len(clean)
        elif ch == "]":
            if start is not None:
                spans.append((start, len(clean)))
                start = None
        else:
            clean.append(ch)
    return "".join(clean), spans


def find_db_candidates(clean, maxlen, name_src):
    """본문 전체에서 DB 최장일치 후보(좌→우, 후보끼리 비겹침). 기존 태깅과의
    겹침은 허용 — 겹치는 경우는 상위에서 선택지로 제시한다."""
    out = []
    n = len(clean)
    i = 0
    while i < n:
        hit = 0
        for L in range(min(maxlen, n - i), 1, -1):  # 2자 이상
            if clean[i:i + L] in name_src:
                hit = L
                break
        if hit:
            out.append((i, i + hit))
            i += hit
        else:
            i += 1
    return out


def group_overlaps(spans):
    """겹치는 span끼리 묶어 그룹 리스트로 반환(검수통합본과 동일 판정)."""
    cand = sorted(set(spans), key=lambda x: x[0])
    groups = []
    if not cand:
        return groups
    group = [cand[0]]
    group_end = cand[0][1]
    for s, e in cand[1:]:
        if s < group_end:
            group.append((s, e))
            group_end = max(group_end, e)
        else:
            groups.append(group)
            group = [(s, e)]
            group_end = e
    groups.append(group)
    return groups


def _label(clean, sp, existing, db, name_src):
    """후보 출처 표시. 모델 태깅·DB·양쪽 구분, DB면 출전 병기."""
    in_m, in_d = sp in existing, sp in db
    src = name_src.get(clean[sp[0]:sp[1]], "")
    if in_m and in_d:
        return f"모델·DB({src})"
    if in_m:
        return "모델"
    return f"DB({src})"


def _ask_group_choice(ordered, clean, gstart, gend):
    """번호(쉼표·공백 복수, 0=선택 안 함) 또는 한자 직접 입력을 받는다.
    한자 입력은 그룹 외곽 본문(clean[gstart:gend])에서 찾아 span으로 만든다.
    번호 span이 서로 겹치거나, 직접 입력이 범위에 없으면 거부하고 다시 묻는다."""
    while True:
        sel = input(f"    선택 (1-{len(ordered)}, 복수는 쉼표, 0=선택 안 함, 또는 한자 직접 입력): ").strip()
        if not sel:
            print("    입력하십시오.")
            continue

        # 한자 직접 입력: 숫자·구분자 외 글자가 있으면 직접 입력으로 간주.
        # 탐색 범위는 해당 자리의 연속 한자 구간(구두점으로 끊긴 연속 한자)까지 확장.
        if any(not (c.isdigit() or c in " ,") for c in sel):
            text = sel.replace(",", "").replace(" ", "")
            cs, ce = _cjk_chunk(clean, gstart, gend)
            off = clean[cs:ce].find(text)
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


def review_db_line(clean, spans, name_src, maxlen, line_no):
    """기존 [] 와 DB 후보를 합쳐 겹침 그룹으로 묶고, 발산 그룹만 선택받는다.
    Returns: 최종 채택 span 리스트."""
    existing = set(spans)
    db = set(find_db_candidates(clean, maxlen, name_src))
    groups = group_overlaps(existing | db)

    chosen = []
    header_done = False
    for group in groups:
        # 모델 단독(또는 모델≡DB 완전 일치) → 통과, 묻지 않음
        if len(group) == 1 and group[0] in existing:
            chosen.append(group[0])
            continue

        # DB 후보가 모델 span에 포함됨(모델이 그룹 전체를 감쌈) → 모델 최장 자동 채택
        enclosing = next((sp for sp in group if sp in existing
                          and all(sp[0] <= o[0] and o[1] <= sp[1] for o in group)), None)
        if enclosing is not None:
            chosen.append(enclosing)
            continue

        # DB 후보 단독(모델 누락) 또는 경계 겹침 → 선택지 제시
        if not header_done:
            print(f"\n[줄 {line_no}]")
            print(f"  추론: {annotate(clean, existing, db)}")
            header_done = True

        ordered = sorted(group, key=lambda x: (x[0], -(x[1] - x[0])))
        gstart = min(s for s, _ in group)
        gend = max(e for _, e in group)
        for i, (s, e) in enumerate(ordered, 1):
            left = ("[" if (s, e) in existing else "") + ("<" if (s, e) in db else "")
            right = (">" if (s, e) in db else "") + ("]" if (s, e) in existing else "")
            print(f"    {i}: {left}{clean[s:e]}{right}  ({_label(clean, (s, e), existing, db, name_src)})")
        chosen.extend(_ask_group_choice(ordered, clean, gstart, gend))

    # 직접 입력이 연속 한자 구간으로 확장돼 이미 채택된 span을 삼키는 경우, 피포함 span 제거
    chosen = [sp for sp in chosen
              if not any(o != sp and o[0] <= sp[0] and sp[1] <= o[1] for o in chosen)]
    return sorted(chosen, key=lambda x: x[0])


def bracketize(clean, spans):
    out = clean
    for s, e in sorted(spans, reverse=True):
        out = out[:s] + "[" + out[s:e] + "]" + out[e:]
    return out


def annotate(clean, model_spans, db_spans):
    """검수 화면 전용: 모델 태깅은 [], DB 후보는 <>로 한 줄에 함께 표시.
    완전 일치는 [<지명>], 부분 겹침은 두 마커가 교차해 경계차가 드러난다.
    저장 출력에는 쓰지 않는다(파일은 bracketize로 [] 만 기록)."""
    n = len(clean)
    pre = [""] * (n + 1)
    post = [""] * (n + 1)
    for s, e in model_spans:
        pre[s] += "["
    for s, e in db_spans:
        pre[s] += "<"
    for s, e in db_spans:
        post[e - 1] += ">"
    for s, e in model_spans:
        post[e - 1] += "]"
    return "".join(pre[i] + clean[i] + post[i] for i in range(n))


def _cjk_chunk(clean, gstart, gend):
    """gstart~gend가 속한 연속 한자 구간의 [s,e)로 좌우 확장(비한자에서 멈춤).
    직접 입력 시 구두점 너머는 묶지 않되, 같은 연속 한자 구간 안에서는 전후를 흡수한다."""
    def is_cjk(ch):
        c = ord(ch)
        return (0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or
                0xF900 <= c <= 0xFADF or 0x20000 <= c <= 0x2A6DF)
    s, e = gstart, gend
    while s > 0 and is_cjk(clean[s - 1]):
        s -= 1
    while e < len(clean) and is_cjk(clean[e]):
        e += 1
    return s, e


def main():
    kind = ask_input_kind()
    inputs = collect_inputs(kind)
    if not inputs:
        print("처리할 입력 TXT가 없습니다.")
        raise SystemExit
    db_path = pick_file("정제 지명DB CSV 선택 (name,출전)", [("CSV", "*.csv")])
    if not db_path:
        raise SystemExit

    name_src, maxlen = load_db(db_path)
    print(f"입력 파일 {len(inputs)}개, DB 표제어 {len(name_src):,}개 (최대 {maxlen}자)")

    for path in inputs:
        base, ext = os.path.splitext(path)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        print(f"\n=== DB 검수: {os.path.basename(path)} "
              f"(발산 사례만 선택, 기존 태깅은 자동 유지) ===")
        rows = []
        for idx, line in enumerate(lines, 1):
            clean, spans = parse_bracketed(line.rstrip("\n"))
            final = review_db_line(clean, spans, name_src, maxlen, idx)
            rows.append(bracketize(clean, final))
        with open(f"{base}{OUT_SUFFIX}{ext}", "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[저장] {base}{OUT_SUFFIX}{ext}")


if __name__ == "__main__":
    main()