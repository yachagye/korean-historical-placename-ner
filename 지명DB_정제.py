# -*- coding: utf-8 -*-
"""
지명DB 정제 (반복 적용용)

지명DB는 출처 통합으로 계속 갱신되므로, 매 정제마다 결과를 감사할 수 있도록
제외 사유별 집계·표본과 잔존 비한자 문자 분포를 보고서로 남긴다.
새 출처에서 미지의 잡음 패턴이 들어오면 보고서의 [비한자 문자 분포]와
[제외 표본]에 드러나며, 이를 근거로 정제 규칙(EXCLUDE_MARKS 등)을 갱신한다.

정제 규칙:
1. 병기 분리: 쉼표(,·、)로 구분된 표기는 각각 등재 (예: 英城,營城 → 2항목)
2. 결손·생략 표지(? ■ ○ - ㅡ) 포함 부분은 복원 불가로 제외
3. 각 부분은 선두 한자 연쇄만 취함 (한글 접미·괄호 주기 제거: 函岩산 → 函岩)
   ※ 괄호 내용을 벗기지 않고 선두만 취하는 이유: 內[土同]里 → 內里 같은
     허위 항목 생성을 차단 (자형 조합 표기와 지역 주기를 기계적으로
     구별할 수 없으므로 보수적으로 처리)
4. 결과가 2자 미만이면 제외 (1자 지명은 사전 매칭 대상이 아님)
5. 거리·방위 파편 제외: 방위(東西南北)·한자 숫자·里로만 구성된 표제어
   (東北·十五里·東二·東七里 등) + 행정단위(郡縣府州)+방위 2자(郡東·縣西).
   집합 밖 글자가 하나라도 있으면 통과 — 南山·東萊·東面·江東·三田渡 등 보존.
6. 중복 제거, 정렬, utf-8-sig(BOM) 저장 — Windows 호환

입력: 원본 CSV 2필드(출전, 지명). 지명에만 정제 규칙 적용, 출전은 보존.

출력:
- 정제본 CSV (헤더 name,출전 — 같은 표면형이 여러 출전을 가지면 ·로 묶음)
- 정제_통계.txt (정제본과 같은 폴더)
"""

import csv
from pathlib import Path
from collections import Counter
import tkinter as tk
from tkinter import filedialog

EXCLUDE_MARKS = set('?■○-ㅡ')  # 결손·생략 표지 (신규 발견 시 여기에 추가)
NUMERALS = set('一二三四五六七八九十百千萬')  # 한자 숫자
BEARINGS = set('東西南北')                    # 방위 한자
FILLER = NUMERALS | BEARINGS | {'里'}         # 거리·방위 파편 판별: 이 집합으로만 구성되면 제외
ADMIN = set('郡縣府州')                        # 행정단위 기준 글자(郡東·縣西 등 방위 서술용; 화이트리스트라 江東·山東 보존)
SAMPLE_LIMIT = 30


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


def leading_cjk(s):
    """선두 한자 연쇄 추출"""
    lead = []
    for ch in s:
        if is_chinese_char(ch):
            lead.append(ch)
        else:
            break
    return ''.join(lead)


def clean_row(raw, stats):
    """한 행 → 정제 항목 리스트. 제외 사유와 잔존 문자를 집계."""
    items = []
    parts = raw.replace('、', ',').split(',')
    if len(parts) > 1:
        stats['split_rows'] += 1

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if any(m in part for m in EXCLUDE_MARKS):
            stats['excluded_mark'] += 1
            if len(stats['sample_mark']) < SAMPLE_LIMIT:
                stats['sample_mark'].append(part)
            continue

        lead = leading_cjk(part)
        if len(lead) < 2:
            stats['excluded_short'] += 1
            if len(stats['sample_short']) < SAMPLE_LIMIT:
                stats['sample_short'].append(part)
            continue

        # 거리·방위 파편 제외: 방위·숫자·里로만 구성 (東北·十五里·東二·二十 등)
        if all(c in FILLER for c in lead):
            stats['excluded_fragment'] += 1
            if len(stats['sample_fragment']) < SAMPLE_LIMIT:
                stats['sample_fragment'].append(lead)
            continue

        # 행정단위+방위 제외: 郡東·縣西 등 '○○의 방위' 서술 (江東·山東 보호 위해 ADMIN 화이트리스트)
        if len(lead) == 2 and lead[0] in ADMIN and lead[1] in BEARINGS:
            stats['excluded_fragment'] += 1
            if len(stats['sample_fragment']) < SAMPLE_LIMIT:
                stats['sample_fragment'].append(lead)
            continue

        if lead != part:
            stats['trimmed'] += 1
            if len(stats['sample_trimmed']) < SAMPLE_LIMIT:
                stats['sample_trimmed'].append(f"{part} → {lead}")
            for ch in part[len(lead):]:
                if not is_chinese_char(ch):
                    stats['residual_chars'][ch] += 1

        items.append(lead)
    return items


def main():
    root = tk.Tk()
    root.withdraw()
    in_path = filedialog.askopenfilename(
        title="지명DB 원본 CSV 선택", filetypes=[("CSV", "*.csv")])
    if not in_path:
        print("파일이 선택되지 않았습니다.")
        return
    out_path = filedialog.asksaveasfilename(
        title="정제본 저장", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
    root.destroy()
    if not out_path:
        print("저장 경로가 지정되지 않았습니다.")
        return

    try:
        text = open(in_path, encoding='utf-8-sig').read()
    except UnicodeDecodeError:
        text = open(in_path, encoding='cp949').read()

    # 원본 2필드(출전, 지명). 지명에 병기 쉼표가 비인용으로 칼럼을 쪼갰을 수 있어
    # 2번째 칼럼 이후를 다시 쉼표로 합쳐 raw 지명으로 둔다(clean_row가 분리).
    rows = []
    for row in csv.reader(text.splitlines()):
        if len(row) < 2 or not row[1].strip():
            continue
        rows.append((row[0].strip(), ','.join(c.strip() for c in row[1:])))
    if rows and (rows[0][0] == '출전' or rows[0][1] == '지명'):
        rows = rows[1:]  # 헤더 제거

    stats = {
        'split_rows': 0,
        'excluded_mark': 0,
        'excluded_short': 0,
        'excluded_fragment': 0,
        'trimmed': 0,
        'residual_chars': Counter(),
        'sample_mark': [],
        'sample_short': [],
        'sample_fragment': [],
        'sample_trimmed': [],
    }

    cleaned = {}  # 표면형 → {출전, ...}
    fully_excluded = 0
    for i, (src, raw) in enumerate(rows):
        items = clean_row(raw, stats)
        if items:
            for it in items:
                cleaned.setdefault(it, set()).add(src)
        else:
            fully_excluded += 1
        if (i + 1) % 50000 == 0:
            print(f"[{i + 1}/{len(rows)}]")

    out_path = Path(out_path)
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', '출전'])
        for t in sorted(cleaned):
            w.writerow([t, '·'.join(sorted(cleaned[t]))])

    # 감사 보고
    lens = Counter(len(t) for t in cleaned)
    L = []
    L.append("=" * 60)
    L.append("지명DB 정제 통계")
    L.append("=" * 60)
    L.append(f"입력: {in_path}")
    L.append(f"원본 행: {len(rows):,}")
    L.append(f"정제 후 고유 항목: {len(cleaned):,} → {out_path.name}")
    L.append(f"완전 제외 행: {fully_excluded:,}")
    L.append(f"\n[처리 내역]")
    L.append(f"병기 분리된 행: {stats['split_rows']:,}")
    L.append(f"결손·생략 표지로 제외: {stats['excluded_mark']:,}건")
    L.append(f"선두 한자 2자 미만으로 제외: {stats['excluded_short']:,}건")
    L.append(f"거리·방위 파편(방위·숫자·里) 제외: {stats['excluded_fragment']:,}건")
    L.append(f"선두 한자만 취한 항목(접미 제거): {stats['trimmed']:,}건")
    L.append(f"\n[비한자 문자 분포 — 신규 잡음 패턴 감지용]")
    for ch, cnt in stats['residual_chars'].most_common(30):
        L.append(f"  {ch!r} (U+{ord(ch):04X}): {cnt:,}회")
    L.append(f"\n[정제 후 길이 분포]")
    for ln in sorted(lens):
        L.append(f"  {ln}자: {lens[ln]:,}")
    for title, key in (("제외 표본: 결손·생략 표지", 'sample_mark'),
                       ("제외 표본: 선두 한자 부족", 'sample_short'),
                       ("제외 표본: 거리·방위 파편", 'sample_fragment'),
                       ("접미 제거 표본", 'sample_trimmed')):
        L.append(f"\n[{title}]")
        for s in stats[key]:
            L.append(f"  {s}")

    report = '\n'.join(L)
    print('\n' + report)

    stats_path = out_path.parent / '정제_통계.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f"\n통계 저장: {stats_path}")


if __name__ == "__main__":
    main()