# -*- coding: utf-8 -*-
"""
1번: NER 학습데이터 독립 검증 (XML 변환 출력 대상)

변환기(0번) 통계와 별개로, 출력 JSONL 자체를 학습 투입 전에 검사한다.

검증 항목:
1. 형식 정합: c/l/n 키 존재, n == len(c) == len(l), 라벨 값 {0,1,2}, 한자 외 문자 잔존
2. BIO 전이 이상: O→I 시작(B 없는 I), 시퀀스 첫 글자 I
3. 분할 누수: 동일 기사 ID가 train과 val 양쪽에 출현하는지
4. 'p'(이름) 메타필드: 좌표 범위 정합, 지명 라벨과의 겹침 집계
5. 분포: 출처별 청크·개체 수, 라벨 비율(B/I/O), 지명 길이 분포, 청크 길이 분포
6. 육안 검수 표본: 무작위 청크의 라벨 복원 출력

출력: 검증_보고.txt (입력 폴더에 저장)
"""

import json
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm

SAMPLE_COUNT = 30  # 육안 검수 표본 수
RNG_SEED = 42


def is_chinese_char(char):
    """한자 판별 (0번 변환기와 동일 기준)"""
    code = ord(char)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or
            0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F or
            0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or
            0x2CEB0 <= code <= 0x2EBEF or 0x30000 <= code <= 0x3134F or
            0x31350 <= code <= 0x323AF or 0x2F800 <= code <= 0x2FA1F or
            0xF900 <= code <= 0xFADF or 0x2F00 <= code <= 0x2FDF or
            0x2E80 <= code <= 0x2EFF)


def extract_entities(text, labels):
    """라벨에서 (시작, 끝) 개체 구간 복원"""
    entities = []
    start = None
    for i, lab in enumerate(labels):
        v = lab[0]
        if v == 1:
            if start is not None:
                entities.append((start, i))
            start = i
        elif v == 0:
            if start is not None:
                entities.append((start, i))
                start = None
    if start is not None:
        entities.append((start, len(labels)))
    return entities


def src_group(src):
    """파일명에서 출처 식별 (0번 변환 스크립트와 동일하게 유지할 것)"""
    s = str(src).lower()
    if 'sjw' in s:
        return '승정원일기'
    if s.startswith('kr'):
        return '고려사'
    if s.startswith('kj'):
        return '고려사절요'
    if s.startswith('sg'):
        return '삼국사기'
    if s.startswith('sy'):
        return '삼국유사'
    if s.startswith('sa'):
        return '한국사료총서'
    if s.startswith(('2nd_w', 'w')):
        return '실록'
    return f'기타({src[:8]})'


def validate_file(path, split_name, report):
    stats = {
        'chunks': 0,
        'format_errors': [],
        'label_value_errors': 0,
        'bio_errors': 0,
        'bio_error_samples': [],
        'noncjk_chars': Counter(),
        'p_range_errors': 0,
        'p_geo_overlap': 0,
        'p_total': 0,
        'label_counts': Counter(),
        'entity_len': Counter(),
        'chunk_len': Counter(),
        'src_chunks': Counter(),
        'src_entities': Counter(),
        'article_ids': set(),
        'samples': [],
    }
    rng = random.Random(RNG_SEED)

    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(tqdm(f, desc=f"{split_name} 검사"), 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                stats['format_errors'].append(f"{line_no}행: JSON 파싱 실패")
                continue

            stats['chunks'] += 1

            # 1. 형식 정합
            if not all(k in r for k in ('c', 'l', 'n')):
                stats['format_errors'].append(f"{line_no}행: 필수 키 누락")
                continue
            c, l, n = r['c'], r['l'], r['n']
            if not (n == len(c) == len(l)):
                stats['format_errors'].append(
                    f"{line_no}행: 길이 불일치 n={n}, len(c)={len(c)}, len(l)={len(l)}")
                continue

            ok = True
            for lab in l:
                if not (isinstance(lab, list) and len(lab) == 1 and lab[0] in (0, 1, 2)):
                    stats['label_value_errors'] += 1
                    ok = False
                    break
            if not ok:
                continue

            for ch in c:
                if not is_chinese_char(ch):
                    stats['noncjk_chars'][ch] += 1

            # 2. BIO 전이 이상: I가 B 또는 I의 후속이 아닌 위치에 출현
            prev = 0
            for i, lab in enumerate(l):
                v = lab[0]
                if v == 2 and prev == 0:
                    stats['bio_errors'] += 1
                    if len(stats['bio_error_samples']) < 10:
                        s = max(0, i - 10)
                        stats['bio_error_samples'].append(
                            f"{line_no}행 {i}위치: ...{c[s:i + 5]}...")
                    break
                prev = v

            # 3. 분할 누수용 기사 ID 수집
            if 'id' in r:
                stats['article_ids'].add(r['id'])

            # 4. p 메타필드 정합 및 지명 라벨 겹침
            geo_positions = set()
            for s_, e_ in extract_entities(c, l):
                geo_positions.update(range(s_, e_))
                stats['entity_len'][e_ - s_] += 1

            for ns, ne in r.get('p', []):
                stats['p_total'] += 1
                if not (0 <= ns < ne <= n):
                    stats['p_range_errors'] += 1
                    continue
                if any(i in geo_positions for i in range(ns, ne)):
                    stats['p_geo_overlap'] += 1

            # 5. 분포
            for lab in l:
                stats['label_counts'][lab[0]] += 1
            stats['chunk_len'][min(n // 32 * 32, 256)] += 1
            src = r.get('src', '미상')
            g = src_group(src)
            stats['src_chunks'][g] += 1
            stats['src_entities'][g] += len(extract_entities(c, l))

            # 6. 육안 검수 표본 (저수지 샘플링)
            if len(stats['samples']) < SAMPLE_COUNT:
                stats['samples'].append(r)
            else:
                j = rng.randint(0, stats['chunks'] - 1)
                if j < SAMPLE_COUNT:
                    stats['samples'][j] = r

    # 보고 작성
    report.append(f"\n{'=' * 70}\n[{split_name}] {path.name}\n{'=' * 70}")
    report.append(f"청크 수: {stats['chunks']:,}개")
    report.append(f"형식 오류: {len(stats['format_errors'])}건")
    for e in stats['format_errors'][:10]:
        report.append(f"  {e}")
    report.append(f"라벨 값 오류: {stats['label_value_errors']}건")
    report.append(f"BIO 전이 이상(B 없는 I): {stats['bio_errors']}건")
    for s in stats['bio_error_samples']:
        report.append(f"  {s}")

    if stats['noncjk_chars']:
        report.append(f"한자 외 문자 잔존: {sum(stats['noncjk_chars'].values()):,}개")
        for ch, cnt in stats['noncjk_chars'].most_common(10):
            report.append(f"  {ch!r} (U+{ord(ch):04X}): {cnt:,}회")
    else:
        report.append("한자 외 문자 잔존: 없음")

    report.append(f"p 메타필드: 총 {stats['p_total']:,}건, 좌표 오류 {stats['p_range_errors']}건, "
                  f"지명 라벨과 겹침 {stats['p_geo_overlap']:,}건")

    total_labels = sum(stats['label_counts'].values())
    if total_labels:
        report.append("라벨 분포:")
        for v, name in ((0, 'O'), (1, 'B-LOC'), (2, 'I-LOC')):
            cnt = stats['label_counts'][v]
            report.append(f"  {name}: {cnt:,} ({cnt / total_labels * 100:.2f}%)")

    report.append("지명 길이 분포 (상위 10):")
    for length, cnt in sorted(stats['entity_len'].items())[:10]:
        report.append(f"  {length}자: {cnt:,}개")
    long_entities = sum(c for ln, c in stats['entity_len'].items() if ln > 10)
    report.append(f"  10자 초과: {long_entities:,}개")

    report.append("청크 길이 분포 (32자 구간):")
    for bucket in sorted(stats['chunk_len']):
        report.append(f"  {bucket}~{bucket + 31}자: {stats['chunk_len'][bucket]:,}개")

    report.append("출처별 분포:")
    for g in sorted(stats['src_chunks']):
        report.append(f"  {g}: 청크 {stats['src_chunks'][g]:,}개, "
                      f"개체 {stats['src_entities'][g]:,}개")

    return stats


def render_samples(samples, report):
    report.append(f"\n{'=' * 70}\n[육안 검수 표본]\n{'=' * 70}")
    for r in samples:
        ents = [r['c'][s:e] for s, e in extract_entities(r['c'], r['l'])]
        names = [r['c'][s:e] for s, e in r.get('p', [])]
        report.append(f"\nid={r.get('id', '미상')} (src={r.get('src', '미상')}, n={r['n']})")
        report.append(f"  본문: {r['c'][:120]}{'...' if r['n'] > 120 else ''}")
        report.append(f"  지명: {ents}")
        report.append(f"  이름(메타): {names[:10]}{' ...' if len(names) > 10 else ''}")


def main():
    import tkinter as tk
    from tkinter import filedialog

    print("=" * 70)
    print("1번: NER 학습데이터 독립 검증")
    print("=" * 70)

    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(title="train.jsonl / val.jsonl이 있는 폴더 선택")
    root.destroy()
    if not data_dir:
        print("폴더가 선택되지 않았습니다.")
        return

    data_path = Path(data_dir)
    train_file = data_path / 'train.jsonl'
    val_file = data_path / 'val.jsonl'
    if not train_file.exists() or not val_file.exists():
        print("train.jsonl 또는 val.jsonl이 없습니다.")
        return

    report = []
    train_stats = validate_file(train_file, 'train', report)
    val_stats = validate_file(val_file, 'val', report)

    # 분할 누수 검사
    leaked = train_stats['article_ids'] & val_stats['article_ids']
    report.append(f"\n{'=' * 70}\n[분할 누수 검사]\n{'=' * 70}")
    report.append(f"train 기사 ID: {len(train_stats['article_ids']):,}개")
    report.append(f"val 기사 ID: {len(val_stats['article_ids']):,}개")
    report.append(f"양쪽 출현(누수): {len(leaked)}개")
    for aid in list(leaked)[:10]:
        report.append(f"  {aid}")

    ratio = len(val_stats['article_ids']) / max(
        1, len(train_stats['article_ids']) + len(val_stats['article_ids']))
    report.append(f"val 기사 비율: {ratio * 100:.2f}% (목표 10% 내외)")

    render_samples(train_stats['samples'], report)

    text = '\n'.join(report)
    print('\n' + text)

    out_file = data_path / '검증_보고.txt'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f"\n검증 보고 저장: {out_file}")

    # 종합 판정
    critical = (len(train_stats['format_errors']) + train_stats['label_value_errors'] +
                train_stats['bio_errors'] + len(leaked) +
                len(val_stats['format_errors']) + val_stats['label_value_errors'] +
                val_stats['bio_errors'])
    print(f"\n{'✅ 치명 결함 없음 — 학습 투입 가능' if critical == 0 else f'⚠️ 치명 결함 {critical}건 — 위 보고 확인 필요'}")


if __name__ == "__main__":
    main()