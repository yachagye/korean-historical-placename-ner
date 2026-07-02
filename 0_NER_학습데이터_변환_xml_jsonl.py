# -*- coding: utf-8 -*-
"""
국사편찬위원회 색인 XML → 지명 NER 학습데이터(JSONL) 변환

신규 문헌 6종 표본 점검 반영: 고려사·고려사절요·삼국사기·삼국유사·한국사료총서:
1. 편찬 부속물 제외: 기사 id의 $ 표지(간행사·범례·해제 섹션) 필터
2. 비한문 기사 가드: 한글·라틴 실질문자 비율 > 5%면 기사 제외 (번역문·영문 사료 차단)
3. 잔존 한글·라틴 연쇄는 결락과 동일하게 분절 경계(SENTINEL) 처리 — 허위 인접 차단
4. 출처 식별 확장: kr(고려사)/kj(절요)/sg(삼국사기)/sy(삼국유사)/sa(사료총서)
5. 관례 점검 추가: 시설·국호 점검 목록의 출처별 태깅률 집계 (색인 관례 충돌 정량화)
6. 출처별 지명 밀도 보고 (태깅 희소 문헌 탐지)

전체 변환 통계 및 미등록 태그 표본 판정 반영:
1. 태그 처리 정책 전환: 미등록 태그 기본값을 '포함'에서 '제외+보고'로 변경
   - 포함: quotation(본문 인용), postScript(사론), proofreading(이문, 텍스트는 원문)
   - 제외: explanation(편찬 주기), noteTitle(교감 교정안), reference/pTitle(개수 전거),
           illustration/caption/image(그림), link(목차), name(개수자 서명)
2. annotation은 type별 분기: 원주만 인라인, 교감주 등 그 외 유형은 제외
3. 결락 분절: missing·newChar 요소 및 원문 □(U+25A1)를 분절 경계로 처리
   - 허위 인접 문맥 차단, 결락에 걸친 개체는 라벨 제외(허위 개체 차단)
4. 이스케이프된 문자 참조(&#x...;) 복원: 확장 영역 한자 회복
5. index 내부 중첩 구조 수용: 내부 교감주는 제외하되 그 앞뒤 텍스트는 개체 구간에 포함

설계 원칙:
- 단일 패스, 기사 단위 자동 탐지(직속 <text>를 가진 levelN), 무표점(한자만 보존),
  기사 ID 해시 분할, 문자 단위 BIO([0]/[1]/[2]), 이름 구간 'p' 메타필드 보존

출력:
- train.jsonl / val.jsonl : {'c', 'l', 'n', 'id', 'src', 'p'}
- 변환_통계.txt : 검증 통계
"""

import json
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# === 고정 파라미터 ===
MAX_LEN = 256
OVERLAP = 64
STEP = MAX_LEN - OVERLAP
MAX_ENTITY_LEN = 17
MIN_CHUNK_LEN = 5
TRAIN_RATIO = 0.9

# === 태그 처리 정책 (미등록 태그 표본 판정 결과) ===
# 본문으로 포함하여 재귀 처리하는 태그
INCLUDE_TAGS = {'text', 'content', 'paragraph', 'index', 'annotation',
                'noteContent', 'quotation', 'postScript', 'proofreading',
                'qna'}
# 서브트리 전체를 제외하는 태그 (출현 횟수는 통계로 보고)
EXCLUDE_TAGS = {'explanation', 'noteTitle', 'reference', 'pTitle',
                'illustration', 'caption', 'image', 'link', 'name'}
# 결락 표지: 해당 지점에서 텍스트를 분절 (허위 인접 차단)
LACUNA_TAGS = {'missing', 'newChar'}
LACUNA_CHARS = {'□'}  # U+25A1, 원문 결락 표지
SENTINEL = '\x00'  # 분절 경계 내부 표지 (출력에 남지 않음)

# 인라인을 허용하는 annotation type (그 외 유형은 제외+보고)
ANNOTATION_INLINE_TYPES = {'원주'}

# 별호 태깅 규약 점검용 목록 (판정 완료 항목이나 회귀 확인용으로 유지)
BYEOLHO_CHECK = ['沁都', '松都', '松京', '完山', '箕城', '漢師']

# v4: 비한문 기사 가드 — 한글·라틴 실질문자가 이 비율을 넘으면 기사 제외
NONCJK_GUARD_PATTERN = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z]+')
NONCJK_RATIO = 0.05

# v4: 관례 점검 목록 — 출처(문헌)별 태깅률을 비교해 색인 관례 충돌을 정량화
FACILITY_CHECK = ['宗廟', '社稷', '文廟', '太廟', '聖廟', '太學', '皇壇',
                  '昌德宮', '景福宮', '乾德殿']
COUNTRY_CHECK = ['倭', '唐', '胡', '虜', '遼', '契丹', '日本',
                 '高麗', '新羅', '百濟', '高句麗']


def src_group(src):
    """파일명에서 출처 식별 (1번 검증 스크립트와 동일하게 유지할 것)"""
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


HANGUL_PATTERN = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
CHAR_REF_PATTERN = re.compile(r'&#[xX]([0-9A-Fa-f]{4,6});')


def is_chinese_char(char):
    """한자 판별 - 모든 CJK 영역 포함 (표점 파이프라인과 동일 기준)"""
    code = ord(char)
    return (0x4E00 <= code <= 0x9FFF or  # CJK Unified Ideographs
            0x3400 <= code <= 0x4DBF or  # Extension A
            0x20000 <= code <= 0x2A6DF or  # Extension B
            0x2A700 <= code <= 0x2B73F or  # Extension C
            0x2B740 <= code <= 0x2B81F or  # Extension D
            0x2B820 <= code <= 0x2CEAF or  # Extension E
            0x2CEB0 <= code <= 0x2EBEF or  # Extension F
            0x30000 <= code <= 0x3134F or  # Extension G
            0x31350 <= code <= 0x323AF or  # Extension H
            0x2F800 <= code <= 0x2FA1F or  # Compatibility Supplement
            0xF900 <= code <= 0xFADF or  # Compatibility Ideographs
            0x2F00 <= code <= 0x2FDF or  # Kangxi Radicals
            0x2E80 <= code <= 0x2EFF)  # CJK Radicals Supplement


def find_articles(root):
    """기사 단위 자동 탐지: 직속 <text> 자식을 가진 levelN 요소"""
    articles = []
    for elem in root.iter():
        if elem.tag.startswith('level') and elem.find('text') is not None:
            articles.append(elem)
    return articles


WHITESPACE_PATTERN = re.compile(r'[\s\u3000\u00A0\uFEFF]+')


def normalize_node_text(s, stats):
    """텍스트 노드 단위 정규화: 공백류 제거, 문자 참조 복원, □→분절 표지"""
    s = WHITESPACE_PATTERN.sub('', s)

    def restore(m):
        try:
            ch = chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)
        if is_chinese_char(ch):
            stats['char_ref_restored'] += 1
            return ch
        return m.group(0)

    s = CHAR_REF_PATTERN.sub(restore, s)
    for lc in LACUNA_CHARS:
        if lc in s:
            stats['lacuna_chars'] += s.count(lc)
            s = s.replace(lc, SENTINEL)
    return s


def extract_article(article, stats):
    """기사의 직속 text 영역을 문서 순서대로 추출.
    포함/제외/결락 정책 적용. Returns: (문자열, [(start, end, type)])
    결락 지점은 SENTINEL로 표시되어 후단에서 분절됨.
    """
    chars = []
    spans = []

    def append_text(s):
        if s:
            chars.extend(normalize_node_text(s, stats))

    def recurse(elem):
        tag = elem.tag

        if tag in LACUNA_TAGS:
            stats['lacuna_tags'][tag] += 1
            chars.append(SENTINEL)  # 내부 텍스트(缺 등)는 버리고 경계만 남김
            return

        if tag in EXCLUDE_TAGS:
            stats['excluded_tags'][tag] += 1
            return

        if tag == 'annotation':
            ann_type = elem.get('type') or '무유형'
            stats['annotation_types'][ann_type] += 1
            if ann_type not in ANNOTATION_INLINE_TYPES:
                return  # 교감주 등은 서브트리 제외

        if tag not in INCLUDE_TAGS:
            stats['unknown_tags'][tag] += 1
            return  # v2: 미등록 태그는 제외가 기본값

        if tag == 'index':
            start = len(chars)
            append_text(elem.text)
            for child in elem:  # index 내부 중첩(교감주 등) 정책 적용
                recurse(child)
                append_text(child.tail)
            spans.append((start, len(chars), elem.get('type')))
            stats['index_types'][elem.get('type')] += 1
            return

        append_text(elem.text)
        for child in elem:
            recurse(child)
            append_text(child.tail)

    for text_elem in article.findall('text'):
        recurse(text_elem)

    return ''.join(chars), spans


def split_at_lacunae(text, spans, stats):
    """SENTINEL 지점에서 텍스트를 분절. 결락에 걸친 구간은 라벨에서 제외.
    Returns: [(segment_text, segment_spans)]
    """
    if SENTINEL not in text:
        return [(text, spans)]

    segments = []
    seg_start = 0
    boundaries = [i for i, ch in enumerate(text) if ch == SENTINEL] + [len(text)]

    for b in boundaries:
        seg_text = text[seg_start:b]
        seg_spans = [(s - seg_start, e - seg_start, t)
                     for s, e, t in spans
                     if s >= seg_start and e <= b]  # 결락에 걸친 구간은 자연 탈락
        segments.append((seg_text, seg_spans))
        seg_start = b + 1

    # 걸친 구간 수 집계 (분절 전 기준)
    placed = sum(len(s) for _, s in segments)
    stats['lacuna_dropped_spans'] += len(spans) - placed
    return segments


def cjk_filter(text, spans, art_id, stats):
    """한자만 보존하며 위치 매핑 구축 후 구간 오프셋 보정.
    구간 가장자리의 비한자(괄호·표점 등) 제거는 무해하므로 침묵 처리.
    구간 내부의 실질 문자 제거(평문 교정 괄호 등)는 개체 병합·왜곡을 만들므로
    지명이면 라벨에서 제외하고, 이름이면 보존하되 집계한다.
    """
    pos_map = {}
    out = []
    for i, ch in enumerate(text):
        if is_chinese_char(ch):
            pos_map[i] = len(out)
            out.append(ch)

    new_spans = []
    for start, end, typ in spans:
        kept_orig = [i for i in range(start, end) if i in pos_map]
        if not kept_orig:
            stats['span_fully_lost'] += 1
            continue

        # 내부 결손 판정: 보존된 첫/끝 위치 사이에 제거된 문자가 있는가
        internal_loss = (kept_orig[-1] - kept_orig[0] + 1) != len(kept_orig)
        if internal_loss:
            if typ == '지명':
                stats['geo_dropped_internal'] += 1
                if stats['geo_dropped_internal'] <= 50:
                    stats['warnings'].append(
                        f"지명 내부 결손→라벨 제외 ({art_id}): {text[start:end]!r}")
                continue
            stats['nongeo_internal_loss'][typ] += 1

        new_spans.append((pos_map[kept_orig[0]], pos_map[kept_orig[-1]] + 1, typ))

    return ''.join(out), new_spans


def encode_bio(length, spans):
    """지명 구간 → 문자 단위 BIO 라벨 ([0]/[1]/[2]), 이름 구간 → 별도 리스트"""
    labels = [[0] for _ in range(length)]
    name_spans = []
    for start, end, typ in spans:
        if typ == '지명':
            labels[start] = [1]
            for i in range(start + 1, end):
                labels[i] = [2]
        elif typ == '이름':
            name_spans.append([start, end])
    return labels, name_spans


def safe_start(labels, s):
    """청크 시작이 개체 중간(I)에 걸리지 않도록 앞으로 당김"""
    if s <= 0:
        return 0
    if s >= len(labels):
        return len(labels)
    while s > 0 and labels[s] == [2]:
        s -= 1
    return s


def safe_end(labels, s, e, total):
    """청크 끝이 개체 중간에 걸리지 않도록 뒤로 물림"""
    if e >= total:
        return total
    back_limit = max(s, e - (MAX_ENTITY_LEN - 1))
    while e > back_limit and labels[e - 1] == [2]:
        e -= 1
    return e


def chunk_segment(text, labels, name_spans, art_id, src):
    """텍스트 조각을 256자 청크(중첩 64자)로 분할. 개체 경계 보정 적용."""
    total = len(text)
    records = []

    def make_record(s, e):
        chunk_names = []
        for ns, ne in name_spans:
            cs, ce = max(ns, s), min(ne, e)
            if cs < ce:
                chunk_names.append([cs - s, ce - s])
        return {
            'c': text[s:e],
            'l': labels[s:e],
            'n': e - s,
            'id': art_id,
            'src': src,
            'p': chunk_names,
        }

    if total <= MAX_LEN:
        if total >= MIN_CHUNK_LEN:
            records.append(make_record(0, total))
        return records

    start = 0
    prev_start = -1
    while start < total:
        start = safe_start(labels, start)
        if start == prev_start:
            break
        prev_start = start

        end = min(start + MAX_LEN, total)
        end = safe_end(labels, start, end, total)
        if end <= start or end - start < MIN_CHUNK_LEN:
            end = min(start + MAX_LEN, total)

        if end - start >= MIN_CHUNK_LEN:
            records.append(make_record(start, end))

        if end >= total:
            break

        next_start = end - OVERLAP
        if next_start <= start:
            next_start = start + STEP
        if next_start <= start:
            break
        start = next_start

    return records


def is_train(art_id):
    """기사 ID 해시 기반 결정적 분할 (재실행 시에도 동일 분할 보장)"""
    h = int(hashlib.md5(art_id.encode('utf-8')).hexdigest(), 16)
    return (h % 100) < int(TRAIN_RATIO * 100)


def check_terms(text, spans, stats, grp):
    """점검 목록(별호·시설·국호)의 출현 위치가 지명 태깅 구간에
    포함되는지 집계. 별호는 전역, 시설·국호는 출처별로 집계하여
    문헌 간 색인 관례 차이를 정량화한다."""
    geo_positions = set()
    for start, end, typ in spans:
        if typ == '지명':
            geo_positions.update(range(start, end))

    def count(term, counter, key_prefix):
        idx = text.find(term)
        while idx != -1:
            covered = all(i in geo_positions
                          for i in range(idx, idx + len(term)))
            key = 'tagged' if covered else 'untagged'
            counter[f"{key_prefix}_{key}"] += 1
            idx = text.find(term, idx + 1)

    for term in BYEOLHO_CHECK:
        count(term, stats['byeolho'], term)
    for term in FACILITY_CHECK + COUNTRY_CHECK:
        count(term, stats['convention'], f"{grp}|{term}")


def process_folder(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_path.glob('**/*.xml'))
    print(f"발견된 XML 파일: {len(xml_files)}개")
    if not xml_files:
        print("XML 파일이 없습니다.")
        return

    stats = {
        'files': 0,
        'parse_errors': 0,
        'articles': 0,
        'skipped_short': 0,
        'segments': 0,
        'chunks_train': 0,
        'chunks_val': 0,
        'total_chars': 0,
        'geo_entities': 0,
        'name_entities': 0,
        'char_ref_restored': 0,
        'lacuna_chars': 0,
        'lacuna_dropped_spans': 0,
        'span_fully_lost': 0,
        'geo_dropped_internal': 0,
        'nongeo_internal_loss': Counter(),
        'index_types': Counter(),
        'unknown_tags': Counter(),
        'excluded_tags': Counter(),
        'lacuna_tags': Counter(),
        'annotation_types': Counter(),
        'byeolho': Counter(),
        'convention': Counter(),       # v4: 출처별 관례 점검
        'skipped_aux': 0,              # v4: $ 표지 부속물 제외
        'skipped_noncjk': Counter(),   # v4: 비한문 가드 제외 (출처별)
        'chars_by_src': Counter(),     # v4: 출처별 한자 수 (밀도 계산용)
        'geo_by_src': Counter(),       # v4: 출처별 지명 개체 수
        'hangul_residue': 0,
        'entity_counter': Counter(),
        'warnings': [],
    }

    train_file = output_path / 'train.jsonl'
    val_file = output_path / 'val.jsonl'

    with open(train_file, 'w', encoding='utf-8') as f_train, \
         open(val_file, 'w', encoding='utf-8') as f_val:

        for xml_file in tqdm(xml_files, desc="XML 변환"):
            try:
                tree = ET.parse(xml_file)
            except ET.ParseError as e:
                stats['parse_errors'] += 1
                stats['warnings'].append(f"파싱 실패: {xml_file.name} ({e})")
                continue

            stats['files'] += 1
            root = tree.getroot()
            src = xml_file.stem
            grp = src_group(src)

            for article in find_articles(root):
                art_id = article.get('id') or f"{src}_noid_{stats['articles']}"

                # v4: 편찬 부속물(간행사·범례·해제) 제외 — id의 $ 표지
                if '$' in art_id:
                    stats['skipped_aux'] += 1
                    continue

                raw, raw_spans = extract_article(article, stats)

                # v4: 비한문 기사 가드 — 한글·라틴 비율로 번역문·영문 사료 차단
                substantive = len(raw) - raw.count(SENTINEL)
                noncjk = sum(len(m) for m in NONCJK_GUARD_PATTERN.findall(raw))
                if substantive > 0 and noncjk / substantive > NONCJK_RATIO:
                    stats['skipped_noncjk'][grp] += 1
                    continue

                # v4: 미량 잔존 한글·라틴은 결락과 동일하게 분절 경계로 처리
                # (1:1 치환으로 길이 보존 → 구간 좌표 무영향, 허위 인접 차단)
                if noncjk:
                    raw = NONCJK_GUARD_PATTERN.sub(
                        lambda m: SENTINEL * len(m.group()), raw)

                segments = split_at_lacunae(raw, raw_spans, stats)

                article_used = False
                target = f_train if is_train(art_id) else f_val

                for seg_text, seg_spans in segments:
                    clean, spans = cjk_filter(seg_text, seg_spans, art_id, stats)
                    if len(clean) < MIN_CHUNK_LEN:
                        continue

                    article_used = True
                    stats['segments'] += 1
                    stats['total_chars'] += len(clean)
                    stats['chars_by_src'][grp] += len(clean)

                    if HANGUL_PATTERN.search(clean):
                        stats['hangul_residue'] += 1

                    check_terms(clean, spans, stats, grp)
                    labels, name_spans = encode_bio(len(clean), spans)

                    for start, end, typ in spans:
                        if typ == '지명':
                            stats['geo_entities'] += 1
                            stats['geo_by_src'][grp] += 1
                            stats['entity_counter'][clean[start:end]] += 1
                        elif typ == '이름':
                            stats['name_entities'] += 1

                    for record in chunk_segment(clean, labels, name_spans,
                                                art_id, src):
                        target.write(json.dumps(record, ensure_ascii=False) + '\n')
                        if target is f_train:
                            stats['chunks_train'] += 1
                        else:
                            stats['chunks_val'] += 1

                if article_used:
                    stats['articles'] += 1
                else:
                    stats['skipped_short'] += 1

    write_stats(output_path, stats, train_file, val_file)


def write_stats(output_path, stats, train_file, val_file):
    lines = []
    lines.append("=" * 70)
    lines.append("XML → NER 학습데이터 변환 통계")
    lines.append("=" * 70)
    lines.append(f"처리 파일: {stats['files']:,}개 (파싱 실패 {stats['parse_errors']}개)")
    lines.append(f"기사 수: {stats['articles']:,}개 (유효 조각 없음 제외 {stats['skipped_short']:,}개)")
    lines.append(f"텍스트 조각 수: {stats['segments']:,}개 (결락 분절 반영)")
    lines.append(f"총 한자 수: {stats['total_chars']:,}자")
    lines.append(f"지명 개체: {stats['geo_entities']:,}개 / 이름 개체(메타): {stats['name_entities']:,}개")
    lines.append(f"고유 지명 종류: {len(stats['entity_counter']):,}개")
    lines.append(f"학습 청크: {stats['chunks_train']:,}개 → {train_file}")
    lines.append(f"검증 청크: {stats['chunks_val']:,}개 → {val_file}")

    lines.append("\n[정제 처리 내역]")
    lines.append(f"문자 참조 복원: {stats['char_ref_restored']:,}자")
    lines.append(f"결락 표지(□): {stats['lacuna_chars']:,}개")
    for tag, cnt in stats['lacuna_tags'].most_common():
        lines.append(f"결락 요소 <{tag}>: {cnt:,}회")
    lines.append(f"결락 걸침으로 라벨 제외된 구간: {stats['lacuna_dropped_spans']:,}개")
    lines.append(f"구간 전체 소실: {stats['span_fully_lost']:,}개")
    lines.append(f"지명 내부 결손→라벨 제외: {stats['geo_dropped_internal']:,}개")
    for typ, cnt in stats['nongeo_internal_loss'].most_common():
        lines.append(f"내부 결손 보존({typ}): {cnt:,}개")

    lines.append("\n[제외 태그 처리 내역]")
    for tag, cnt in stats['excluded_tags'].most_common():
        lines.append(f"  <{tag}>: {cnt:,}회 제외")
    lines.append("[annotation type 분포]")
    for typ, cnt in stats['annotation_types'].most_common():
        inline = "인라인" if typ in ANNOTATION_INLINE_TYPES else "제외"
        lines.append(f"  {typ}: {cnt:,}회 ({inline})")

    lines.append("\n[index type 분포]")
    for typ, cnt in stats['index_types'].most_common():
        lines.append(f"  {typ}: {cnt:,}")

    lines.append("\n[검증 항목]")
    lines.append(f"한글 잔존 조각: {stats['hangul_residue']:,}개")
    if stats['unknown_tags']:
        lines.append("미등록 태그 출현 (서브트리 제외됨, 신규 판정 필요):")
        for tag, cnt in stats['unknown_tags'].most_common():
            lines.append(f"  <{tag}>: {cnt:,}회")
    else:
        lines.append("미등록 태그: 없음")

    lines.append("\n[별호 태깅률 점검]")
    for term in BYEOLHO_CHECK:
        t = stats['byeolho'].get(f"{term}_tagged", 0)
        u = stats['byeolho'].get(f"{term}_untagged", 0)
        if t + u > 0:
            lines.append(f"  {term}: 태깅 {t:,} / 무태깅 {u:,} (태깅률 {t / (t + u) * 100:.1f}%)")

    lines.append("\n[v4 가드 처리 내역]")
    lines.append(f"부속물($ 표지) 제외 기사: {stats['skipped_aux']:,}개")
    for grp, cnt in stats['skipped_noncjk'].most_common():
        lines.append(f"비한문 가드 제외 ({grp}): {cnt:,}개")

    lines.append("\n[출처별 지명 밀도 (1,000자당 지명 개체)]")
    for grp in sorted(stats['chars_by_src']):
        chars = stats['chars_by_src'][grp]
        geo = stats['geo_by_src'][grp]
        if chars > 0:
            lines.append(f"  {grp}: {geo / chars * 1000:.2f} "
                         f"(개체 {geo:,} / 한자 {chars:,})")

    lines.append("\n[관례 점검: 시설·국호 태깅률 (출처별)]")
    groups = sorted({k.split('|')[0] for k in stats['convention']})
    for grp in groups:
        lines.append(f"  ◆ {grp}")
        for term in FACILITY_CHECK + COUNTRY_CHECK:
            t = stats['convention'].get(f"{grp}|{term}_tagged", 0)
            u = stats['convention'].get(f"{grp}|{term}_untagged", 0)
            if t + u > 0:
                lines.append(f"    {term}: 태깅 {t:,} / 무태깅 {u:,}"
                             f" (태깅률 {t / (t + u) * 100:.1f}%)")

    lines.append(f"\n[경고: {len(stats['warnings'])}건]")
    for w in stats['warnings'][:100]:
        lines.append(f"  {w}")
    if len(stats['warnings']) > 100:
        lines.append(f"  ... 외 {len(stats['warnings']) - 100:,}건")

    lines.append("\n[상위 30개 지명]")
    for entity, count in stats['entity_counter'].most_common(30):
        lines.append(f"  {entity}: {count:,}회")

    report = '\n'.join(lines)
    print('\n' + report)

    stats_file = output_path / '변환_통계.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(f"\n통계 파일 저장: {stats_file}")


def main():
    import tkinter as tk
    from tkinter import filedialog

    print("=" * 70)
    print("국사편찬위원회 색인 XML → 지명 NER 학습데이터 변환")
    print("=" * 70)

    root = tk.Tk()
    root.withdraw()

    print("\n1. XML 파일이 있는 폴더 선택 (하위 폴더 포함)...")
    input_dir = filedialog.askdirectory(title="XML 폴더 선택")
    if not input_dir:
        print("폴더가 선택되지 않았습니다.")
        return

    print("2. 출력 폴더 선택...")
    output_dir = filedialog.askdirectory(title="출력 폴더 선택")
    if not output_dir:
        print("폴더가 선택되지 않았습니다.")
        return

    root.destroy()

    process_folder(input_dir, output_dir)
    print("\n✅ 변환 완료")


if __name__ == "__main__":
    main()