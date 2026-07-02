# -*- coding: utf-8 -*-
"""
XML 메타데이터 분석 + 지명 표면형 출처별 태깅률 전수 census

출력(선택 폴더 안):
  1) xml_파일별.csv         : 파일 단위 (제목·출처·subjectClass·type별·고유종·한자·밀도)
  2) xml_끝글자_출처별.csv   : 행=끝글자, 열=출처, 값=지명 개수 (절단 없음, 합계 포함)
  3) xml_끝글자_장르별.csv   : 행=끝글자, 열=subjectClass, 값=지명 개수 (절단 없음)
  4) xml_type_집계.csv       : 출처·장르별 index type(지명/이름/관서/관직/서명) 전수
  5) xml_요약.txt            : 출처·장르 목록과 규모 (탐색용 인덱스)
  6) xml_표면형_태깅률.csv   : 태깅된 적 있는 모든 지명 표면형의 출처별 태깅/출현/태깅률 (전수)
  7) xml_태깅률_분산.csv     : 2개 이상 출처에 출현하는 표면형의 출처간 태깅률 편차 (큰 순)

태깅률 정의: 표면형 S에 대해
  출현(S, 출처) = 그 출처 본문에서 문자열 S의 출현 횟수
  태깅(S, 출처) = 그 출현 중 <index type='지명'> 구간에 완전히 덮인 횟수
  태깅률 = 태깅 / 출현
출처별 태깅률 편차가 큰 표면형이 색인 관례 충돌 후보 (倭 유형).

실행: Run → XML 루트 폴더 선택 (하위 전체 재귀)
  ※ census 위해 XML 재파싱(2패스). 코퍼스 규모에 따라 수 분 소요 가능.
"""

import os
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from tkinter import Tk, filedialog

_WS = set(' \t\r\n\u3000\xa0\ufeff')
MAX_FORM_LEN = 17        # census 매칭 표면형 길이 상한 (파이프라인 MAX_ENTITY_LEN과 일치)
MIN_OCC_FOR_RATE = 3     # 분산 요약에서 출처별 태깅률을 신뢰 단위로 인정하는 최소 출현


def is_cjk(ch):
    c = ord(ch)
    return (0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or
            0x20000 <= c <= 0x2A6DF or 0xF900 <= c <= 0xFAFF)


def cjk_ratio(s):
    return sum(1 for ch in s if is_cjk(ch)) / len(s) if s else 0.0


def analyze_file(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    def ftext(tag):
        e = root.find('.//' + tag)
        return (e.text or '').strip() if e is not None and e.text else ''
    title = ftext('mainTitle')
    sc_e = root.find('.//subjectClass')
    subject = (sc_e.text or '').strip() if sc_e is not None and sc_e.text else '미상'

    type_cnt = Counter()
    place_names = []
    body_chars = 0
    for te in root.iter('text'):
        for s in te.itertext():
            body_chars += sum(1 for ch in s if is_cjk(ch))

    for idx in root.iter('index'):
        t = idx.get('type') or '무유형'
        type_cnt[t] += 1
        if t == '지명':
            full = re.sub(r'\s+', '', ''.join(idx.itertext()).strip())
            if full:
                place_names.append(full)

    last_cnt = Counter(n[-1] for n in place_names if n)
    noncjk = sum(1 for n in place_names if cjk_ratio(n) < 0.5)
    return {
        'title': title, 'subject': subject, 'type_cnt': type_cnt,
        'place_total': len(place_names), 'place_uniq': len(set(place_names)),
        'last_cnt': last_cnt, 'noncjk': noncjk, 'body_chars': body_chars,
        'names': place_names,
    }


def extract_body_geo(root):
    """본문(text 요소 내부)을 문서 순서로 잇고, 각 문자가 <index type='지명'>
    내부인지 여부(1/0)를 함께 반환. 공백류 제거."""
    chars = []
    geo = []

    def emit(s, ctx):
        for ch in s:
            if ch in _WS:
                continue
            chars.append(ch)
            geo.append(ctx)

    def walk(elem, in_text, in_geo):
        is_text = in_text or (elem.tag == 'text')
        cur_geo = 1 if (in_geo or (elem.tag == 'index'
                                   and elem.get('type') == '지명')) else 0
        if is_text and elem.text:
            emit(elem.text, cur_geo)
        for child in elem:
            walk(child, is_text, cur_geo)
            if is_text and child.tail:
                emit(child.tail, cur_geo)

    walk(root, False, 0)
    return ''.join(chars), geo


def census_file(root, forms_by_len, lengths, tot, tg):
    """한 파일 본문에서 모든 표면형의 출현/태깅을 길이버킷 슬라이딩으로 집계.
    tot/tg는 호출자가 넘긴 해당 출처 Counter (in-place 갱신)."""
    body, geo = extract_body_geo(root)
    B = len(body)
    if B == 0:
        return
    pref = [0] * (B + 1)          # 덮임 판정 O(1)용 prefix 합
    for i in range(B):
        pref[i + 1] = pref[i] + geo[i]
    for i in range(B):
        for L in lengths:        # lengths 오름차순 → 범위 초과 시 break
            j = i + L
            if j > B:
                break
            sub = body[i:j]
            if sub in forms_by_len[L]:
                tot[sub] += 1
                if pref[j] - pref[i] == L:
                    tg[sub] += 1


def write_crosstab(path, last_by_key, key_order):
    total_by_char = Counter()
    for k in key_order:
        total_by_char += last_by_key[k]
    chars_sorted = [c for c, _ in total_by_char.most_common()]
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['끝글자', '총합'] + key_order)
        for ch in chars_sorted:
            w.writerow([ch, total_by_char[ch]] + [last_by_key[k].get(ch, 0) for k in key_order])


def file_src(path, folder):
    rel = os.path.relpath(path, folder)
    return rel.split(os.sep)[0] if os.sep in rel else '(root)'


def main():
    root_t = Tk(); root_t.withdraw()
    folder = filedialog.askdirectory(title="XML 루트 폴더 선택 (하위 전체 재귀)")
    root_t.destroy()
    if not folder:
        print("취소됨"); return

    xml_files = []
    for dp, _, fns in os.walk(folder):
        for fn in fns:
            if fn.lower().endswith('.xml'):
                xml_files.append(os.path.join(dp, fn))
    print(f"XML {len(xml_files)}개 발견\n")

    # === 패스 1: 메타데이터 집계 + 전역 지명 표면형 수집 ===
    rows = []
    src_last = defaultdict(Counter)
    genre_last = defaultdict(Counter)
    src_type = defaultdict(Counter)
    genre_type = defaultdict(Counter)
    src_meta = defaultdict(lambda: {'files': 0, 'place': 0, 'chars': 0, 'noncjk': 0})
    genre_meta = defaultdict(lambda: {'files': 0, 'place': 0, 'chars': 0})
    all_forms = set()
    errors = 0

    print("패스 1/2: 메타데이터 집계")
    for i, path in enumerate(xml_files, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(xml_files)}]")
        r = analyze_file(path)
        if r is None:
            errors += 1
            continue
        src = file_src(path, folder)
        sub = r['subject']
        rows.append({
            '파일': os.path.basename(path), '출처': src, '제목': r['title'],
            'subjectClass': sub, '한자수': r['body_chars'],
            '지명총': r['place_total'], '지명고유': r['place_uniq'], '비한자지명': r['noncjk'],
            '이름': r['type_cnt'].get('이름', 0), '관서': r['type_cnt'].get('관서', 0),
            '관직': r['type_cnt'].get('관직', 0), '서명': r['type_cnt'].get('서명', 0),
            '밀도_천자당': round(r['place_total'] / r['body_chars'] * 1000, 2) if r['body_chars'] else 0,
        })
        src_last[src] += r['last_cnt']; genre_last[sub] += r['last_cnt']
        src_type[src] += r['type_cnt']; genre_type[sub] += r['type_cnt']
        sm = src_meta[src]; sm['files'] += 1; sm['place'] += r['place_total']
        sm['chars'] += r['body_chars']; sm['noncjk'] += r['noncjk']
        gm = genre_meta[sub]; gm['files'] += 1; gm['place'] += r['place_total']
        gm['chars'] += r['body_chars']
        for n in r['names']:
            if 1 <= len(n) <= MAX_FORM_LEN:
                all_forms.add(n)

    src_order = sorted(src_meta, key=lambda k: -src_meta[k]['chars'])
    genre_order = sorted(genre_meta, key=lambda k: -genre_meta[k]['place'])

    # 기존 출력 1~5 (수치 불변)
    with open(os.path.join(folder, 'xml_파일별.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    write_crosstab(os.path.join(folder, 'xml_끝글자_출처별.csv'), src_last, src_order)
    write_crosstab(os.path.join(folder, 'xml_끝글자_장르별.csv'), genre_last, genre_order)
    with open(os.path.join(folder, 'xml_type_집계.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['구분', '키', '파일수', '한자수', '지명', '이름', '관서', '관직', '서명', '밀도_천자당'])
        for s in src_order:
            t = src_type[s]; m = src_meta[s]
            w.writerow(['출처', s, m['files'], m['chars'], t.get('지명', 0), t.get('이름', 0),
                        t.get('관서', 0), t.get('관직', 0), t.get('서명', 0),
                        round(m['place']/m['chars']*1000, 2) if m['chars'] else 0])
        for g in genre_order:
            t = genre_type[g]; m = genre_meta[g]
            w.writerow(['장르', g, m['files'], m['chars'], t.get('지명', 0), t.get('이름', 0),
                        t.get('관서', 0), t.get('관직', 0), t.get('서명', 0),
                        round(m['place']/m['chars']*1000, 2) if m['chars'] else 0])
    with open(os.path.join(folder, 'xml_요약.txt'), 'w', encoding='utf-8') as f:
        f.write(f"XML {len(xml_files)}개 / 파싱오류 {errors}\n")
        f.write("※ 판단 근거는 끝글자 교차표 및 태깅률 CSV. 이 파일은 출처·장르 인덱스용.\n\n[출처]\n")
        for s in src_order:
            m = src_meta[s]
            f.write(f"  {s}: 파일 {m['files']}, 한자 {m['chars']:,}, 지명 {m['place']:,}, 비한자 {m['noncjk']:,}\n")
        f.write("\n[subjectClass]\n")
        for g in genre_order:
            m = genre_meta[g]
            f.write(f"  {g}: 파일 {m['files']}, 한자 {m['chars']:,}, 지명 {m['place']:,}\n")

    # === 패스 2: 표면형 출처별 태깅률 census ===
    lengths = sorted({len(s) for s in all_forms})
    if not lengths:
        print("태깅된 지명 표면형이 없어 census를 건너뜁니다."); return
    forms_by_len = {L: set() for L in lengths}
    for s in all_forms:
        forms_by_len[len(s)].add(s)
    print(f"\n패스 2/2: 표면형 태깅률 census "
          f"(대상 {len(all_forms):,}종, 길이 {lengths[0]}~{lengths[-1]})")

    tot = defaultdict(Counter)   # 출처 → 표면형 → 출현
    tg = defaultdict(Counter)    # 출처 → 표면형 → 태깅
    for i, path in enumerate(xml_files, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(xml_files)}]")
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        src = file_src(path, folder)
        census_file(root, forms_by_len, lengths, tot[src], tg[src])

    # 6) 표면형 출처별 태깅률 (전수, long)
    forms_sorted = sorted(all_forms, key=lambda s: -sum(tot[x][s] for x in src_order))
    with open(os.path.join(folder, 'xml_표면형_태깅률.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['표면형', '출처', '태깅', '출현', '태깅률'])
        for s in forms_sorted:
            for src in src_order:
                o = tot[src][s]
                if o:
                    w.writerow([s, src, tg[src][s], o, round(tg[src][s] / o * 100, 1)])

    # 7) 태깅률 분산 (2개 이상 신뢰 출처, 편차 큰 순)
    summ = []
    for s in all_forms:
        rates = []
        tot_o = tot_t = 0
        for src in src_order:
            o = tot[src][s]
            if o:
                tot_o += o; tot_t += tg[src][s]
                if o >= MIN_OCC_FOR_RATE:
                    rates.append((tg[src][s] / o * 100, src))
        if len(rates) >= 2:
            rmin = min(rates); rmax = max(rates)
            summ.append((s, len(rates), tot_o, tot_t, round(tot_t / tot_o * 100, 1),
                         round(rmin[0], 1), rmin[1], round(rmax[0], 1), rmax[1],
                         round(rmax[0] - rmin[0], 1)))
    summ.sort(key=lambda x: -x[-1])
    with open(os.path.join(folder, 'xml_태깅률_분산.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['표면형', '신뢰출처수', '총출현', '총태깅', '전체태깅률',
                    '최소태깅률', '최소출처', '최대태깅률', '최대출처', '태깅률폭'])
        w.writerows(summ)

    print("\n완료: xml_파일별 / xml_끝글자_출처별 / xml_끝글자_장르별 / "
          "xml_type_집계 / xml_요약 / xml_표면형_태깅률 / xml_태깅률_분산")


if __name__ == "__main__":
    main()