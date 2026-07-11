#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""모델 스타일 팩 감사 (P2 파일럿) — "코덱스/클로드 티" 측정·제거 처방.

지문(작가 복제)과 반대 방향의 도구다. 팩은 모델이 대량 생성한 코퍼스에서
인간 대조군 대비 과하게 나타나는 서명(요인·특징·분포 마커)을 고정한 것이고,
이 모듈은 임의의 글에 그 서명이 얼마나 묻어 있는지 점수화하고 제거를 처방한다.

팩 서명은 판별 실적 관문(|Cohen's d| ≥ 0.5)을 통과한 항목만 담는다 —
"모델이 좀 더 쓴다"가 아니라 "인간과 유의하게 갈린다"만 마커로 인정.

사용:
  python3 style_pack.py score packs/codex-style.json 글.txt      # 코덱스니스 점수
  python3 style_pack.py strip packs/codex-style.json 글.txt      # 제거 처방(LLM 지시문)
"""
import json
import math
import re
import sys
from collections import Counter

from features_surface import split_sentences, chunk_features
from features_ext import ext_features_from_sents

_kiwi = None

def _morph_stream(text):
    """kiwi form 스트림 (표현 사전 매칭용). kiwi 없으면 None."""
    global _kiwi
    if _kiwi is None:
        try:
            from features_morph import get_kiwi
            _kiwi = get_kiwi()
        except Exception:
            _kiwi = False
    if not _kiwi:
        return None
    return ' '.join(t.form for t in _kiwi.tokenize(text))


def _content_lemmas(text):
    """내용어 lemma Counter (내용어 사전 매칭용). kiwi 없으면 None."""
    global _kiwi
    if _kiwi is None:
        _morph_stream('')
    if not _kiwi:
        return None
    tags = {'NNG', 'VV', 'VA', 'XR'}
    return Counter(t.form for t in _kiwi.tokenize(text) if t.tag in tags)

FEATURE_KO = {
    'CX_not_a_but_b_sent': '"~이 아니라 ~이다" 대조 구문',
    'CX_hal_su_sent': '"~할 수 있다" 구문',
    'CX_geot_end_sent': '"~것이다" 종결',
    'CX_meta_discourse_sent': '"결론적으로/요컨대" 메타 담화 표지',
    'CX_triple_list_sent': '삼항 병렬 나열',
    'CX_conditional_sent': '조건절("~다면")',
    'L7_first_person_sent': '1인칭 노출',
    'L7_booster_sent': '단정 강조어(반드시·분명)',
    'L7_hedge_sent': '완화 표지(아마·~듯)',
    'L2_etm_ej': '관형절(-은/-는/-던) 밀도',
    'L2_etm_chain_max': '관형절 연쇄 깊이',
    'L2_verb_density_ej': '용언 밀도',
    'L2_nominalization_ej': '명사화(-것/-음/-기)',
    'L3_sensory_ej': '감각어',
}


def load_pack(path):
    return json.load(open(path, encoding='utf-8'))


def measure(text, pack):
    sents = split_sentences(text)
    if len(sents) < 8:
        raise ValueError('문장이 너무 적음 (팩 감사 불가)')
    ext_rows = []
    for i in range(0, len(sents) - 14, 15):
        ext_rows.append(ext_features_from_sents(sents[i:i + 15]))
    if not ext_rows:                           # 15문장 미만이지만 8+ → 전체 1창
        ext_rows = [ext_features_from_sents(sents)]
    ext = {k: sum(r.get(k, 0) for r in ext_rows) / len(ext_rows)
           for k in ext_rows[0]}
    cf = chunk_features(sents)

    hits = []
    # 특징 마커: 코덱스 방향(d>0=과다)으로 인간 이상이면 적중
    for k, m in pack['feature_markers'].items():
        val = ext.get(k)
        if val is None:
            continue
        d = m['d']
        exceed = (val > m['human'] if d > 0 else val < m['human'])
        if exceed:
            span = abs(m['codex'] - m['human']) or 1e-9
            frac = min(abs(val - m['human']) / span, 2.0)
            hits.append({'kind': 'feature', 'key': k,
                         'name': FEATURE_KO.get(k, k),
                         'severity': round(frac * abs(d), 2),
                         'val': round(val, 3), 'codex': m['codex'],
                         'human': m['human']})
    # 분포 마커: 어미·연결·부사 과다 사용
    for fam, label in (('ending_overuse', '종결어미'),
                       ('connective_overuse', '연결어미'),
                       ('adverb_overuse', '부사·접속어')):
        counts = cf[{'ending_overuse': 'endings',
                     'connective_overuse': 'connectives',
                     'adverb_overuse': 'adverbs'}[fam]]
        tot = sum(counts.values()) or 1
        for marker, ref in pack.get(fam, {}).items():
            r = counts.get(marker, 0) / tot
            if r >= ref['human'] and counts.get(marker, 0) > 0:
                hits.append({'kind': 'dist', 'family': label, 'marker': marker,
                             'severity': round((r - ref['human']) /
                                               (ref['codex'] - ref['human'] or 1e-9), 2),
                             'val': round(r, 3), 'codex': ref['codex'],
                             'human': ref['human']})
    # 내용어 사전: 코덱스 편애 내용어 밀도 (kiwi 필요)
    lemmas = _content_lemmas(text)
    if lemmas is not None and pack.get('content_words'):
        n_lem = max(sum(lemmas.values()), 1)
        used = []
        for w, m in pack['content_words'].items():
            c = lemmas.get(w, 0)
            if c == 0:
                continue
            rate = 10000 * c / n_lem
            if rate >= m['human_per10k']:
                # 도달률 기반: 개별 단어 밀도 폭발(짧은 글)을 코덱스 수준으로 클립.
                # 내용어는 '얼마나 진하게'보다 '코덱스 편애어를 몇 종 쓰나'가 신호라
                # 단어당 상한을 낮게(≤0.7) 둬 폐쇄류·구문 마커를 압도하지 않게 한다.
                reach = min(rate, m['codex_per10k'] * 1.5) / \
                    (m['codex_per10k'] or 1e-9)
                sev = min(reach, 1.0) * min(m['z'] / 5.0, 0.7)
                used.append({'kind': 'content', 'word': w,
                             'severity': round(sev, 2), 'count': c,
                             'rate': round(rate, 1),
                             'codex': m['codex_per10k'], 'human': m['human_per10k']})
        used.sort(key=lambda h: -h['severity'])
        hits.extend(used[:15])                   # 상위만 (내용어는 개수 많음)

    # 표현 사전: 문법 표현 적중 (kiwi 필요)
    stream = _morph_stream(text)
    if stream is not None and pack.get('phrases'):
        n_words = max(len(text.split()), 1)
        for name, m in pack['phrases'].items():
            c = len(re.findall(m['pattern'], stream))
            if c == 0:
                continue
            rate = 10000 * c / n_words
            if rate >= m['human_per10k']:
                sev = min(rate / (m['codex_per10k'] or 1e-9), 1.5) * \
                      min(math.log(m['ratio']) if m['ratio'] > 1 else 0.1, 1.5)
                hits.append({'kind': 'phrase', 'phrase': name,
                             'severity': round(sev, 2), 'count': c,
                             'ratio': m['ratio']})

    hits.sort(key=lambda h: -h['severity'])
    codex_score = round(sum(h['severity'] for h in hits), 2)
    return {'codex_score': codex_score, 'n_hits': len(hits),
            'n_sent': len(sents), 'hits': hits,
            'kiwi': lemmas is not None}


def prescribe(text, pack):
    m = measure(text, pack)
    lines = []
    for h in m['hits']:
        if h['severity'] < 0.3:
            continue
        if h['kind'] == 'feature':
            lines.append(f"[{h['severity']}] {h['name']} 과다 "
                         f"(현재 {h['val']}, 인간 기준 {h['human']}) — 줄여라")
        elif h['kind'] == 'dist':
            lines.append(f"[{h['severity']}] {h['family']} '{h['marker']}' 과다 "
                         f"({h['val']:.0%}, 인간 {h['human']:.0%}) — 다른 표현으로 분산")
        elif h['kind'] == 'content':
            lines.append(f"[{h['severity']}] 내용어 '{h['word']}' {h['count']}회 "
                         f"(만어절당 {h['rate']}, 인간 {h['human']}) — 동의어·구체어로 대체")
        elif h['kind'] == 'phrase':
            lines.append(f"[{h['severity']}] 표현 '{h['phrase']}' {h['count']}회 "
                         f"(인간의 {h['ratio']}배) — 다른 구문으로")
    return m, lines


def main():
    cmd, pack_path, text_path = sys.argv[1], sys.argv[2], sys.argv[3]
    pack = load_pack(pack_path)
    text = open(text_path, encoding='utf-8').read()
    if cmd == 'score':
        r = measure(text, pack)
        print(json.dumps(r, ensure_ascii=False, indent=1))
    elif cmd == 'strip':
        m, lines = prescribe(text, pack)
        print(f"코덱스니스 점수: {m['codex_score']} ({m['n_hits']}개 마커 적중, "
              f"{m['n_sent']}문장)\n")
        print(f"제거 처방 ({pack['meta']['pack']}):")
        for l in lines:
            print('  ' + l)
        if not lines:
            print('  (유의한 코덱스 마커 없음)')


if __name__ == '__main__':
    main()
