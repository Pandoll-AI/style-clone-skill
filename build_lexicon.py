#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""모델 스타일 팩의 내용어·표현 사전 빌더 (P2b, dev — kiwipiepy 필요).

두 사전을 추가한다:
- content_words: 모델이 인간 대비 즐겨 쓰는 내용어(명사·용언 어간).
  방법: weighted log-odds-ratio with informative Dirichlet prior
  (Monroe, Colaresi & Quinn 2008) — 희귀어 편향을 배경분포 사전으로 보정한
  계량언어학 표준. 관문: |z|≥Z_MIN ∧ 주제 분산 df≥TOPIC_MIN.
  **주제 분산 관문이 핵심**: 특정 주제 셀에만 몰린 단어(배터리·독서)는
  프롬프트 지문이지 문체가 아니므로 탈락시킨다.
- phrases: 문법 표현(의존명사·보조용언·연결 구성)의 빈도비.
  형태소 form 스트림 정규식으로 매칭(종성 자모 ㄹ/ㄴ까지 정확).

입력: 코덱스 코퍼스는 주제별 파일(gen/{genre}_{topic}.txt), 인간 대조군(human/).
출력: 기존 packs/codex-style.json에 content_words·phrases 블록을 병합.

사용: build_lexicon.py <gen_dir> <human_dir> <pack.json>
"""
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

from features_morph import get_kiwi

CONTENT_TAGS = {'NNG', 'VV', 'VA', 'XR'}     # 고유명사(NNP)는 주제어 위험 → 제외
Z_MIN = 2.0
TOPIC_MIN = 4          # 12주제 중 최소 등장 수 (주제 독립성 관문)
MIN_COUNT = 8          # 최소 절대 빈도 (희귀어 노이즈 컷)
PRIOR = 500            # Dirichlet 사전 총질량 a0

# 문법 표현 후보 — 형태소 form 스트림 정규식 (kiwi가 ㄹ→ᆯ, ㄴ→ᆫ 조합형 분해)
PHRASE_CANDIDATES = {
    '수 있다': r'수 있', '수 없다': r'수 없',
    '것이다': r'것 이 다|것 입니다', '-는 것': r'것',
    '뿐만 아니라': r'뿐 만? 아니 라', '-기도 하다': r'기 도 하',
    '-어지다': r'[어아여] 지(?= )', '-어 주다': r'[어아여] 주',
    '-어 버리다': r'[어아여] 버리', '-게 되다': r'게 되',
    '-곤 하다': r'곤 하', '-ㄴ 채': r'ᆫ 채|은 채|는 채',
    '-ㄹ 만큼': r'ᆯ 만큼|을 만큼', '-는 셈이다': r'셈 이',
    '-기 마련이다': r'기 마련', '-ㄹ 뿐이다': r'ᆯ 뿐|을 뿐',
    '-기 때문': r'기 때문', '-ㄴ 덕분': r'ᆫ 덕분|은 덕분',
    '-는 동안': r'는 동안|ᆫ 동안', '인지도 모른다': r'ᆫ지 도 모르|는지 도 모르',
    '-와/과 같이': r'같이|처럼', '-에 따라': r'에 따르|에 따라',
    '-을 통해': r'을 통하|를 통하', '-지 않다': r'지 않',
    '-ㄴ다는 것': r'ᆫ다는|는다는', '-ㄹ 수밖에': r'ᆯ 수밖에|을 수밖에',
}


def topic_of(path):
    m = re.search(r'_(\d+)\.txt$', path)
    return m.group(1) if m else os.path.basename(path)


def lemmas_and_stream(text, kiwi):
    toks = kiwi.tokenize(text)
    lem = [t.form for t in toks if t.tag in CONTENT_TAGS]
    stream = ' '.join(t.form for t in toks)
    return lem, stream


def build_content(gen_paths, hu_paths, kiwi):
    cx, cx_top, hu = Counter(), defaultdict(set), Counter()
    for p in gen_paths:
        lem, _ = lemmas_and_stream(open(p, encoding='utf-8').read(), kiwi)
        t = topic_of(p)
        for w in lem:
            cx[w] += 1
            cx_top[w].add(t)
    for p in hu_paths:
        lem, _ = lemmas_and_stream(open(p, encoding='utf-8').read(), kiwi)
        for w in lem:
            hu[w] += 1
    Nc, Nh = sum(cx.values()), sum(hu.values())
    bg = cx + hu
    Nbg = sum(bg.values())
    out = {}
    for w in bg:
        if cx[w] < MIN_COUNT or len(w) < 2:      # 1음절 파편 제외
            continue
        aw = PRIOR * bg[w] / Nbg
        li = math.log((cx[w] + aw) / (Nc + PRIOR - cx[w] - aw))
        lj = math.log((hu[w] + aw) / (Nh + PRIOR - hu[w] - aw))
        z = (li - lj) / math.sqrt(1 / (cx[w] + aw) + 1 / (hu[w] + aw))
        if z >= Z_MIN and len(cx_top[w]) >= TOPIC_MIN:
            out[w] = {'z': round(z, 2), 'topics': len(cx_top[w]),
                      'codex_per10k': round(10000 * cx[w] / Nc, 2),
                      'human_per10k': round(10000 * hu[w] / Nh, 2)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['z']))


def build_phrases(gen_paths, hu_paths, kiwi):
    def stream_all(paths):
        streams, topics = [], []
        for p in paths:
            _, s = lemmas_and_stream(open(p, encoding='utf-8').read(), kiwi)
            streams.append(s)
            topics.append(topic_of(p))
        return streams, topics
    cx_s, cx_t = stream_all(gen_paths)
    hu_s, _ = stream_all(hu_paths)
    cx_words = sum(len(open(p, encoding='utf-8').read().split()) for p in gen_paths)
    hu_words = sum(len(open(p, encoding='utf-8').read().split()) for p in hu_paths)
    out = {}
    for name, pat in PHRASE_CANDIDATES.items():
        rx = re.compile(pat)
        cc = sum(len(rx.findall(s)) for s in cx_s)
        hc = sum(len(rx.findall(s)) for s in hu_s)
        topics = len({t for t, s in zip(cx_t, cx_s) if rx.search(s)})
        cr = 10000 * cc / cx_words
        hr = 10000 * hc / hu_words
        ratio = (cr + 0.5) / (hr + 0.5)
        if ratio >= 1.5 and topics >= TOPIC_MIN and cr >= 3.0:
            out[name] = {'pattern': pat, 'ratio': round(ratio, 2),
                         'topics': topics, 'codex_per10k': round(cr, 2),
                         'human_per10k': round(hr, 2)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['ratio']))


def main():
    gen_dir, hu_dir, pack_path = sys.argv[1], sys.argv[2], sys.argv[3]
    kiwi = get_kiwi()
    gen = sorted(glob.glob(f'{gen_dir}/*.txt'))
    hu = sorted(glob.glob(f'{hu_dir}/*.txt'))
    print(f'코덱스 {len(gen)}편 / 인간 {len(hu)}편')

    content = build_content(gen, hu, kiwi)
    phrases = build_phrases(gen, hu, kiwi)
    print(f'내용어 사전: {len(content)}개 (관문 z≥{Z_MIN}, 주제≥{TOPIC_MIN})')
    print(f'  상위: {list(content)[:12]}')
    print(f'표현 사전: {len(phrases)}개')
    for k, v in phrases.items():
        print(f"  {k:16s} {v['ratio']:4.1f}배 (주제{v['topics']})")

    pack = json.load(open(pack_path, encoding='utf-8'))
    pack['content_words'] = content
    pack['phrases'] = phrases
    pack['meta']['lexicon_method'] = (
        'content: weighted log-odds (Monroe 2008) + 주제분산 관문; '
        'phrases: 형태소 form 스트림 정규식 빈도비')
    json.dump(pack, open(pack_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'→ {pack_path} 병합 완료')


if __name__ == '__main__':
    main()
