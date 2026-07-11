#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""표층 접미 패턴 기반 L1 특징 추출 (analyzer='surface').

v1 stylometry.py에서 분리. kiwipiepy가 없는 환경의 폴백이자,
형태소 분석기 도입 전후 판별 정확도 비교(설계감사 #11)의 기준선.
오검출은 비교 대상 전체에 동일 규칙로 적용되므로 판별 목적에는 상쇄됨.
"""
import math
import re
from collections import Counter

NAME = 'surface'

# ---------------------------------------------------------------- 문장 분리
_SENT_SPLIT = re.compile(r'(?<=[.!?…])(?=[”"’\']?\s)|(?<=[.!?…][”"’\'])(?=\s)')
_HANGUL = re.compile(r'[가-힣]')

def split_sentences(text):
    sents = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        for s in _SENT_SPLIT.split(line):
            s = s.strip()
            if s and _HANGUL.search(s):
                sents.append(s)
    return sents

# ---------------------------------------------------------------- 특징 정의
# 연결어미(문중 어절 접미). 긴 패턴 우선 매칭.
CONNECTIVES = ['자마자', '으면서', '면서', '는데', '지만', '니까', '다가',
               '려고', '도록', '든지', '거나', '어서', '아서', '으며', '며',
               '고']
# 조사(어절 접미). 긴 패턴 우선.
PARTICLES = ['에서는', '에게서', '으로써', '으로는', '까지', '부터', '조차',
             '마저', '처럼', '보다', '밖에', '에서', '에게', '한테', '께서',
             '마다', '이나', '으로', '하고', '이며', '은', '는', '이', '가',
             '을', '를', '에', '도', '만', '와', '과', '의', '로', '나']

# 부사·접속어 (폐쇄류, 어절 완전일치) — 주제에 덜 오염되는 문체 표지
ADVERBS = ['그리고', '그러나', '그런데', '그래서', '하지만', '또', '또한',
           '다시', '이제', '지금', '아직', '벌써', '이미', '늘', '가끔',
           '문득', '별안간', '갑자기', '결국', '결코', '도무지', '도시',
           '아주', '매우', '몹시', '퍽', '썩', '좀', '꼭', '흠씬', '가장',
           '제법', '거반', '모두', '다', '왜', '어찌', '과연', '물론',
           '역시', '차라리', '오히려', '무릇', '비로소', '어느덧']

_PUNCT_STRIP = re.compile(r'[“”"‘’\'.,!?…―\-)(』」》〉\]]+$')
_REDUP = re.compile(r'^(..)\1')          # 어정어정, 데굴데굴

def final_ending(sentence):
    """문장 마지막 어절에서 종결어미 표층형(끝 1~2음절)을 뽑는다."""
    eojeols = sentence.split()
    if not eojeols:
        return None
    last = _PUNCT_STRIP.sub('', eojeols[-1])
    syls = [c for c in last if '가' <= c <= '힣']
    if not syls:
        return None
    return ''.join(syls[-2:]) if len(syls) >= 2 else syls[-1]

def eojeol_suffix_counts(eojeols, patterns):
    c = Counter()
    for e in eojeols:
        e = _PUNCT_STRIP.sub('', e)
        for p in patterns:
            if len(e) > len(p) and e.endswith(p):
                c[p] += 1
                break
    return c

# ---------------------------------------------------------------- 공용 스칼라
def scalar_features(sentences):
    """문장부호·길이 계열 스칼라 — 분석기와 무관한 텍스트 층위.

    features_morph도 이 함수를 공유해 스칼라 정의를 통일한다.
    """
    n_sent = len(sentences)
    lens, n_dialog, n_comma, n_excl, n_quest, n_ellip, n_dash, n_redup = \
        [], 0, 0, 0, 0, 0, 0, 0
    n_ej = 0
    for s in sentences:
        ej = s.split()
        lens.append(len(ej))
        n_ej += len(ej)
        if s.lstrip().startswith(('“', '"', '‘')):
            n_dialog += 1
        n_comma += s.count(',')
        n_excl += s.count('!')
        n_quest += s.count('?')
        n_ellip += s.count('……')
        n_dash += s.count('―')
        for w in ej:
            w2 = _PUNCT_STRIP.sub('', w)
            if len(w2) >= 4 and _REDUP.match(w2):
                n_redup += 1
    n_ej = max(n_ej, 1)
    mean_len = sum(lens) / n_sent
    var_len = sum((x - mean_len) ** 2 for x in lens) / n_sent
    sl = sorted(lens)
    return n_ej, {
        'sent_len_mean': mean_len,
        'sent_len_std': math.sqrt(var_len),
        'sent_len_p90': sl[int(0.9 * (n_sent - 1))],
        'dialog_ratio': n_dialog / n_sent,
        'comma_per_sent': n_comma / n_sent,
        'excl_per_sent': n_excl / n_sent,
        'quest_per_sent': n_quest / n_sent,
        'ellipsis_per_sent': n_ellip / n_sent,
        'dash_per_1k_ej': 1000 * n_dash / n_ej,
        'redup_per_1k_ej': 1000 * n_redup / n_ej,   # 첩어(의태어) 밀도
    }

# ---------------------------------------------------------------- 청크 특징
def chunk_features(sentences):
    """문장 리스트(청크 하나) → 특징 딕셔너리."""
    all_eojeols, mid_eojeols = [], []
    endings = Counter()
    for s in sentences:
        ej = s.split()
        all_eojeols += ej
        mid_eojeols += ej[:-1]                      # 마지막 어절 제외 → 연결어미용
        e = final_ending(s)
        if e:
            endings[e] += 1
    n_ej, scalars = scalar_features(sentences)
    conn = eojeol_suffix_counts(mid_eojeols, CONNECTIVES)
    part = eojeol_suffix_counts(all_eojeols, PARTICLES)
    adv_set = set(ADVERBS)
    adv = Counter(w2 for w in all_eojeols
                  if (w2 := _PUNCT_STRIP.sub('', w)) in adv_set)
    return {
        'n_sent': len(sentences), 'n_eojeol': n_ej,
        'endings': dict(endings), 'connectives': dict(conn),
        'particles': dict(part), 'adverbs': dict(adv),
        'scalars': scalars,
    }
