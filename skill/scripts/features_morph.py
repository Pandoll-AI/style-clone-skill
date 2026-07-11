#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kiwipiepy 형태소 태그 기반 L1 특징 추출 (analyzer='kiwi').

로드맵 P0: 표층 접미 패턴을 EF(종결어미)/EC(연결어미)/J*(조사)/MAG·MAJ(부사)
태그로 교체. 문장 분리도 정규식 대신 Kiwi.split_into_sents 사용 (설계감사 #12).

스칼라(문장부호·길이) 계열은 분석기와 무관한 텍스트 층위이므로
features_surface.scalar_features를 공유해 정의를 통일한다 — 표층 vs 형태소
벤치마크 비교에서 범주 분포 계열만 순수하게 달라지게 하기 위함.
"""
import re
from collections import Counter

from features_surface import scalar_features

NAME = 'kiwi'

_HANGUL = re.compile(r'[가-힣]')
_kiwi = None


def get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


# ---------------------------------------------------------------- 문장 분리
def split_sentences(text):
    kiwi = get_kiwi()
    sents = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        for s in kiwi.split_into_sents(line):
            t = s.text.strip()
            if t and _HANGUL.search(t):
                sents.append(t)
    return sents


# ---------------------------------------------------------------- 태그 집합
PARTICLE_TAGS = {'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'}
ADVERB_TAGS = {'MAG', 'MAJ'}
NOMINAL_TAGS = {'NNG', 'NNP', 'NNB', 'NR', 'NP', 'XSN', 'ETN', 'SN'}


def _tokens(sentence):
    return get_kiwi().tokenize(sentence)


def final_ending(sentence):
    """문장의 종결어미 형태소(EF 형태)를 뽑는다. 명사 종결은 '명사종결'로 범주화."""
    toks = _tokens(sentence)
    if not toks:
        return None
    last_ef = None
    for t in toks:
        if t.tag == 'EF':
            last_ef = t.form
    if last_ef:
        return last_ef
    # EF 없음 → 마지막 실질 토큰으로 명사 종결 여부 판정
    for t in reversed(toks):
        if t.tag.startswith('S'):        # 문장부호·기호는 건너뜀
            continue
        return '명사종결' if t.tag in NOMINAL_TAGS else None
    return None


# ---------------------------------------------------------------- 청크 특징
def chunk_features(sentences):
    """문장 리스트(청크 하나) → 특징 딕셔너리 (features_surface와 동일 스키마)."""
    endings, conn, part, adv = Counter(), Counter(), Counter(), Counter()
    for s in sentences:
        toks = _tokens(s)
        last_ef_i = None
        for i, t in enumerate(toks):
            if t.tag == 'EF':
                last_ef_i = i
            if t.tag == 'EC':
                conn[t.form] += 1
            elif t.tag in PARTICLE_TAGS:
                part[t.form] += 1
            elif t.tag in ADVERB_TAGS:
                adv[t.form] += 1
        if last_ef_i is not None:
            endings[toks[last_ef_i].form] += 1
        else:
            for t in reversed(toks):
                if t.tag.startswith('S'):
                    continue
                if t.tag in NOMINAL_TAGS:
                    endings['명사종결'] += 1
                break
    n_ej, scalars = scalar_features(sentences)
    return {
        'n_sent': len(sentences), 'n_eojeol': n_ej,
        'endings': dict(endings), 'connectives': dict(conn),
        'particles': dict(part), 'adverbs': dict(adv),
        'scalars': scalars,
    }
