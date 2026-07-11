#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1–T4 정밀 특징의 인터페이스 명세 (모델 필요 — 이 샌드박스 외부에서 실행).

각 함수는 시그니처·수식·권장 모델·검증 기준을 고정한 스텁이다.
discourse.py 의 T0 프록시와 이름을 맞춰 두었으므로, 구현 후
지문 스키마에 그대로 끼워 넣으면 된다.
"""

# 권장 모델 (2026-07 기준, 로컬 실행 가능 규모)
MODELS = {
    'morph':     'kiwipiepy >= 0.18',                  # T1
    'embedding': 'BAAI/bge-m3 또는 BM-K/KoSimCSE-roberta',  # T2
    'lm':        'EleutherAI/polyglot-ko-1.3b',        # T3
    'nli':       'klue/roberta-large + KLUE-NLI 파인튜닝',  # T4
}

def semantic_jump(sentences, embed_fn):
    """j_i = 1 - cos(e(s_{i-1}), e(s_i)).

    반환: {'mean','std','p75','p90','tail_index'} — 추론 보폭 분포.
    검증: discourse.B_jump_mean과의 스피어만 상관 보고. 상관 > 0.8이면
    T0 유지 권장(해석 가능성 이득), 미만이면 교체.
    """
    raise NotImplementedError

def topic_drift_curve(sentences, embed_fn, max_lag=8):
    """D(k) = mean_i cos(e(s_i), e(s_{i+k})). 감쇠율(지수 적합 계수)이
    '주제를 오래 붙드는 필자 vs 빨리 미끄러지는 필자'를 가른다."""
    raise NotImplementedError

def entailment_gap(sentences, nli_fn):
    """gap_i = 1 - max(P_entail(ctx_i -> s_i), P_entail(s_i -> ctx_i)),
    ctx_i = s_{i-2}+s_{i-1}. '천재 스킵'의 조작적 정의:
    gap 상위 25% & topic_drift 정상 & 블라인드 독자 판별 통과."""
    raise NotImplementedError

def surprisal_contour(text, lm_score_fn):
    """어절 단위 surprisal 시계열 → {'mean'(정보 밀도),
    'cv'(템포 일정함), 'autocorr1', 'lowfreq_ratio'(저주파 스펙트럼
    비율 = 리듬 주기성)}. 근거: UID + Harmonic Structure of
    Information Contours (arXiv:2506.03902)."""
    raise NotImplementedError

def idea_density(text, tagger):
    """kiwipiepy 태그에서 (VV+VA+MAG+MAJ+XSV) / 어절수.
    CPIDR의 한국어 근사."""
    raise NotImplementedError

def syntax_profile(text, tagger):
    """관형형 어미(ETM) 밀도, '것' 명사화 밀도, 주어 생략률
    (JKS 부재 문장 비율), 피동 접미(XSV-피동) 비율."""
    raise NotImplementedError

def entity_grid(text, tagger, coref_fn=None):
    """개체(NNG/NNP 반복어)의 문장별 역할(S/O/X) 행렬 →
    역할 전이 bigram 분포 (Barzilay & Lapata 2008). 일관성의 지문."""
    raise NotImplementedError

def hedge_booster_ratio(text):
    """T0 가능. hedge = {~일지도, ~인 듯, 아마, 어쩌면, ~같다, ~수 있다},
    booster = {반드시, 분명, 결코, 절대, 확실히, 틀림없이}.
    비율 + 위치(사분위)별 분포 — 확신의 템포."""
    raise NotImplementedError
