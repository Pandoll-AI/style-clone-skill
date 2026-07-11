#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""확장 원자 특징 (P1.5 S1) — L2 통사 · L3 어휘 · L7 화용 · CX 구문 패턴.

요인 분석(docs/06)의 원자 재료. 설계 원칙:
- T0 특징: 정규식·소사전 — 의존성 없이 항상 계산.
- T1 특징: kiwipiepy 태그 시퀀스 — 없으면 해당 키 부재(None 아님, 생략).
  요인 채점부는 부재 특징에 의존하는 요인을 건너뛰고 보고한다.
- 단위: `_ej` 접미 = 어절당 ×100, `_sent` 접미 = 문장당 비율.
- 모든 특징은 QG1(비퇴화·비중복)과 요인 관문(QG3)을 통과해야 채점에 편입.
  여기 있다는 것만으로는 채용이 아니다 (docs/04 채용 관문).

한계 기록: 고유어/한자어 어종 구분은 어종 사전 부재로 이번 단계 보류
(kiwipiepy는 어원 정보를 주지 않음) — 추상도 접미·한자 병기·로마자 밀도로 근사.
"""
import re

# ---------------------------------------------------------------- T0 사전
HEDGES = ['아마', '어쩌면', '자칫', '혹시', '듯하', '듯싶', '성싶', '모양이',
          '지도 모른', '것 같', '인 듯', '싶다', '싶었', '글쎄']
BOOSTERS = ['반드시', '분명', '결코', '확실히', '틀림없', '절대', '단연',
            '기어이', '기필코', '어김없']
FIRST_PERSON = {'나', '나는', '나를', '나의', '나도', '나만', '내', '내가',
                '우리', '우리는', '우리의', '우리가', '저는', '제가'}
SECOND_PERSON = {'당신', '당신은', '당신이', '여러분', '그대', '그대는', '너는', '네가'}
COLLOQUIAL = ['뭐', '되게', '진짜', '그냥', '막상', '얘기', '이랬', '저랬', '요즘']
SENSORY = ['반짝', '어둑', '붉', '푸르', '누렇', '검붉', '향기', '냄새', '고요',
           '소리', '차갑', '뜨겁', '부드럽', '거칠', '달콤', '씁쓸', '비릿',
           '눅눅', '서늘', '따뜻', '싸늘', '환하', '컴컴']
META_DISCOURSE = ['결론적으로', '요컨대', '정리하면', '즉', '다시 말해', '한마디로',
                  '바꾸어 말하면', '요약하면']

_ROMAN = re.compile(r'[A-Za-z]')
_HANJA = re.compile(r'[一-鿿]')
_DIGIT = re.compile(r'[0-9]')
_NOT_A_BUT_B = re.compile(r'[이가는] 아니라|[이가] 아니고|는 아니다')
_HAL_SU = re.compile(r'[할알볼갈올쓸열될낼줄] 수 (있|없)')
_GEOT_END = re.compile(r'것[이입]?[다니]?[다까]?[.!?…]*\s*$')
_TRIPLE = re.compile(r'\S+,\s*\S+,\s*\S+')
_COND = re.compile(r'(다면|라면|으면 몰라도)')
_RHET_Q = re.compile(r'(는가|ㄴ가|을까|ㄹ까|랴|는고)[.…]\s*$')
_PAREN_COLON = re.compile(r'[()：:]')

_SYL = re.compile(r'[가-힣]')


def _rate(count, base):
    return count / base if base else 0.0


# ---------------------------------------------------------------- T0 본체
def t0_features(sents):
    n_sent = len(sents)
    eojeols = [w for s in sents for w in s.split()]
    n_ej = max(len(eojeols), 1)
    text = '\n'.join(sents)

    def per_ej(count):
        return 100.0 * _rate(count, n_ej)

    def per_sent(count):
        return _rate(count, n_sent)

    hedge = sum(text.count(h) for h in HEDGES)
    boost = sum(text.count(b) for b in BOOSTERS)
    fp = sum(1 for w in eojeols if w.strip('.,!?…"“”') in FIRST_PERSON)
    sp = sum(1 for w in eojeols if w.strip('.,!?…"“”') in SECOND_PERSON)
    colloq = sum(text.count(c) for c in COLLOQUIAL)
    sensory = sum(text.count(s) for s in SENSORY)
    meta = sum(1 for s in sents
               if any(s.lstrip('“"‘ ').startswith(m) for m in META_DISCOURSE))
    syl_lens = [len(_SYL.findall(w)) for w in eojeols]
    return {
        'L3_sensory_ej': per_ej(sensory),
        'L3_colloquial_ej': per_ej(colloq),
        'L3_roman_ej': per_ej(sum(1 for w in eojeols if _ROMAN.search(w))),
        'L3_hanja_ej': per_ej(sum(1 for w in eojeols if _HANJA.search(w))),
        'L3_digit_ej': per_ej(sum(1 for w in eojeols if _DIGIT.search(w))),
        'L3_word_syllables': _rate(sum(syl_lens), max(len(syl_lens), 1)),
        'L7_hedge_sent': per_sent(hedge),
        'L7_booster_sent': per_sent(boost),
        'L7_first_person_sent': per_sent(fp),
        'L7_second_person_sent': per_sent(sp),
        'CX_not_a_but_b_sent': per_sent(len(_NOT_A_BUT_B.findall(text))),
        'CX_hal_su_sent': per_sent(len(_HAL_SU.findall(text))),
        'CX_geot_end_sent': per_sent(sum(1 for s in sents if _GEOT_END.search(s))),
        'CX_triple_list_sent': per_sent(sum(1 for s in sents if _TRIPLE.search(s))),
        'CX_meta_discourse_sent': per_sent(meta),
        'CX_conditional_sent': per_sent(len(_COND.findall(text))),
        'CX_rhet_q_period_sent': per_sent(sum(1 for s in sents if _RHET_Q.search(s))),
        'CX_paren_colon_sent': per_sent(len(_PAREN_COLON.findall(text))),
    }


# ---------------------------------------------------------------- T1 (kiwi)
VERB_TAGS = {'VV', 'VA', 'VX', 'VCP', 'VCN'}
SUBJ_TAGS = {'JKS'}          # 주격조사. 보조사 '은/는' 주어는 근사에서 제외(엄격판)
ABSTRACT_SUFFIX = {'성', '화', '도', '론', '적'}


def t1_features(sents):
    """kiwipiepy 필요. 실패 시 빈 dict (호출부가 부재를 처리)."""
    try:
        from features_morph import get_kiwi
        kiwi = get_kiwi()
    except Exception:
        return {}
    n_sent = len(sents)
    n_etn = n_geot = n_etm = n_xsv = n_verb = n_ic = n_jkq = n_abs = 0
    n_ej = 0
    no_subj = 0
    etm_chain_max_sum = 0
    for s in sents:
        toks = kiwi.tokenize(s)
        n_ej += max(len(s.split()), 1)
        has_subj = False
        chain = max_chain = 0
        for t in toks:
            tag, form = t.tag, t.form
            if tag == 'ETN':
                n_etn += 1
            elif tag == 'ETM':
                n_etm += 1
                chain += 1
                max_chain = max(max_chain, chain)
                continue
            elif tag == 'XSV':
                n_xsv += 1
            elif tag in VERB_TAGS:
                n_verb += 1
            elif tag == 'IC':
                n_ic += 1
            elif tag == 'JKQ':
                n_jkq += 1
            elif tag in SUBJ_TAGS:
                has_subj = True
            if tag == 'NNB' and form == '것':
                n_geot += 1
            if tag in ('XSN', 'NNG') and form in ABSTRACT_SUFFIX:
                n_abs += 1
            chain = 0
        if not has_subj:
            no_subj += 1
        etm_chain_max_sum += max_chain
    n_ej = max(n_ej, 1)
    return {
        'L2_nominalization_ej': 100.0 * (n_etn + n_geot) / n_ej,
        'L2_etm_ej': 100.0 * n_etm / n_ej,
        'L2_xsv_ej': 100.0 * n_xsv / n_ej,
        'L2_verb_density_ej': 100.0 * n_verb / n_ej,
        'L2_subject_ellipsis_sent': no_subj / n_sent,
        'L2_etm_chain_max': etm_chain_max_sum / n_sent,
        'L3_abstract_suffix_ej': 100.0 * n_abs / n_ej,
        'L7_interjection_sent': n_ic / n_sent,
        'L7_quote_particle_sent': n_jkq / n_sent,
    }


def ext_features_from_sents(sents, use_kiwi=True):
    """확장 특징 딕셔너리. kiwi 부재 시 T1 키는 생략된다."""
    if len(sents) < 2:
        raise ValueError('문장이 너무 적음')
    out = t0_features(sents)
    if use_kiwi:
        out.update(t1_features(sents))
    return out


T0_KEYS = ['L3_sensory_ej', 'L3_colloquial_ej', 'L3_roman_ej', 'L3_hanja_ej',
           'L3_digit_ej', 'L3_word_syllables', 'L7_hedge_sent',
           'L7_booster_sent', 'L7_first_person_sent', 'L7_second_person_sent',
           'CX_not_a_but_b_sent', 'CX_hal_su_sent', 'CX_geot_end_sent',
           'CX_triple_list_sent', 'CX_meta_discourse_sent',
           'CX_conditional_sent', 'CX_rhet_q_period_sent', 'CX_paren_colon_sent']
T1_KEYS = ['L2_nominalization_ej', 'L2_etm_ej', 'L2_xsv_ej',
           'L2_verb_density_ej', 'L2_subject_ellipsis_sent',
           'L2_etm_chain_max', 'L3_abstract_suffix_ej',
           'L7_interjection_sent', 'L7_quote_particle_sent']
ALL_KEYS = T0_KEYS + T1_KEYS
