#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""담화·논리 층 문체 특징 추출기 (v2 — 표층 프록시 구현).

v1(stylometry.py)이 '어떻게 쓰는가'의 형태 층을 측정한다면,
이 모듈은 '어떻게 생각을 잇는가'를 측정한다. 외부 모델 없이
계산 가능한 프록시만 구현했고, 임베딩·LM 기반 정밀판의 인터페이스는
embedding_features_spec.py 에 명세되어 있다.

측정 축
=======
A. 논리 연결 구조  — 문두 연결 표지의 유형 분포 + 전이 행렬 + 무표지율.
   '탄탄한 논리'는 명시 연결이 많고 전이가 규칙적이며,
   '사고의 비행'은 무표지 + 큰 의미 도약으로 나타난다.
B. 의미 도약(추론 보폭) — 인접 문장 내용어 중첩의 역수. 분포의 평균이
   보폭, 꼬리가 비행, 도약 후 복귀율이 '비행 vs 단절'을 가른다.
C. 전개 템포 — 문장 길이·도약 크기 시계열의 변동계수, 자기상관,
   부호 교대율. '템포가 일정한 필자'는 자기상관과 교대 패턴이 안정적.
D. 정보 흐름 — 문장별 신규 내용어 도입률, 어휘 성장 곡선의 사분위 기울기.
E. 거시 구조(기승전결) — 사분위별 특징 프로파일 + 수미상관(echo) 점수.
F. 산포·평탄화 — MTLD-lite, 내용어 burstiness, 문장 길이 Gini.
   LLM 평탄화(균질화)는 실증된 현상이므로, 채점기는 평균뿐 아니라
   산포를 일치시켜야 하며 산포 미달은 명시적으로 감점해야 한다.
"""
import json
import math
import re
import sys
from collections import Counter

from features_surface import split_sentences, _PUNCT_STRIP

# ---------------------------------------------------------------- 연결 표지
CONNECTIVE_TYPES = {
    '인과': ['그래서', '따라서', '그러므로', '왜냐하면', '결국', '그러니까',
           '그리하여', '이때문에', '덕분에'],
    '역접': ['하지만', '그러나', '그런데', '반면', '오히려', '그렇지만',
           '반대로', '하나'],
    '부연': ['즉', '다시', '사실', '실제로', '말하자면', '요컨대', '결국은'],
    '예시': ['예를', '가령', '이를테면'],
    '전환': ['그렇다면', '한편', '이제', '그러면', '자'],
    '첨가': ['그리고', '또한', '게다가', '더구나', '뿐만'],
}
_TYPE_OF = {w: t for t, ws in CONNECTIVE_TYPES.items() for w in ws}
TYPES = list(CONNECTIVE_TYPES) + ['무표지']

# 내용어 추출용: 조사 접미 제거(표층 근사)
_PARTICLE_TAIL = re.compile(
    r'(에서는|에게서|으로써|까지|부터|조차|마저|처럼|보다|밖에|에서|에게|한테|'
    r'께서|마다|이나|으로|하고|이며|은|는|이|가|을|를|에|도|만|와|과|의|로|나)$')
_STOP = set('그 이 저 것 수 등 때 년 월 일 더 덜 안 못 왜 다 또 좀 잘'.split())

def content_stems(sentence):
    out = set()
    for w in sentence.split():
        w = _PUNCT_STRIP.sub('', w)
        w = re.sub(r'^[“”"‘’\'『「《〈(]+', '', w)
        w2 = _PARTICLE_TAIL.sub('', w)
        if len(w2) >= 2 and w2 not in _STOP:
            out.add(w2[:3])          # 표층 어간 근사(앞 3음절)
    return out

def sentence_connective(sentence):
    first = _PUNCT_STRIP.sub('', sentence.split()[0]) if sentence.split() else ''
    first = re.sub(r'^[“”"‘’\']+', '', first)
    for w, t in _TYPE_OF.items():
        if first.startswith(w):
            return t
    return '무표지'

# ---------------------------------------------------------------- 유틸
def _mean(xs): return sum(xs) / len(xs) if xs else 0.0
def _std(xs):
    if len(xs) < 2: return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def _autocorr1(xs):
    if len(xs) < 3: return 0.0
    m, s = _mean(xs), _std(xs)
    if s == 0: return 0.0
    return _mean([(xs[i] - m) * (xs[i + 1] - m) for i in range(len(xs) - 1)]) / (s * s)

def _gini(xs):
    xs = sorted(xs)
    n, s = len(xs), sum(xs)
    if n == 0 or s == 0: return 0.0
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * s)

# 지문 통합(stylometry.py)에서 청크 분산을 낼 스칼라 특징 키.
# 사분위(D_/E_*_by_quartile)·전이(top_transitions)는 전체 글 단위 특징이라 제외.
SCALAR_KEYS = ['A_implicit_ratio', 'B_jump_mean', 'B_jump_std', 'B_jump_p90',
               'B_flight_return_rate', 'C_len_cv', 'C_len_autocorr',
               'C_len_alternation', 'C_jump_autocorr', 'E_echo_head_tail',
               'F_mtld_lite', 'F_burstiness_fano', 'F_len_gini']

# 산포 계열 (측정명세 v2 §채점규칙 2): 후보 값 < 작가 평균이면 '과평탄' —
# 비대칭 벌점 α의 대상. LLM 오류 모드가 평탄화로 치우친다는 문헌 실증 근거.
DISPERSION_KEYS = {'B_jump_std', 'C_len_cv', 'F_mtld_lite',
                   'F_burstiness_fano', 'F_len_gini'}

MIN_SENTS = 4            # 이하면 담화 특징 계산 불가
QUARTILE_MIN_SENTS = 40  # 미만이면 사분위 프로파일 불안정 → 전역 백오프 (측정명세 §한계)


# ---------------------------------------------------------------- 본체
def discourse_features(text):
    return discourse_features_from_sents(split_sentences(text))


def discourse_features_from_sents(sents):
    n = len(sents)
    if n < MIN_SENTS:
        raise ValueError('문장이 너무 적음')
    stems = [content_stems(s) for s in sents]
    lens = [len(s.split()) for s in sents]

    # A. 논리 연결 구조
    conns = [sentence_connective(s) for s in sents]
    conn_dist = {t: conns.count(t) / n for t in TYPES}
    trans = Counter(zip(conns, conns[1:]))
    top_trans = {f'{a}→{b}': c / max(len(conns) - 1, 1)
                 for (a, b), c in trans.most_common(5)}

    # B. 의미 도약 (인접 문장 내용어 비중첩)
    jumps = []
    for i in range(1, n):
        a, b = stems[i - 1], stems[i]
        if not a or not b:
            continue
        jumps.append(1.0 - len(a & b) / min(len(a), len(b)))
    # 도약 후 복귀: i-1과 무관해도 i-2/i-3과 이어지면 '비행 후 귀환'
    returns = 0; big = 0
    for i in range(3, n):
        a, b = stems[i - 1], stems[i]
        if not a or not b: continue
        j = 1.0 - len(a & b) / min(len(a), len(b))
        if j > 0.8:
            big += 1
            back = stems[i - 2] | stems[i - 3]
            if len(stems[i] & back) / max(len(stems[i]), 1) > 0.2:
                returns += 1
    return_rate = returns / big if big else 1.0

    # C. 템포
    diffs = [lens[i + 1] - lens[i] for i in range(n - 1)]
    alternation = _mean([1.0 if diffs[i] * diffs[i + 1] < 0 else 0.0
                         for i in range(len(diffs) - 1) if diffs[i] and diffs[i+1]])

    # D. 정보 흐름 (신규 내용어 도입)
    seen = set(); newness = []
    for st in stems:
        if st:
            newness.append(len(st - seen) / len(st)); seen |= st
    q = max(n // 4, 1)
    new_by_q = [_mean(newness[i * q:(i + 1) * q]) for i in range(4)]

    # E. 거시 구조 (사분위 프로파일 + 수미상관)
    def qprofile(vals):
        return [round(_mean(vals[i * q:(i + 1) * q]), 3) for i in range(4)]
    quest = [1.0 if re.search(r'(까|가|냐|는가)\s*[.?]?$', s) or '?' in s else 0.0
             for s in sents]
    d = max(n // 10, 2)
    head = set().union(*stems[:d]); tail = set().union(*stems[-d:])
    echo = len(head & tail) / max(min(len(head), len(tail)), 1)

    # F. 산포·평탄화
    tokens = [w for st in stems for w in st]
    seg = 50
    ttrs = [len(set(tokens[i:i + seg])) / seg
            for i in range(0, len(tokens) - seg + 1, seg)]
    top_words = [w for w, _ in Counter(tokens).most_common(10)]
    fanos = []
    for w in top_words:
        occ = [1.0 if w in st else 0.0 for st in stems]
        m = _mean(occ)
        if m > 0:
            fanos.append(_std(occ) ** 2 / m)

    return {
        'n_sent': n,
        'A_connective_dist': {k: round(v, 3) for k, v in conn_dist.items()},
        'A_implicit_ratio': round(conn_dist['무표지'], 3),
        'A_top_transitions': top_trans,
        'B_jump_mean': round(_mean(jumps), 3),
        'B_jump_std': round(_std(jumps), 3),
        'B_jump_p90': round(sorted(jumps)[int(0.9 * (len(jumps) - 1))], 3),
        'B_flight_return_rate': round(return_rate, 3),
        'C_len_cv': round(_std(lens) / _mean(lens), 3),
        'C_len_autocorr': round(_autocorr1(lens), 3),
        'C_len_alternation': round(alternation, 3),
        'C_jump_autocorr': round(_autocorr1(jumps), 3),
        'D_new_info_by_quartile': [round(x, 3) for x in new_by_q],
        'E_question_by_quartile': qprofile(quest),
        'E_len_by_quartile': qprofile([float(x) for x in lens]),
        'E_echo_head_tail': round(echo, 3),
        'F_mtld_lite': round(_mean(ttrs), 3) if ttrs else None,
        'F_burstiness_fano': round(_mean(fanos), 3) if fanos else None,
        'F_len_gini': round(_gini([float(x) for x in lens]), 3),
    }

if __name__ == '__main__':
    for path in sys.argv[1:]:
        f = discourse_features(open(path, encoding='utf-8').read())
        print(f'== {path}')
        print(json.dumps(f, ensure_ascii=False, indent=1))
