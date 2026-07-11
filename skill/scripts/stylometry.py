#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""문체 지문(Style Fingerprint) 추출·채점 엔진.

설계 원칙
=========
1. 지문 = 측정층 + 생성층.
   측정층: 종결어미·연결어미·조사·문장부호·문장길이 등 분포 벡터
           + '문서(청크) 단위 분산' — 수용 영역 판정의 근거.
   생성층: 각 종결어미 범주의 실제 예문 뱅크 + 금지 목록.
2. 채점은 '평균과의 거리 최소화'가 아니라 '작가 자신의 청크 분산으로
   정규화한 z-점수' → 캐리커처(작가보다 더 작가스러움) 방지.
3. 재순위화용 특징(형태 표층 분포)과 판별용 특징(문자 bigram)을
   분리 → Goodhart 순환 차단.

분석기 이중화 (P0)
==================
- analyzer='surface': 표층 접미 패턴 (features_surface) — 의존성 없는 폴백.
- analyzer='kiwi':    kiwipiepy EF/EC/J*/MAG 태그 (features_morph) — 정밀판.
지문 JSON에 analyzer가 기록되며, 채점은 지문의 analyzer를 따라간다.
서로 다른 분석기의 지문끼리 d를 비교하는 것은 오류다(분포 vocab이 다른
공간에 살기 때문) — 벤치마크에서는 분석기별로 지문 세트를 통일할 것.
"""
import json
import math
import re
import sys
from collections import Counter

import features_surface

# ---------------------------------------------------------------- 분석기 등록
_ANALYZERS = {'surface': features_surface}

def get_analyzer(name):
    if name not in _ANALYZERS:
        if name == 'kiwi':
            import features_morph
            _ANALYZERS['kiwi'] = features_morph
        else:
            raise ValueError(f'unknown analyzer: {name!r} (surface|kiwi)')
    return _ANALYZERS[name]

# 하위 호환 재수출 (validate.py 등 v1 코드용 — surface 기준)
split_sentences = features_surface.split_sentences
chunk_features = features_surface.chunk_features
final_ending = features_surface.final_ending
CONNECTIVES = features_surface.CONNECTIVES
PARTICLES = features_surface.PARTICLES
ADVERBS = features_surface.ADVERBS
_HANGUL = re.compile(r'[가-힣]')

# ---------------------------------------------------------------- 분포 도구
def _normalize(counter, vocab):
    tot = sum(counter.get(k, 0) for k in vocab) + 1e-9
    return [counter.get(k, 0) / tot for k in vocab]

def js_divergence(p, q):
    def kl(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)
    p = [x + 1e-12 for x in p]; q = [x + 1e-12 for x in q]
    sp, sq = sum(p), sum(q)
    p = [x / sp for x in p]; q = [x / sq for x in q]
    m = [(x + y) / 2 for x, y in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

CAT_FAMILIES = ['endings', 'connectives', 'particles', 'adverbs']

# ---------------------------------------------------------------- 지문 구축
def build_fingerprint(text, author, chunk_sents=15, contrast_text=None,
                      example_bank_size=3, analyzer='surface',
                      with_discourse=True):
    """with_discourse=True(기본): 담화층(L4~L9) T0 특징을 지문에 통합 (P1).

    담화층의 채용 근거는 판별 정확도가 아니라 실패 모드 검출이다 —
    v1 승자 C2의 평탄화 오버피팅을 산포 지표가 잡아냈다 (README §실증 4).
    v1 수치 재현이 필요한 회귀 실험은 False로 끈다 (validate.py).
    """
    mod = get_analyzer(analyzer)
    sents = mod.split_sentences(text)
    chunks = [sents[i:i + chunk_sents]
              for i in range(0, len(sents) - chunk_sents + 1, chunk_sents)]
    feats = [mod.chunk_features(c) for c in chunks]
    corpus = mod.chunk_features(sents)

    fp = {'author': author, 'analyzer': analyzer, 'n_chunks': len(chunks),
          'chunk_sents': chunk_sents, 'n_sentences': len(sents)}

    # (1) 범주 분포: 코퍼스 평균 + 청크별 JS 산포 (수용 영역)
    fp['cat'] = {}
    for fam in CAT_FAMILIES:
        vocab = sorted(corpus[fam], key=corpus[fam].get, reverse=True)[:25]
        mean_dist = _normalize(Counter(corpus[fam]), vocab)
        js_list = [js_divergence(_normalize(Counter(f[fam]), vocab), mean_dist)
                   for f in feats]
        mu = sum(js_list) / len(js_list)
        sd = math.sqrt(sum((x - mu) ** 2 for x in js_list) / len(js_list))
        fp['cat'][fam] = {'vocab': vocab, 'mean_dist': mean_dist,
                          'js_mean': mu, 'js_std': max(sd, 1e-6)}

    # (2) 스칼라: 청크 평균·표준편차 (작가 내 변동 = 수용 영역의 폭)
    #     std 하한 = max(절대 하한, 평균의 15%) → 상수 특징의 z 폭발 방지
    fp['scalars'] = {}
    for k in corpus['scalars']:
        vals = [f['scalars'][k] for f in feats]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
        fp['scalars'][k] = {'mean': mu,
                            'std': max(sd, 0.15 * abs(mu), 0.05)}

    # (3) 금지 목록: 대조 코퍼스에서 흔한데 이 작가는 안 쓰는 표지
    fp['ban_list'] = {}
    if contrast_text:
        contrast = mod.chunk_features(mod.split_sentences(contrast_text))
        for fam in CAT_FAMILIES:
            mine, other = corpus[fam], contrast[fam]
            tot_o = sum(other.values()) + 1e-9
            tot_m = sum(mine.values()) + 1e-9
            banned = [k for k, v in other.items()
                      if v / tot_o > 0.01 and mine.get(k, 0) / tot_m < 0.001]
            fp['ban_list'][fam] = banned

    # (4) 예문 뱅크: 상위 종결어미별 실제 문장
    bank = {}
    top_endings = fp['cat']['endings']['vocab'][:12]
    for e in top_endings:
        ex = [s for s in sents
              if mod.final_ending(s) == e and 5 <= len(s.split()) <= 25]
        bank[e] = ex[:example_bank_size]
    fp['example_bank'] = bank

    # (5) 담화층 (P1): 스칼라는 청크 분산 포함, 연결 유형은 분포+JS 산포,
    #     사분위 프로파일은 전체 코퍼스 기준(분산 없음 — d 전용).
    if with_discourse:
        import discourse as disc
        d_feats = []
        for c in chunks:
            try:
                d_feats.append(disc.discourse_features_from_sents(c))
            except ValueError:
                continue
        d_corpus = disc.discourse_features_from_sents(sents)
        dsc = {'scalars': {}, 'quartile': {
            'D_new_info': d_corpus['D_new_info_by_quartile'],
            'E_question': d_corpus['E_question_by_quartile'],
            'E_len': d_corpus['E_len_by_quartile'],
        }}
        for k in disc.SCALAR_KEYS:
            vals = [f[k] for f in d_feats if f.get(k) is not None]
            if len(vals) < 2:
                continue
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
            # corpus: 전체 코퍼스 단위 실측값. 산포 계열은 창 크기에 따라
            # 값이 커지므로(짧은 청크의 cv < 전체 cv) 과평탄 판정의 기준은
            # 청크 평균이 아니라 이 값을 쓴다 — v2 리포트의 C2 실측
            # (len_cv 0.34 vs 작가 0.53)과 같은 비교 축.
            dsc['scalars'][k] = {'mean': mu,
                                 'std': max(sd, 0.15 * abs(mu), 0.02),
                                 'corpus': d_corpus.get(k)}
        conn_dists = [f['A_connective_dist'] for f in d_feats]
        vocab = disc.TYPES
        mean_dist = _normalize(Counter(
            {t: sum(cd.get(t, 0) for cd in conn_dists) for t in vocab}), vocab)
        js_list = [js_divergence(_normalize(Counter(cd), vocab), mean_dist)
                   for cd in conn_dists]
        mu = sum(js_list) / len(js_list)
        sd = math.sqrt(sum((x - mu) ** 2 for x in js_list) / len(js_list))
        dsc['conn'] = {'vocab': vocab, 'mean_dist': mean_dist,
                       'js_mean': mu, 'js_std': max(sd, 1e-6)}
        fp['discourse'] = dsc
    return fp

# ---------------------------------------------------------------- 채점
def score_text(text, fp, detail=False):
    """텍스트를 지문에 대해 채점.

    반환 z: '작가 자신의 청크 분산' 단위의 평균 이탈도.
    z ≈ 1 → 작가의 정상 변동 범위. z >> 2 → 작가답지 않음.

    채점 규칙 v2 (측정명세 §채점규칙):
    - 산포 계열(DISPERSION_KEYS)은 방향 부호를 유지해, 후보 산포가 작가
      평균보다 낮으면(과평탄) α=1.5배 가중 벌점 — LLM 평탄화 편향의 비대칭 대응.
    - 30문장 미만 텍스트는 low_confidence=True (P0 벤치마크 실측: 15문장
      단위 판별 53.6% vs 30문장 75%).
    """
    analyzer = fp.get('analyzer', 'surface')
    mod = get_analyzer(analyzer)
    sents = mod.split_sentences(text)
    f = mod.chunk_features(sents)
    Z_CAP = 6.0                          # 특징 하나가 전체를 지배하지 않도록
    ALPHA_FLAT = 1.5                     # 과평탄 비대칭 벌점 (설계감사 #3)
    FAM_W = {'endings': 3.0, 'connectives': 2.0, 'particles': 2.0,
             'adverbs': 2.0}
    zs, ds, ws, diag = [], [], [], {}
    for fam in CAT_FAMILIES:
        c = fp['cat'][fam]
        d = js_divergence(_normalize(Counter(f[fam]), c['vocab']),
                          c['mean_dist'])
        z = min(max((d - c['js_mean']) / c['js_std'], 0.0), Z_CAP)
        zs.append(z); ds.append(d); ws.append(FAM_W[fam])
        diag[f'js_{fam}'] = round(z, 2)
    for k, st in fp['scalars'].items():
        z = min(abs(f['scalars'][k] - st['mean']) / st['std'], Z_CAP)
        rel = abs(f['scalars'][k] - st['mean']) / (abs(st['mean']) + 0.05)
        zs.append(z); ds.append(min(rel, 3.0)); ws.append(1.0)
        diag[k] = round(z, 2)
    # 담화층 (P1): 지문에 있으면 채점, 텍스트가 너무 짧으면 건너뛰고 보고
    flat_hits = []
    d_extra = []          # (d값, 가중치) — z 없이 d에만 들어가는 항
    discourse_skipped = False
    if 'discourse' in fp:
        import discourse as disc
        if len(sents) < disc.MIN_SENTS:
            discourse_skipped = True
        else:
            df = disc.discourse_features_from_sents(sents)
            dsc = fp['discourse']
            for k, st in dsc['scalars'].items():
                if df.get(k) is None:
                    continue
                # 산포 계열은 코퍼스 전체 값이 기준 (창 크기 편향 회피)
                is_disp = k in disc.DISPERSION_KEYS
                ref = (st.get('corpus') if is_disp and
                       st.get('corpus') is not None else st['mean'])
                signed = (df[k] - ref) / st['std']
                over_flat = is_disp and signed < 0
                alpha = ALPHA_FLAT if over_flat else 1.0
                z = min(abs(signed) * alpha, Z_CAP)
                rel = abs(df[k] - ref) / (abs(ref) + 0.05)
                zs.append(z); ds.append(min(rel * alpha, 3.0)); ws.append(1.0)
                diag[k] = round(z, 2)
                if over_flat and abs(signed) > 1.0:
                    flat_hits.append(f'{k}:{df[k]:.3f}<{ref:.3f}')
            c = dsc['conn']
            d = js_divergence(_normalize(Counter(df['A_connective_dist']),
                                         c['vocab']), c['mean_dist'])
            z = min(max((d - c['js_mean']) / c['js_std'], 0.0), Z_CAP)
            zs.append(z); ds.append(d); ws.append(2.0)
            diag['js_discourse_conn'] = round(z, 2)
            # 사분위 프로파일: 분산 정보가 없어 d에만 반영 (표본 작으면 백오프).
            # z 평균을 희석하지 않도록 z 합산에서는 제외한 별도 항.
            if len(sents) >= disc.QUARTILE_MIN_SENTS:
                qd = []
                for key, prof in (('D_new_info', df['D_new_info_by_quartile']),
                                  ('E_question', df['E_question_by_quartile']),
                                  ('E_len', df['E_len_by_quartile'])):
                    ref = dsc['quartile'][key]
                    denom = sum(abs(x) for x in ref) / 4 + 0.05
                    qd.append(sum(abs(a - b) for a, b in zip(prof, ref))
                              / 4 / denom)
                d_extra.append((min(sum(qd) / len(qd), 3.0), 1.0))
                diag['quartile_d'] = round(sum(qd) / len(qd), 3)

    # 금지 목록 위반: 강한 벌점
    ban_hits = []
    for fam, banned in fp.get('ban_list', {}).items():
        for b in banned:
            if f[fam].get(b, 0) > 0:
                ban_hits.append(f'{fam}:{b}')
    # z: 수용 판정용 — 작가 자신의 청크 분산 단위 이탈도.
    #    (분산 폭이 다른 지문끼리 비교하는 용도가 아님)
    # d: 절대 문체 거리 — 작가 간 판별, best-of-N 재순위화용.
    z_total = (sum(z * w for z, w in zip(zs, ws)) / sum(ws)
               + 2.0 * len(ban_hits))
    d_num = sum(d * w for d, w in zip(ds, ws)) + sum(d * w for d, w in d_extra)
    d_den = sum(ws) + sum(w for _, w in d_extra)
    d_total = d_num / d_den + 0.5 * len(ban_hits)
    out = {'score': round(z_total, 3), 'distance': round(d_total, 4),
           'ban_violations': ban_hits, 'analyzer': analyzer,
           'n_sent': len(sents), 'n_eojeol': f['n_eojeol'],
           'low_confidence': len(sents) < 30}
    if 'discourse' in fp:
        out['flatness_hits'] = flat_hits
        if discourse_skipped:
            out['discourse_skipped'] = True
    if detail:
        out['diagnostics'] = diag
    return out

# ---------------------------------------------------- 독립 판별기 (문자 bigram)
def char_bigram_profile(text, top=300):
    t = re.sub(r'\s+', ' ', text)
    grams = Counter(t[i:i + 2] for i in range(len(t) - 1)
                    if _HANGUL.search(t[i:i + 2]))
    return dict(grams.most_common(top))

def bigram_distance(text, profile):
    vocab = list(profile)
    p = _normalize(Counter(profile), vocab)
    q = _normalize(Counter(char_bigram_profile(text, top=10 ** 6)), vocab)
    return js_divergence(q, p)

# ---------------------------------------------------------------- CLI
def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    analyzer = 'surface'
    for a in sys.argv[1:]:
        if a.startswith('--analyzer='):
            analyzer = a.split('=', 1)[1]
    cmd = args[0]
    if cmd == 'build':
        text = open(args[1], encoding='utf-8').read()
        author = args[2]
        contrast = (open(args[3], encoding='utf-8').read()
                    if len(args) > 3 else None)
        fp = build_fingerprint(text, author, contrast_text=contrast,
                               analyzer=analyzer)
        out = f'fingerprints/{author}.json'
        json.dump(fp, open(out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'{out}: analyzer={analyzer} chunks={fp["n_chunks"]} '
              f'sents={fp["n_sentences"]}')
    elif cmd == 'score':
        fp = json.load(open(args[1], encoding='utf-8'))
        text = open(args[2], encoding='utf-8').read()
        print(json.dumps(score_text(text, fp, detail=True),
                         ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
