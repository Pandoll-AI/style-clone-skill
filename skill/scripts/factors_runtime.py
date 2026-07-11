#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""요인 채점 런타임 (P1.5 S4) — 순수 파이썬, numpy 불필요.

factor_analysis.py(dev, numpy)가 추정한 요인 공간(factors.json)을 읽어
텍스트를 투영한다. 요인 정의는 지문에 통째로 내장되므로(build 시)
지문 JSON 하나만 있으면 어디서든 요인 채점이 된다 (팩 이식성).

이번 단계의 편입 수준(docs/06 R2 결정): 요인은 **진단 층** — score_text의
z_total 산식에는 넣지 않는다. 통합은 벤치마크 근거 확보 후.
"""
import json
import math
import pathlib

from features_surface import chunk_features
from features_ext import ext_features_from_sents
import discourse as disc

DEFAULT_PATH = pathlib.Path(__file__).parent / 'factors.json'
MIN_SENTS = 8            # 미만이면 담화·확장 특징이 무의미 → 요인 생략
CHUNK = 30               # 작가 요인 분산 추정 단위 (요인 공간 추정과 동일)


def load_factor_def(path=None):
    p = pathlib.Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return None
    return json.load(open(p, encoding='utf-8'))


def chunk_vector(sents):
    """청크 → 요인 입력 스칼라 dict. factor_analysis.py와 단일 진실원."""
    row = {}
    f = chunk_features(sents)
    for k, v in f['scalars'].items():
        row[f'L1_{k}'] = v
    d = disc.discourse_features_from_sents(sents)
    for k in disc.SCALAR_KEYS:
        row[k] = d[k] if d.get(k) is not None else 0.0
    for t, v in d['A_connective_dist'].items():
        if t != '무표지':
            row[f'L5_conn_{t}'] = v
    row.update(ext_features_from_sents(sents))
    return row


def project(sents, fdef):
    """문장 리스트 → (요인 점수 dict, 결측 특징 목록).

    kiwi 부재 등으로 빠진 특징은 코퍼스 평균(z=0)으로 대치하고 보고한다.
    """
    if len(sents) < MIN_SENTS:
        raise ValueError('문장이 너무 적음 (요인 투영 불가)')
    row = chunk_vector(sents)
    feats, mu, sd = fdef['features'], fdef['mu'], fdef['sd']
    z, missing = [], []
    for i, k in enumerate(feats):
        if k in row:
            z.append((row[k] - mu[i]) / (sd[i] or 1.0))
        else:
            z.append(0.0)
            missing.append(k)
    scores = {}
    for f in fdef['factors']:
        j = f['index']
        s = sum(z[i] * fdef['loadings'][i][j] for i in range(len(feats)))
        scores[f['name']] = s
    return scores, missing


def author_factor_stats(sents, fdef, chunk=CHUNK):
    """작가 코퍼스 → 요인별 청크 mean/std (수용 영역). 청크 2개 미만이면 None."""
    chunks = [sents[i:i + chunk] for i in range(0, len(sents) - chunk + 1, chunk)]
    if len(chunks) < 2:
        return None
    per = []
    for c in chunks:
        try:
            s, _ = project(c, fdef)
            per.append(s)
        except ValueError:
            continue
    if len(per) < 2:
        return None
    stats = {}
    for name in per[0]:
        vals = [p[name] for p in per]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
        stats[name] = {'mean': round(m, 4), 'std': round(max(sd, 0.35), 4)}
    return stats


def score_factors(sents, fp_factors):
    """후보 문장들을 지문 내장 요인 정의에 대해 채점 → 요인별 z(부호 유지)."""
    fdef = fp_factors['def']
    scores, missing = project(sents, fdef)
    out = {}
    for name, s in scores.items():
        st = fp_factors['author'].get(name)
        if not st:
            continue
        out[name] = round((s - st['mean']) / st['std'], 2)
    return out, missing
