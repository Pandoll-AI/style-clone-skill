#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""요인 추출 (P1.5 S2-S3, dev 전용 — numpy 필요).

Biber MDA 방식: 원자 스칼라 특징 행렬(작가 청크 × 특징) → 표준화 → PCA →
평행 분석으로 요인 수 결정 → varimax 회전 → 요인별 관문(QG3) 심사 →
`factors.json` (런타임 엔진은 이 JSON의 행렬만 씀 — numpy 불필요).

주의(QG2): 표본 = 청크 ~277 × 특징 ~53 (관측/특징 ≈ 5) — 요인 구조의
안정성은 코퍼스 확장 시 재추정 필요. 이 스크립트가 그 재현 레시피다.

사용: .venv/bin/python factor_analysis.py [--chunk=30] [--out=factors.json]
"""
import glob
import json
import sys

import numpy as np

from features_surface import split_sentences
from factors_runtime import chunk_vector as _runtime_chunk_vector

AUTHORS = ['kimyj', 'yisang', 'hyunjg', 'chaems', 'leehs']
SEED = 20260711

# R1 회고 결정: 요인 입력에서 제외 (모듈에는 유지 — 모델 팩 전용 후보)
EXCLUDE = {'CX_meta_discourse_sent',   # 유병률 1.1% (1930s 소설에 부재하는 AI 표지)
           'CX_triple_list_sent',      # F<1 — 이 코퍼스에서 판별 신호 없음
           'L7_quote_particle_sent',   # 유병률 4.7%
           'CX_paren_colon_sent'}      # L3_hanja_ej와 r=0.956 중복


def chunk_vector(sents):
    """요인 입력 벡터 — 런타임(factors_runtime)과 단일 진실원, EXCLUDE만 필터."""
    return {k: v for k, v in _runtime_chunk_vector(sents).items()
            if k not in EXCLUDE}


def build_matrix(chunk_sents=30):
    rows, labels = [], []
    for a in AUTHORS:
        text = '\n'.join(open(p, encoding='utf-8').read()
                         for p in sorted(glob.glob(f'corpus/{a}/*.txt')))
        sents = split_sentences(text)
        for i in range(0, len(sents) - chunk_sents + 1, chunk_sents):
            rows.append(chunk_vector(sents[i:i + chunk_sents]))
            labels.append(a)
    keys = sorted(rows[0])
    X = np.array([[r[k] for k in keys] for r in rows])
    return X, keys, labels


def parallel_analysis(Z, n_iter=200):
    """평행 분석: 무작위 데이터 고유값 95분위를 넘는 요인 수."""
    rng = np.random.default_rng(SEED)
    n, p = Z.shape
    real = np.linalg.eigvalsh(np.corrcoef(Z.T))[::-1]
    rand = np.empty((n_iter, p))
    for i in range(n_iter):
        R = rng.standard_normal((n, p))
        rand[i] = np.linalg.eigvalsh(np.corrcoef(R.T))[::-1]
    thresh = np.percentile(rand, 95, axis=0)
    return int(np.sum(real > thresh)), real, thresh


def varimax(L, gamma=1.0, q=100, tol=1e-6):
    p, k = L.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        Lr = L @ R
        u, s, vt = np.linalg.svd(
            L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag(np.diag(Lr.T @ Lr))))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R


def f_ratio(scores, labels):
    """작가간/작가내 분산비 — 요인 신뢰도·판별 신호 관문."""
    labs = np.array(labels)
    grand = scores.mean()
    groups = [scores[labs == a] for a in AUTHORS]
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(AUTHORS) - 1)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(labs) - len(AUTHORS))
    return float(ssb / ssw) if ssw else 0.0


def centroid_attribution(S, labels):
    """요인 점수만으로 최근접 중심 판별 (leave-one-out) — 관문 (b)."""
    labs = np.array(labels)
    correct = 0
    for i in range(len(labs)):
        cents = {}
        for a in AUTHORS:
            mask = (labs == a)
            mask[i] = False
            cents[a] = S[mask].mean(axis=0)
        pred = min(cents, key=lambda a: np.linalg.norm(S[i] - cents[a]))
        correct += (pred == labs[i])
    return correct / len(labs)


def main():
    chunk = 30
    out = 'factors.json'
    for a in sys.argv[1:]:
        if a.startswith('--chunk='):
            chunk = int(a.split('=')[1])
        elif a.startswith('--out='):
            out = a.split('=')[1]

    X, keys, labels = build_matrix(chunk)
    # 퇴화(상수) 열은 상관 행렬을 오염시키므로 기각하고 보고 (QG1 자동 집행)
    sd0 = X.std(0)
    dead = [keys[i] for i in range(len(keys)) if sd0[i] == 0]
    if dead:
        print(f'퇴화 특징 기각: {dead}')
        keep = [i for i in range(len(keys)) if sd0[i] > 0]
        X = X[:, keep]
        keys = [keys[i] for i in keep]
    print(f'행렬: {X.shape[0]} 청크 × {X.shape[1]} 특징 '
          f'(관측/특징 = {X.shape[0]/X.shape[1]:.1f} — 5 미만이면 불안정 경고)')
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd

    k, eig, thresh = parallel_analysis(Z)
    print(f'평행 분석: 요인 {k}개 (고유값 {np.round(eig[:k+2], 2)} vs 임계 {np.round(thresh[:k+2], 2)})')

    C = np.corrcoef(Z.T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1][:k]
    L = V[:, order] * np.sqrt(w[order])          # 초기 로딩
    L = varimax(L)
    expl = (L ** 2).sum(0) / X.shape[1]

    # 요인 점수 (로딩 가중 합 → 표준화)
    S = Z @ L
    S = (S - S.mean(0)) / S.std(0)

    print(f'\n요인별 관문 심사 (F≥2, 상위 로딩):')
    factors = []
    for j in range(k):
        F = f_ratio(S[:, j], labels)
        top = sorted(zip(keys, L[:, j]), key=lambda x: -abs(x[1]))[:6]
        top_s = ', '.join(f'{n}({v:+.2f})' for n, v in top)
        verdict = 'PASS' if F >= 2.0 else 'FAIL(F<2)'
        print(f'  F{j+1}: 분산 {expl[j]:.1%}  F비 {F:6.1f}  {verdict}')
        print(f'      {top_s}')
        factors.append({'index': j, 'f_ratio': round(F, 2),
                        'explained': round(float(expl[j]), 4),
                        'top_loadings': [[n, round(float(v), 3)] for n, v in top],
                        'gate_reliability': F >= 2.0})

    acc = centroid_attribution(S, labels)
    print(f'\n요인 점수만의 5인 판별 (LOO 최근접 중심): {acc:.1%} (관문: ≥40%)')

    json.dump({
        'meta': {'chunk_sents': chunk, 'n_obs': int(X.shape[0]),
                 'n_features': int(X.shape[1]), 'seed': SEED,
                 'attribution_loo': round(acc, 4),
                 'corpus': '1930s-fiction-5authors',
                 'caveat': '요인 공간은 이 코퍼스에서 추정 — 현대 산문 일반화 미검증'},
        'features': keys,
        'mu': [round(float(v), 6) for v in mu],
        'sd': [round(float(v), 6) for v in sd],
        'loadings': [[round(float(v), 6) for v in row] for row in L],
        'factors': factors,
    }, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'→ {out} 저장 (이름은 관문 통과 후 수동 명명)')


if __name__ == '__main__':
    main()
