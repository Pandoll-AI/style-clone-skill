#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""통합 스타일 팩 빌더 (P2 자동화) — 코퍼스 디렉토리 → 완전한 팩 JSON.

지금까지 코덱스 팩은 여러 스크립트(요인 투영·build_lexicon·수동 폐쇄류 계산)로
조각조각 만들었다. 이 스크립트가 그 전 과정을 하나로 통합한다. 어떤 모델의
코퍼스든 동일 프로토콜로 팩을 뽑아, claude-style 등 새 팩을 재현 가능하게 만든다.

입력 규약:
  <gen_dir>/{genre}_{topic}.txt   모델 생성 코퍼스 (topic 인덱스로 주제 분산 관문)
  <human_dir>/*.txt               인간 대조군 (같은 주제 권장)

산출 팩 블록 (전부 판별 실적 관문 통과분만):
  factor_signature   요인축 Cohen's d (|d|≥0.5)
  feature_markers    확장 특징 Cohen's d (표기 계열은 장르 교란으로 제외)
  ending/connective/adverb_overuse  폐쇄류 분포 마커
  content_words      내용어 (weighted log-odds + 주제분산, build_lexicon)
  phrases            문법 표현 (형태소 스트림, build_lexicon)
  meta.discrimination  15문장 청크 52특징 vs 인간 판별 정확도

사용: build_pack.py <gen_dir> <human_dir> <out.json> --name codex-style --model "..."
"""
import glob
import json
import math
import os
import random
import statistics
import sys
from collections import Counter

from features_surface import split_sentences, chunk_features
from features_ext import ext_features_from_sents, ALL_KEYS
from factor_analysis import chunk_vector          # 요인 입력 벡터 (EXCLUDE 반영)
from factors_runtime import load_factor_def, project
import build_lexicon as bl

CHUNK = 15
GENRE_CONFOUND = {'L3_roman_ej', 'L3_hanja_ej', 'L3_digit_ej',
                  'CX_paren_colon_sent'}
SEED = 20260711


def _cohens_d(a, b):
    sp = math.sqrt((statistics.pvariance(a) + statistics.pvariance(b)) / 2) or 1e-9
    return (statistics.mean(a) - statistics.mean(b)) / sp


def _load(paths, fdef, ch=CHUNK):
    """청크 단위 (확장특징, 요인점수, 요인입력벡터) 수집."""
    ext, fac, vec = [], [], []
    for p in paths:
        sents = split_sentences(open(p, encoding='utf-8').read())
        for i in range(0, len(sents) - ch + 1, ch):
            c = sents[i:i + ch]
            try:
                ext.append(ext_features_from_sents(c))
                s, _ = project(c, fdef)
                fac.append(s)
                vec.append(chunk_vector(c))
            except (ValueError, Exception):
                continue
    return ext, fac, vec


def _dist(cf, fam):
    t = sum(cf[fam].values()) or 1
    return {k: v / t for k, v in cf[fam].items()}


def _overuse(cx_cf, hu_cf, fam, thresh=1.8, floor=0.03):
    cd, hd = _dist(cx_cf, fam), _dist(hu_cf, fam)
    return {k: {'codex': round(v, 3), 'human': round(hd.get(k, 0), 3)}
            for k, v in cd.items()
            if v >= floor and v >= thresh * hd.get(k, 1e-4)}


def _discrimination(cx_vec, hu_vec, n_boot=1000):
    """52특징 벡터 LOO 최근접중심 판별 + 부트스트랩 CI."""
    keys = sorted(set(cx_vec[0]) & set(hu_vec[0]))
    allr = [(v, 'm') for v in cx_vec] + [(v, 'h') for v in hu_vec]
    mu = {k: statistics.mean(v[k] for v, _ in allr) for k in keys}
    sd = {k: statistics.pstdev(v[k] for v, _ in allr) or 1.0 for k in keys}
    Z = [([(v[k] - mu[k]) / sd[k] for k in keys], lab) for v, lab in allr]
    n, dim = len(Z), len(keys)
    hits = []
    for i in range(n):
        cents = {}
        for lab in ('m', 'h'):
            pts = [Z[j][0] for j in range(n) if Z[j][1] == lab and j != i]
            cents[lab] = [sum(p[k] for p in pts) / len(pts) for k in range(dim)]
        pred = min(cents, key=lambda l: sum((Z[i][0][k] - cents[l][k]) ** 2
                                            for k in range(dim)))
        hits.append(1 if pred == Z[i][1] else 0)
    rng = random.Random(SEED)
    accs = sorted(sum(rng.choice(hits) for _ in range(n)) / n
                  for _ in range(n_boot))
    return {'acc': round(sum(hits) / n, 3),
            'ci95': [round(accs[int(.025 * n_boot)], 3),
                     round(accs[int(.975 * n_boot)], 3)],
            'n': n}


def build(gen_dir, human_dir, name, model):
    fdef = load_factor_def()
    kiwi = bl.get_kiwi()
    gen = sorted(glob.glob(f'{gen_dir}/*.txt'))
    hu = sorted(glob.glob(f'{human_dir}/*.txt'))
    cx_ext, cx_fac, cx_vec = _load(gen, fdef)
    hu_ext, hu_fac, hu_vec = _load(hu, fdef)

    # 요인 서명
    factor_sig = {}
    for f in fdef['factors']:
        d = _cohens_d([s[f['name']] for s in cx_fac],
                      [s[f['name']] for s in hu_fac])
        if abs(d) >= 0.5:
            factor_sig[f['name']] = round(d, 2)

    # 특징 마커 (표기 교란 제외)
    markers = {}
    for k in ALL_KEYS:
        if k in GENRE_CONFOUND:
            continue
        ca = [r[k] for r in cx_ext if k in r]
        ha = [r[k] for r in hu_ext if k in r]
        if not ca or not ha:
            continue
        d = _cohens_d(ca, ha)
        if abs(d) >= 0.5:
            markers[k] = {'d': round(d, 2), 'codex': round(statistics.mean(ca), 3),
                          'human': round(statistics.mean(ha), 3)}

    # 폐쇄류 분포
    cx_all = split_sentences('\n'.join(open(p, encoding='utf-8').read()
                                       for p in gen))
    hu_all = split_sentences('\n'.join(open(p, encoding='utf-8').read()
                                       for p in hu))
    cxcf, hucf = chunk_features(cx_all), chunk_features(hu_all)

    # 사전 (build_lexicon 재사용)
    content = bl.build_content(gen, hu, kiwi)
    phrases = bl.build_phrases(gen, hu, kiwi)

    disc = _discrimination(cx_vec, hu_vec)

    return {
        'meta': {
            'pack': name, 'model': model, 'generated': 'AUTO',
            'genres_detected': sorted({os.path.basename(p).split('_')[0]
                                       for p in gen}),
            'n_gen': len(gen), 'n_human': len(hu),
            'discrimination_15sent': {'acc': disc['acc'], 'ci95': disc['ci95'],
                                      'n': disc['n']},
            'method': 'build_pack.py 통합 파이프라인 (요인+특징+폐쇄류+사전+판별)',
            'caveat': '표기 계열 제외(장르 교란). 요인 공간은 1930s 소설 추정. '
                      '대조군 장르에 특화 — 다장르 인간 대조군 권장.',
        },
        'factor_signature': factor_sig,
        'feature_markers': markers,
        'ending_overuse': _overuse(cxcf, hucf, 'endings'),
        'connective_overuse': _overuse(cxcf, hucf, 'connectives'),
        'adverb_overuse': _overuse(cxcf, hucf, 'adverbs'),
        'content_words': content,
        'phrases': phrases,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = {}
    for a in sys.argv[1:]:
        if a.startswith('--'):
            k, _, v = a[2:].partition('=')
            opts[k] = v or True
    # --name X 형태도 지원
    it = iter(sys.argv[1:])
    for a in it:
        if a == '--name':
            opts['name'] = next(it)
        elif a == '--model':
            opts['model'] = next(it)
    gen_dir, human_dir, out = args[0], args[1], args[2]
    pack = build(gen_dir, human_dir, opts.get('name', 'style'),
                 opts.get('model', 'unknown'))
    json.dump(pack, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    m = pack['meta']
    print(f"팩 '{m['pack']}' → {out}")
    print(f"  판별 {m['discrimination_15sent']['acc']:.1%} "
          f"CI{m['discrimination_15sent']['ci95']} (n={m['discrimination_15sent']['n']})")
    print(f"  요인서명 {len(pack['factor_signature'])} · 특징마커 {len(pack['feature_markers'])} "
          f"· 내용어 {len(pack['content_words'])} · 표현 {len(pack['phrases'])}")
    print(f"  어미과다 {list(pack['ending_overuse'])[:6]}")
    print(f"  부사과다 {list(pack['adverb_overuse'])[:6]}")


if __name__ == '__main__':
    main()
