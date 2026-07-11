#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 표준 벤치마크 — 작가 5인 held-out 청크 판별.

로드맵 P0: "작가 5인 × held-out 판별을 표준 과제로 삼고 모든 특징 추가를
이 수치로 심사한다." 특징 채용의 유일한 관문:

  새 특징은 held-out 판별 정확도를 유의하게 올리거나, 기존 특징이 못 잡는
  실패 사례를 검출해야 편입된다. 정확도에는 부트스트랩 95% CI를 함께 보고
  하고, CI가 겹치면 '유의한 향상'으로 치지 않는다.

비교 대상 (판별기 3종, 서로 특징군 분리 — Goodhart 규율):
  surface  표층 접미 L1 지문, argmin d
  kiwi     형태소 태그 L1 지문, argmin d       ← P0 심사 대상 (설계감사 #11)
  bigram   문자 bigram JS, argmin              ← 독립 상한선 참조

주의: analyzer가 다른 지문끼리 d 비교는 무의미하므로 판별은 분석기별로
지문 5개 세트를 통일해 수행한다.

사용법: python3 benchmark.py [--analyzers=surface,kiwi] [--boot=2000] [--chunk=15]

실측 (2026-07-11, 설계감사 #2·#11):
  청크 15문장: surface 53.6% / kiwi 49.6% / bigram 88.4%
  청크 30문장: surface 75.0% / kiwi 75.0% / bigram 89.7%
  청크 60문장: surface 66.7% / kiwi 78.8% / bigram 93.9% (n=33, CI 넓음)
→ 판별 단위가 클수록 정확도 상승(Eder 2015 부합). kiwi 이득은 큰 단위에서만
  나타나며 현 코퍼스 규모에서는 CI 겹침 — '유의한 향상 실패'로 정직 기록.
"""
import glob
import json
import random
import sys
from collections import defaultdict

from stylometry import (build_fingerprint, score_text, get_analyzer,
                        char_bigram_profile, bigram_distance)

AUTHORS = ['kimyj', 'yisang', 'hyunjg', 'chaems', 'leehs']
CHUNK = 15          # 문장/청크 (설계감사 #2 — 스윕 대상)
HOLD_EVERY = 4      # 4청크마다 1개 held-out (v1과 동일 프로토콜)
SEED = 20260711


def load_author(author, analyzer):
    paths = sorted(glob.glob(f'corpus/{author}/*.txt'))
    if not paths:
        raise FileNotFoundError(f'corpus/{author}/ 에 작품이 없음')
    text = '\n'.join(open(p, encoding='utf-8').read() for p in paths)
    sents = get_analyzer(analyzer).split_sentences(text)
    chunks = [sents[i:i + CHUNK] for i in range(0, len(sents) - CHUNK + 1, CHUNK)]
    train = [c for i, c in enumerate(chunks) if i % HOLD_EVERY != HOLD_EVERY - 1]
    held = [c for i, c in enumerate(chunks) if i % HOLD_EVERY == HOLD_EVERY - 1]
    return train, held


def as_text(chunks):
    return '\n'.join('\n'.join(c) for c in chunks)


def bootstrap_ci(hits, n_boot, seed=SEED):
    """hits: 0/1 리스트 → (정확도, 95% CI 하한, 상한)."""
    rng = random.Random(seed)
    n = len(hits)
    accs = sorted(sum(rng.choice(hits) for _ in range(n)) / n
                  for _ in range(n_boot))
    return (sum(hits) / n,
            accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)])


def run_measurement(analyzer, n_boot):
    """지문 측정층(argmin d) 판별."""
    train, held, fps = {}, {}, {}
    for a in AUTHORS:
        train[a], held[a] = load_author(a, analyzer)
    for a in AUTHORS:
        contrast = '\n'.join(as_text(train[b]) for b in AUTHORS if b != a)
        fps[a] = build_fingerprint(as_text(train[a]), a,
                                   contrast_text=contrast, analyzer=analyzer)
    hits, confusion = [], defaultdict(lambda: defaultdict(int))
    for true_a in AUTHORS:
        for c in held[true_a]:
            t = '\n'.join(c)
            d = {b: score_text(t, fps[b])['distance'] for b in AUTHORS}
            pred = min(d, key=d.get)
            confusion[true_a][pred] += 1
            hits.append(1 if pred == true_a else 0)
    acc, lo, hi = bootstrap_ci(hits, n_boot)
    return {'accuracy': acc, 'ci95': [lo, hi], 'n_held': len(hits),
            'confusion': {a: dict(confusion[a]) for a in AUTHORS},
            'held_counts': {a: len(held[a]) for a in AUTHORS}}, fps


def run_bigram(n_boot):
    """독립 판별기(문자 bigram JS). 문장 분리가 필요 없어 surface 분리 사용."""
    train, held, profiles = {}, {}, {}
    for a in AUTHORS:
        train[a], held[a] = load_author(a, 'surface')
        profiles[a] = char_bigram_profile(as_text(train[a]))
    hits, confusion = [], defaultdict(lambda: defaultdict(int))
    for true_a in AUTHORS:
        for c in held[true_a]:
            t = '\n'.join(c)
            d = {b: bigram_distance(t, profiles[b]) for b in AUTHORS}
            pred = min(d, key=d.get)
            confusion[true_a][pred] += 1
            hits.append(1 if pred == true_a else 0)
    acc, lo, hi = bootstrap_ci(hits, n_boot)
    return {'accuracy': acc, 'ci95': [lo, hi], 'n_held': len(hits),
            'confusion': {a: dict(confusion[a]) for a in AUTHORS}}


def fmt(r):
    return (f"{r['accuracy']*100:5.1f}%  "
            f"[{r['ci95'][0]*100:.1f}, {r['ci95'][1]*100:.1f}]  "
            f"n={r['n_held']}")


def main():
    global CHUNK
    analyzers = ['surface', 'kiwi']
    n_boot = 2000
    for a in sys.argv[1:]:
        if a.startswith('--analyzers='):
            analyzers = a.split('=', 1)[1].split(',')
        elif a.startswith('--boot='):
            n_boot = int(a.split('=', 1)[1])
        elif a.startswith('--chunk='):
            CHUNK = int(a.split('=', 1)[1])

    results = {'protocol': {'chunk_sents': CHUNK, 'hold_every': HOLD_EVERY,
                            'authors': AUTHORS, 'seed': SEED,
                            'n_boot': n_boot}}
    print(f'=== P0 벤치마크: 작가 {len(AUTHORS)}인 held-out 판별 '
          f'(청크 {CHUNK}문장, 1/{HOLD_EVERY} held-out) ===\n')

    for analyzer in analyzers:
        r, fps = run_measurement(analyzer, n_boot)
        results[f'measurement_{analyzer}'] = r
        print(f'[측정층 d, analyzer={analyzer}]  {fmt(r)}')
        for a in AUTHORS:
            row = r['confusion'][a]
            n = sum(row.values())
            ok = row.get(a, 0)
            wrong = {b: v for b, v in row.items() if b != a and v}
            print(f'    {a}: {ok}/{n}' + (f'  오귀속 {wrong}' if wrong else ''))
        for a in AUTHORS:
            json.dump(fps[a], open(f'fingerprints/{a}_{analyzer}.json', 'w',
                                   encoding='utf-8'),
                      ensure_ascii=False, indent=1)
        print()

    r = run_bigram(n_boot)
    results['bigram'] = r
    print(f'[독립 판별기 문자 bigram]  {fmt(r)}')
    for a in AUTHORS:
        row = r['confusion'][a]
        n = sum(row.values())
        wrong = {b: v for b, v in row.items() if b != a and v}
        print(f'    {a}: {row.get(a, 0)}/{n}' + (f'  오귀속 {wrong}' if wrong else ''))

    json.dump(results, open('benchmark_results.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n→ benchmark_results.json 저장')

    ms, mk = results.get('measurement_surface'), results.get('measurement_kiwi')
    if ms and mk:
        overlap = not (mk['ci95'][0] > ms['ci95'][1] or
                       ms['ci95'][0] > mk['ci95'][1])
        verdict = ('CI 겹침 — 유의한 차이 아님' if overlap
                   else 'CI 분리 — 유의한 차이')
        print(f'\n심사(#11): surface {ms["accuracy"]*100:.1f}% vs '
              f'kiwi {mk["accuracy"]*100:.1f}% → {verdict}')


if __name__ == '__main__':
    main()
