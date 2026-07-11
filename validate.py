#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""held-out 판별 검증.

각 작가 코퍼스를 청크로 나눠 75%로 지문을 만들고, 나머지 25% held-out
청크를 두 지문 모두에 채점 → argmin으로 귀속. 자기 작가 지문에서는
수용 영역 안(z 낮음), 타 작가 지문에서는 밖(z 높음)이어야 한다.

동시에 독립 판별기(문자 bigram JS)로도 같은 귀속을 수행해
'특징군 분리' 하에서도 판별이 성립하는지 확인한다.
"""
import json
from stylometry import (split_sentences, build_fingerprint, score_text,
                        char_bigram_profile, bigram_distance)

# v1 원 실험 구성 그대로 (봄봄·만무방 vs 날개) — 회귀 기준선.
# 확장 코퍼스(작가 5인) 벤치마크는 benchmark.py가 담당한다.
CORPORA = {
    'kimyj':  ['corpus/kimyj/봄봄.txt', 'corpus/kimyj/만무방.txt'],
    'yisang': ['corpus/yisang/날개.txt'],
}
CHUNK = 15          # 문장/청크
HOLD_EVERY = 4      # 4청크마다 1개 held-out

def load(author):
    text = '\n'.join(open(p, encoding='utf-8').read() for p in CORPORA[author])
    sents = split_sentences(text)
    chunks = [sents[i:i + CHUNK] for i in range(0, len(sents) - CHUNK + 1, CHUNK)]
    train = [c for i, c in enumerate(chunks) if i % HOLD_EVERY != HOLD_EVERY - 1]
    held = [c for i, c in enumerate(chunks) if i % HOLD_EVERY == HOLD_EVERY - 1]
    return train, held

def as_text(chunks):
    return '\n'.join('\n'.join(c) for c in chunks)

def main():
    train, held, fps, bigram_fps = {}, {}, {}, {}
    for a in CORPORA:
        train[a], held[a] = load(a)
        print(f'{a}: train_chunks={len(train[a])} held={len(held[a])}')
    for a in CORPORA:
        other = [b for b in CORPORA if b != a][0]
        fps[a] = build_fingerprint(as_text(train[a]), a,
                                   contrast_text=as_text(train[other]),
                                   with_discourse=False)  # v1 수치 재현용
        bigram_fps[a] = char_bigram_profile(as_text(train[a]))
        json.dump(fps[a], open(f'fingerprints/{a}.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)

    print('\n=== held-out 귀속 (측정층: 절대 거리 d) ===')
    confusion = {a: {b: 0 for b in CORPORA} for a in CORPORA}
    all_d = {a: {b: [] for b in CORPORA} for a in CORPORA}
    all_z = {a: {b: [] for b in CORPORA} for a in CORPORA}
    for true_a in CORPORA:
        for c in held[true_a]:
            t = '\n'.join(c)
            res = {b: score_text(t, fps[b]) for b in CORPORA}
            pred = min(res, key=lambda b: res[b]['distance'])
            confusion[true_a][pred] += 1
            for b in CORPORA:
                all_d[true_a][b].append(res[b]['distance'])
                all_z[true_a][b].append(res[b]['score'])
    for a in CORPORA:
        n = sum(confusion[a].values())
        print(f'  실제={a}: ' + ', '.join(
            f'{b}로 귀속 {confusion[a][b]}/{n}' for b in CORPORA))
    for a in CORPORA:
        for b in CORPORA:
            v, w = all_d[a][b], all_z[a][b]
            print(f'  실제 {a} → 지문 {b}: d_mean={sum(v)/len(v):.3f} '
                  f'z_mean={sum(w)/len(w):.2f} z_max={max(w):.2f}')

    print('\n=== held-out 귀속 (독립 판별기: 문자 bigram JS) ===')
    confusion2 = {a: {b: 0 for b in CORPORA} for a in CORPORA}
    for true_a in CORPORA:
        for c in held[true_a]:
            t = '\n'.join(c)
            d = {b: bigram_distance(t, bigram_fps[b]) for b in CORPORA}
            confusion2[true_a][min(d, key=d.get)] += 1
    for a in CORPORA:
        n = sum(confusion2[a].values())
        print(f'  실제={a}: ' + ', '.join(
            f'{b}로 귀속 {confusion2[a][b]}/{n}' for b in CORPORA))

    json.dump({'measurement_layer': confusion, 'bigram_verifier': confusion2},
              open('validation_results.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main()
