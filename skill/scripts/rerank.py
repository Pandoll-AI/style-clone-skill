#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""문단 단위 best-of-N 재순위화 (v1 남은 과제 5).

v1 데모는 글 전체 단위 best-of-N이었다. 이 모듈은 후보들을 문단 슬롯으로
정렬해 슬롯별 최적 문단을 고르고, 조합 결과에 3중 검증을 건다:

  1. 수용 판정  — 조합 전체의 z ≤ 임계 (기본 1.0, 작가 정상 변동 범위)
  2. 평탄화 감사 — flatness_hits (산포 비대칭 벌점, 측정명세 §채점규칙 2)
  3. 캐리커처 가드 — 조합 d < 작가 held-out d 하한이면 정지·경고.
     기준선은 점추정이 아니라 baseline 텍스트의 30문장 청크 d 분포의
     10% 분위수 (설계감사 #13: 분위수 기준선).

전제: 후보들은 같은 개요(문단 순서)로 생성됐다 — 슬롯 i의 후보 풀은 각
후보의 i번째 문단. 문단 수가 다르면 짧은 후보는 해당 슬롯에서 빠진다.
문단 채점은 저신뢰(low_confidence)일 수밖에 없으므로 (P0 실측: 판별
단위가 작을수록 부정확) 슬롯 선택은 d로만 하고, 최종 판정은 조합
전체에 대해서만 내린다.

사용: python3 rerank.py <지문.json> <출력.txt> <후보1> <후보2> ...
      [--baseline=작가텍스트] [--z-accept=1.0]
"""
import json
import sys

from stylometry import score_text, get_analyzer

Z_ACCEPT = 1.0          # 수용 임계 (v1 실측: 작가 본인 held-out z≈0.31)
GUARD_CHUNK = 30        # 기준선 청크 크기 (P0 실측: 30문장부터 신뢰 가능)
GUARD_Q = 0.10          # 캐리커처 가드 분위수


def paragraphs(text):
    return [p.strip() for p in text.split('\n\n') if p.strip()]


def baseline_d_bound(baseline_text, fp):
    """작가 held-out 청크 d 분포의 하위 분위수 — 캐리커처 가드 기준선.

    baseline이 30문장 청크를 못 만들 만큼 짧으면 15문장으로 폴백한다
    (짧은 단위 d는 분산이 커서 하한이 느슨해질 뿐 — 가드가 없는 것보다 낫다).
    """
    mod = get_analyzer(fp.get('analyzer', 'surface'))
    sents = mod.split_sentences(baseline_text)
    for size in (GUARD_CHUNK, GUARD_CHUNK // 2):
        chunks = [sents[i:i + size]
                  for i in range(0, len(sents) - size + 1, size)]
        if len(chunks) >= 2:
            ds = sorted(score_text('\n'.join(c), fp)['distance']
                        for c in chunks)
            return ds[int(GUARD_Q * (len(ds) - 1))], ds
    return None, []


def rerank(fp, candidates, baseline_text=None, z_accept=Z_ACCEPT):
    """candidates: {이름: 텍스트}. → 리포트 딕셔너리 (composite 포함)."""
    # 1) 후보 전체 채점
    whole = {name: score_text(t, fp) for name, t in candidates.items()}

    # 2) 슬롯별 문단 선택 (d 최소)
    paras = {name: paragraphs(t) for name, t in candidates.items()}
    n_slots = max(len(p) for p in paras.values())
    slots, chosen = [], []
    for i in range(n_slots):
        pool = {name: ps[i] for name, ps in paras.items() if i < len(ps)}
        scored = {name: score_text(p, fp)['distance']
                  for name, p in pool.items()}
        best = min(scored, key=scored.get)
        slots.append({'slot': i, 'chosen': best,
                      'd': {k: round(v, 4) for k, v in scored.items()}})
        chosen.append(pool[best])
    composite = '\n\n'.join(chosen)

    # 3) 조합 전체 검증
    comp_score = score_text(composite, fp, detail=True)
    verdict = {'z_accept': comp_score['score'] <= z_accept,
               'flatness_ok': not comp_score.get('flatness_hits'),
               'ban_ok': not comp_score['ban_violations']}
    if baseline_text:
        bound, dist = baseline_d_bound(baseline_text, fp)
        if bound is None:
            verdict['caricature_guard'] = 'baseline_too_short'
        else:
            verdict['caricature_guard'] = (
                'STOP: 캐리커처 의심 (d < 작가 하한)'
                if comp_score['distance'] < bound else 'ok')
            verdict['baseline_d_bound'] = round(bound, 4)
            verdict['baseline_d_chunks'] = [round(x, 4) for x in dist]
    else:
        verdict['caricature_guard'] = 'skipped: baseline 미제공'

    best_whole = min(whole, key=lambda k: whole[k]['distance'])
    return {
        'whole_scores': {k: {'z': v['score'], 'd': v['distance'],
                             'ban': v['ban_violations'],
                             'flat': v.get('flatness_hits', [])}
                         for k, v in whole.items()},
        'best_single': best_whole,
        'slots': slots,
        'composite': composite,
        'composite_score': comp_score,
        'composite_beats_best_single':
            comp_score['distance'] <= whole[best_whole]['distance'],
        'verdict': verdict,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    fp = json.load(open(args[0], encoding='utf-8'))
    out_path = args[1]
    candidates = {p: open(p, encoding='utf-8').read() for p in args[2:]}
    baseline = (open(opts['baseline'], encoding='utf-8').read()
                if 'baseline' in opts else None)
    r = rerank(fp, candidates, baseline,
               z_accept=float(opts.get('z-accept', Z_ACCEPT)))
    open(out_path, 'w', encoding='utf-8').write(r['composite'])
    report = {k: v for k, v in r.items() if k != 'composite'}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f'\n→ 조합 저장: {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
