#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""교정 루프 공식화 (루프 2) — 진단 → 처방 → 재채점.

v1에서 수동으로 실증한 회전(z 6.35→0.15, 2회전)을 코드화한다.
score_text의 z 진단은 크기만 주므로, 여기서 방향(과다/과소)을 복원해
LLM에게 줄 한국어 처방문으로 번역한다. 임베딩 점수와 달리 수제 특징은
처방을 줄 수 있다는 것(해석 가능성)이 이 루프의 존재 이유다 (로드맵 P5).

정지 조건:
  1. 수용: z ≤ z_accept (기본 1.0)
  2. 캐리커처 가드: d < 작가 held-out 하한 → 개선이 아니라 위조 심화 (STOP)
  3. 최대 회전 수는 호출자(스킬)가 관리 — 권장 2회전 (v1 실증 수렴 속도)

사용: python3 loop.py <지문.json> <글.txt> [--baseline=작가텍스트] [--z-accept=1.0]
출력: JSON {score, verdict, prescriptions[]} — prescriptions를 LLM 재작성
      지시문에 그대로 붙이면 된다.
"""
import json
import sys
from collections import Counter

import discourse as disc
from stylometry import (score_text, get_analyzer, _normalize, CAT_FAMILIES)

Z_PRESCRIBE = 1.5       # 이 이상 이탈한 특징만 처방 (사소한 지적 남발 방지)
TOP_K = 6               # 처방 최대 개수 — 한 회전에 고칠 수 있는 만큼만

FAM_KO = {'endings': '종결어미', 'connectives': '연결어미',
          'particles': '조사', 'adverbs': '부사·접속어'}

# 스칼라 특징 → (한국어 이름, 낮을 때 지시, 높을 때 지시)
SCALAR_KO = {
    'sent_len_mean': ('평균 문장 길이(어절)',
                      '문장을 더 길게 이어 붙여라', '문장을 짧게 쪼개라'),
    'sent_len_std': ('문장 길이 변화폭',
                     '긴 문장과 짧은 문장을 섞어 리듬을 만들어라',
                     '문장 길이를 고르게 하라'),
    'sent_len_p90': ('긴 문장(상위 10%) 길이',
                     '이따금 긴 호흡의 문장을 넣어라', '가장 긴 문장들을 잘라라'),
    'dialog_ratio': ('대화문 비율', '직접 인용을 더 써라', '직접 인용을 줄여라'),
    'comma_per_sent': ('쉼표 밀도', '쉼표를 더 써라', '쉼표를 줄여라'),
    'excl_per_sent': ('느낌표 밀도', '감탄을 더 써라', '느낌표를 줄여라'),
    'quest_per_sent': ('물음표 밀도', '물음표를 더 써라', '물음표를 줄여라'),
    'ellipsis_per_sent': ('말줄임표 밀도', '말줄임표를 더 써라', '말줄임표를 줄여라'),
    'dash_per_1k_ej': ('줄표(―) 밀도', '줄표 삽입구를 더 써라', '줄표를 줄여라'),
    'redup_per_1k_ej': ('첩어(의태어) 밀도', '첩어를 더 써라', '첩어를 줄여라'),
    'A_implicit_ratio': ('무표지 연결 비율',
                         '문두 접속사를 줄이고 어휘 반복·생략으로 이어라',
                         '논리 표지를 조금 더 깔아라'),
    'B_jump_mean': ('문장 간 의미 도약(추론 보폭)',
                    '문장 사이 논리 간격을 넓혀라 — 설명을 덜어내라',
                    '문장 사이 간격을 좁혀라 — 다리를 놓아라'),
    'B_jump_std': ('도약 변화폭', '도약 크기에 변화를 줘라', '도약을 고르게 하라'),
    'B_flight_return_rate': ('도약 후 복귀율(나선 구조)',
                             '크게 벗어났다가 앞 문맥으로 돌아오는 나선을 만들어라',
                             '한 번 떠난 화제로 덜 돌아가라'),
    'C_len_cv': ('문장 길이 산포(CV)',
                 '단문과 장문의 낙차를 키워라 (과평탄)', '길이 낙차를 줄여라'),
    'C_len_autocorr': ('길이 리듬의 관성',
                       '비슷한 길이의 문장을 몇 개씩 뭉쳐 파동을 만들어라',
                       '같은 길이 문장의 연속을 끊어라'),
    'C_len_alternation': ('장단 교대율',
                          '장단 교대를 늘려라', '기계적 장단 교대를 끊어라'),
    'C_jump_autocorr': ('전개 템포의 관성',
                        '전개 속도의 완급을 뭉쳐라', '완급 전환을 더 자주'),
    'E_echo_head_tail': ('수미상관(첫·끝 어휘 중첩)',
                         '결말에서 서두의 핵심어를 되받아라', '수미상관을 풀어라'),
    'F_mtld_lite': ('어휘 다양성(MTLD)',
                    '같은 단어 반복을 줄이고 어휘를 다양화하라 (과평탄)',
                    '어휘를 절제하라'),
    'F_burstiness_fano': ('핵심어 집중도(burstiness)',
                          '핵심어를 특정 구간에 몰아 써라 (과평탄)',
                          '핵심어를 고르게 분산하라'),
    'F_len_gini': ('길이 불평등(Gini)',
                   '문장 길이의 빈부차를 키워라 (과평탄)', '길이 차를 줄여라'),
}


def _cat_deviations(fam, cand_counter, cat, top=3):
    """분포 이탈의 방향: 과다/과소 표지 상위."""
    vocab = cat['vocab']
    cd = _normalize(Counter(cand_counter), vocab)
    devs = sorted(zip(vocab, cd, cat['mean_dist']),
                  key=lambda x: abs(x[1] - x[2]), reverse=True)[:top]
    return [f"'{v}' {'과다' if c > m else '과소'} ({c:.0%}→목표 {m:.0%})"
            for v, c, m in devs if abs(c - m) > 0.02]


def diagnose(fp, text, z_accept=1.0, baseline_bound=None):
    res = score_text(text, fp, detail=True)
    diag = res['diagnostics']
    mod = get_analyzer(fp.get('analyzer', 'surface'))
    sents = mod.split_sentences(text)
    f = mod.chunk_features(sents)
    try:
        df = (disc.discourse_features_from_sents(sents)
              if len(sents) >= disc.MIN_SENTS else {})
    except ValueError:
        df = {}

    presc = []

    # 1) 금지 목록 위반 — 무조건 최우선 처방
    for v in res['ban_violations']:
        fam, marker = v.split(':', 1)
        presc.append({'z': 99, 'feature': v,
                      'action': f"{FAM_KO.get(fam, fam)} '{marker}'는 이 작가가 "
                                f"쓰지 않는 표지다 — 전부 제거·치환하라"})

    # 2) 범주 분포 (방향 복원: 과다/과소 표지 명시)
    for fam in CAT_FAMILIES:
        z = diag.get(f'js_{fam}', 0)
        if z >= Z_PRESCRIBE:
            items = _cat_deviations(fam, f[fam], fp['cat'][fam])
            bank = fp.get('example_bank', {})
            hint = ''
            if fam == 'endings' and bank:
                ex = [s for v in list(bank.values())[:3] for s in v[:1]]
                if ex:
                    hint = ' 예문: ' + ' / '.join(ex[:2])
            presc.append({'z': z, 'feature': f'js_{fam}',
                          'action': f"{FAM_KO[fam]} 분포 조정: "
                                    + '; '.join(items) + hint})

    # 3) 스칼라 + 담화 스칼라 (방향 = 후보값 vs 기준값)
    refs = {}
    for k, st in fp['scalars'].items():
        refs[k] = (st['mean'], f['scalars'].get(k))
    for k, st in fp.get('discourse', {}).get('scalars', {}).items():
        ref = (st.get('corpus') if k in disc.DISPERSION_KEYS else None)
        refs[k] = (ref if ref is not None else st['mean'], df.get(k))
    for k, (ref, val) in refs.items():
        z = diag.get(k, 0)
        if z < Z_PRESCRIBE or val is None or k not in SCALAR_KO:
            continue
        name, low_fix, high_fix = SCALAR_KO[k]
        action = (low_fix if val < ref else high_fix)
        presc.append({'z': z, 'feature': k,
                      'action': f"{name}: 현재 {val:.2f}, 목표 {ref:.2f} — {action}"})

    presc.sort(key=lambda p: -p['z'])
    presc = presc[:TOP_K]

    verdict = 'accept' if res['score'] <= z_accept else 'revise'
    if baseline_bound is not None and res['distance'] < baseline_bound:
        verdict = 'STOP_caricature'

    # 요인 층 요약 (P1.5): 어느 축이 흔들리는지 한 줄 요지 — 원자 처방의 '왜'
    factor_notes = []
    fdefs = {f['name']: f
             for f in fp.get('factors', {}).get('def', {}).get('factors', [])}
    for name, z in sorted(res.get('factors', {}).items(),
                          key=lambda x: -abs(x[1])):
        if abs(z) >= 1.5:
            desc = fdefs.get(name, {}).get('description', '')
            side = '작가보다 높음' if z > 0 else '작가보다 낮음'
            factor_notes.append(f"요인 '{name}' z={z:+.1f} ({side}) — {desc}")

    return {'score': res['score'], 'distance': res['distance'],
            'n_sent': res['n_sent'], 'low_confidence': res['low_confidence'],
            'flatness_hits': res.get('flatness_hits', []),
            'factor_notes': factor_notes,
            'verdict': verdict, 'prescriptions': presc}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
    fp = json.load(open(args[0], encoding='utf-8'))
    text = open(args[1], encoding='utf-8').read()
    bound = None
    if 'baseline' in opts:
        from rerank import baseline_d_bound
        bound, _ = baseline_d_bound(
            open(opts['baseline'], encoding='utf-8').read(), fp)
    r = diagnose(fp, text, z_accept=float(opts.get('z-accept', 1.0)),
                 baseline_bound=bound)
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
