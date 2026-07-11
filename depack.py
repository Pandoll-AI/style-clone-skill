#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""자동 디팩 루프 — 모델에게 '자기 스타일 티 제거'를 시키는 교정 루프.

저자와 심판을 분리한다: 스타일 팩(style_pack)이 진단·채점(심판)하고,
재작성(저자)은 외부 모델(codex 등)이 한다. 이 스크립트는 오케스트레이터다:

  원문 → 팩 strip 처방 → [모델 재작성] → 재채점 → (여전히 높으면) 반복

핵심 설계:
- **저자 ≠ 심판**: Claude가 개작하면 Claude 습관이 새로 끼므로(코덱스 팩은
  그걸 못 봄), 코덱스 글의 디팩은 코덱스에게 시킨다. 심판은 결정론적 팩.
- 수렴 판정: 점수 ≤ 목표 or 개선폭 < ε or 최대 회전. 개선이 멈추면 정지
  (모델이 자기 습관을 스스로는 못 지운다는 것도 유효한 결과 — 숨기지 않음).
- 재작성기는 커맨드 템플릿으로 주입 (codex 외 모델도 가능).

사용:
  python3 depack.py <pack.json> <원문.txt> [--out=결과.txt] [--rounds=3]
      [--target=15] [--effort=medium]
결과 JSON을 stdout에, 최종 텍스트를 --out에 쓴다.
"""
import json
import subprocess
import sys
import tempfile

from style_pack import load_pack, measure, prescribe

PROMPT_TMPL = """다음 한국어 글을 문체만 바꿔 다시 써라. 뜻과 문단 수, 분량은 그대로 둔다. \
아래 표현들을 덜 쓰고, 문장 길이를 짧고 길게 섞어 더 자연스럽게 고친다. 결과물 본문만 출력하라.

덜 쓸 것: {prescriptions}

원문:
{text}
"""


def run_codex(text, prescriptions, effort='medium'):
    """codex exec으로 재작성. 프롬프트는 인자로 직접 전달(쉘 이스케이프 없음)."""
    prompt = PROMPT_TMPL.format(prescriptions=prescriptions, text=text)
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as of:
        out_path = of.name
    proc = subprocess.run(
        ['codex', 'exec', '--skip-git-repo-check', '-s', 'read-only',
         '-c', f'model_reasoning_effort="{effort}"', '-o', out_path, prompt],
        capture_output=True, text=True, timeout=600)
    out = open(out_path, encoding='utf-8').read().strip()
    if not out:
        raise RuntimeError(f'codex 빈 출력: {proc.stderr[-300:]}')
    # codex가 재작성 대신 작업 보고를 낸 경우 차단 (원문의 40% 미만이면 거부)
    if len(out) < 0.4 * len(text):
        raise RuntimeError(f'재작성 아닌 응답으로 보임(짧음): {out[:80]!r}')
    return out


def depack(pack, text, effort='medium', rounds=3, target=15.0, eps=1.5):
    history = []
    cur = text
    base = measure(cur, pack)
    history.append({'round': 0, 'score': base['codex_score'],
                    'n_hits': base['n_hits'], 'author': 'original'})
    for r in range(1, rounds + 1):
        m = measure(cur, pack)
        if m['codex_score'] <= target:
            break
        # 처방을 단어/마커 나열로 압축 (codex가 개발 태스크로 오해하지 않도록)
        toks = []
        for h in m['hits'][:12]:
            t = (h.get('marker') or h.get('word') or h.get('phrase')
                 or h.get('name'))
            if t and t not in toks:
                toks.append(t)
        presc = ', '.join(toks)
        # 재작성 + 채점, 짧은/거부 출력은 최대 2회 재시도
        nm = new = None
        last_err = ''
        for attempt in range(4):
            try:
                cand = run_codex(cur, presc, effort)
                nm = measure(cand, pack)
                new = cand
                break
            except (ValueError, RuntimeError) as e:
                last_err = str(e)
        if nm is None:
            history.append({'round': r, 'author': 'rewriter',
                            'stopped': f'재작성 실패: {last_err[:80]}'})
            break
        history.append({'round': r, 'score': nm['codex_score'],
                        'n_hits': nm['n_hits'], 'author': 'rewriter',
                        'delta': round(m['codex_score'] - nm['codex_score'], 2)})
        improved = m['codex_score'] - nm['codex_score']
        if nm['codex_score'] < m['codex_score']:      # 개선된 경우만 채택
            cur = new
        if improved < eps:                            # 개선 정체 → 정지
            history[-1]['stopped'] = 'improvement < eps'
            break
    final = measure(cur, pack)
    return {'final_score': final['codex_score'], 'final_hits': final['n_hits'],
            'text': cur, 'history': history}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = {}
    for a in sys.argv[1:]:
        if a.startswith('--'):
            k, _, v = a[2:].partition('=')
            opts[k] = v
    pack = load_pack(args[0])
    text = open(args[1], encoding='utf-8').read()
    res = depack(pack, text, effort=opts.get('effort', 'medium'),
                 rounds=int(opts.get('rounds', 3)),
                 target=float(opts.get('target', 15)))
    if 'out' in opts:
        open(opts['out'], 'w', encoding='utf-8').write(res['text'])
    print(json.dumps({k: v for k, v in res.items() if k != 'text'},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
