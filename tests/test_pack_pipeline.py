# -*- coding: utf-8 -*-
"""통합 팩 빌더·디팩 루프 스모크 테스트 (외부 모델 없이 순수 로직만)."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build_pack  # noqa: E402
import depack  # noqa: E402
from style_pack import load_pack  # noqa: E402


def test_cohens_d_direction():
    assert build_pack._cohens_d([3, 3, 3], [1, 1, 1]) > 0
    assert build_pack._cohens_d([1, 1, 1], [3, 3, 3]) < 0


def test_discrimination_separates_obvious():
    """서로 다른 두 분포를 완벽히 가르면 정확도 1.0에 근접."""
    cx = [{'a': 10.0 + i * 0.1, 'b': 0.0} for i in range(8)]
    hu = [{'a': 0.0, 'b': 10.0 + i * 0.1} for i in range(8)]
    r = build_pack._discrimination(cx, hu, n_boot=100)
    assert r['acc'] >= 0.9


def test_overuse_gate():
    """분포 마커는 임계(1.8배)·바닥(3%) 관문을 넘어야 채택."""
    from collections import Counter
    cx = {'endings': Counter({'다': 90, '요': 10})}
    hu = {'endings': Counter({'다': 50, '요': 50})}
    over = build_pack._overuse(cx, hu, 'endings')
    assert '다' in over and '요' not in over        # 다 90% vs 50% 채택, 요는 과소


def test_depack_prompt_is_concise():
    """디팩 프롬프트는 codex를 개발 태스크로 오해시키지 않도록 간결해야 한다."""
    p = depack.PROMPT_TMPL.format(prescriptions='물론, 그래서', text='원문.')
    assert 'verifier' not in p and '검증' not in p
    assert '결과물 본문만 출력하라' in p


def test_short_output_guard():
    """재작성 출력이 원문의 40% 미만이면 거부 (작업 보고 차단) — 로직 확인."""
    # run_codex 내부 가드를 직접 검증하긴 어려우므로 임계 상수만 고정
    import inspect
    src = inspect.getsource(depack.run_codex)
    assert '0.4' in src and 'len(text)' in src
