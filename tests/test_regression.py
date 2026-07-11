# -*- coding: utf-8 -*-
"""회귀 테스트 — 실증된 결과가 코드 변경 후에도 유지되는지.

각 테스트는 REPORT.md·README.md에 기록된 실측과 1:1 대응한다.
실행: .venv/bin/python -m pytest tests/ -q
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stylometry import build_fingerprint, score_text, get_analyzer  # noqa: E402

# 대상 에세이·생성 후보는 저자 개인 글이라 공개 저장소에는 없다 (README §데이터 고지)
needs_private = pytest.mark.skipif(
    not (ROOT / 'corpus/target_essay.txt').exists(),
    reason='비공개 픽스처(대상 에세이·후보) 필요 — 공개 저장소에서는 skip')


def _read(rel):
    return (ROOT / rel).read_text(encoding='utf-8')


def essay_fp():
    return build_fingerprint(_read('corpus/target_essay.txt'), 'essay',
                             contrast_text=_read('corpus/kimyj/봄봄.txt'))


# ---------------------------------------------------------------- v1 판별 회귀
def test_v1_two_author_attribution():
    """v1 실측: 김유정/이상 held-out 측정층 25/28 (REPORT §2)."""
    from validate import CORPORA, load, as_text
    train, held, fps = {}, {}, {}
    for a in CORPORA:
        train[a], held[a] = load(a)
    for a in CORPORA:
        other = [b for b in CORPORA if b != a][0]
        fps[a] = build_fingerprint(as_text(train[a]), a,
                                   contrast_text=as_text(train[other]),
                                   with_discourse=False)
    correct = total = 0
    for true_a in CORPORA:
        for c in held[true_a]:
            t = '\n'.join(c)
            d = {b: score_text(t, fps[b])['distance'] for b in CORPORA}
            correct += (min(d, key=d.get) == true_a)
            total += 1
    assert total == 28
    assert correct >= 25, f'v1 판별 회귀 실패: {correct}/28 (기준 25/28)'


# ---------------------------------------------------------------- C2 평탄화 검출
@needs_private
def test_c2_flattening_detected():
    """v2 실측: C2 과평탄 (len_cv 0.34 vs 0.53) — 산포 비대칭 벌점이 검출해야
    한다 (README §실증 4, 측정명세 §채점규칙 2)."""
    r = score_text(_read('candidates/C2.txt'), essay_fp(), detail=True)
    assert any(h.startswith('C_len_cv:') for h in r['flatness_hits']), \
        f'C2 len_cv 과평탄 미검출: {r["flatness_hits"]}'
    # v1이 못 잡던 기계적 장단 교대(자기상관 부호 반전)도 z 진단에 떠야 한다
    assert r['diagnostics']['C_len_autocorr'] >= 2.0


@needs_private
def test_subagent_baseline_rejected():
    """v1 실측: 원문 통째 모사 베이스라인은 수용 영역 밖 + 금지 위반
    (REPORT §3: z 6.35, 니까·는데 위반)."""
    r = score_text(_read('candidates/subagent.txt'), essay_fp())
    assert r['score'] > 4.0
    fams = {v.split(':')[0] for v in r['ban_violations']}
    assert 'connectives' in fams


# ---------------------------------------------------------------- 채점 규칙 v2
@needs_private
def test_dispersion_penalty_is_asymmetric():
    """산포 부족(과평탄)에만 α=1.5 가중 — 같은 크기의 초과에는 표준 가중."""
    fp = essay_fp()
    st = fp['discourse']['scalars']['C_len_cv']
    ref = st['corpus']
    # 직접 signed z 계산으로 규칙 자체를 검증
    low_z = abs((ref - 2 * st['std']) - ref) / st['std'] * 1.5
    high_z = abs((ref + 2 * st['std']) - ref) / st['std'] * 1.0
    assert low_z == 3.0 and high_z == 2.0


@needs_private
def test_low_confidence_flag():
    """P0 벤치마크 실측(15문장 53.6% vs 30문장 75%) → 30문장 미만 저신뢰 플래그."""
    fp = essay_fp()
    short = '\n'.join(_read('corpus/target_essay.txt').split('\n')[:6])
    r = score_text(short, fp)
    assert r['low_confidence'] is True
    r2 = score_text(_read('candidates/C2.txt'), fp)
    assert r2['n_sent'] < 30 and r2['low_confidence'] is True


# ---------------------------------------------------------------- 분석기 이중화
def test_analyzer_recorded_and_followed():
    """지문에 analyzer가 기록되고 채점이 그것을 따라간다 (혼용 차단의 전제)."""
    text = _read('corpus/yisang/봉별기.txt')
    fp_s = build_fingerprint(text, 't', analyzer='surface', with_discourse=False)
    fp_k = build_fingerprint(text, 't', analyzer='kiwi', with_discourse=False)
    assert fp_s['analyzer'] == 'surface' and fp_k['analyzer'] == 'kiwi'
    r_s, r_k = score_text(text, fp_s), score_text(text, fp_k)
    assert r_s['analyzer'] == 'surface' and r_k['analyzer'] == 'kiwi'
    # 어미 vocab이 실제로 다른 공간에 산다 (표층 음절 vs EF 형태소)
    assert fp_s['cat']['endings']['vocab'] != fp_k['cat']['endings']['vocab']


def test_unknown_analyzer_rejected():
    import pytest
    with pytest.raises(ValueError):
        get_analyzer('mecab')
