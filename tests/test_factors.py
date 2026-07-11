# -*- coding: utf-8 -*-
"""P1.5 요인 구조 검증 — 확장 특징(QG1)·요인 런타임·통합(QG4)."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features_ext import ext_features_from_sents, t0_features  # noqa: E402
from factors_runtime import load_factor_def, project, score_factors  # noqa: E402
from stylometry import build_fingerprint, score_text  # noqa: E402
from features_surface import split_sentences  # noqa: E402

needs_private = pytest.mark.skipif(
    not (ROOT / 'corpus/target_essay.txt').exists(),
    reason='비공개 픽스처 필요 — 공개 저장소에서는 skip')


def _read(rel):
    return (ROOT / rel).read_text(encoding='utf-8')


# ---------------------------------------------------------------- 확장 특징
def test_ext_detects_constructions():
    sents = ['메모는 기억술이 아니라 사고술이다.',
             '우리는 그것을 할 수 있다.',
             '아마 그것은 습관일 것이다.',
             '나는 반드시 그곳에 간다.']
    f = t0_features(sents)
    assert f['CX_not_a_but_b_sent'] > 0
    assert f['CX_hal_su_sent'] > 0
    assert f['CX_geot_end_sent'] > 0
    assert f['L7_hedge_sent'] > 0 and f['L7_booster_sent'] > 0
    assert f['L7_first_person_sent'] > 0


def test_ext_t1_kiwi_keys():
    sents = split_sentences(_read('corpus/yisang/봉별기.txt'))[:40]
    f = ext_features_from_sents(sents)
    assert 'L2_nominalization_ej' in f and 'L2_verb_density_ej' in f
    assert f['L2_verb_density_ej'] > 0


def test_ext_rejects_tiny_input():
    with pytest.raises(ValueError):
        ext_features_from_sents(['한 문장.'])


# ---------------------------------------------------------------- 요인 런타임
def test_factor_projection_and_separation():
    """요인 점수가 계산되고, 본인 텍스트가 타자 텍스트보다 요인 공간에서 가깝다."""
    fdef = load_factor_def()
    assert fdef is not None and len(fdef['factors']) >= 5
    fp = build_fingerprint(_read('corpus/kimyj/봄봄.txt'), 'kimyj')
    assert 'factors' in fp
    self_z, _ = score_factors(
        split_sentences(_read('corpus/kimyj/만무방.txt')), fp['factors'])
    other_z, _ = score_factors(
        split_sentences(_read('corpus/yisang/날개.txt')), fp['factors'])
    mean_abs = lambda d: sum(abs(v) for v in d.values()) / len(d)
    assert mean_abs(self_z) < mean_abs(other_z), \
        f'요인 분리 실패: 본인 {mean_abs(self_z):.2f} vs 타자 {mean_abs(other_z):.2f}'


def test_factor_projection_needs_enough_sentences():
    fdef = load_factor_def()
    with pytest.raises(ValueError):
        project(['하나.', '둘.', '셋.'], fdef)


# ---------------------------------------------------------------- 통합 (QG4)
@needs_private
def test_score_output_has_factors_and_z_total_unchanged():
    """요인은 진단 층 — z_total 산식은 with_factors 여부와 무관해야 한다."""
    essay = _read('corpus/target_essay.txt')
    kimyj = _read('corpus/kimyj/봄봄.txt')
    fp_on = build_fingerprint(essay, 'e', contrast_text=kimyj, with_factors=True)
    fp_off = build_fingerprint(essay, 'e', contrast_text=kimyj, with_factors=False)
    c2 = _read('candidates/C2.txt')
    r_on, r_off = score_text(c2, fp_on), score_text(c2, fp_off)
    assert r_on['score'] == r_off['score']          # z_total 불변 (R2 결정)
    assert r_on['distance'] == r_off['distance']
    assert 'factors' in r_on and 'factors' not in r_off


@needs_private
def test_c2_flaws_named_by_factors():
    """C2의 실증된 결함(평탄·기계적 교대)이 요인 이름으로 표면화된다."""
    fp = build_fingerprint(_read('corpus/target_essay.txt'), 'e',
                           contrast_text=_read('corpus/kimyj/봄봄.txt'))
    r = score_text(_read('candidates/C2.txt'), fp)
    assert r['factors']['길이 균질성'] > 2.0        # 평탄화
    assert abs(r['factors']['장단 교대성']) > 2.0   # 기계적 장단 교대
