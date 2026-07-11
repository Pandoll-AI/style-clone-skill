# -*- coding: utf-8 -*-
"""Phase 4 검증 — 종결어미 후처리·문단 재순위화·교정 루프."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from postprocess import convert_text, convert_sentence  # noqa: E402
from stylometry import build_fingerprint  # noqa: E402
from rerank import rerank, baseline_d_bound  # noqa: E402
from loop import diagnose  # noqa: E402

# 대상 에세이·생성 후보는 저자 개인 글이라 공개 저장소에는 없다 (README §데이터 고지)
needs_private = pytest.mark.skipif(
    not (ROOT / 'corpus/target_essay.txt').exists(),
    reason='비공개 픽스처(대상 에세이·후보) 필요 — 공개 저장소에서는 skip')


def _read(rel):
    return (ROOT / rel).read_text(encoding='utf-8')


# ---------------------------------------------------------------- 후처리
HANDA = "나는 밥을 먹었다. 그는 학교에 간다. 하늘이 참 예쁘다. 이것이 핵심이다."


def test_speech_level_roundtrip():
    """한다체 → 합쇼체 → 한다체 왕복 무손실 (계획 Verification 4)."""
    h, rep = convert_text(HANDA, 'hapsyo')
    assert rep['rate'] == 1.0
    assert '먹었습니다' in h and '갑니다' in h and '예쁩니다' in h and '핵심입니다' in h
    back, _ = convert_text(h, 'haera')
    assert back == HANDA


def test_sentence_type_preserved():
    """청유·명령·의문 유형이 평서로 왜곡되지 않는다 (실험에서 확인한 위험)."""
    out, _ = convert_text("우리는 내일 떠나자. 빨리 일어나라. 너는 어디에 가느냐?",
                          'hapsyo')
    assert '떠납시다' in out and '일어나십시오' in out and '갑니까' in out


def test_dialogue_and_nominal_untouched():
    conv, why = convert_sentence('“성례구 뭐구 미처 자라야지!”', 'hapsyo')
    assert conv is None and why == 'dialogue'
    conv, why = convert_sentence('결론은 하나.', 'hapsyo')
    assert conv is None and why == 'no_ef'


def test_unconvertible_reported_not_hidden():
    """변환 불가 문장은 원문 보존 + 사유 카운트 (침묵 실패 금지)."""
    text = "결론은 하나. 나는 간다."
    out, rep = convert_text(text, 'hapsyo')
    assert '결론은 하나.' in out and rep['total'] == 2 and rep['converted'] == 1
    assert rep['reasons'].get('no_ef') == 1


# ---------------------------------------------------------------- 재순위화
def essay_fp():
    return build_fingerprint(_read('corpus/target_essay.txt'), 'essay',
                             contrast_text=_read('corpus/kimyj/봄봄.txt'))


@needs_private
def test_rerank_composite_not_worse():
    """문단 조합은 d 기준 최선 단일 후보보다 나쁘지 않아야 한다."""
    fp = essay_fp()
    cands = {f'C{i}': _read(f'candidates/C{i}.txt') for i in (1, 2, 3, 4)}
    r = rerank(fp, cands)
    assert r['composite_beats_best_single']
    assert r['verdict']['ban_ok'] and r['verdict']['flatness_ok']
    assert len(r['slots']) == max(len(c.split('\n\n')) for c in cands.values())


def test_caricature_guard_quantile_baseline():
    """가드 기준선은 baseline 청크 d 분포의 분위수 (설계감사 #13)."""
    fp = build_fingerprint(_read('corpus/kimyj/봄봄.txt'), 'kimyj',
                           with_discourse=False)
    bound, dist = baseline_d_bound(_read('corpus/kimyj/만무방.txt'), fp)
    assert bound is not None and len(dist) >= 3
    assert min(dist) <= bound <= sorted(dist)[len(dist) // 2]


# ---------------------------------------------------------------- 교정 루프
@needs_private
def test_loop_prescribes_ban_first_and_directions():
    fp = essay_fp()
    d = diagnose(fp, _read('candidates/subagent.txt'))
    assert d['verdict'] == 'revise'
    assert d['prescriptions'][0]['z'] == 99          # 금지 위반 최우선
    assert any('니까' in p['action'] for p in d['prescriptions'])
    assert any('목표' in p['action'] for p in d['prescriptions'] if p['z'] < 99)


@needs_private
def test_loop_accepts_good_candidate():
    fp = essay_fp()
    d = diagnose(fp, _read('candidates/C2.txt'))
    assert d['verdict'] == 'accept'
    assert d['score'] <= 1.0
