# -*- coding: utf-8 -*-
"""모델 스타일 팩 감사 검증 (P2 파일럿)."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from style_pack import load_pack, measure, prescribe  # noqa: E402

PACK = ROOT / 'packs' / 'codex-style.json'
pytestmark = pytest.mark.skipif(not PACK.exists(), reason='codex-style 팩 없음')


def _read(rel):
    return (ROOT / rel).read_text(encoding='utf-8')


def test_pack_schema():
    p = load_pack(PACK)
    assert p['factor_signature'] and p['feature_markers']
    # 모든 마커는 판별 관문 |d|>=0.5 통과 (팩 계약)
    for m in p['feature_markers'].values():
        assert abs(m['d']) >= 0.5


def test_codex_sample_scores_higher_than_human():
    """코덱스 생성 샘플이 인간 위키보다 코덱스니스 점수가 높아야 한다."""
    pack = load_pack(PACK)
    codex = measure(_read('packs/corpus_samples/codex_essay_sample.txt'), pack)
    human = measure(_read('corpus/hyunjg/운수좋은날.txt'), pack)
    assert codex['codex_score'] > human['codex_score']


def test_strip_prescription_targets_markers():
    """제거 처방은 실제 적중 마커를 겨냥한다 (침묵/헛처방 금지)."""
    pack = load_pack(PACK)
    m, lines = prescribe(_read('packs/corpus_samples/codex_essay_sample.txt'), pack)
    assert m['codex_score'] > 0
    assert lines and all('줄여라' in l or '분산' in l for l in lines)


def test_short_text_rejected():
    pack = load_pack(PACK)
    with pytest.raises(ValueError):
        measure('한 문장. 두 문장. 세 문장.', pack)
