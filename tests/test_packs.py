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
    # 모든 처방은 방향 동사로 끝난다 (침묵/헛처방 금지)
    verbs = ('줄여라', '분산', '대체', '다른 구문으로')
    assert lines and all(any(v in l for v in verbs) for l in lines)


def test_short_text_rejected():
    pack = load_pack(PACK)
    with pytest.raises(ValueError):
        measure('한 문장. 두 문장. 세 문장.', pack)


def test_lexicons_present_and_gated():
    """내용어·표현 사전이 존재하고 주제 분산 관문을 통과했는지."""
    p = load_pack(PACK)
    assert p.get('content_words') and p.get('phrases')
    for w, m in p['content_words'].items():
        assert m['topics'] >= 4 and m['z'] >= 2.0        # 주제 독립 + 판별 관문
        assert m['codex_per10k'] > m['human_per10k']
    for name, m in p['phrases'].items():
        assert m['ratio'] >= 1.5 and m['topics'] >= 4


def test_content_words_are_topic_independent():
    """주제어(배터리·독서·산책)는 사전에 없어야 한다 (주제 오염 배제 확인)."""
    p = load_pack(PACK)
    for topic_word in ('배터리', '독서', '산책', '커피', '김치'):
        assert topic_word not in p['content_words']


def test_lexicon_scoring_lifts_codex_over_novel():
    """사전 포함 채점에서 코덱스 샘플이 문학 산문보다 높아야 한다."""
    pack = load_pack(PACK)
    codex = measure(_read('packs/corpus_samples/codex_essay_sample.txt'), pack)
    novel = measure(_read('corpus/yisang/봉별기.txt'), pack)
    assert codex['codex_score'] > novel['codex_score']
    # 사전 히트가 실제로 잡히는지 (kiwi 환경)
    if codex.get('kiwi'):
        kinds = {h['kind'] for h in codex['hits']}
        assert 'content' in kinds or 'phrase' in kinds
