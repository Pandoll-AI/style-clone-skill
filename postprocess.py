#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""종결어미 스피치 레벨 결정론적 변환 (루프 1 — 규칙 기반 투영).

kiwipiepy EF 태그로 문장 종결을 찾아 목표 스피치 레벨(합쇼체·한다체·해요체)로
치환하고 Kiwi.join으로 재조립한다. join이 음운 이형태(ㅂ니다/습니다,
ㄴ다/는다, 아요/어요)를 처리한다는 것을 실험으로 확인함.

원칙 (침묵 실패 금지):
- 문장 유형(평서·의문·명령·청유)을 보존한다 — 유형을 모르는 어미는 변환하지
  않고 unconverted로 보고한다 (실험: 떠나자→떠납니다 같은 유형 왜곡 방지).
- 대화 인용("...")는 화자의 말 — 지문(地文)만 변환 대상. 문장 전체가 인용이면
  건너뛴다.
- 한계(문헌지도 §8, Hong et al. 2018): 종결어미만의 통사 변환은 부분 해법.
  어휘·어조까지 포함한 전신 변환은 LLM 재작성 + 재순위화가 담당하고,
  이 모듈은 생성 후 잔여 이탈의 교정과 스타일 강제에 쓴다.

사용: python3 postprocess.py <hapsyo|haera|haeyo> <파일> [--report]
API : convert_text(text, level) -> (변환문, 리포트)
"""
import json
import sys

LEVELS = ('hapsyo', 'haera', 'haeyo')

# 알려진 EF 형태 → 문장 유형. kiwi는 이형태를 부분 정규화하므로 변형 병기.
# 유형: decl(평서) inter(의문) imper(명령) prop(청유)
EF_TYPE = {
    # 합쇼체
    'ᆸ니다': 'decl', '습니다': 'decl', 'ᆸ니까': 'inter', '습니까': 'inter',
    '십시오': 'imper', 'ᆸ시다': 'prop', '읍시다': 'prop',
    # 해라체/한다체
    '다': 'decl', 'ᆫ다': 'decl', '는다': 'decl',
    '냐': 'inter', '느냐': 'inter', '니': 'inter', 'ᆫ가': 'inter',
    '는가': 'inter', '은가': 'inter', 'ᆫ지': 'inter',
    '어라': 'imper', '아라': 'imper', '거라': 'imper', '려무나': 'imper',
    '자': 'prop',
    # 해요체 (요 통합형)
    '어요': 'decl', '아요': 'decl', '에요': 'decl', '예요': 'decl',
    '세요': 'imper', 'ᆸ시오': 'imper',
    # 해체(반말) — 평서/의문 중의적: 물음표로 재분류
    '어': 'decl', '아': 'decl', '지': 'decl', '야': 'decl', '네': 'decl',
    '군': 'decl', '구나': 'decl', '데': 'decl', '거든': 'decl', '잖아': 'decl',
    # 하오·하게체
    '오': 'decl', '소': 'decl', '게': 'imper', '세': 'prop', '나': 'inter',
}

AMBIG_WITH_QUESTION = {'어', '아', '지', '야', '요', '네', '나', '데'}


def _target_ef(level, sent_type, prev_tag):
    """목표 (form, tag) — None이면 변환 불가 유형."""
    if level == 'hapsyo':
        return {'decl': 'ᆸ니다', 'inter': 'ᆸ니까',
                'imper': '십시오', 'prop': 'ᆸ시다'}[sent_type]
    if level == 'haera':
        if sent_type == 'decl':
            # 동사 현재만 ㄴ다/는다, 나머지(형용사·계사·선어말 뒤)는 다
            return 'ᆫ다' if prev_tag.startswith('VV') or prev_tag == 'VX' \
                else '다'
        return {'inter': '느냐', 'imper': '어라', 'prop': '자'}[sent_type]
    if level == 'haeyo':
        if sent_type == 'imper':
            return '세요'
        # 계사 뒤는 '에요' (이어요→이에요 표준형)
        return '에요' if prev_tag == 'VCP' else '어요'
    raise ValueError(f'unknown level: {level!r} ({"|".join(LEVELS)})')


_kiwi = None

def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def convert_sentence(sentence, level):
    """한 문장 변환 → (변환문 또는 None, 사유)."""
    s = sentence.strip()
    if not s:
        return None, 'empty'
    # 인용부호가 있는 문장은 통째로 보존: 화자의 말이거나, 문장 분리가
    # 인용 경계와 어긋나 join 재조립 시 부호가 소실될 수 있음(실험 확인)
    if any(q in s for q in '“”"‘’'):
        return None, 'dialogue'
    kiwi = _get_kiwi()
    toks = kiwi.tokenize(s)
    ef_i = None
    for i, t in enumerate(toks):
        if t.tag == 'EF':
            ef_i = i
    if ef_i is None:
        return None, 'no_ef'             # 명사 종결 등
    src = toks[ef_i].form
    sent_type = EF_TYPE.get(src)
    if sent_type is None:
        return None, f'unknown_ef:{src}'
    if src in AMBIG_WITH_QUESTION and ('?' in s or '까' in src):
        sent_type = 'inter'
    prev_tag = toks[ef_i - 1].tag if ef_i > 0 else ''
    tgt = _target_ef(level, sent_type, prev_tag)
    if tgt == src:
        return None, 'already'
    morphs = [(t.form, t.tag) for t in toks]
    morphs[ef_i] = (tgt, 'EF')
    try:
        out = kiwi.join(morphs)
    except Exception as e:               # join 실패 = 변환 불가로 보고
        return None, f'join_fail:{e}'
    # 의문형 물음표 정리: 평서→의문 등 유형은 보존되므로 부호는 그대로 둔다
    return out, 'converted'


def convert_text(text, level):
    """문단 구조(줄 단위)를 보존하며 변환. → (텍스트, 리포트)."""
    if level not in LEVELS:
        raise ValueError(f'unknown level: {level!r} ({"|".join(LEVELS)})')
    from features_surface import split_sentences
    out_lines, report = [], {'converted': 0, 'total': 0, 'reasons': {}}
    for line in text.split('\n'):
        if not line.strip():
            out_lines.append(line)
            continue
        parts = []
        for s in split_sentences(line):
            report['total'] += 1
            conv, why = convert_sentence(s, level)
            report['reasons'][why.split(':')[0]] = \
                report['reasons'].get(why.split(':')[0], 0) + 1
            if conv is not None:
                report['converted'] += 1
                parts.append(conv)
            else:
                parts.append(s)
        out_lines.append(' '.join(parts))
    report['rate'] = (round(report['converted'] / report['total'], 3)
                      if report['total'] else 0.0)
    return '\n'.join(out_lines), report


def main():
    level, path = sys.argv[1], sys.argv[2]
    text = open(path, encoding='utf-8').read()
    out, report = convert_text(text, level)
    print(out)
    if '--report' in sys.argv:
        print('\n---', json.dumps(report, ensure_ascii=False), file=sys.stderr)


if __name__ == '__main__':
    main()
