#!/usr/bin/env python3
"""davincimap 원문 페이지(markdown 덤프)에서 본문만 추출.

본문은 '- | | 1 제목 2 문단...' 형태의 초장문 라인 하나에 들어 있고,
문단 경계가 ' N ' (순차 증가 정수) 토큰으로 표시된다.
순차성 검사로 본문 내 숫자('33번지' 등)와 구분한다.
"""
import re, sys

def clean(raw_path, out_path):
    lines = open(raw_path, encoding='utf-8').read().split('\n')
    def hangul(s):
        return sum('가' <= c <= '힣' for c in s)
    body = max(lines, key=hangul)  # 본문 = 한글이 가장 많은 라인
    body = re.sub(r'-{3,}', ' ', body)  # 표 구분선 제거
    body = body.lstrip('- ').strip('| ').strip()
    # markdown 링크/이미지 제거
    body = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', body)
    body = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', body)
    body = body.replace('|', ' ')
    tokens = body.split(' ')
    segs, cur, expect = [], [], 1
    for t in tokens:
        if t == str(expect):
            if cur:
                segs.append(' '.join(cur)); cur = []
            expect += 1
        elif t:
            cur.append(t)
    if cur:
        segs.append(' '.join(cur))
    # 세그먼트 1은 제목 반복 → 짧으면 버림
    if segs and len(segs[0]) < 30:
        segs = segs[1:]
    text = '\n'.join(segs)
    open(out_path, 'w', encoding='utf-8').write(text)
    n_ej = len(text.split())
    print(f'{out_path}: segs={len(segs)} eojeol={n_ej} chars={len(text)}')
    print('HEAD:', text[:150].replace('\n', ' / '))
    print('TAIL:', text[-150:].replace('\n', ' / '))

if __name__ == '__main__':
    clean(sys.argv[1], sys.argv[2])
