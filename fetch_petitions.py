#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""청와대 국민청원(Korpora korean_petitions) → 주장문(argue) 장르 인간 대조군.

라이선스: CC0 1.0 (퍼블릭 도메인) — 재배포·상업사용 자유. 승인 절차 없이
Korpora 패키지로 즉시 다운로드. 시민이 직접 쓴 현대 한국어 주장·설득문.

품질 필터 (개조식·저품질 청원 배제, '잘 쓰인' 것만):
- num_agree(동의 수) 상위 — 호응 = 품질의 대리 지표
- 어절 220~650, 문장 10개 이상, 평균 문장 길이 7어절 이상(나열식 배제)
- 카테고리당 최대 N편 (한 주제 편중 방지 → 문체 다양성 확보)

장르 주의(docs/07): 청원은 '주장/설득(argue)' 장르지 성찰적 에세이가 아니다.
AI 대조도 '주장문' 생성분과 매칭해야 한다 (essay 팩에 넣으면 장르 미스매칭).

사용: python3 fetch_petitions.py <out_dir> [--n=36] [--per-cat=6]
"""
import os
import sys

from features_surface import split_sentences


def main():
    out_dir = sys.argv[1]
    n_target = 36
    per_cat = 6
    for a in sys.argv[2:]:
        if a.startswith("--n="):
            n_target = int(a.split("=")[1])
        elif a.startswith("--per-cat="):
            per_cat = int(a.split("=")[1])
    os.makedirs(out_dir, exist_ok=True)

    from Korpora import Korpora
    tr = Korpora.load("korean_petitions").train

    cands = []
    for i in range(len(tr)):
        ex = tr[i]
        body = ex.text.strip()
        n = len(body.split())
        if not (220 <= n <= 650):
            continue
        sents = split_sentences(body)
        if len(sents) < 10 or n / len(sents) < 7:
            continue
        try:
            agree = int(ex.num_agree)
        except (ValueError, TypeError):
            agree = 0
        cands.append((agree, ex.category, body))
        if len(cands) > 8000:                    # 앞부분에서 충분히 모으면 중단
            break

    cands.sort(key=lambda x: -x[0])              # 호응 상위
    saved, seen = 0, {}
    for agree, cat, body in cands:
        if seen.get(cat, 0) >= per_cat:
            continue
        with open(f"{out_dir}/argue_{saved:04d}.txt", "w", encoding="utf-8") as f:
            f.write(body)
        seen[cat] = seen.get(cat, 0) + 1
        saved += 1
        if saved >= n_target:
            break
    print(f"argue 인간 코퍼스: {saved}편 (카테고리: {seen})")


if __name__ == "__main__":
    main()
