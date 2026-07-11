#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""모두의 말뭉치 '신문' JSON → 장르별 인간 대조군 텍스트.

NIKL 신문 말뭉치 표준 구조 (문서 확인분):
  {"document": [
     {"id": "...",
      "metadata": {"title","author","publisher","date",
                   "topic": "정치|경제|사회|문화|오피니언|...",
                   "original_topic": "..."},
      "paragraph": [{"id":"...","form":"문장/문단 텍스트"}, ...]}
  ]}

장르 매핑 (우리 AI 장르에 대응):
  오피니언/칼럼/사설/논설  → essay  (우리가 부족한 에세이·논설 장르)
  그 외 정치/경제/사회/... → news   (위키뉴스 보강)

각 문서를 문단당 한 줄 평문으로 <out>/<genre>_NNNN.txt 저장. 표기 계열이
장르 교란을 일으키지 않도록(build_pack이 이미 제외하지만) 원문은 손대지 않는다.

사용: python3 parse_modu_news.py <json_dir_or_file> <out_dir>
      [--per-genre=120] [--min-eojeol=180]
"""
import glob
import json
import os
import re
import sys

OPINION = ("오피니언", "칼럼", "사설", "논설", "기고", "시론", "opinion",
           "column", "editorial")


def genre_of(topic):
    t = (topic or "").lower()
    return "essay" if any(o.lower() in t for o in OPINION) else "news"


def doc_text(doc):
    paras = []
    for p in doc.get("paragraph", []):
        form = p.get("form", "").strip()
        if form and re.search(r"[가-힣]", form):
            paras.append(" ".join(form.split()))
    return "\n".join(paras)


def iter_documents(path):
    files = ([path] if os.path.isfile(path)
             else sorted(glob.glob(f"{path}/**/*.json", recursive=True)))
    for fp in files:
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as e:
            print(f"SKIP {fp}: {e}", file=sys.stderr)
            continue
        for doc in data.get("document", []):
            yield doc


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    src, out_dir = args[0], args[1]
    per_genre = int(opts.get("per-genre", 120))
    min_ej = int(opts.get("min-eojeol", 180))

    os.makedirs(out_dir, exist_ok=True)
    count = {"essay": 0, "news": 0}
    for doc in iter_documents(src):
        g = genre_of(doc.get("metadata", {}).get("topic")
                     or doc.get("metadata", {}).get("original_topic"))
        if count[g] >= per_genre:
            if all(count[k] >= per_genre for k in count):
                break
            continue
        body = doc_text(doc)
        if len(body.split()) < min_ej:
            continue
        idx = count[g]
        with open(f"{out_dir}/{g}_{idx:04d}.txt", "w", encoding="utf-8") as f:
            f.write(body)
        count[g] += 1
    print(f"저장: essay(칼럼·사설) {count['essay']}편, news(기사) {count['news']}편 "
          f"→ {out_dir}")
    print("→ build_pack.py <out_dir> ... --genre essay / --genre news 로 팩 생성")


if __name__ == "__main__":
    main()
