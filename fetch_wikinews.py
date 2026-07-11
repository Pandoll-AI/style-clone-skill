#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""한국 위키뉴스에서 현대 뉴스 기사를 수집 (뉴스 장르 인간 대조군).

라이선스: 위키뉴스는 CC BY 2.5. 본 용도는 게재·재배포·모델 학습이 아니라
문체 통계(집계)만 추출 — 원문은 팩에 저장하지 않는다.

allpages(namespace 0)로 기사 목록을 받고, 각 본문을 위키 마크업 제거 후
문단당 한 줄 평문으로 저장. 출처 링크·날짜 템플릿·표는 제거한다.

사용: python3 fetch_wikinews.py <out_dir> [--limit=120] [--min-eojeol=180]
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "styleclone-research/0.1 (personal research; stats-only)"}
API = "https://ko.wikinews.org/w/api.php?"


def api(params):
    req = urllib.request.Request(API + urllib.parse.urlencode(params), headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=30))


def list_articles(limit):
    titles, cont = [], None
    while len(titles) < limit:
        p = {"action": "query", "list": "allpages", "apnamespace": "0",
             "aplimit": "50", "apfilterredir": "nonredirects", "format": "json"}
        if cont:
            p["apcontinue"] = cont
        r = api(p)
        titles += [m["title"] for m in r["query"]["allpages"]]
        cont = r.get("continue", {}).get("apcontinue")
        if not cont:
            break
        time.sleep(0.6)
    return titles[:limit]


def strip_markup(wt):
    prev = None
    while prev != wt:
        prev = wt
        wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)          # 템플릿(날짜·출처박스)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<[^>]+>", " ", wt)
    wt = re.sub(r"\[\[(?:분류|Category):[^\]]*\]\]", " ", wt)
    wt = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", wt)
    wt = re.sub(r"\[\[([^\]]*)\]\]", r"\1", wt)
    wt = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", wt)   # 외부링크 표시텍스트만
    wt = re.sub(r"https?://\S+", " ", wt)
    wt = re.sub(r"'{2,}", "", wt)
    wt = re.sub(r"^[=*#:;].*$", " ", wt, flags=re.M)         # 소제목·목록·출처절
    # 위키뉴스 상투 꼬리(관련 기사/출처/공유) 절 제거
    wt = re.split(r"(관련 기사|출처|이 기사는|공유하기)", wt)[0]
    return wt


def to_lines(text):
    out = []
    for block in re.split(r"\n\s*\n", text):
        line = " ".join(block.split())
        # 날짜 헤더 라인(예: 2020년 4월 1일) 스킵
        if line and re.search(r"[가-힣]", line) and not re.match(
                r"^\d{4}년\s*\d{1,2}월", line):
            out.append(line)
    return out


def main():
    out_dir = sys.argv[1]
    limit = 120
    min_ej = 180
    for a in sys.argv[2:]:
        if a.startswith("--limit="):
            limit = int(a.split("=")[1])
        elif a.startswith("--min-eojeol="):
            min_ej = int(a.split("=")[1])
    import os
    os.makedirs(out_dir, exist_ok=True)

    titles = list_articles(limit * 3)      # 여유있게 목록 (일부는 너무 짧아 탈락)
    saved, i = 0, 0
    for t in titles:
        if saved >= limit:
            break
        if "뉴스브리핑" in t:              # 여러 기사 모음 — 단일 문체 아님, 제외
            continue
        body = None
        for gap in (1.5, 5, 12):           # 429 지수 백오프
            try:
                r = api({"action": "parse", "page": t, "prop": "wikitext",
                         "format": "json", "redirects": 1})
                body = "\n".join(to_lines(strip_markup(r["parse"]["wikitext"]["*"])))
                break
            except Exception as e:
                if "429" in str(e):
                    time.sleep(gap)
                else:
                    print(f"FAIL\t{t}\t{e}")
                    break
        if body is None:
            continue
        n = len(body.split())
        if n < min_ej:
            time.sleep(0.4)
            continue
        fname = re.sub(r"[^가-힣A-Za-z0-9]+", "_", t)[:40]
        with open(f"{out_dir}/news_{i:03d}_{fname}.txt", "w", encoding="utf-8") as f:
            f.write(body)
        saved += 1
        i += 1
        print(f"OK\t{n}어절\t{t[:50]}")
        time.sleep(1.5)
    print(f"\n수집 완료: {saved}편 → {out_dir}")


if __name__ == "__main__":
    main()
