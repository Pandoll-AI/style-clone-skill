#!/usr/bin/env python3
"""한국 위키문헌에서 저작권 만료 작품을 수집해 corpus/<작가>/<작품>.txt로 저장.

- API: action=parse&prop=wikitext (개인 연구 목적, 요청 간 1초 간격)
- 정제: 위키 마크업 제거 → 문단당 한 줄 평문 (기존 corpus 형식과 동일)
- 안전장치: 동음이의 문서·비정상적으로 짧은 본문은 저장하지 않고 보고
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "styleclone-research/0.1 (personal research; contact: local)"}
API = "https://ko.wikisource.org/w/api.php?"

# 작가 → [(위키문헌 페이지명, 저장 파일명)]
WORKS = {
    "kimyj": [("동백꽃", "동백꽃"), ("금 따는 콩밭", "금따는콩밭"),
              ("소낙비", "소낙비"), ("땡볕", "땡볕"), ("따라지", "따라지")],
    "yisang": [("봉별기", "봉별기"), ("종생기", "종생기"), ("지주회시", "지주회시")],
    "hyunjg": [("운수 좋은 날", "운수좋은날"), ("빈처", "빈처"),
               ("술 권하는 사회", "술권하는사회")],
    "chaems": [("레디메이드 인생", "레디메이드인생"), ("치숙", "치숙"),
               ("미스터 방", "미스터방")],
    "leehs": [("메밀꽃 필 무렵", "메밀꽃필무렵"), ("수탉", "수탉"),
              ("산 (이효석)", "산")],
}


def api(params):
    url = API + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=30))


def fetch_wikitext(title):
    r = api({"action": "parse", "page": title, "prop": "wikitext",
             "format": "json", "redirects": 1})
    return r["parse"]["wikitext"]["*"]


def strip_markup(wt):
    # 템플릿 {{...}} — 중첩 대응 반복 제거
    prev = None
    while prev != wt:
        prev = wt
        wt = re.sub(r"\{\{[^{}]*\}\}", " ", wt)
    wt = re.sub(r"<!--.*?-->", " ", wt, flags=re.S)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", " ", wt, flags=re.S)
    wt = re.sub(r"<[^>]+>", " ", wt)                       # 나머지 HTML 태그
    wt = re.sub(r"\[\[(?:분류|Category):[^\]]*\]\]", " ", wt)
    wt = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", wt)   # [[a|b]] → b
    wt = re.sub(r"\[\[([^\]]*)\]\]", r"\1", wt)            # [[a]] → a
    wt = re.sub(r"'{2,}", "", wt)                          # 굵게/기울임
    wt = re.sub(r"^=+.*?=+\s*$", " ", wt, flags=re.M)      # == 소제목 ==
    wt = re.sub(r"^[*#:;]+\s*", "", wt, flags=re.M)        # 목록 마커
    wt = re.sub(r"__[A-Z]+__", " ", wt)
    return wt


def to_paragraph_lines(text):
    paras = []
    for block in re.split(r"\n\s*\n", text):
        line = " ".join(block.split())
        if line and re.search(r"[가-힣]", line):
            paras.append(line)
    return paras


def hangul_count(s):
    return sum("가" <= c <= "힣" for c in s)


def main():
    import os
    report = []
    delay_idx = 0
    gaps = [1, 3, 5, 10]
    for author, works in WORKS.items():
        os.makedirs(f"corpus/{author}", exist_ok=True)
        for title, fname in works:
            try:
                wt = fetch_wikitext(title)
            except Exception as e:
                report.append((author, title, "FETCH_FAIL", str(e)))
                delay_idx = min(delay_idx + 1, len(gaps) - 1)
                time.sleep(gaps[delay_idx])
                continue
            if "동음이의" in wt or "{{disambiguation" in wt.lower():
                report.append((author, title, "DISAMBIG", "동음이의 문서"))
                time.sleep(1)
                continue
            paras = to_paragraph_lines(strip_markup(wt))
            body = "\n".join(paras)
            n_h = hangul_count(body)
            if n_h < 2000:
                report.append((author, title, "TOO_SHORT", f"hangul={n_h}"))
                time.sleep(1)
                continue
            out = f"corpus/{author}/{fname}.txt"
            with open(out, "w", encoding="utf-8") as f:
                f.write(body)
            report.append((author, title, "OK",
                           f"paras={len(paras)} eojeol={len(body.split())}"))
            time.sleep(1)
    for r in report:
        print("\t".join(r))


if __name__ == "__main__":
    main()
