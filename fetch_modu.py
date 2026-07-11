#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""국립국어원 모두의 말뭉치 OpenAPI 다운로더.

인증키(keyVal, 32자리)는 사용자가 언어정보나눔터에서 개별 신청·승인 후 발급받는다
(봇이 대신할 수 없는 단계). 키만 확보되면 다운로드는 자동:

  키 → API 호출 → 다운로드 URL 획득 → 말뭉치 파일 저장

인증키 우선순위 (Local Secret Rule):
  1) 환경변수 KLI_API_KEY
  2) 프로젝트 .env.local 의 KLI_API_KEY=... (untracked)
키 값은 로그·출력·에러 어디에도 찍지 않는다.

사용: python3 fetch_modu.py [--out=corpus_modu/newspaper.zip]
"""
import json
import os
import pathlib
import sys
import urllib.request

API = "https://kli.korean.go.kr/restapi/v1/corpus/download"


def load_key():
    key = os.environ.get("KLI_API_KEY")
    if key:
        return key.strip()
    envf = pathlib.Path(__file__).parent / ".env.local"
    if envf.exists():
        for line in envf.read_text(encoding="utf-8").splitlines():
            if line.startswith("KLI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def main():
    out = "corpus_modu/download.bin"
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out = a.split("=", 1)[1]
    key = load_key()
    if not key or len(key) < 16:
        print("인증키 없음. KLI_API_KEY를 환경변수 또는 .env.local에 설정하라.",
              file=sys.stderr)
        print("(언어정보나눔터 회원가입 → 말뭉치 신청 → 승인 → 32자리 키 발급)",
              file=sys.stderr)
        sys.exit(2)

    # 1) API 호출 → 다운로드 URL (응답 형식이 문서에 없어 유연 파싱)
    url = f"{API}?keyVal={key}"
    try:
        raw = urllib.request.urlopen(url, timeout=60).read()
    except Exception as e:
        print(f"API 호출 실패: {e}", file=sys.stderr)   # 키 값은 e에 안 실림
        sys.exit(1)
    dl_url = None
    try:
        j = json.loads(raw)
        # 흔한 키 후보들을 순회
        for k in ("download_url", "url", "downloadUrl", "fileUrl", "data"):
            v = j.get(k) if isinstance(j, dict) else None
            if isinstance(v, str) and v.startswith("http"):
                dl_url = v
                break
        if dl_url is None:
            print("응답 JSON에서 다운로드 URL을 못 찾음. 원문 구조:",
                  list(j) if isinstance(j, dict) else type(j), file=sys.stderr)
            print(raw[:400].decode("utf-8", "replace"), file=sys.stderr)
            sys.exit(1)
    except json.JSONDecodeError:
        # JSON이 아니면 응답 자체가 파일일 수 있음 → 그대로 저장
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_bytes(raw)
        print(f"파일 저장(비 JSON 응답): {out} ({len(raw)} bytes)")
        return

    # 2) 다운로드 URL에서 파일 받기
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(dl_url, out)
    print(f"다운로드 완료: {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
