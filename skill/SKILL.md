---
name: style-clone
description: 문체 지문(stylometric fingerprint) 기반 한국어 문체 복제·진단 스킬. 대상 작가의 코퍼스에서 측정 가능한 지문(종결어미·연결어미·조사 분포, 문장 리듬, 담화 구조, 산포)을 추출하고, 그 지문을 조건으로 best-of-N 생성→문단 재순위화→교정 루프를 돌려 글을 산출(clone)하거나, 기존 글의 문체 이탈을 정량 진단(audit)한다. 트리거 — "문체 복제", "스타일 클론", "이 문체로 써줘", "OO 작가 문체로", "내 문체로 생성", "문체 진단", "문체 점수", "작가 판별", "style clone", "문체 감사". 원문 통째 모사("이 글처럼 써줘"에 원문만 던지는 방식)보다 블라인드 심판 2:0으로 우세함이 실증된 파이프라인.
---

# style-clone — 문체 지문 복제·진단

핵심 원리: 문체를 서술이 아니라 **측정과 재현의 대상**으로 다룬다.
- **z (수용 판정)**: 작가 자신의 청크 분산으로 정규화한 이탈도. z ≤ 1 이면 작가의 정상 변동 범위.
- **d (절대 거리)**: 재순위화·판별용. **z와 d를 섞지 말 것.**
- **캐리커처 가드**: d가 작가 본인 held-out 하한보다 낮으면 "작가보다 더 작가스러운 위조" — 정지.
- **산포 비대칭 벌점**: LLM은 체계적으로 평탄화한다(문헌 실증). 산포 부족은 α=1.5배 감점.

모든 스크립트는 이 스킬의 `scripts/`에 있다. `SKILL_DIR=~/.claude/skills/style-clone` 기준으로 서술.

## 요구 사항

- Python 3.10+. `kiwipiepy` 있으면 형태소 정밀 모드(`--analyzer=kiwi`), 없으면 표층 폴백(`surface`, 기본값) — 자동 감지: `python3 -c "import kiwipiepy" 2>/dev/null && echo kiwi || echo surface`
- 종결어미 변환(postprocess.py)은 kiwipiepy **필수**. 없으면 이 단계를 건너뛰고 사용자에게 알린다.
- 코퍼스 최소량: **60문장(≈지문 청크 4개)**. 미만이면 지문 분산 추정이 무의미 — 진행하되 결과에 저신뢰 경고를 명시.

## 작업 디렉토리

산출물(지문 JSON, 후보, 리포트)은 사용자 프로젝트나 세션 스크래치 디렉토리에 만든다. 스킬 디렉토리에 쓰지 않는다.

## Clone 모드 (코퍼스 + 주제 → 글)

### 1. 코퍼스 분할 (캐리커처 가드용)
코퍼스를 지문용 75% / 기준선용 25%로 나눈다 (작품 단위 분할 권장 — 한 작품을 통째로 기준선으로).
한 파일뿐이면 뒤쪽 25% 문단을 잘라 `baseline.txt`로.

### 2. 지문 추출
```bash
python3 $SKILL_DIR/scripts/stylometry.py build 지문용코퍼스.txt <이름> [대조코퍼스.txt] [--analyzer=kiwi]
# → fingerprints/<이름>.json (실행 위치 기준 fingerprints/ 필요: mkdir -p fingerprints)
```
대조 코퍼스(다른 필자의 글)가 있으면 금지 목록이 생성된다 — 있는 것이 훨씬 강력.

### 3. 지문 읽고 생성 조건 구성
지문 JSON에서 다음을 읽어 생성 지시문으로 번역한다:
- `cat.endings.vocab`+`mean_dist` 상위 5개 → "종결어미 비율: -이다 17%, -한다 14%, ..."
- `cat.connectives` 상위 → "연결어미는 사실상 '고'와 '지만'만"
- `scalars.sent_len_mean/std` → "평균 N어절, 변동폭 ±M"
- `ban_list` → "절대 금지: 는데, 면서, 니까, ..."
- `example_bank` → 실제 예문 4~6개를 스타일 앵커로 제시
- `discourse.scalars` 중 특징적인 것 (A_implicit_ratio 높음 → "접속사 없이 이어라", E_echo_head_tail 높음 → "수미상관") 
- **산포 지시 필수**: "문장 길이 산포(CV) ~X를 유지하라 — 균질한 문장 금지" (평탄화 방지)

### 4. best-of-N 생성 (N ≥ 4)
**같은 개요·같은 문단 수**로 후보 N개를 직접 생성한다 (문단 구분은 빈 줄).
주의: 원문 문장의 재사용 금지 — 재현할 것은 표현이 아니라 **습관의 분포**다.

### 5. 문단 재순위화 + 3중 검증
```bash
python3 $SKILL_DIR/scripts/rerank.py fingerprints/<이름>.json composite.txt cand1.txt cand2.txt ... --baseline=baseline.txt
```
출력 verdict: `z_accept`(수용) / `flatness_ok`(평탄화) / `ban_ok` / `caricature_guard`.

### 6. 교정 루프 (최대 2회전 — v1 실증 수렴 속도)
verdict가 revise면:
```bash
python3 $SKILL_DIR/scripts/loop.py fingerprints/<이름>.json composite.txt --baseline=baseline.txt
```
`prescriptions`의 지시(방향 포함)를 그대로 반영해 **재작성**하고 재채점. 
- `verdict: accept` → 종료. `STOP_caricature` → 즉시 정지하고 직전 버전 채택.
- 2회전 후에도 revise면 현재 최선본과 남은 이탈을 함께 보고 (숨기지 않는다).

### 7. (선택) 종결어미 레벨 강제
사용자가 명시적으로 존댓말/반말 레벨을 지정한 경우:
```bash
python3 $SKILL_DIR/scripts/postprocess.py <hapsyo|haera|haeyo> composite.txt > final.txt
```

### 8. 최종 보고
최종 글 + 채점표(z, d, 금지 위반, 평탄화, 가드 상태, low_confidence 여부)를 함께 제시.
**절대 하지 말 것**: 점수가 나쁜데 좋다고 보고. 30문장 미만 채점은 반드시 "저신뢰" 명시.

## Audit 모드 (글 + 코퍼스 → 진단 리포트)

1. 지문 추출 (위 2단계 동일. 이미 지문 JSON이 있으면 재사용)
2. 진단:
```bash
python3 $SKILL_DIR/scripts/loop.py <지문.json> <검사할글.txt> [--baseline=...]
python3 $SKILL_DIR/scripts/stylometry.py score <지문.json> <검사할글.txt>   # 상세 z 진단표
```
3. 리포트 구성: 총평(z/d/verdict) → 층위별 이탈(진단표 z 상위) → 금지 위반 → 평탄화 여부 → 처방 목록.

## 여러 작가 판별이 필요할 때

같은 analyzer로 작가별 지문을 만들고 d(distance) argmin으로 귀속한다.
**analyzer가 다른 지문끼리 d 비교는 무의미** (vocab 공간이 다름). 판별 대상이 30문장 미만이면 신뢰 불가.

## 참고 수치 (연구 실증)

| 지표 | 값 |
|---|---|
| 작가 본인 held-out z | ≈ 0.31 (수용 목표선) |
| 원문 통째 모사 베이스라인 z | 6.35 (수용 영역 밖) |
| 판별 정확도 (30문장 단위, 5인) | 75% (측정층) / 90% (bigram) |
| 교정 루프 수렴 | z 6.35→0.15, 2회전 |

연구 기반 전체(문헌·설계 근거·벤치마크): https://github.com/pandoll-ai/style-clone-skill (docs/01~05, REPORT.md)
