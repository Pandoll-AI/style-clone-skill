#!/bin/bash
# 연구 레포의 엔진을 스킬 scripts/로 동기화하고 ~/.claude/skills에 설치.
# 사용: bash skill/sync.sh  (프로젝트 루트에서)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKILL_SRC="$ROOT/skill"
DEST="$HOME/.claude/skills/style-clone"

ENGINE=(stylometry.py features_surface.py features_morph.py discourse.py
        postprocess.py rerank.py loop.py)

mkdir -p "$SKILL_SRC/scripts"
for f in "${ENGINE[@]}"; do
  cp "$ROOT/$f" "$SKILL_SRC/scripts/$f"
done

mkdir -p "$DEST"
rsync -a --delete "$SKILL_SRC/" "$DEST/" --exclude sync.sh

echo "동기화 완료:"
echo "  소스: $SKILL_SRC/scripts/ (${#ENGINE[@]}개 엔진)"
echo "  설치: $DEST"
