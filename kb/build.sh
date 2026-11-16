#!/usr/bin/env bash
# Render a knowledge-base repo (markdown folders at its root) with MkDocs Material.
#
#   kb/build.sh serve <content-repo> [mkdocs opts]  live preview with reload
#   kb/build.sh build <content-repo> <out-dir>  static HTML into <out-dir>
#
# Site title = first "# " heading of the repo's README.md (falls back to repo name).
# Optional <content-repo>/.kb.yml is appended to the generated config
# (e.g. a custom site_name or nav:). KB_REPO / KB_BRANCH override what is
# otherwise read from the repo's git remote.
set -euo pipefail

KB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-}"
SRC="${2:-}"
if [[ ! "$CMD" =~ ^(serve|build)$ || -z "$SRC" || ( "$CMD" == build && -z "${3:-}" ) ]]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
SRC="$(cd "$SRC" && pwd)"

# Use mkdocs from PATH (CI), otherwise a private venv next to this script.
if ! command -v mkdocs >/dev/null; then
  if [[ ! -x "$KB_DIR/.venv/bin/mkdocs" ]]; then
    python3 -m venv "$KB_DIR/.venv"
    "$KB_DIR/.venv/bin/pip" install -q -r "$KB_DIR/requirements.txt"
  fi
  PATH="$KB_DIR/.venv/bin:$PATH"
fi

REPO="${KB_REPO:-$(git -C "$SRC" remote get-url origin | sed -E 's#^.*github\.com[:/]##; s#\.git$##')}"
OWNER="${REPO%%/*}"
NAME="${REPO#*/}"
BRANCH="${KB_BRANCH:-$(git -C "$SRC" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)}"
BRANCH="${BRANCH:-main}"
TITLE="$(sed -n 's/^# \{1,\}//p' "$SRC/README.md" 2>/dev/null | head -n1 | sed 's/[[:space:]]*$//')"
TITLE="${TITLE:-$NAME}"

# MkDocs refuses docs_dir == config dir, so the config lives in a temp dir
# and docs_dir is a symlink to the repo root.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ln -s "$SRC" "$WORK/docs"

cat > "$WORK/mkdocs.yml" <<EOF
INHERIT: $KB_DIR/mkdocs.base.yml
site_name: "${TITLE//\"/\\\"}"
site_url: https://${OWNER,,}.github.io/$NAME/
repo_url: https://github.com/$REPO
repo_name: $REPO
edit_uri: edit/$BRANCH/
docs_dir: docs
theme:
  custom_dir: $KB_DIR/overrides
EOF
if [[ -f "$SRC/.kb.yml" ]]; then
  cat "$SRC/.kb.yml" >> "$WORK/mkdocs.yml"
fi

case "$CMD" in
  serve) mkdocs serve -f "$WORK/mkdocs.yml" "${@:3}" ;;
  build) mkdocs build -f "$WORK/mkdocs.yml" -d "$(mkdir -p "$3" && cd "$3" && pwd)" ;;
esac
