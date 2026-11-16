#!/usr/bin/env bash
# Render a knowledge-base repo (markdown folders at its root) with MkDocs Material.
#
#   kb/build.sh serve <content-repo> [mkdocs opts]  live preview with reload
#   kb/build.sh build <content-repo> <out-dir>  static HTML into <out-dir>
#   kb/build.sh check <content-repo>            lint + strict build, nothing written
#
# Markdown standard (enforced by check, and by build when KB_STRICT=1, as in CI):
#   - plain markdown links/images only, no Obsidian [[wikilinks]] / ![[embeds]]
#   - every relative link/image must point to an existing file (mkdocs --strict)
# Site title = first "# " heading of the repo's README.md (falls back to repo name).
# Optional <content-repo>/.kb.yml is appended to the generated config
# (e.g. a custom site_name or nav:). KB_REPO / KB_BRANCH override what is
# otherwise read from the repo's git remote.
set -euo pipefail

KB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="${1:-}"
SRC="${2:-}"
if [[ ! "$CMD" =~ ^(serve|build|check)$ || -z "$SRC" || ( "$CMD" == build && -z "${3:-}" ) ]]; then
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
SRC="$(cd "$SRC" && pwd)"

# Obsidian-style [[...]] links, ignoring fenced code blocks and `inline code`.
# Prints one line per hit; returns 1 when anything was found.
lint_wikilinks() {
  local hits
  hits="$(cd "$SRC" && find . \( -path '*/.*' -o -name venv -o -name site -o -name node_modules \) -prune -o -name '*.md' -print0 \
    | xargs -0 -r awk '
        FNR == 1 { fence = 0 }
        /^[[:space:]]*(```|~~~)/ { fence = !fence; next }
        fence { next }
        {
          line = $0; gsub(/`[^`]*`/, "", line)
          if (match(line, /!?\[\[[^] ][^]]*\]\]/))
            printf "%s:%d: %s\n", substr(FILENAME, 3), FNR, substr(line, RSTART, RLENGTH)
        }')"
  [[ -z "$hits" ]] && return 0
  echo "Obsidian-style links found (use [text](path.md) / ![alt](path.png) instead):" >&2
  while IFS= read -r h; do
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      local f="${h%%:*}" rest="${h#*:}"
      echo "::error file=$f,line=${rest%%:*}::Obsidian-style link ${rest#*: }" >&2
    fi
    echo "  $h" >&2
  done <<< "$hits"
  return 1
}

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

STRICT=()
[[ "$CMD" == check || "${KB_STRICT:-}" == 1 ]] && STRICT=(--strict)
# Lint always reports; it only fails the run in strict mode (after mkdocs has
# also had its say, so one run shows every problem).
LINT_OK=1
lint_wikilinks || LINT_OK=0

case "$CMD" in
  serve) mkdocs serve -f "$WORK/mkdocs.yml" "${@:3}" ;;
  build) mkdocs build "${STRICT[@]}" -f "$WORK/mkdocs.yml" -d "$(mkdir -p "$3" && cd "$3" && pwd)"
         [[ $LINT_OK == 1 || ${#STRICT[@]} == 0 ]] || exit 1 ;;
  check) rc=0
         out="$(mkdocs build --strict -f "$WORK/mkdocs.yml" -d "$WORK/site" 2>&1)" || rc=$?
         grep -E 'WARNING|ERROR|Aborted' <<< "$out" || true
         [[ $LINT_OK == 0 && $rc == 0 ]] && rc=1
         [[ $rc == 0 ]] && echo "OK: $NAME follows the markdown standard"
         exit "$rc" ;;
esac
