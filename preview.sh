#!/usr/bin/env bash
# Local preview of the whole https://kubow.github.io/ :
#   /         profile page, regenerated from data/ + source/templates/
#   /<repo>/  every projects.yaml entry with  site: https://kubow.github.io/<repo>/
#             built with kb/build.sh from a checkout found via KB_SEARCH
#             (default: ../<repo>, then ../../Space/<repo>)
# Links to https://kubow.github.io/ are rewritten to local paths, so the
# profile and the knowledge bases link to each other. Nothing in the repos
# is modified; output goes to a temp dir. Re-run after changes.
#
#   ./preview.sh [port]      (default 8000)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8000}"
HOST="kubow.github.io"

# One venv for both the profile generator and MkDocs.
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/source/requirements.txt" -r "$ROOT/kb/requirements.txt"
fi
PATH="$ROOT/.venv/bin:$PATH"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "== profile page"
cp -r "$ROOT/docs/." "$OUT/"
if ! log="$(python "$ROOT/source/build.py" --out "$OUT" 2>&1)"; then
  echo "$log"
  exit 1
fi
rm -f "$OUT/README.md"

repos="$(python - "$ROOT/data/projects.yaml" "$HOST" <<'EOF'
import re, sys, yaml
host = re.escape(sys.argv[2])
for group in yaml.safe_load(open(sys.argv[1])):
    for p in group.get("sub_projects", []) + group.get("items", []):
        m = re.fullmatch(rf"https://{host}/([^/]+)/?", p.get("site", ""))
        if m:
            print(m.group(1))
EOF
)"
# Where to look for checkouts (colon-separated); first match wins.
KB_SEARCH="${KB_SEARCH:-$ROOT/..:$ROOT/../../Space}"
for repo in $repos; do
  src=""
  IFS=: read -ra dirs <<< "$KB_SEARCH"
  for d in "${dirs[@]}"; do
    if [[ -d "$d/$repo" ]]; then src="$(cd "$d/$repo" && pwd)"; break; fi
  done
  if [[ -n "$src" ]]; then
    echo "== $repo ($src)"
    "$ROOT/kb/build.sh" build "$src" "$OUT/$repo" 2>&1 | grep -E 'WARNING|ERROR|Doc file' || true
  else
    echo "== $repo: no checkout in $KB_SEARCH, link will 404"
  fi
done

grep -rlF --include='*.html' "https://$HOST/" "$OUT" | xargs -r sed -i "s#https://$HOST/#/#g"

echo
echo "Serving on http://127.0.0.1:$PORT/  (Ctrl+C to stop)"
python -m http.server --bind 127.0.0.1 --directory "$OUT" "$PORT" 2> /dev/null
