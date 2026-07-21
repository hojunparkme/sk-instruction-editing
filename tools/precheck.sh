#!/usr/bin/env bash
# Sanity checks to run before the first push.
set -u
cd "$(dirname "$0")/.."
fail=0

echo "1. absolute paths / usernames"
if grep -rn '/workspace' src/ results/ data/ figures/ analysis/ config.py 2>/dev/null; then
  echo "   FAIL - remove these before pushing"; fail=1
else
  echo "   ok"
fi

echo "2. large files (>5 MB)"
big=$(find . -path ./.git -prune -o -type f -size +5M -print 2>/dev/null)
if [ -n "$big" ]; then echo "$big" | sed 's/^/   /'; echo "   check these are meant to be public"; else echo "   ok"; fi

echo "3. images or weights that should not be committed"
if find . -path ./.git -prune -o \( -name '*.png' -o -name '*.jpg' -o -name '*.pt' -o -name '*.safetensors' \) -print 2>/dev/null | grep -q .; then
  find . -path ./.git -prune -o \( -name '*.png' -o -name '*.jpg' -o -name '*.pt' -o -name '*.safetensors' \) -print | sed 's/^/   /'
  echo "   .gitignore should already exclude these"
else
  echo "   ok"
fi

echo "4. placeholders still to fill"
grep -rn '<AUTHORS>\|TODO' LICENSE README.md 2>/dev/null | sed 's/^/   /' || echo "   ok"

echo "5. results reproduce the paper"
python analysis/statistics.py results/flux_results.json results/ip2p_results.json 2>/dev/null \
  | grep -E 'SK\+LLM vs LLM-only' | head -1 | sed 's/^/   /' \
  || { echo "   FAIL - statistics.py did not run"; fail=1; }

echo
[ "$fail" -eq 0 ] && echo "ready to push" || echo "fix the FAIL items first"
