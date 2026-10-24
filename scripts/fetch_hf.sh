#!/usr/bin/env bash
# Download the safetensors + json files of a Hugging Face model repo at a pinned revision.
# Usage: scripts/fetch_hf.sh <org/name> <revision> <dest-dir>
# Gated repos: accept the licence on huggingface.co, then export HF_TOKEN or run `hf auth login`.
# Resumable: re-run the same command after an interruption; complete files are skipped.
# The destination must be empty or already hold this exact <org/name>@<revision>: file sizes alone
# cannot tell revisions apart. Pin a commit hash, not a branch name, so the label stays true.
# FETCH_INCLUDE: regex of files to fetch (default: all .safetensors and .json files).
set -euo pipefail
repo="$1"; rev="$2"; dest="$3"
token="${HF_TOKEN:-$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
auth=(); [ -n "$token" ] && auth=(-H "Authorization: Bearer $token")
want="$repo@$rev"
for marker in .source .source.partial; do
  if [ -f "$dest/$marker" ] && [ "$(cat "$dest/$marker")" != "$want" ]; then
    echo "error: $dest already holds $(cat "$dest/$marker"), not $want; use an empty directory" >&2; exit 1
  fi
done
if [ ! -f "$dest/.source" ] && [ ! -f "$dest/.source.partial" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
  echo "error: $dest is not empty and was not fetched by this script; use an empty directory" >&2; exit 1
fi
mkdir -p "$dest"
echo "$want" > "$dest/.source.partial"
curl -fsSL ${auth[@]+"${auth[@]}"} "https://huggingface.co/api/models/$repo/revision/$rev?blobs=true" \
  | INCLUDE="${FETCH_INCLUDE:-\.(safetensors|json)\$}" python3 -I -c 'import json, os, re, sys
for s in json.load(sys.stdin)["siblings"]:
    if re.search(os.environ["INCLUDE"], s["rfilename"]):
        print(s.get("size", -1), s["rfilename"])' \
  | while read -r size f; do
      have=$( [ -f "$dest/$f" ] && wc -c < "$dest/$f" | tr -d ' ' || echo 0 )
      if [ "$have" = "$size" ]; then echo "== $f (complete)"; continue; fi
      echo "== $f"
      mkdir -p "$dest/$(dirname "$f")"
      curl -fL --retry 5 -C - ${auth[@]+"${auth[@]}"} -o "$dest/$f" "https://huggingface.co/$repo/resolve/$rev/$f" < /dev/null
    done
mv "$dest/.source.partial" "$dest/.source"
