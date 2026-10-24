#!/usr/bin/env bash
# Download the safetensors + json files of a Hugging Face model repo at a pinned revision.
# Usage: scripts/fetch_hf.sh <org/name> <revision> <dest-dir>
# Gated repos: accept the licence on huggingface.co, then export HF_TOKEN or run `hf auth login`.
# Resumable: re-run the same command after an interruption; complete files are skipped.
set -euo pipefail
repo="$1"; rev="$2"; dest="$3"
token="${HF_TOKEN:-$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)}"
auth=(); [ -n "$token" ] && auth=(-H "Authorization: Bearer $token")
mkdir -p "$dest"
curl -fsSL ${auth[@]+"${auth[@]}"} "https://huggingface.co/api/models/$repo/revision/$rev?blobs=true" \
  | python3 -I -c 'import json,sys
for s in json.load(sys.stdin)["siblings"]:
    if s["rfilename"].endswith((".safetensors", ".json")):
        print(s.get("size", -1), s["rfilename"])' \
  | while read -r size f; do
      have=$( [ -f "$dest/$f" ] && wc -c < "$dest/$f" | tr -d ' ' || echo 0 )
      if [ "$have" = "$size" ]; then echo "== $f (complete)"; continue; fi
      echo "== $f"
      mkdir -p "$dest/$(dirname "$f")"
      curl -fL --retry 5 -C - ${auth[@]+"${auth[@]}"} -o "$dest/$f" "https://huggingface.co/$repo/resolve/$rev/$f" < /dev/null
    done
echo "$repo@$rev" > "$dest/.source"
