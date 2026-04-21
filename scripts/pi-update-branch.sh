#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Beacon-Low-Training-Pi-Camera-System}"
BRANCH="${1:-}"

if [[ -z "$BRANCH" ]]; then
  echo "Usage: $0 <branch-name>"
  exit 2
fi

cd "$REPO_DIR"

git fetch --prune origin

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH" "origin/$BRANCH"
fi

git pull --ff-only origin "$BRANCH"

echo "Updated $REPO_DIR to branch $BRANCH"
echo "Recommended next steps:"
echo "  python3 -m inspection_system.app.capture_test pilot-readiness"
echo "  python3 -m inspection_system.app.capture_test quick-check"