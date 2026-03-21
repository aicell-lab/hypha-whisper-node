#!/bin/bash
# git-commit-with-kimi.sh
# Usage: ./scripts/git-commit-with-kimi.sh "Your commit message"
# Automatically adds Kimi as a co-author

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Your commit message\""
    exit 1
fi

MESSAGE="$1

Co-authored-by: Kimi <kimi@moonshot.cn>"

git commit -m "$MESSAGE"
echo "✅ Committed with Kimi as co-author"
