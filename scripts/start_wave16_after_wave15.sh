#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
cd "$ROOT" || exit 1
mkdir -p logs

echo "=== wave16 watcher start $(date -Is) ==="

while pgrep -f '[w]ave15_gpu3_runner.sh' >/dev/null; do
  sleep 60
done

if pgrep -f '[w]ave16_gpu3_runner.sh' >/dev/null; then
  echo "=== wave16 already running; watcher done $(date -Is) ==="
  exit 0
fi

echo "=== starting wave16 $(date -Is) ==="
nohup bash scripts/wave16_gpu3_runner.sh > logs/wave16_gpu3_launcher.log 2>&1 < /dev/null &
echo $! > logs/wave16_gpu3.pid
echo "=== wave16 launched pid=$(cat logs/wave16_gpu3.pid) $(date -Is) ==="
