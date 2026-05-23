#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
cd "$ROOT" || exit 1
mkdir -p logs

echo "=== wave13 watcher start $(date -Is) ==="

while pgrep -f '[w]ave10_gpu3_runner.sh' >/dev/null || \
      pgrep -f '[w]ave11_gpu3_runner.sh' >/dev/null || \
      pgrep -f '[w]ave12_gpu3_runner.sh' >/dev/null; do
  sleep 60
done

if pgrep -f '[w]ave13_gpu3_runner.sh' >/dev/null; then
  echo "=== wave13 already running; watcher done $(date -Is) ==="
  exit 0
fi

echo "=== starting wave13 $(date -Is) ==="
nohup bash scripts/wave13_gpu3_runner.sh > logs/wave13_gpu3_launcher.log 2>&1 < /dev/null &
echo $! > logs/wave13_gpu3.pid
echo "=== wave13 launched pid=$(cat logs/wave13_gpu3.pid) $(date -Is) ==="
