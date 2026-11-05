#!/usr/bin/env bash
# Cron entry point (weekdays, 3 times each evening): mark the paper portfolios, republish the snapshot.
# Idempotent: daily() only processes trading days after the last mark, so extra runs are harmless and a
# missed day (Mac asleep, DNS down) is caught up on the next successful run with real closes.
set -uo pipefail
ulimit -n 8192
cd "$(dirname "$0")"
# Local, git-ignored settings (SEC_CONTACT_EMAIL for the SEC User-Agent).
[ -f .env ] && { set -a; . ./.env; set +a; }
{
  echo "== $(date)"
  for i in 1 2 3; do
    if .venv/bin/python -W ignore -m src.live.run daily; then break; fi
    echo "daily failed (attempt $i); retrying in 120s"; sleep 120
  done
  .venv/bin/python -W ignore -m src.publish && "$HOME/kalshi-paper/scripts/refresh_sp500_artifacts.sh" --no-publish
} >> data/live/cron.log 2>&1
