#!/usr/bin/env bash
# One-shot progress of HoVer-Net runs on LM1 (local) and LM2 + GPU usage.   Usage: scripts/status.sh
cd "$(dirname "$0")/.."; source scripts/hosts.sh
summ() { for f in "$@"; do [ -f "$f" ] || continue; printf '%-28s ' "$(basename "$f")"
  grep '"epoch"' "$f" | tail -1 | python3 -c 'import sys,json
l=sys.stdin.read().strip()
if l: d=json.loads(l); print("phase",d["phase"],"epoch",d["epoch"],"loss %.3f"%d["train/loss"],"val %.3f"%d.get("val/loss",float("nan")),"%.0fs/ep"%d["time"])
else: print("starting")'
  grep -E "^mPQ|Error|Traceback" "$f" | tail -2; done; }
echo "== LM1"; summ logs/hovernet_split*.log
echo "== LM2"; ssh "${SSH_OPTS[@]}" -p $LM2_PORT $LM2_USER_HOST "cd $LM2_ROOT && $(declare -f summ); summ logs/hovernet_split*.log"
