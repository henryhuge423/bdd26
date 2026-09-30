# Source me FROM LM2: direct LM2 -> ugrad connectivity (2026-09-30).
# LM2 has its own dedicated key (~/.my_ssh/id_ed25519_ugrad, authorized on both ugrad
# hosts); the LM1 private key never leaves LM1. Usage identical to scripts/hosts.sh:
#   source scripts/hosts_lm2.sh
#   ssh "${SSH_OPTS[@]}" $UGRAD_USER@$UGRAD_HOSTS[0] 'hostname'
# GOTCHA (cost 40 min on 2026-09-30): never put `pkill -f <pattern>` and the launch line
# for the same script in ONE ssh command — the whole command is one `bash -c` cmdline, so
# the pkill regex matches the unbracketed launch text and kills the session (silent exit
# 255). Bracket every occurrence ('queu[e]') or split clean/launch into two ssh sessions.
KEY=$HOME/.my_ssh/id_ed25519_ugrad
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30 -i "$KEY")
UGRAD_USER=cgf2604
UGRAD_ROOT=/tmp/cgf2604
UGRAD_HOSTS=(ugradx.cs.jhu.edu ugradv.cs.jhu.edu)
