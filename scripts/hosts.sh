# Shared host config; sourced by the sync scripts (run from LM1).
KEY="$HOME/my_ssh/id_ed25519"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=30 -i "$KEY")
UGRAD_USER=cgf2604
UGRAD_HOSTS=(ugradx.cs.jhu.edu ugradv.cs.jhu.edu)
# Everything big lives on each ugrad machine's LOCAL /tmp (not the shared 15G NFS home quota).
# /tmp is purged after 10 days without access -> ugrad_bootstrap.sh installs a keep-alive cron,
# and LM1 stays the source of truth (push_ugrad.sh restores everything idempotently).
UGRAD_ROOT=/tmp/cgf2604
LM2_USER_HOST=jinxinhao@101.6.69.148
LM2_PORT=35167
LM2_ROOT=/data6/jinxinhao/bdd26
LM1_ROOT=/data7/jinxinhao/bdd26
