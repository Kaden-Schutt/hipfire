#!/usr/bin/env bash
for p in "rollover_v2" "oracle_profile" "agent_exec.sh" "codex exec" "grok -p" "queue_driver.sh" "release/examples/daemon" "hipcc" "stack_verify" "fire_gfx12_ring"; do pkill -9 -f "$p" 2>/dev/null; done
sleep 2
echo "remaining: rollover=$(pgrep -cf rollover_v2) codex=$(pgrep -cf 'codex exec') grok=$(pgrep -cf 'grok -p') daemon=$(pgrep -cf 'release/examples/daemon') driver=$(pgrep -cf queue_driver)"
echo "baseline: $(git -C ~/hipfire/.aw/sw_card0 rev-parse --short HEAD 2>/dev/null); loop_baseline advanced + intact"
