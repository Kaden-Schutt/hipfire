#!/usr/bin/env bash
set -uo pipefail
cd /home/kaden/hipfire-ds4-gfx1151-opt
source scripts/gpu-lock.sh
gpu_acquire "ds4-clk-ab" || exit 9
export HIP_VISIBLE_DEVICES=1
C=/sys/class/drm/card1/device     # gfx1151, 103.1 GB, pci bf:00.0
BIN=./target/release/examples/bench_e8_soa_correctness

cur() { grep '\*' $C/pp_dpm_$1 2>/dev/null | grep -oP '\d+(?=Mhz)' | head -1; }

run_arm() {
  echo "  level=$(cat $C/power_dpm_force_performance_level) idle: sclk=$(cur sclk) fclk=$(cur fclk) mclk=$(cur mclk)"
  echo "0 0 0" > /tmp/clkpeak
  ( ms=0; mf=0; mm=0
    while :; do
      s=$(cur sclk); f=$(cur fclk); m=$(cur mclk)
      [ -n "${s:-}" ] && [ "$s" -gt "$ms" ] && ms=$s
      [ -n "${f:-}" ] && [ "$f" -gt "$mf" ] && mf=$f
      [ -n "${m:-}" ] && [ "$m" -gt "$mm" ] && mm=$m
      echo "$ms $mf $mm" > /tmp/clkpeak; sleep 0.5
    done ) & local sp=$!
  $BIN 2>&1 | grep -E "WAVESWEEP|GB/s" | head -14
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  echo "  PEAK under load: sclk/fclk/mclk = $(cat /tmp/clkpeak) MHz"
}

echo "### ARM 1: auto"
echo auto | sudo tee $C/power_dpm_force_performance_level >/dev/null; run_arm
echo "### ARM 2: high"
echo high | sudo tee $C/power_dpm_force_performance_level >/dev/null; run_arm
echo "### ARM 3: manual, max sclk+fclk+mclk"
echo manual | sudo tee $C/power_dpm_force_performance_level >/dev/null
echo 2 | sudo tee $C/pp_dpm_sclk >/dev/null 2>&1 || echo "  (sclk force rejected)"
echo 5 | sudo tee $C/pp_dpm_fclk >/dev/null 2>&1 || echo "  (fclk force rejected)"
echo 2 | sudo tee $C/pp_dpm_mclk >/dev/null 2>&1 || echo "  (mclk force rejected)"
run_arm
echo "### restore"
echo auto | sudo tee $C/power_dpm_force_performance_level >/dev/null
echo "  level=$(cat $C/power_dpm_force_performance_level)"
gpu_release
echo "DONE"
