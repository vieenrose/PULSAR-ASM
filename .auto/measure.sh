#!/bin/bash
# Objective (2026-11-11): combined conversation RTF = LFM3 S2T + PrimeTTS T2S on the Nano,
# zh and en turns. The harness lives in .auto/combined/measure.sh.
# The previous objective's harness (LFM3 speech-out / S2S realtime) is preserved and still
# runnable as .auto/lfm3/measure_s2s.sh.
exec bash "$(dirname "$0")/combined/measure.sh" "$@"
