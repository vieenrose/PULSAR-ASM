#!/usr/bin/env bash
# Active objective: LFM3 speech-turn realtime (S2S). The S2T harness is .auto/measure_s2t.sh.
exec bash "$(dirname "$0")/lfm3/measure_s2s.sh" "$@"
