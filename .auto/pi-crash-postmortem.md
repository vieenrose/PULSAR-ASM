# Why pi sessions died on training-machine (postmortem, 2026-10-13)

Evidence sources: ~/.pi/agent/sessions/--home-user-PULSAR-ASM--/*.jsonl,
~/.config/herdr/herdr-{server,client}.log, journalctl (kernel + systemd), ps.

## Cause A — SSH-scoped teardown (Oct 4 04:06, Oct 6 12:16, Oct 13 ~03:33)
herdr spawns pi inside a `tailscaled be-child ssh --force-v1-behavior` login session, so pi
lives in `user.slice/user-1000.slice/session-NNNN.scope` (verified: current pi pid 3816409 ->
`0::/user.slice/user-1000.slice/session-2697.scope`). When the SSH connection dies, systemd
reaps the scope and pi dies with it. Decoded last record of the Oct 6 session: a phone-monitor
poll loop (`adb ... seq 1 90`, TOTAL PSS sampling of studio.voxsum.androidtest) started from
luigi-inspiron135330 at 02:47:24, gone at 03:33:05 (45m40s wall):

  03:33:05 systemd[1]: session-c837.scope: Deactivated successfully.
  03:33:05 systemd[1]: session-c837.scope: Consumed 2.656s CPU over 45min40.646s wall, 6.2M peak.
  03:33:05 systemd-logind[1782]: Session c837 logged out.

User's last message: 03:32:55 -> dead 10s later. Oct 13 same shape (laptop conn 02:47:24;
session-2697 predates the 30d journal window = long-lived conn, dies silently, takes every pi
inside it). logind `#KillUserProcesses=no` is the DEFAULT, so scopes are killed on logout.

## Cause B — torn session-file write (Oct 6, ~09:11 UTC)
Record 13329 of the 56.8 MB session jsonl has two records interleaved mid-JSON (valid prefix,
then a second record's bytes spliced in) = interleave during an abrupt process death, not ENOSPC.
Assistant thinking in that region: "Analyzing disk space failure". Box OOM-killed a 9.2 GB python3
Oct 5 05:37 (tailscaled/herdr/thermald invoked the oom-killer).

## Cause C — context exhaustion (not a crash)
Peak single-turn input on the Oct 6 session = 1,004,436 tokens; one literal `prompt is too long`
at 2026-10-13T00:40:12. Session ends cleanly at rec 15907 with the user's "monitor also peak rss"
= hit the window, stopped.

## Ruled out
- No node/pi OOM kill, no pi segfault in 30 days of journal.
- `cicc` segfaulted twice (Oct 12 08:46, 09:03) = CUDA/LLVM compiler bug, unrelated.
- `traps: herdr[2385] general protection fault` at Oct 13 04:32:59 = death of the PREVIOUS
  multiplexer (pid 2385 belongs to no attributable pi session); server restarted 04:33:05,
  pane pid 3816400/3816409 = this session.

## Countermeasure (not yet applied — needs user sign-off)
Run long loops in a scope loginctl will not reap, decoupled from any SSH session:
  setsid nohup <cmd> >log 2>&1 &        # minimal: detach from the controlling terminal
  systemd-run --user --scope -p KillUserProcesses=no ...   # explicit scope
Current autoresearch runs via bg_run (child of this pi) => same exposure as Cause A.
