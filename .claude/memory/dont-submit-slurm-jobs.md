---
name: dont-submit-slurm-jobs
description: Do not submit jobs to the GPU queue — the cluster is shared and the queue is Connor's to manage.
metadata:
  type: feedback
  observed: 2026-07-30
---

*"pls dont submit for gpu node btw"* (2026-07-30).

**Why:** the cluster is shared, and queue position is his call, not mine.
**How to apply:** verify on CPU, and when a device is genuinely required use the interactive
allocation protocol in the `gpu-session` skill rather than submitting. If work needs a queued job,
hand him the script and ask.

Extends past the queue (2026-08-12). I had backgrounded a wait-for-the-job-then-build wrapper on
the login node — no queue involved — and he cut it with *"can you just print the command and ill
run it from screen"*. Salloc/srun inside a work chunk stayed fine all session; what he takes over
is anything **long-lived and unattended**. Default to handing over the command for those.
