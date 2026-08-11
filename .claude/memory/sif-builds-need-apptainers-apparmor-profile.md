---
name: sif-builds-need-apptainers-apparmor-profile
description: Unprivileged .sif builds work here only via apptainer's AppArmor profile; plain user namespaces are blocked, and only some hosts have the profile.
metadata:
  type: reference
  observed: 2026-08-11
---

`unshare -r` fails on every host (`kernel.apparmor_restrict_unprivileged_userns=1`), so nothing
gets a user namespace on its own. Apptainer's deb ships `/etc/apparmor.d/apptainer`, which grants
its binary the exception — that profile, not the sysctl, is what makes `apptainer build --fakeroot`
work. Installing apptainer is therefore the whole fix; no sysctl change is needed.

The pre-existing `singularity-ce 4.1.1` is not a substitute. It could `build` from a `docker://`
source and run the result, but any def file with a `%post` died on `--fakeroot` with "Failed to
create mount namespace". Before apptainer was installed, the working route was **docker → SIF**:
`docker build`, then `singularity build out.sif docker-daemon://tag:latest`, both verified. Worth
remembering if this ever has to run somewhere without the profile, since docker needs no userns.

Host split, which is easy to lose an hour to: **the login node has apptainer but no GPU driver**
(no `/dev/nvidia*`), and **apptainer reached only `n-6`**, not the other compute nodes — the
challenge validator fails preflight on `n-3` with "apptainer not found". So building happens on the
login node and running happens under `salloc --nodelist=n-6`. A driver-less host does not merely
fall back to CPU: see [[nested-tensor-sdpa-needs-a-device]], which is what actually kills the
forward pass there.
