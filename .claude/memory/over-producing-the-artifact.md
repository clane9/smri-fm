---
name: over-producing-the-artifact
description: Connor's most repeated correction — I build more artifact than the job needs, and cutting it in one place tends to relocate it to another.
metadata:
  type: feedback
  observed: 2026-08-11
---

Four corrections in one week (2026-07-29..30), all the same defect: over-producing the artifact.
Comments and docstrings, then commit bodies, then a plot built to a full design spec, then restyling
a draft he handed over.

The pattern worth remembering is the *relocation*: when comments were cut, the material reappeared
in five-paragraph commit bodies. Removing the output in one place did not remove the impulse.

Recurred verbatim on 2026-08-11: he trimmed the comments out of `fomo_tune/build.py`, and the
commit for that same work went out with a five-paragraph body he then asked to cut to one. Every
paragraph of it was already in `src/fomo_tune/README.md`. The tell is that the material has a home
already — if the README or the code says it, the commit body repeating it is the relocation, not
context.

Recurred on 2026-08-12 as over-*verification* rather than over-writing: after a three-line addition
of a timing field across three files, I proposed exercising the new record and log formatting, and
he cut it with *"should be no need to exercise, it's very minor"*. Same impulse pointed at process
instead of prose — the size of the check should track the risk of the change, and adding a field to
a dict carries almost none. Note this is a calibration, not a licence: the container verification
in the same session was elaborate and he asked for *more* of it, because three leaderboard
submissions were riding on it.

**Why:** he wants the minimum artifact that does the job — extra explanation is not free, it is
something he has to read and maintain.
**How to apply:** default to the smaller version and let him ask for more. Specific forms:
[[no-explanatory-comments]], [[plain-matplotlib-for-internal-plots]], and minimum-diff on his drafts
(covered in the `add-eval-model` skill). Commit messages: concise, per `CODING_STANDARDS.md`.
