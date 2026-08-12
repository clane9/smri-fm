---
name: smri-mae-patch-ids-index-the-grid
description: SmriMaeBackbone's patch_ids index the flattened regular patch grid in C order, so a token's exact voxel block is known without any nearest-neighbour matching — and token_mask must filter padded slots first.
metadata:
  type: reference
  observed: 2026-08-12
---

`backbone.forward` returns `patch_ids` alongside `patch_embeds`. They are indices into the
**flattened regular patch grid**, C order over (x, y, z) — `patchify3d` rearranges
`b c (t u) (h p) (w q) -> b (t h w) (c u p q)`, and `SmriMaeBackbone` builds `grid_coords` with
`rearrange(np.indices(grid_size), "c x y z -> (x y z) c")`, which is the same order. At
img_size 208x240x208 and patch 8 the grid is 26x30x26 = 20280.

So token `p` owns exactly the voxel block `[8gx:8gx+8, 8gy:8gy+8, 8gz:8gz+8]` where
`gx, gy, gz = unravel_index(p, (26, 30, 26))`. Labels can be reduced onto tokens with a single
`einops.reduce(..., "(gx px) (gy py) (gz pz) -> (gx gy gz)", "sum")` and predictions expanded back
with the matching `repeat`.

**Two traps.** `patch_embeds` and `patch_ids` are padded, and the padded slots carry meaningless
indices — filter both by `token_mask` before using either. And a patch is kept when
`patch_num_obs > 0`, i.e. it has **any** voxel inside the transform's mask, not a majority; a
dropped token therefore means a block with no in-brain voxel at all, which is a defensible place
to predict zero.

**Why it matters:** `nanobrain.eval.probe_seg` assigns voxels to patches with a `cKDTree` over
`patch_coords` in world mm, and needs a coverage assert to catch a bad affine. That machinery
exists because the probe is generic over backbones whose tokens are not on a regular grid. For
this backbone none of it is needed — the block is exact, so there is no distance to check and no
axis-swap to miss. See [[seg-probe-design]], [[seg-probe-world-coord-guards]].
