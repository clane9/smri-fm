import argparse
import base64
import itertools
import io
import random
import warnings
from pathlib import Path
from typing import Iterator
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from flask import Flask, jsonify, render_template_string
from PIL import Image
from tqdm import tqdm

GRID_CELL_WIDTH = 128  # px min-width of grid cells
IMAGE_WIDTH = 224
JPEG_QUALITY = 85
DEFAULT_BATCH_SIZE = 16

DEFAULT_DATA_DIR = Path("data/FOMO50K")
DEFAULT_IMAGE_CACHE_DIR = Path("data/FOMO50K_images")

app = Flask(__name__)

_cycle: Iterator = None
_batch_size: int = DEFAULT_BATCH_SIZE
_data_dir: Path = None
_image_cache_dir: Path = None
_view: str = "ax"

HTML = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>sMRI viewer</title>
  <style>
    body {{ margin: 0; background: #111; }}
    #grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax({GRID_CELL_WIDTH}px, 1fr));
      gap: 4px;
      padding: 4px;
    }}
    img {{ width: 100%; display: block; }}
    #sentinel {{ height: 1px; }}
  </style>
</head>
<body>
  <div id="grid"></div>
  <div id="sentinel"></div>
  <script>
    const PREFETCH = 3;
    const grid = document.getElementById('grid');
    const sentinel = document.getElementById('sentinel');
    const queue = [];
    let loading = false;

    function enqueue() {{
      while (queue.length < PREFETCH) {{
        queue.push(fetch('/batch').then(r => r.json()));
      }}
    }}

    async function loadBatch() {{
      if (loading) return;
      loading = true;
      enqueue();
      const images = await queue.shift();
      enqueue();
      for (const b64 of images) {{
        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + b64;
        grid.appendChild(img);
      }}
      loading = false;
      if (sentinel.getBoundingClientRect().top < window.innerHeight) {{
        loadBatch();
      }}
    }}

    const observer = new IntersectionObserver(entries => {{
      if (entries[0].isIntersecting) loadBatch();
    }});

    observer.observe(sentinel);
    loadBatch();
  </script>
</body>
</html>"""


def make_images(nifti_path: Path, view: str = "ax") -> Path | None:
    rel = nifti_path.relative_to(_data_dir)
    stem = rel.name.split(".nii")[0]
    out = {v: _image_cache_dir / rel.parent / f"{stem}_{v}.jpg" for v in ("sag", "cor", "ax")}
    if out[view].exists():
        return out[view]

    try:
        img = nib.load(nifti_path)
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()

        if data.ndim == 4:
            data = data[..., data.shape[3] // 2]

        # crop to foreground bounding box
        mask = data > data.mean()
        where = np.argwhere(mask)
        lo_idx, hi_idx = where.min(axis=0), where.max(axis=0) + 1
        data = data[lo_idx[0] : hi_idx[0], lo_idx[1] : hi_idx[1], lo_idx[2] : hi_idx[2]]
        mask = mask[lo_idx[0] : hi_idx[0], lo_idx[1] : hi_idx[1], lo_idx[2] : hi_idx[2]]

        lo, hi = np.percentile(data[mask], [0.5, 99.5])
        data = np.clip(data, lo, hi)
        data = (data - lo) / (hi - lo)
        data = (data * 255).astype(np.uint8)

        sx, sy, sz = img.header.get_zooms()[:3]
        cx, cy = data.shape[0] // 2, data.shape[1] // 2
        cz = int(mask.sum(axis=(0, 1)).argmax())
        planes = {"sag": data[cx, :, :], "cor": data[:, cy, :], "ax": data[:, :, cz]}
        # (col_spacing, row_spacing) for each view after RAS reorientation
        spacings = {"sag": (sy, sz), "cor": (sx, sz), "ax": (sx, sy)}

        out[view].parent.mkdir(parents=True, exist_ok=True)
        for v, plane in planes.items():
            plane = np.flipud(plane.T)
            h, w = plane.shape
            col_sp, row_sp = spacings[v]
            # scale to fit in square, preserving physical aspect ratio
            scale = IMAGE_WIDTH / max(h * row_sp, w * col_sp)
            new_h = max(1, round(h * row_sp * scale))
            new_w = max(1, round(w * col_sp * scale))
            img = Image.fromarray(plane).resize((new_w, new_h), Image.LANCZOS)
            square = Image.new("L", (IMAGE_WIDTH, IMAGE_WIDTH))
            square.paste(img, ((IMAGE_WIDTH - new_w) // 2, (IMAGE_WIDTH - new_h) // 2))
            square.save(out[v], format="JPEG", quality=JPEG_QUALITY)
        return out[view]

    except Exception as exc:
        warnings.warn(f"Skipping {nifti_path.name}: {exc}")
        return None


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/batch")
def batch():
    images = []
    attempts = 0
    while len(images) < _batch_size and attempts < _batch_size * 10:
        attempts += 1
        nifti_path = next(_cycle)
        img_path = make_images(nifti_path, _view)
        if img_path is None:
            continue
        buf = io.BytesIO(img_path.read_bytes())
        images.append(base64.b64encode(buf.getvalue()).decode())
    return jsonify(images)


def main():
    global _cycle, _batch_size, _data_dir, _image_cache_dir, _view

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-cache-dir", type=Path, default=DEFAULT_IMAGE_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--view", choices=["ax", "cor", "sag"], default="ax")
    parser.add_argument("--prefill-cache", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    args.image_cache_dir.mkdir(parents=True, exist_ok=True)
    filelist = args.image_cache_dir / "filelist.txt"
    if filelist.exists():
        volumes = [Path(p) for p in filelist.read_text().splitlines()]
    else:
        volumes = sorted(args.data_dir.glob("**/*.nii.gz"))
        filelist.write_text("\n".join(str(p) for p in volumes))
    if not volumes:
        raise SystemExit(f"No .nii.gz files found under {args.data_dir}")

    if args.shuffle:
        random.Random(args.seed).shuffle(volumes)

    print(f"Found {len(volumes)} volumes")

    _view = args.view
    _data_dir = args.data_dir
    _image_cache_dir = args.image_cache_dir
    _batch_size = args.batch_size
    _cycle = itertools.cycle(volumes)

    if args.prefill_cache:
        with ProcessPoolExecutor(max_workers=8) as pool:
            for _ in tqdm(pool.map(make_images, volumes, chunksize=64), total=len(volumes)):
                pass
        return

    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
