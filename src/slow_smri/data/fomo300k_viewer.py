import argparse
import base64
import io
import importlib.resources
from threading import Thread
from queue import Queue

import nibabel as nib
import numpy as np
from flask import Flask, jsonify, render_template_string
from PIL import Image, ImageDraw, ImageFont

from slow_smri.data.fomo300k import Fomo300K

GRID_CELL_WIDTH = 128  # px min-width of grid cells
IMAGE_WIDTH = 224
JPEG_QUALITY = 85
BATCH_SIZE = 16

VIEWS = ("ax", "cor", "sag")

app = Flask(__name__)

_queue: Queue = Queue(maxsize=2048)
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
    img {{ width: 100%; aspect-ratio: 1 / 1; display: block; }}
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
    let done = false;

    function enqueue() {{
      while (!done && queue.length < PREFETCH) {{
        queue.push(fetch('/batch').then(r => r.json()));
      }}
    }}

    const visible = () => sentinel.getBoundingClientRect().top < window.innerHeight;

    async function loadBatch() {{
      if (loading || done) return;
      loading = true;
      while (!done && visible()) {{
        enqueue();
        const images = await queue.shift();
        if (images.length === 0) {{
          done = true;
          observer.disconnect();
          break;
        }}
        for (const b64 of images) {{
          const img = document.createElement('img');
          img.src = 'data:image/jpeg;base64,' + b64;
          grid.appendChild(img);
        }}
      }}
      loading = false;
    }}

    const observer = new IntersectionObserver(entries => {{
      if (entries[0].isIntersecting) loadBatch();
    }});

    observer.observe(sentinel);
  </script>
</body>
</html>"""


def make_image(img: nib.Nifti1Image, view: str = "ax", label: str | None = None) -> Image.Image:
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()

    mask = data > data.mean()
    where = np.argwhere(mask)
    lo_idx, hi_idx = where.min(axis=0), where.max(axis=0) + 1
    data = data[lo_idx[0] : hi_idx[0], lo_idx[1] : hi_idx[1], lo_idx[2] : hi_idx[2]]
    mask = mask[lo_idx[0] : hi_idx[0], lo_idx[1] : hi_idx[1], lo_idx[2] : hi_idx[2]]

    lo, hi = np.percentile(data[mask], [1.0, 99.0])
    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo)
    data = (data * 255).astype(np.uint8)

    sx, sy, sz = img.header.get_zooms()[:3]
    cx, cy = data.shape[0] // 2, data.shape[1] // 2
    cz = int(mask.sum(axis=(0, 1)).argmax())
    planes = {"sag": data[cx, :, :], "cor": data[:, cy, :], "ax": data[:, :, cz]}
    spacings = {"sag": (sy, sz), "cor": (sx, sz), "ax": (sx, sy)}
    font = ImageFont.load_default()

    plane = np.flipud(planes[view].T)
    h, w = plane.shape
    col_sp, row_sp = spacings[view]
    # scale to fit in square, preserving physical aspect ratio
    scale = IMAGE_WIDTH / max(h * row_sp, w * col_sp)
    new_h = max(1, round(h * row_sp * scale))
    new_w = max(1, round(w * col_sp * scale))
    off = ((IMAGE_WIDTH - new_w) // 2, (IMAGE_WIDTH - new_h) // 2)

    tile = Image.fromarray(plane).resize((new_w, new_h), Image.LANCZOS)
    square = Image.new("L", (IMAGE_WIDTH, IMAGE_WIDTH))
    square.paste(tile, off)
    if label:
        ImageDraw.Draw(square).text((3, 2), label, fill=255, font=font)
    return square


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/batch")
def batch():
    images = []
    while len(images) < BATCH_SIZE:
        img: Image.Image | None = _queue.get()
        if img is None:  # sentinel: stream exhausted
            _queue.put(None)  # keep it for subsequent requests
            break
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        images.append(base64.b64encode(buf.getvalue()).decode())
    return jsonify(images)


def main():
    global _view, _queue

    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=VIEWS, default="ax", help="view (default: ax)")
    parser.add_argument("--port", type=int, default=5023)
    args = parser.parse_args()

    _view = args.view

    ds = Fomo300K(
        "hf://datasets/FOMO-MRI/FOMO300K",
        filelist=importlib.resources.files("slow_smri.config").joinpath(
            "fomo300k_full_filelist.txt"
        ),
        max_workers=8,
        shuffle=True,
        random_state=42,
    )

    def fn():
        for name, img in ds:
            suffix = name.removesuffix(".nii.gz").split("_")[-1]
            label = f"{name[:32]} {suffix}"
            square = make_image(img, view=_view, label=label)
            _queue.put(square)
        _queue.put(None)  # sentinel

    thread = Thread(target=fn)
    thread.start()

    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
