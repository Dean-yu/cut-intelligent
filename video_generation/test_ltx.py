from __future__ import annotations

import base64
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
root = Path.cwd()
run_id = os.environ.get("GITHUB_RUN_ID", "local")
out = root / "ltx_test" / run_id
out.mkdir(parents=True, exist_ok=True)


def locate(value: Any) -> Path | None:
    if isinstance(value, dict):
        for key in ("video", "path", "url", "name"):
            if key in value:
                p = locate(value[key])
                if p:
                    return p
        for item in value.values():
            p = locate(item)
            if p:
                return p
    elif isinstance(value, (list, tuple)):
        for item in value:
            p = locate(item)
            if p:
                return p
    elif isinstance(value, str):
        p = Path(value)
        if p.exists() and p.is_file():
            return p
    return None

try:
    text = "".join((root / "video_generation/inputs/tianyuan.b64").read_text(encoding="utf-8").split())
    text += "=" * ((4 - len(text) % 4) % 4)
    raw = base64.b64decode(text, validate=False)
    source = out / "tianyuan_source.jpg"
    source.write_bytes(raw)
    with Image.open(source) as im:
        im = im.convert("RGB")
        w, h = im.size
        ratio = 16 / 9
        if w / h > ratio:
            nw = int(h * ratio)
            x = (w - nw) // 2
            im = im.crop((x, 0, x + nw, h))
        elif w / h < ratio:
            nh = int(w / ratio)
            y = (h - nh) // 2
            im = im.crop((0, y, w, y + nh))
        im.resize((704, 396), Image.Resampling.LANCZOS).save(source, quality=92)

    client = Client("Lightricks/ltx-video-distilled", verbose=True)
    result = client.predict(
        prompt="Live-action cinematic science-fantasy. A bright cultivation-industrial metropolis comes alive. The camera flies forward between layered skyways and descends into a crowded public plaza. Real human pedestrians walk naturally, robes and coats move in the wind, crystal trains glide overhead, flying craft cross the frame with strong parallax, physically plausible motion, 35mm anamorphic lens, continuous camera movement.",
        negative_prompt="static image, frozen frame, slideshow, simple zoom, motionless people, duplicate people, warped face, deformed hands, melting buildings, flicker, jitter, blur, cartoon, anime, subtitles, logo, watermark",
        input_image_filepath=handle_file(str(source)),
        input_video_filepath=None,
        height_ui=396,
        width_ui=704,
        mode="image-to-video",
        duration_ui=2.0,
        ui_frames_to_use=9,
        seed_ui=71623,
        randomize_seed=False,
        ui_guidance_scale=1.5,
        improve_texture_flag=True,
        api_name="/image_to_video",
    )
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    src = locate(result)
    if src is None:
        raise RuntimeError(f"No output video found in result: {result!r}")
    shutil.copy2(src, out / "tianyuan_ltx_test.mp4")
    print(f"SUCCESS: {out / 'tianyuan_ltx_test.mp4'}")
except Exception as exc:
    (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    print(traceback.format_exc(), flush=True)
    raise
