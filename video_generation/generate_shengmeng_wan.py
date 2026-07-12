from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path.cwd()
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")
OUT = ROOT / "wan_shengmeng" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)

NEGATIVE = (
    "static image, frozen frame, slideshow, motionless people, deformed human, duplicate people, "
    "bad anatomy, warped face, melting buildings, flicker, jitter, low quality, blur, cartoon, anime, "
    "subtitles, text, logo, watermark"
)
PROMPT = (
    "Live-action cinematic science-fantasy short story in the Covenant Alliance from Forty Millenniums of Cultivation. "
    "A vast perfectly symmetrical violet-white city lies beneath a glowing geometric energy web. Thousands of real human citizens "
    "in identical white uniforms walk in absolute synchronization. The camera glides forward and finds one young white-haired woman. "
    "She suddenly hears a small child crying and stops while everyone else continues moving. The camera circles her as overhead symbols "
    "turn red and patrol drones descend. Emotion appears on her face for the first time. She takes the child's hand and runs through the "
    "ordered crowd into a dark maintenance tunnel while the entire formation turns to watch. Begin with a grand wide establishing shot, "
    "move into a medium character shot, then dynamic handheld pursuit and end on a close emotional shot in the tunnel. Realistic humans, "
    "natural walking and running, coherent faces, cloth and hair motion, purple energy reflections, strong depth and parallax, physically "
    "plausible motion, 35mm anamorphic cinema, suspenseful lighting, no text."
)


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def locate(value: Any) -> Path | None:
    if isinstance(value, dict):
        for key in ("video", "path", "url", "name"):
            if key in value:
                found = locate(value[key])
                if found:
                    return found
        for item in value.values():
            found = locate(item)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = locate(item)
            if found:
                return found
    elif isinstance(value, str):
        path = Path(value)
        if path.exists() and path.is_file():
            return path
    return None


def prepare_source(output: Path) -> None:
    source_b64 = ROOT / "video_generation/inputs/shengmeng.b64"
    text = "".join(source_b64.read_text(encoding="utf-8").split())
    text += "=" * ((4 - len(text) % 4) % 4)
    output.write_bytes(base64.b64decode(text, validate=False))
    with Image.open(output) as image:
        image = image.convert("RGB")
        width, height = image.size
        ratio = 16 / 9
        if width / height > ratio:
            new_width = int(height * ratio)
            left = (width - new_width) // 2
            image = image.crop((left, 0, left + new_width, height))
        elif width / height < ratio:
            new_height = int(width / ratio)
            top = (height - new_height) // 2
            image = image.crop((0, top, width, top + new_height))
        image.resize((832, 468), Image.Resampling.LANCZOS).save(output, quality=94)


def generate(source: Path, raw_video: Path) -> dict[str, Any]:
    spaces = [
        "Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom",
        "dream2589632147/Dream-wan2-2-fp8da-aoti-preview-2",
        "prashant-AI-ML/Wan-2.2-pro-Superb",
    ]
    failures: list[str] = []
    for space in spaces:
        for attempt in range(1, 3):
            try:
                print(f"CONNECT {space} attempt={attempt}", flush=True)
                client = Client(space, verbose=True)
                seed = 71877 + (attempt - 1) * 1009
                result = client.predict(
                    input_image=handle_file(str(source)),
                    last_image=None,
                    prompt=PROMPT,
                    steps=4,
                    negative_prompt=NEGATIVE,
                    duration_seconds=5.0,
                    guidance_scale=2.0,
                    guidance_scale_2=1.0,
                    seed=seed,
                    randomize_seed=False,
                    quality=6,
                    scheduler="FlowMatchEulerDiscrete",
                    flow_shift=3.0,
                    frame_multiplier=16,
                    safe_mode=False,
                    video_component=True,
                    api_name="/generate_video",
                )
                generated = locate(result)
                if generated is None:
                    raise RuntimeError(f"No video path found in result: {result!r}")
                shutil.copy2(generated, raw_video)
                if raw_video.stat().st_size < 80_000:
                    raise RuntimeError(f"Generated video too small: {raw_video.stat().st_size}")
                return {"space": space, "seed": seed, "result": result, "bytes": raw_video.stat().st_size}
            except Exception as exc:
                message = f"{space} attempt {attempt}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(message, flush=True)
                print(traceback.format_exc(), flush=True)
                time.sleep(8 * attempt)
    raise RuntimeError(" | ".join(failures))


def edit(raw_video: Path, final_video: Path) -> None:
    graph = (
        "[0:v]trim=start=0:end=1.2,setpts=1.12*(PTS-STARTPTS),fps=24,"
        "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p[v0];"
        "[0:v]trim=start=1.2:end=2.4,setpts=1.12*(PTS-STARTPTS),fps=24,"
        "scale=1430:804:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p[v1];"
        "[0:v]trim=start=2.4:end=3.6,setpts=1.15*(PTS-STARTPTS),fps=24,"
        "scale=1360:765:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p[v2];"
        "[0:v]trim=start=3.6:end=4.8,setpts=1.22*(PTS-STARTPTS),fps=24,"
        "scale=1530:861:force_original_aspect_ratio=increase,crop=1280:720,"
        "eq=contrast=1.05:saturation=1.06,format=yuv420p[v3];"
        "[v0][v1][v2][v3]concat=n=4:v=1:a=0,fade=t=in:st=0:d=0.18,fade=t=out:st=5.25:d=0.35[v];"
        "[1:a]volume=0.05,afade=t=in:st=0:d=0.4,afade=t=out:st=5.1:d=0.45[a]"
    )
    run([
        "ffmpeg", "-y", "-i", str(raw_video),
        "-f", "lavfi", "-i", "sine=frequency=71:sample_rate=48000:duration=6.0",
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", "5.65",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(final_video),
    ])


def probe(video: Path) -> dict[str, Any]:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_name,width,height,r_frame_rate",
        "-of", "json", str(video),
    ], text=True))


def main() -> None:
    source = OUT / "shengmeng_source.jpg"
    raw_video = OUT / "shengmeng_wan_raw.mp4"
    final_video = OUT / "03_圣盟_第一次违抗.mp4"
    report: dict[str, Any] = {"run_id": RUN_ID, "status": "started"}
    try:
        prepare_source(source)
        generation = generate(source, raw_video)
        edit(raw_video, final_video)
        report.update({
            "status": "completed",
            "generation": generation,
            "final": final_video.name,
            "probe": probe(final_video),
        })
    except Exception as exc:
        report.update({
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
