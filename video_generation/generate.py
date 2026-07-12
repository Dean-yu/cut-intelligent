from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path.cwd()
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")
WORK = ROOT / "work" / RUN_ID
OUT = ROOT / "generated_videos" / RUN_ID
LOG = ROOT / "generation_logs" / RUN_ID
for p in (WORK, OUT, LOG):
    p.mkdir(parents=True, exist_ok=True)
LOGFILE = LOG / "generation.log"
REPORT = LOG / "report.json"
LOCK = threading.Lock()

SOURCES = {
    "tianyuan": ROOT / "video_generation/inputs/tianyuan.b64",
    "jitian": ROOT / "video_generation/inputs/jitian.b64",
    "shengmeng": ROOT / "video_generation/inputs/shengmeng.b64",
}
TITLES = {
    "tianyuan": "01_天元界_回风符盘",
    "jitian": "02_极天界_密令",
    "shengmeng": "03_圣盟_偏离",
}
PROMPTS = {
    "tianyuan": [
        "Live-action cinematic science-fantasy in Tianyuan Realm, a bright cultivation-industrial metropolis. Wide aerial crane descends through layered skyways into a crowded plaza. Real human pedestrians walk naturally, robes and coats move in wind, crystal trains glide overhead, flying craft cross with strong parallax. Smooth forward dolly, 35mm anamorphic, realistic daylight, deep physical space, continuous shot.",
        "Continue seamlessly. Shoulder-height tracking behind a young human cultivator with a blue circular energy gauntlet. The giant blue rune ring above the plaza becomes unstable; he runs through the moving crowd toward it. People turn, step aside and shield their faces; blue sparks and wind ripple through clothes and banners. Fast stable gimbal follow, believable running, continuous shot.",
        "Continue seamlessly. The cultivator plants his feet and thrusts the glowing blue gauntlet forward. A translucent circular talisman shield unfolds and catches a falling crystal shuttle just above a child. Dust, paper and robes blast outward; the crowd ducks then looks up in relief. Smooth 120-degree orbit ending on the rescued child and living skyline, realistic physics, cinematic climax, continuous shot.",
    ],
    "jitian": [
        "Live-action imperial science-fantasy in Jitian Realm at sunset. A colossal black-gold palace city rises in terraces, an imperial procession fills the avenue, armored human nobles and soldiers move through incense and banners, giant warships drift above. Epic crane down and slow push toward the palace gate, 40mm anamorphic, oppressive scale, natural people, continuous shot.",
        "Continue seamlessly. Low shoulder-level tracking follows a lone human courier in a dark cloak moving against the procession. He grips a sealed glowing jade decree while guards scan faces and mechanical sentinels pivot toward him. Horses stamp, banners whip, nobles cross foreground. Camera stays close as he slips through the gate, controlled suspense, continuous shot.",
        "Continue seamlessly. In the shadowed gate the courier secretly swaps the glowing jade decree. The new order activates and warships above bank away from a civilian district. A guard notices; heads turn and pursuers run. Push into the courier's hand, whip-pan to the turning fleet, pull back as the chase begins, political-thriller climax, continuous shot.",
    ],
    "shengmeng": [
        "Live-action dystopian science-fantasy in the Covenant city. Perfectly symmetrical cold-purple megacity beneath an artificial starry sky. Thousands of real human citizens in identical uniforms march in synchronized lines while luminous neural geometry pulses between towers. Central-axis glide forward then descend into the crowd, pristine reflections, subtle mist, eerie controlled movement, continuous shot.",
        "Continue seamlessly. Close lateral tracking beside one young human woman in the synchronized crowd. Every citizen turns their head at exactly the same instant except her; she freezes after hearing a child cry from a side passage. Her eyes show the first genuine emotion while everyone else continues in perfect rhythm. Purple command light on her face, natural hair and cloth, continuous shot.",
        "Continue seamlessly. The woman breaks formation, runs to the side passage and pulls a frightened child behind a crystal pillar. The purple command lattice flashes red; thousands stop and turn toward her in exact unison, then pursue. She takes the child's hand and runs into an alley as camera follows, then rises to reveal the ordered city reacting like one organism, emotional escape climax, continuous shot.",
    ],
}
NEG = "static image, frozen frame, slideshow, simple zoom, pan over photograph, motionless people, mannequin, cloned crowd, duplicate people, warped face, deformed hands, extra fingers, extra limbs, melting architecture, flicker, jitter, camera shake, blur, cartoon, anime, painting, subtitles, captions, logo, watermark, new text, reverse walking, impossible physics"
SPACES = ["zerogpu-aoti/wan2-2-fp8da-aoti-faster", "r3gm/wan2-2-fp8da-aoti-preview-2c"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOCK:
        print(line, flush=True)
        with LOGFILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cmd: list[str]) -> None:
    log("RUN " + " ".join(cmd))
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with LOCK:
        with LOGFILE.open("a", encoding="utf-8") as f:
            f.write((p.stdout or "") + "\n")
    if p.returncode:
        raise RuntimeError(f"command failed {p.returncode}: {' '.join(cmd)}")


def source_image(world: str) -> Path:
    text = "".join(SOURCES[world].read_text(encoding="utf-8").split())
    text += "=" * ((4 - len(text) % 4) % 4)
    raw = base64.b64decode(text, validate=False)
    p = WORK / f"{world}_source.jpg"
    p.write_bytes(raw)
    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        r = 16 / 9
        if w / h > r:
            nw = int(h * r); x = (w - nw) // 2; im = im.crop((x, 0, x + nw, h))
        elif w / h < r:
            nh = int(w / r); y = (h - nh) // 2; im = im.crop((0, y, w, y + nh))
        im.resize((832, 468), Image.Resampling.LANCZOS).save(p, quality=93, optimize=True)
    return p


def find_file(v: Any) -> Path | None:
    if isinstance(v, dict):
        for key in ("path", "url", "video", "name"):
            if key in v:
                p = find_file(v[key])
                if p: return p
        for x in v.values():
            p = find_file(x)
            if p: return p
    elif isinstance(v, (list, tuple)):
        for x in v:
            p = find_file(x)
            if p: return p
    elif isinstance(v, str):
        p = Path(v)
        if p.is_file() and p.stat().st_size: return p
        if v.startswith(("http://", "https://")):
            p = WORK / f"download_{abs(hash(v))}.mp4"
            urllib.request.urlretrieve(v, p)
            if p.is_file() and p.stat().st_size: return p
    return None


def generate(image: Path, prompt: str, out: Path, seed: int) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(3):
        for space in SPACES:
            try:
                log(f"GENERATE {out.name} space={space} attempt={attempt + 1} seed={seed}")
                kw: dict[str, Any] = dict(
                    input_image=handle_file(str(image)), prompt=prompt, steps=6,
                    negative_prompt=NEG, duration_seconds=4.0,
                    guidance_scale=1.0, guidance_scale_2=1.0,
                    seed=seed, randomize_seed=False, api_name="/generate_video",
                )
                if space.startswith("r3gm/"):
                    kw["last_image"] = None
                result = Client(space, verbose=False).predict(**kw)
                src = find_file(result)
                if not src: raise RuntimeError(f"no video in result {result!r}")
                shutil.copy2(src, out)
                if out.stat().st_size < 100000: raise RuntimeError("video file too small")
                log(f"DONE {out.name} bytes={out.stat().st_size}")
                return {"space": space, "seed": seed, "bytes": out.stat().st_size}
            except Exception as e:
                errors.append(f"{space}: {type(e).__name__}: {e}")
                log("ERROR " + errors[-1])
                time.sleep(12 + attempt * 12)
    raise RuntimeError(" | ".join(errors))


def last_frame(video: Path, image: Path) -> None:
    run(["ffmpeg", "-y", "-sseof", "-0.08", "-i", str(video), "-frames:v", "1", "-update", "1", "-vf", "scale=832:468:force_original_aspect_ratio=increase,crop=832:468", "-q:v", "2", str(image)])


def edit(world: str, clips: list[Path]) -> Path:
    norm: list[Path] = []
    for i, src in enumerate(clips, 1):
        dst = WORK / f"{world}_{i}_norm.mp4"
        run(["ffmpeg", "-y", "-i", str(src), "-t", "4", "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=24,format=yuv420p", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19", str(dst)])
        norm.append(dst)
    silent = WORK / f"{world}_silent.mp4"
    fg = "[0:v][1:v]xfade=transition=fade:duration=.25:offset=3.75[v1];[v1][2:v]xfade=transition=fade:duration=.25:offset=7.5,fade=t=in:st=0:d=.35,fade=t=out:st=11.1:d=.4,format=yuv420p[v]"
    run(["ffmpeg", "-y", "-i", str(norm[0]), "-i", str(norm[1]), "-i", str(norm[2]), "-filter_complex", fg, "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(silent)])
    final = OUT / f"{TITLES[world]}.mp4"
    freq = {"tianyuan": "110", "jitian": "55", "shengmeng": "82"}[world]
    af = f"anoisesrc=color=brown:amplitude=.028:duration=11.5[a0];sine=frequency={freq}:sample_rate=48000:duration=11.5[a1];[a0]lowpass=f=700[a0f];[a1]volume=.12[a1v];[a0f][a1v]amix=inputs=2,afade=t=in:st=0:d=.7,afade=t=out:st=10.7:d=.8[a]"
    run(["ffmpeg", "-y", "-i", str(silent), "-filter_complex", af, "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(final)])
    return final


def world_job(world: str) -> dict[str, Any]:
    log(f"START {world}")
    frame = source_image(world)
    clips, shots = [], []
    base = {"tianyuan": 71623, "jitian": 71653, "shengmeng": 71683}[world]
    for i, prompt in enumerate(PROMPTS[world], 1):
        clip = WORK / f"{world}_{i}_raw.mp4"
        shots.append({"shot": i, **generate(frame, prompt, clip, base + i * 97)})
        clips.append(clip)
        if i < 3:
            frame = WORK / f"{world}_{i}_last.jpg"
            last_frame(clip, frame)
    final = edit(world, clips)
    log(f"FINISH {world} {final.name} bytes={final.stat().st_size}")
    return {"final": str(final), "bytes": final.stat().st_size, "shots": shots}


report: dict[str, Any] = {"run_id": RUN_ID, "worlds": {}, "errors": {}}
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(world_job, w): w for w in SOURCES}
    for f in as_completed(futures):
        w = futures[f]
        try:
            report["worlds"][w] = f.result()
        except Exception as e:
            report["errors"][w] = {"message": str(e), "traceback": traceback.format_exc()}
            log(f"FATAL {w}: {e}\n{traceback.format_exc()}")
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
finals = sorted(OUT.glob("*.mp4"))
if finals:
    run(["zip", "-j", "-9", str(OUT / "修真四万年_三界电影短片.zip"), *map(str, finals)])
if len(finals) != 3:
    raise SystemExit(f"Expected 3 films, got {len(finals)}")
