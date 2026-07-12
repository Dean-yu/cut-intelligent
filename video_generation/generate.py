from __future__ import annotations

import base64
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
import traceback
import urllib.request
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image
from gradio_client import Client, handle_file

ROOT = Path.cwd()
WORK = ROOT / "work"
OUT = ROOT / "generated_videos"
LOG_DIR = ROOT / "generation_logs"
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "generation.log"
REPORT_FILE = LOG_DIR / "report.json"
lock = threading.Lock()

def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    with lock:
        print(line, flush=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    log("RUN " + " ".join(cmd))
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.stdout:
        with lock:
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(result.stdout + "\n")
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result

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

STORY_PROMPTS = {
    "tianyuan": [
        "Live-action cinematic science-fantasy in Tianyuan Realm, a bright open cultivation-industrial metropolis. A wide aerial crane shot descends between layered skyways into a crowded public plaza. Hundreds of real human pedestrians walk naturally, robes and modern coats moving in the wind; crystal trains glide overhead; small flying craft cross the frame with strong parallax. Smooth forward dolly, slight tilt down, realistic daylight, 35mm anamorphic lens, physically plausible motion, rich depth, no cut within the shot.",
        "Continue seamlessly from the previous moment. The camera drops to shoulder height and tracks behind a young human cultivator wearing a blue circular energy gauntlet. He notices the enormous blue rune ring above the plaza flickering out of control, then runs through the moving crowd toward it. Pedestrians turn, step aside and shield their faces; blue sparks and wind ripple through clothes and banners. Fast but stable gimbal follow shot, natural running motion, real people, cinematic urgency, no cut within the shot.",
        "Continue seamlessly. The young cultivator plants his feet and thrusts the glowing blue gauntlet forward. A translucent circular talisman shield rapidly unfolds in the air and catches a falling crystal shuttle just above a small child. The impact sends dust, loose paper and robes outward; the crowd ducks, then looks up in relief. The camera performs a smooth 120-degree orbit around the hero and ends on the rescued child with the living city skyline behind them. Realistic physics, emotional cinematic climax, no cut within the shot.",
    ],
    "jitian": [
        "Live-action cinematic imperial science-fantasy in Jitian Realm. Epic sunset establishing shot: a colossal black-gold palace city rises in terraces, an imperial procession fills the central avenue, armored human nobles and soldiers move through incense and banners, and giant warships drift above the rooftops. The camera cranes down from the sky and slowly pushes toward the palace gate, deep atmospheric perspective, 40mm anamorphic, real human motion, oppressive scale, no cut within the shot.",
        "Continue seamlessly. A low shoulder-level tracking shot follows a lone human courier in a dark cloak moving against the flow of the grand procession. He grips a sealed glowing jade decree inside his sleeve while imperial guards scan faces and mechanical sentinels pivot toward him. Horses stamp, silk banners whip in the wind, nobles pass in foreground occlusion. The camera stays close behind him as he slips through the gate, controlled suspense, realistic body motion, no cut within the shot.",
        "Continue seamlessly. Inside the shadowed gate, the courier pauses beside a bronze pillar and secretly swaps the glowing jade decree for another seal. As the new order activates, the warships high above bank away from a densely populated civilian district. A guard notices; heads turn and the first pursuers break into a run. The camera pushes into the courier's hand for a brief close detail, then whip-pans upward to the turning fleet and pulls back as the chase begins. Cinematic political-thriller climax, natural humans, no cut within the shot.",
    ],
    "shengmeng": [
        "Live-action cinematic dystopian science-fantasy in the Covenant city. Perfectly symmetrical cold-purple megacity under a starry artificial sky. Thousands of real human citizens in identical white and black uniforms march in synchronized lines while luminous neural geometry pulses between towers. The camera glides forward on the central axis, then slowly descends into the crowd. Controlled movements, pristine reflective surfaces, subtle mist, eerie silence and scale, no cut within the shot.",
        "Continue seamlessly. The camera becomes a close tracking shot beside one young human woman in the synchronized crowd. Every citizen turns their head at exactly the same instant except her; she freezes after hearing a small child cry from a side passage. Her eyes move with the first genuine emotion while everyone else continues in perfect rhythm. Purple command light reflects across her face; cloth and hair move naturally. Slow lateral dolly, intimate tension, no cut within the shot.",
        "Continue seamlessly. The woman abruptly breaks formation, runs to the side passage and pulls the frightened child behind a crystal pillar. The citywide purple command lattice flashes red; thousands of citizens stop and turn toward her in exact unison, then begin pursuing. She takes the child's hand and runs into a narrow alley as the camera follows close behind, then rises to reveal the ordered city reacting like one organism. Real people, believable running and cloth motion, emotional escape climax, no cut within the shot.",
    ],
}

NEGATIVE = (
    "static image, still frame, frozen motion, slideshow, simple zoom, pan over a photograph, "
    "motionless people, mannequin, duplicate person, cloned crowd, warped face, deformed hands, "
    "extra fingers, extra limbs, melting architecture, unstable buildings, flicker, jitter, camera shake, "
    "low resolution, blur, oversaturated, cartoon, anime, painting, subtitles, captions, logo, watermark, "
    "new text, abrupt cut, reverse walking, impossible physics"
)

SPACE_CHOICES = {
    "tianyuan": [
        "zerogpu-aoti/wan2-2-fp8da-aoti-faster",
        "r3gm/wan2-2-fp8da-aoti-preview-2c",
    ],
    "jitian": [
        "r3gm/wan2-2-fp8da-aoti-preview-2c",
        "zerogpu-aoti/wan2-2-fp8da-aoti-faster",
    ],
    "shengmeng": [
        "zerogpu-aoti/wan2-2-fp8da-aoti-faster",
        "r3gm/wan2-2-fp8da-aoti-preview-2c",
    ],
}

def prepare_source(world: str) -> Path:
    raw = base64.b64decode(SOURCES[world].read_text(encoding="utf-8").strip())
    source_jpg = WORK / f"{world}_source.jpg"
    source_jpg.write_bytes(raw)
    with Image.open(source_jpg) as im:
        im = im.convert("RGB")
        w, h = im.size
        ratio = 16 / 9
        if w / h > ratio:
            nw = int(h * ratio)
            left = (w - nw) // 2
            im = im.crop((left, 0, left + nw, h))
        elif w / h < ratio:
            nh = int(w / ratio)
            top = (h - nh) // 2
            im = im.crop((0, top, w, top + nh))
        im = im.resize((832, 468), Image.Resampling.LANCZOS)
        im.save(source_jpg, quality=93, optimize=True)
    return source_jpg

def locate_file(value: Any) -> Path | None:
    if isinstance(value, dict):
        for key in ("path", "url", "video", "name"):
            if key in value:
                found = locate_file(value[key])
                if found:
                    return found
        for item in value.values():
            found = locate_file(item)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = locate_file(item)
            if found:
                return found
    elif isinstance(value, str):
        p = Path(value)
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
        if value.startswith("http://") or value.startswith("https://"):
            suffix = Path(value.split("?", 1)[0]).suffix or ".mp4"
            dst = WORK / f"download_{abs(hash(value))}{suffix}"
            urllib.request.urlretrieve(value, dst)
            if dst.exists() and dst.stat().st_size > 0:
                return dst
    return None

def generate_one(image_path: Path, prompt: str, output_path: Path, seed: int, spaces: list[str]) -> dict[str, Any]:
    if output_path.exists() and output_path.stat().st_size > 100_000:
        log(f"SKIP existing {output_path.name}")
        return {"status": "existing", "path": str(output_path)}

    errors: list[str] = []
    for cycle in range(3):
        for space in spaces:
            try:
                log(f"Generate {output_path.name} via {space}; cycle={cycle + 1}; seed={seed}")
                client = Client(space, verbose=False)
                result = client.predict(
                    input_image=handle_file(str(image_path)),
                    prompt=prompt,
                    steps=6,
                    negative_prompt=NEGATIVE,
                    duration_seconds=4.0,
                    guidance_scale=1.0,
                    guidance_scale_2=1.0,
                    seed=seed,
                    randomize_seed=False,
                    api_name="/generate_video",
                )
                src = locate_file(result)
                if not src:
                    raise RuntimeError(f"No video file in result: {result!r}")
                shutil.copy2(src, output_path)
                if output_path.stat().st_size < 100_000:
                    raise RuntimeError(f"Generated file too small: {output_path.stat().st_size}")
                log(f"DONE {output_path.name} via {space}: {output_path.stat().st_size} bytes")
                return {
                    "status": "generated",
                    "space": space,
                    "seed": seed,
                    "bytes": output_path.stat().st_size,
                    "path": str(output_path),
                }
            except Exception as exc:
                msg = f"{space} cycle {cycle + 1}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                log("ERROR " + msg)
                log(traceback.format_exc())
                time.sleep(15 + cycle * 20)
    raise RuntimeError("All generation routes failed for " + output_path.name + " | " + " | ".join(errors))

def extract_last_frame(video: Path, image: Path) -> None:
    run([
        "ffmpeg", "-y", "-sseof", "-0.08", "-i", str(video),
        "-frames:v", "1", "-vf", "scale=832:468:force_original_aspect_ratio=increase,crop=832:468",
        "-q:v", "2", str(image),
    ])
    if not image.exists() or image.stat().st_size < 10_000:
        raise RuntimeError(f"Unable to extract final frame from {video}")

def normalize_clip(src: Path, dst: Path) -> None:
    run([
        "ffmpeg", "-y", "-i", str(src), "-t", "4.0",
        "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,fps=24,format=yuv420p",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-movflags", "+faststart", str(dst),
    ])

def make_audio(world: str, path: Path, duration: float = 11.5, sr: int = 48000) -> None:
    rng = random.Random({"tianyuan": 101, "jitian": 202, "shengmeng": 303}[world])
    base_freq = {"tianyuan": 110.0, "jitian": 55.0, "shengmeng": 82.0}[world]
    frames = int(duration * sr)
    data = array("h")
    for i in range(frames):
        t = i / sr
        fade = min(1.0, t / 0.8, (duration - t) / 0.8)
        fade = max(0.0, fade)
        drone = 0.38 * math.sin(2 * math.pi * base_freq * t)
        drone += 0.18 * math.sin(2 * math.pi * base_freq * 1.5 * t + 0.7)
        shimmer = 0.0
        if world == "tianyuan":
            shimmer = 0.12 * math.sin(2 * math.pi * 440 * t) * (0.5 + 0.5 * math.sin(2 * math.pi * 0.2 * t))
        elif world == "jitian":
            for beat in (0.6, 1.7, 3.6, 4.0, 5.1, 7.5, 7.9, 9.2, 10.6):
                dt = t - beat
                if 0 <= dt < 0.22:
                    shimmer += 0.65 * math.sin(2 * math.pi * (72 - 120 * dt) * dt) * math.exp(-14 * dt)
        else:
            shimmer = 0.10 * math.sin(2 * math.pi * 246 * t) * (1 if int(t * 2) % 2 == 0 else 0.2)
            for beat in (4.0, 4.65, 7.6, 8.15, 8.7, 9.25, 9.8):
                dt = t - beat
                if 0 <= dt < 0.18:
                    shimmer += 0.45 * math.sin(2 * math.pi * 62 * dt) * math.exp(-16 * dt)
        noise = (rng.random() * 2 - 1) * 0.035
        sample = (drone + shimmer + noise) * 0.22 * fade
        left = int(max(-1, min(1, sample * (0.98 + 0.02 * math.sin(t)))) * 32767)
        right = int(max(-1, min(1, sample * (0.98 - 0.02 * math.sin(t)))) * 32767)
        data.extend((left, right))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())

def edit_world(world: str, raw_clips: list[Path]) -> Path:
    normalized: list[Path] = []
    for idx, clip in enumerate(raw_clips, 1):
        dst = WORK / f"{world}_{idx}_norm.mp4"
        normalize_clip(clip, dst)
        normalized.append(dst)

    silent = WORK / f"{world}_silent.mp4"
    filter_graph = (
        "[0:v][1:v]xfade=transition=fade:duration=0.25:offset=3.75[v01];"
        "[v01][2:v]xfade=transition=fade:duration=0.25:offset=7.50,"
        "fade=t=in:st=0:d=0.35,fade=t=out:st=11.10:d=0.40,format=yuv420p[v]"
    )
    run([
        "ffmpeg", "-y",
        "-i", str(normalized[0]), "-i", str(normalized[1]), "-i", str(normalized[2]),
        "-filter_complex", filter_graph,
        "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-movflags", "+faststart", str(silent),
    ])

    audio = WORK / f"{world}_score.wav"
    make_audio(world, audio)
    final = OUT / f"{TITLES[world]}.mp4"
    run([
        "ffmpeg", "-y", "-i", str(silent), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(final),
    ])
    return final

def generate_world(world: str) -> dict[str, Any]:
    log(f"=== START WORLD {world} ===")
    current_frame = prepare_source(world)
    raw_clips: list[Path] = []
    shot_reports: list[dict[str, Any]] = []
    base_seed = {"tianyuan": 71623, "jitian": 71653, "shengmeng": 71683}[world]
    for idx, prompt in enumerate(STORY_PROMPTS[world], 1):
        clip = WORK / f"{world}_{idx}_raw.mp4"
        report = generate_one(
            current_frame,
            prompt,
            clip,
            base_seed + idx * 97,
            SPACE_CHOICES[world],
        )
        report["shot"] = idx
        shot_reports.append(report)
        raw_clips.append(clip)
        if idx < 3:
            next_frame = WORK / f"{world}_{idx}_last.jpg"
            extract_last_frame(clip, next_frame)
            current_frame = next_frame
    final = edit_world(world, raw_clips)
    log(f"=== DONE WORLD {world}: {final.name} ({final.stat().st_size} bytes) ===")
    return {
        "world": world,
        "final": str(final),
        "bytes": final.stat().st_size,
        "shots": shot_reports,
    }

reports: dict[str, Any] = {"worlds": {}, "errors": {}}
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(generate_world, world): world for world in SOURCES}
    for future in as_completed(futures):
        world = futures[future]
        try:
            reports["worlds"][world] = future.result()
        except Exception as exc:
            reports["errors"][world] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            log(f"FATAL WORLD ERROR {world}: {exc}")
            log(traceback.format_exc())

REPORT_FILE.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

finals = sorted(OUT.glob("*.mp4"))
if finals:
    zip_path = OUT / "修真四万年_三界电影短片.zip"
    run(["zip", "-j", "-9", str(zip_path), *[str(p) for p in finals]])
if len(finals) != 3:
    raise SystemExit(f"Expected 3 final films, produced {len(finals)}. See generation_logs.")
