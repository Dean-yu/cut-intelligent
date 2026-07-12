from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path.cwd()
RUN_ID = os.environ.get("GITHUB_RUN_ID", "local")
OUT = ROOT / "ltx_test" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)

NEGATIVE = (
    "static image, frozen frame, slideshow, simple zoom, motionless people, duplicate people, "
    "warped face, deformed hands, melting buildings, flicker, jitter, heavy blur, cartoon, anime, "
    "subtitles, title, logo, watermark, text"
)

WORLDS = {
    "tianyuan": {
        "source": ROOT / "video_generation/inputs/tianyuan.b64",
        "filename": "01_天元界_回风符盘.mp4",
        "tone": 58,
        "seed": 71623,
        "prompts": [
            "Live-action cinematic science-fantasy in Tianyuan Realm. Establishing shot: a bright cultivation-industrial metropolis with traditional eastern roofs fused into immense crystal towers and layered skyways. The camera dives between flying craft toward a crowded public plaza, then finds the SAME young male artifact apprentice in a dark blue workshop coat carrying a small brass wind-talisman disk. Real pedestrians move naturally, robes and banners react to wind, crystal trains glide overhead, strong parallax, 35mm anamorphic lens, physically plausible motion.",
            "Continue from the exact previous frame with the SAME young artifact apprentice and the SAME city plaza. An unmanned crystal skiff suddenly loses control and dives toward a child. The apprentice sprints through the scattering crowd, slides on one knee and opens the brass wind-talisman disk. Blue spiritual wind forms a powerful vortex, diverts the skiff and lifts the child to safety. A city-guard cultivator notices him in the background. Dynamic tracking camera, real human movement, wind, dust and cloth simulation, cinematic action, coherent faces and architecture.",
        ],
    },
    "jitian": {
        "source": ROOT / "video_generation/inputs/jitian.b64",
        "filename": "02_极天界_密令.mp4",
        "tone": 43,
        "seed": 71750,
        "prompts": [
            "Live-action cinematic imperial science-fantasy in Jitian Realm. Establishing shot of a colossal black-gold imperial capital under a storm-lit sunset, monumental eastern palaces, suspended warships and endless military banners. The camera tracks behind the SAME young male imperial courier in a black-and-gold cloak as he crosses a grand procession toward the command palace. Crowds bow in rigid hierarchy, soldiers march, ships move overhead, smoke and fabric move naturally, oppressive scale, 40mm anamorphic lens, realistic humans and materials.",
            "Continue from the exact previous frame with the SAME imperial courier inside the black-gold command palace. He opens a glowing red order tablet and sees a civilian district marked for sacrifice. Close on his restrained reaction, then a tense tracking move as he reaches the fleet dispatch console, secretly swaps the command seal and redirects the warships away from the city. Armored guards surround him as the fleet turns in the sky; he remains still and accepts arrest. Realistic acting, suspenseful camera movement, coherent architecture, cinematic lighting and physical motion.",
        ],
    },
    "shengmeng": {
        "source": ROOT / "video_generation/inputs/shengmeng.b64",
        "filename": "03_圣盟_第一次违抗.mp4",
        "tone": 71,
        "seed": 71877,
        "prompts": [
            "Live-action cinematic science-fantasy in the Covenant Alliance. A vast perfectly symmetrical violet-white city beneath a geometric energy web. Thousands of human citizens in identical white uniforms walk in absolute synchronization. The camera glides down the central avenue and finds the SAME young white-haired woman among them. Faces are calm and emotionless, drones patrol overhead, polished surfaces reflect purple light, precise collective motion, realistic humans, unsettling order, slow controlled dolly, 50mm anamorphic lens.",
            "Continue from the exact previous frame with the SAME white-haired woman in the SAME ordered avenue. She hears a small child crying and suddenly stops while every other citizen keeps moving in perfect synchronization. The camera makes a tense orbit; overhead symbols turn red and patrol drones descend. She shows emotion for the first time, takes the child's hand and runs into a dark maintenance tunnel as the formation turns to watch. Dynamic follow camera, believable running, cloth and hair motion, coherent faces, violet energy flashes, suspenseful cinematic realism.",
        ],
    },
}


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


def prepare_source(source_b64: Path, output: Path) -> None:
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
        image.resize((704, 396), Image.Resampling.LANCZOS).save(output, quality=93)


def extract_last_frame(video: Path, output: Path) -> None:
    run([
        "ffmpeg", "-y", "-sseof", "-0.10", "-i", str(video), "-frames:v", "1",
        "-vf", "scale=704:396:force_original_aspect_ratio=increase,crop=704:396",
        "-q:v", "2", "-update", "1", str(output),
    ])


def generate_clip(client: Client, source: Path, prompt: str, output: Path, seed: int) -> dict[str, Any]:
    failures: list[str] = []
    for attempt in range(1, 4):
        try:
            actual_seed = seed + (attempt - 1) * 1009
            print(f"GENERATE {output.name} attempt={attempt} seed={actual_seed}", flush=True)
            result = client.predict(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                input_image_filepath=handle_file(str(source)),
                input_video_filepath=None,
                height_ui=396,
                width_ui=704,
                mode="image-to-video",
                duration_ui=3.0,
                ui_frames_to_use=9,
                seed_ui=actual_seed,
                randomize_seed=False,
                ui_guidance_scale=1.7,
                improve_texture_flag=True,
                api_name="/image_to_video",
            )
            generated = locate(result)
            if generated is None:
                raise RuntimeError(f"No output video in result: {result!r}")
            shutil.copy2(generated, output)
            if output.stat().st_size < 50_000:
                raise RuntimeError(f"Generated file is unexpectedly small: {output.stat().st_size}")
            return {
                "seed": actual_seed,
                "bytes": output.stat().st_size,
                "result": result,
            }
        except Exception as exc:
            failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            print(traceback.format_exc(), flush=True)
            time.sleep(8 * attempt)
    raise RuntimeError(" | ".join(failures))


def edit_story(first: Path, second: Path, output: Path, tone: int) -> None:
    filter_graph = (
        "[0:v]trim=start=0:end=1.4,setpts=1.12*(PTS-STARTPTS),"
        "fps=24,scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p[v0];"
        "[0:v]trim=start=1.4:end=2.8,setpts=1.12*(PTS-STARTPTS),"
        "fps=24,scale=1450:816:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p[v1];"
        "[1:v]trim=start=0:end=1.4,setpts=1.12*(PTS-STARTPTS),"
        "fps=24,scale=1340:754:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p[v2];"
        "[1:v]trim=start=1.4:end=2.8,setpts=1.18*(PTS-STARTPTS),"
        "fps=24,scale=1520:855:force_original_aspect_ratio=increase,crop=1280:720,"
        "eq=contrast=1.04:saturation=1.04,format=yuv420p[v3];"
        "[v0][v1][v2][v3]concat=n=4:v=1:a=0,fade=t=in:st=0:d=0.18,"
        "fade=t=out:st=6.05:d=0.35[v];"
        "[2:a]volume=0.055,afade=t=in:st=0:d=0.4,afade=t=out:st=5.9:d=0.55[a]"
    )
    run([
        "ffmpeg", "-y", "-i", str(first), "-i", str(second),
        "-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000:duration=7.0",
        "-filter_complex", filter_graph,
        "-map", "[v]", "-map", "[a]", "-t", "6.4",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output),
    ])


def probe(video: Path) -> dict[str, Any]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=codec_name,width,height,r_frame_rate",
        "-of", "json", str(video),
    ], text=True)
    return json.loads(raw)


def main() -> None:
    client = Client("Lightricks/ltx-video-distilled", verbose=True)
    report: dict[str, Any] = {"run_id": RUN_ID, "model": "Lightricks/ltx-video-distilled", "worlds": {}, "errors": {}}

    for key, spec in WORLDS.items():
        try:
            source = OUT / f"{key}_source.jpg"
            prepare_source(spec["source"], source)
            clip1 = OUT / f"{key}_01_setup.mp4"
            clip2 = OUT / f"{key}_02_action.mp4"
            continuation = OUT / f"{key}_continuation.jpg"

            first_info = generate_clip(client, source, spec["prompts"][0], clip1, spec["seed"])
            extract_last_frame(clip1, continuation)
            second_info = generate_clip(client, continuation, spec["prompts"][1], clip2, spec["seed"] + 97)

            final = OUT / spec["filename"]
            edit_story(clip1, clip2, final, spec["tone"])
            report["worlds"][key] = {
                "final": final.name,
                "bytes": final.stat().st_size,
                "probe": probe(final),
                "generated_clips": [first_info, second_info],
            }
            print(f"WORLD COMPLETE {key}: {final}", flush=True)
        except Exception as exc:
            report["errors"][key] = {
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"WORLD FAILED {key}\n{traceback.format_exc()}", flush=True)

    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    completed = [OUT / spec["filename"] for key, spec in WORLDS.items() if key in report["worlds"]]
    if completed:
        package = OUT / "修真四万年_三界真实时序短片.zip"
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
            for video in completed:
                archive.write(video, arcname=video.name)
            archive.write(report_path, arcname="生成校验报告.json")

    if report["errors"]:
        raise RuntimeError("Incomplete generation: " + ", ".join(report["errors"]))


if __name__ == "__main__":
    main()
