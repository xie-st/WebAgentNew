"""Generate the tiny synthetic media fixtures used by multimodal tests and docs.

These are deliberately small, content-free synthetic clips — a short sine tone
and a moving test pattern — so the multimodal examples and tests have *real*
audio/video to load without shipping large or copyrighted media. They are
committed to the repo so the cookbook snippets and tests work from a clean
checkout. Regenerate them with:

    python tests/fixtures/multimodal/generate_fixtures.py

Requires PyAV (``pip install av``), which bundles its own ffmpeg libraries.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

HERE = Path(__file__).parent


def make_audio(path: Path, seconds: float = 1.5, freq: int = 440, rate: int = 44100) -> None:
    """Write a short mono sine-tone MP3."""
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float32) / rate
    samples = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mp3", rate=rate)
        stream.bit_rate = 64000
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="s16p", layout="mono"
        )
        frame.rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def make_video(
    path: Path, seconds: float = 2.0, fps: int = 6, width: int = 160, height: int = 120
) -> None:
    """Write a short MP4 with a moving colour bar (silent)."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        total = int(seconds * fps)
        for i in range(total):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            # A gradient background plus a vertical bar that moves each frame.
            img[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
            img[:, :, 1] = int(255 * i / max(total - 1, 1))
            bar = (i * (width // total + 1)) % width
            img[:, bar : bar + 12, 2] = 255
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def main() -> None:
    make_audio(HERE / "sample_audio.mp3")
    make_video(HERE / "sample_video.mp4")
    for f in ("sample_audio.mp3", "sample_video.mp4"):
        p = HERE / f
        print(f"wrote {p.relative_to(HERE.parents[2])} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
