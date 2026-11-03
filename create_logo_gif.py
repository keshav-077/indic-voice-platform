"""Regenerate VoiceBharat Pro PaperLoom-style logo banner GIF."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H, N, DUR = 920, 211, 32, 90
PAPER, INK, MUTED, SOFT, MARK = (251, 250, 246), (23, 26, 34), (113, 118, 111), (69, 74, 68), (23, 26, 34)
ACCENT, ACCENT2 = (232, 145, 58), (19, 136, 8)
TITLE, UI = r"C:\Windows\Fonts\georgiai.ttf", r"C:\Windows\Fonts\segoeui.ttf"


def fnt(p: str, s: int):
    return ImageFont.truetype(p, s)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def underline(frame: int) -> int:
    t = frame / (N - 1)
    if t < 0.45:
        return int(420 * ease(t / 0.45))
    if t < 0.75:
        return 420
    return int(420 * (1 - ease((t - 0.75) / 0.25)))


def pulse(frame: int) -> float:
    return 0.5 + 0.5 * math.sin(2 * math.pi * (frame * DUR / 1000) / 1.7)


def frame(i: int) -> Image.Image:
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        d.line([(0, y), (W, y)], fill=(int(251 - 9 * t), int(250 - 11 * t), int(246 - 15 * t)))
    ox, oy = 72, 36
    d.rounded_rectangle([ox + 8, oy + 8, ox + 140, oy + 140], radius=20, fill=MARK)
    ix, iy = ox + 8, oy + 8
    cx, cy = ix + 66, iy + 66
    for k in range(3):
        phase = (i / N + k * 0.22) % 1.0
        rad = 18 + 28 * phase
        alpha = int(220 * (1 - phase))
        ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(ring).ellipse(
            [cx - rad, cy - rad, cx + rad, cy + rad],
            outline=(*(ACCENT if k % 2 == 0 else ACCENT2), alpha),
            width=3,
        )
        img = Image.alpha_composite(img.convert("RGBA"), ring).convert("RGB")
        d = ImageDraw.Draw(img)
    for s in range(12):
        a = math.radians(s * 30 + i * (360 / N) * 0.35)
        d.line(
            [(cx + 34 * math.cos(a), cy + 34 * math.sin(a)), (cx + 42 * math.cos(a), cy + 42 * math.sin(a))],
            fill=(242, 239, 231),
            width=2,
        )
    d.text((cx, cy - 4), "V", fill=(251, 250, 246), font=fnt(TITLE, 48), anchor="mm")
    r = 7 + 2 * pulse(i)
    node = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(node).ellipse([ix + 108 - r, iy + 20 - r, ix + 108 + r, iy + 20 + r], fill=(*ACCENT, int(140 + 115 * pulse(i))))
    img = Image.alpha_composite(img.convert("RGBA"), node).convert("RGB")
    d = ImageDraw.Draw(img)
    d.text((260, 62), "VoiceBharat Pro", fill=INK, font=fnt(TITLE, 46))
    d.text((262, 118), "ENTERPRISE VOICE AI FOR 22 INDIAN LANGUAGES", fill=MUTED, font=fnt(UI, 12))
    uw = underline(i)
    if uw:
        d.rounded_rectangle([260, 148, 260 + min(uw, 480), 151], radius=1, fill=ACCENT)
    d.text((260, 168), "ASR · MT · TTS · LID · NLU on one platform.", fill=SOFT, font=fnt(UI, 15))
    return img


def main() -> None:
    frames = [frame(i) for i in range(N)]
    q = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for f in frames]
    out = ROOT / "docs" / "images" / "logo-banner.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=DUR, loop=0, optimize=True, disposal=2)
    q[0].save(ROOT / "VoiceBharat_Pro_Logo.gif", save_all=True, append_images=q[1:], duration=DUR, loop=0, optimize=True, disposal=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
