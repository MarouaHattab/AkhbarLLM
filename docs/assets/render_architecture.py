"""Render Miro / Excel-style architecture diagrams for the README."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
W, H = 1920, 1080
NAVY = (24, 36, 62)
INK = (18, 28, 48)
MUTED = (70, 84, 108)
GRID = (188, 208, 228)
PAPER = (247, 250, 252)
SHADOW = (24, 36, 62, 32)

YELLOW = (255, 236, 140)
ORANGE = (255, 196, 130)
BLUE = (164, 208, 255)
GREEN = (164, 226, 168)
PURPLE = (206, 186, 242)
TEAL = (150, 220, 212)
PINK = (255, 186, 208)
WHITE = (255, 255, 255)
RED = (255, 214, 214)
OK = (196, 240, 208)

FONT_UI = r"C:\Windows\Fonts\segoeui.ttf"
FONT_UI_B = r"C:\Windows\Fonts\segoeuib.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def dotted_board(width: int = W, height: int = H) -> Image.Image:
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    step = 28
    for y in range(18, height, step):
        for x in range(18, width, step):
            draw.ellipse((x, y, x + 3, y + 3), fill=GRID)
    return image


def rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill, outline=NAVY, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def card_shadow(base: Image.Image, box, radius: int = 22, offset: int = 5) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(
        (x0 + offset, y0 + offset, x1 + offset, y1 + offset),
        radius=radius,
        fill=SHADOW,
    )
    composed = Image.alpha_composite(base.convert("RGBA"), overlay)
    base.paste(composed.convert("RGB"))


def text_size(draw: ImageDraw.ImageDraw, text: str, used_font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=used_font)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw, cx, cy, text, used_font, fill=INK):
    tw, th = text_size(draw, text, used_font)
    draw.text((cx - tw / 2, cy - th / 2), text, font=used_font, fill=fill)


def wrap_lines(draw, text: str, used_font, max_width: int) -> list[str]:
    words = text.replace("/", " / ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        tw, _ = text_size(draw, trial, used_font)
        if tw <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def center_block(draw, cx, top, lines, used_font, fill, gap: int = 6):
    y = top
    heights = []
    for line in lines:
        _, th = text_size(draw, line, used_font)
        heights.append(th)
        center_text(draw, cx, y + th / 2, line, used_font, fill)
        y += th + gap
    return y


def arrow(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, dashed=False, width=5):
    if dashed:
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(int(length / 12), 1)
        for i in range(steps):
            if i % 2:
                continue
            t0 = i / steps
            t1 = min((i + 1) / steps, 1)
            draw.line(
                (
                    x0 + (x1 - x0) * t0,
                    y0 + (y1 - y0) * t0,
                    x0 + (x1 - x0) * t1,
                    y0 + (y1 - y0) * t1,
                ),
                fill=NAVY,
                width=width,
            )
    else:
        draw.line((x0, y0, x1, y1), fill=NAVY, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    size = 14
    left = (
        x1 - size * math.cos(angle - math.pi / 6),
        y1 - size * math.sin(angle - math.pi / 6),
    )
    right = (
        x1 - size * math.cos(angle + math.pi / 6),
        y1 - size * math.sin(angle + math.pi / 6),
    )
    draw.polygon([(x1, y1), left, right], fill=NAVY)


def icon_circle(draw, cx, cy, radius=30):
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=WHITE,
        outline=NAVY,
        width=3,
    )


def icon_document(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.rounded_rectangle((cx - 12, cy - 14, cx + 12, cy + 14), 3, outline=NAVY, width=3)
    draw.line((cx - 7, cy - 5, cx + 7, cy - 5), fill=NAVY, width=3)
    draw.line((cx - 7, cy + 2, cx + 7, cy + 2), fill=NAVY, width=3)
    draw.line((cx - 7, cy + 8, cx + 3, cy + 8), fill=NAVY, width=3)


def icon_sparkle(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    for size, dx, dy in ((10, 0, 0), (5, -12, -8), (5, 11, 8)):
        draw.line((cx + dx - size, cy + dy, cx + dx + size, cy + dy), fill=NAVY, width=3)
        draw.line((cx + dx, cy + dy - size, cx + dx, cy + dy + size), fill=NAVY, width=3)


def icon_list(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    for y in (-9, 0, 9):
        draw.ellipse((cx - 13, cy + y - 4, cx - 5, cy + y + 4), outline=NAVY, width=3)
        draw.line((cx - 1, cy + y, cx + 13, cy + y), fill=NAVY, width=3)


def icon_refresh(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.arc((cx - 13, cy - 13, cx + 13, cy + 13), 25, 310, fill=NAVY, width=4)
    draw.polygon([(cx + 11, cy - 10), (cx + 18, cy - 2), (cx + 6, cy - 1)], fill=NAVY)


def icon_chip(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.rounded_rectangle((cx - 12, cy - 9, cx + 12, cy + 9), 3, outline=NAVY, width=3)
    for x in (-16, 16):
        draw.line((cx + x, cy - 5, cx + x + (5 if x < 0 else -5), cy - 5), fill=NAVY, width=3)
        draw.line((cx + x, cy + 5, cx + x + (5 if x < 0 else -5), cy + 5), fill=NAVY, width=3)


def icon_plug(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.rounded_rectangle((cx - 11, cy - 6, cx + 11, cy + 10), 4, outline=NAVY, width=3)
    draw.line((cx - 5, cy - 13, cx - 5, cy - 6), fill=NAVY, width=3)
    draw.line((cx + 5, cy - 13, cx + 5, cy - 6), fill=NAVY, width=3)


def icon_bars(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.rectangle((cx - 12, cy + 4, cx - 5, cy + 12), outline=NAVY, width=3)
    draw.rectangle((cx - 3, cy - 4, cx + 4, cy + 12), outline=NAVY, width=3)
    draw.rectangle((cx + 6, cy - 12, cx + 13, cy + 12), outline=NAVY, width=3)


def icon_hf(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    center_text(draw, cx, cy + 1, "HF", font(FONT_UI_B, 18), NAVY)


def icon_window(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.rounded_rectangle((cx - 14, cy - 11, cx + 14, cy + 12), 3, outline=NAVY, width=3)
    draw.line((cx - 14, cy - 3, cx + 14, cy - 3), fill=NAVY, width=3)
    draw.ellipse((cx - 10, cy - 8, cx - 6, cy - 4), fill=NAVY)


def icon_server(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.rounded_rectangle((cx - 13, cy - 13, cx + 13, cy + 13), 3, outline=NAVY, width=3)
    draw.line((cx - 13, cy - 3, cx + 13, cy - 3), fill=NAVY, width=3)
    draw.line((cx - 13, cy + 5, cx + 13, cy + 5), fill=NAVY, width=3)
    draw.ellipse((cx + 7, cy - 9, cx + 11, cy - 5), fill=NAVY)
    draw.ellipse((cx + 7, cy - 1, cx + 11, cy + 3), fill=NAVY)


def icon_braces(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    center_text(draw, cx, cy + 1, "{ }", font(FONT_UI_B, 20), NAVY)


def icon_translate(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    center_text(draw, cx - 6, cy - 3, "A", font(FONT_UI_B, 16), NAVY)
    center_text(draw, cx + 7, cy + 7, "ع", font(FONT_UI_B, 16), NAVY)


def icon_gauge(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.arc((cx - 13, cy - 10, cx + 13, cy + 16), 200, 340, fill=NAVY, width=4)
    draw.line((cx, cy + 5, cx + 8, cy - 7), fill=NAVY, width=4)


def icon_schema(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.rounded_rectangle((cx - 12, cy - 13, cx + 12, cy + 13), 3, outline=NAVY, width=3)
    center_text(draw, cx, cy + 1, "{}", font(FONT_UI_B, 16), NAVY)


def icon_globe(draw, cx, cy, _color):
    icon_circle(draw, cx, cy)
    draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), outline=NAVY, width=3)
    draw.ellipse((cx - 6, cy - 12, cx + 6, cy + 12), outline=NAVY, width=3)
    draw.line((cx - 12, cy, cx + 12, cy), fill=NAVY, width=3)


def icon_news(draw, cx, cy, _color):
    icon_circle(draw, cx, cy, 34)
    draw.rounded_rectangle((cx - 14, cy - 14, cx + 14, cy + 14), 3, outline=NAVY, width=3)
    draw.rectangle((cx - 10, cy - 9, cx + 2, cy + 1), outline=NAVY, width=2)
    draw.line((cx + 5, cy - 7, cx + 11, cy - 7), fill=NAVY, width=3)
    draw.line((cx + 5, cy - 1, cx + 11, cy - 1), fill=NAVY, width=3)
    draw.line((cx - 10, cy + 7, cx + 11, cy + 7), fill=NAVY, width=3)


ICONS = {
    "document": icon_document,
    "sparkle": icon_sparkle,
    "list": icon_list,
    "refresh": icon_refresh,
    "chip": icon_chip,
    "plug": icon_plug,
    "bars": icon_bars,
    "hf": icon_hf,
    "window": icon_window,
    "server": icon_server,
    "braces": icon_braces,
    "translate": icon_translate,
    "gauge": icon_gauge,
    "schema": icon_schema,
    "globe": icon_globe,
    "news": icon_news,
}


def draw_card(
    image: Image.Image,
    box,
    fill,
    title: str,
    subtitle: str | None,
    icon: str,
    title_size: int = 26,
    sub_size: int = 18,
):
    card_shadow(image, box)
    draw = ImageDraw.Draw(image)
    rounded(draw, box, 24, fill)
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    inner = int((x1 - x0) - 28)
    ICONS[icon](draw, cx, y0 + 52, fill)
    title_font = font(FONT_UI_B, title_size)
    title_lines = wrap_lines(draw, title, title_font, inner)
    y = center_block(draw, cx, y0 + 96, title_lines, title_font, INK, gap=2)
    if subtitle:
        sub_font = font(FONT_UI, sub_size)
        sub_lines = wrap_lines(draw, subtitle, sub_font, inner)
        center_block(draw, cx, y + 8, sub_lines, sub_font, MUTED, gap=1)
    return box


def header(draw, title: str, subtitle: str, width: int = W):
    center_text(draw, width / 2, 54, title, font(FONT_UI_B, 44), INK)
    center_text(draw, width / 2, 108, subtitle, font(FONT_UI, 24), MUTED)
    draw.line((120, 140, width - 120, 140), fill=GRID, width=3)


def footer_note(image, box, title: str, body: str):
    card_shadow(image, box, radius=18)
    draw = ImageDraw.Draw(image)
    rounded(draw, box, 18, WHITE)
    x0, y0, x1, _y1 = box
    draw.text((x0 + 36, y0 + 22), title, font=font(FONT_UI_B, 26), fill=INK)
    body_font = font(FONT_UI, 22)
    lines = wrap_lines(draw, body, body_font, int(x1 - x0 - 72))
    y = y0 + 64
    for line in lines:
        draw.text((x0 + 36, y), line, font=body_font, fill=MUTED)
        y += 30


def render_overview() -> Image.Image:
    width, height = 1920, 980
    image = dotted_board(width, height)
    draw = ImageDraw.Draw(image)
    center_text(draw, width / 2, 58, "AkhbarLLM", font(FONT_UI_B, 56), INK)
    center_text(
        draw,
        width / 2,
        118,
        "Arabic news  →  local Qwen 1.5B  →  schema-valid JSON",
        font(FONT_UI, 26),
        MUTED,
    )
    draw.line((160, 158, width - 160, 158), fill=GRID, width=3)

    steps = [
        (80, YELLOW, "news", "Arabic news", "Paste a story. Extract details or translate."),
        (520, ORANGE, "sparkle", "Base Qwen is weak", "English titles, markdown fences, fake entities."),
        (960, PURPLE, "chip", "LoRA on Kaggle T4 x 2", "Teacher JSON distilled, then fine-tuned."),
        (1400, GREEN, "braces", "AkhbarLLM JSON", "Arabic extraction + full translation, schema OK."),
    ]
    for x, color, icon, title, body in steps:
        box = (x, 210, x + 400, 470)
        draw_card(image, box, color, title, body, icon, title_size=28, sub_size=20)

    draw = ImageDraw.Draw(image)
    for x in (480, 920, 1360):
        arrow(draw, x, 340, x + 40, 340)

    bad = (80, 530, 900, 760)
    good = (1020, 530, 1840, 760)
    draw_card(image, bad, RED, "Before  ·  base Qwen", "Title in English. Invents Person / Disease. JSON invalid.", "document", 28, 20)
    draw_card(image, good, OK, "After  ·  AkhbarLLM", "Arabic title. Real entities. Raw JSON, schema valid.", "list", 28, 20)

    footer_note(
        image,
        (80, 800, 1840, 930),
        "Trained on Kaggle with 2x NVIDIA T4",
        "LoRA rank 64 on Qwen2.5-1.5B-Instruct. Tracked in Weights & Biases. Adapter on Hugging Face: marouaHattab/ArabLLM-news. Served with vLLM in WSL and a Streamlit UI on Windows.",
    )
    return image


def render_training() -> Image.Image:
    image = dotted_board()
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "Training pipeline",
        "AkhbarLLM  ·  Qwen2.5-1.5B Instruct  ·  LoRA SFT on Kaggle T4 x 2",
    )

    top = 190
    height = 230
    gap = 24
    card_w = 280
    start = 50
    cards = [
        ("Raw news", "Arabic JSONL stories", YELLOW, "document"),
        ("Teacher", "o4-mini labels", ORANGE, "sparkle"),
        ("SFT data", "story + schema + JSON", BLUE, "list"),
        ("Format", "train.json and val.json", GREEN, "refresh"),
        ("Train", "Kaggle T4 x 2", PURPLE, "chip"),
        ("Adapter", "LoRA rank 64", TEAL, "plug"),
    ]
    boxes = []
    x = start
    for title, subtitle, color, icon in cards:
        box = (x, top, x + card_w, top + height)
        draw_card(image, box, color, title, subtitle, icon)
        boxes.append(box)
        x += card_w + gap

    draw = ImageDraw.Draw(image)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, left[2] + 2, top + height / 2, right[0] - 2, top + height / 2)

    wandb = (boxes[4][0] - 40, 500, boxes[4][0] + 250, 720)
    hf = (boxes[4][2] - 80, 500, boxes[4][2] + 280, 720)
    draw_card(image, wandb, PINK, "W&B", "loss, GPU, steps", "bars")
    draw_card(image, hf, BLUE, "Hugging Face", "marouaHattab / ArabLLM-news", "hf")
    draw = ImageDraw.Draw(image)
    arrow(
        draw,
        (boxes[4][0] + boxes[4][2]) / 2 - 50,
        boxes[4][3] + 6,
        (wandb[0] + wandb[2]) / 2,
        wandb[1] - 6,
        dashed=True,
    )
    arrow(
        draw,
        (boxes[4][0] + boxes[4][2]) / 2 + 50,
        boxes[4][3] + 6,
        (hf[0] + hf[2]) / 2,
        hf[1] - 6,
        dashed=True,
    )

    footer_note(
        image,
        (50, 780, 1870, 1020),
        "What is trained",
        "The 1.5B base model stays frozen. Only a LoRA adapter (rank 64, all linear layers) learns to emit schema-valid JSON for Arabic news extraction and translation. Training used 2 NVIDIA T4 GPUs on Kaggle.",
    )
    return image


def render_inference() -> Image.Image:
    image = dotted_board()
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "Inference pipeline",
        "AkhbarLLM  ·  Streamlit on Windows  ·  vLLM in WSL",
    )

    streamlit = (70, 200, 430, 430)
    vllm = (520, 180, 1040, 620)
    extract = (1130, 180, 1490, 370)
    translate = (1130, 400, 1490, 590)
    locust = (1130, 620, 1490, 810)
    schema = (1570, 250, 1870, 540)
    api = (70, 500, 430, 730)

    draw_card(image, streamlit, YELLOW, "Streamlit UI", "Windows client", "window")

    card_shadow(image, vllm, radius=26)
    draw = ImageDraw.Draw(image)
    rounded(draw, vllm, 26, PURPLE)
    cx = (vllm[0] + vllm[2]) / 2
    icon_server(draw, cx, vllm[1] + 62, PURPLE)
    center_text(draw, cx, vllm[1] + 128, "vLLM server", font(FONT_UI_B, 34))
    center_text(draw, cx, vllm[1] + 172, "WSL Linux GPU", font(FONT_UI, 22), MUTED)
    lines = [
        "Qwen2.5-1.5B Instruct",
        "LoRA module  news-lora",
        "CJK token suppressor",
    ]
    chip_font = font(FONT_UI_B, 20)
    for i, line in enumerate(lines):
        y = vllm[1] + 220 + i * 54
        draw.rounded_rectangle(
            (vllm[0] + 48, y, vllm[2] - 48, y + 42),
            12,
            fill=WHITE,
            outline=NAVY,
            width=3,
        )
        center_text(draw, cx, y + 21, line, chip_font)

    draw_card(image, extract, GREEN, "Extract JSON", "Arabic NewsDetails", "braces")
    draw_card(image, translate, BLUE, "Translate JSON", "TranslatedStory", "translate")
    draw_card(image, locust, ORANGE, "Locust test", "/v1/completions", "gauge")
    draw_card(image, schema, WHITE, "Pydantic", "schema-valid JSON", "schema")
    draw_card(image, api, TEAL, "localhost:8000/v1", "OpenAI-compatible API", "globe")

    draw = ImageDraw.Draw(image)
    arrow(draw, streamlit[2] + 4, (streamlit[1] + streamlit[3]) / 2, vllm[0] - 4, 310)
    arrow(draw, vllm[2] + 4, 275, extract[0] - 4, (extract[1] + extract[3]) / 2)
    arrow(draw, vllm[2] + 4, 430, translate[0] - 4, (translate[1] + translate[3]) / 2)
    arrow(draw, vllm[2] + 4, 560, locust[0] - 4, (locust[1] + locust[3]) / 2)
    arrow(draw, extract[2] + 4, (extract[1] + extract[3]) / 2, schema[0] - 4, 330)
    arrow(draw, translate[2] + 4, (translate[1] + translate[3]) / 2, schema[0] - 4, 460)
    arrow(
        draw,
        (api[0] + api[2]) / 2,
        api[1] - 4,
        vllm[0] + 90,
        vllm[3] + 4,
        dashed=True,
    )

    footer_note(
        image,
        (520, 850, 1870, 1030),
        "How a request is served",
        "Streamlit streams chat completions from vLLM. Middleware always attaches the Chinese-token suppressor. Valid JSON is parsed into NewsDetails or TranslatedStory.",
    )
    return image


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overview = render_overview()
    training = render_training()
    inference = render_inference()
    overview.save(OUT_DIR / "project-overview.png", "PNG")
    training.save(OUT_DIR / "training-pipeline.png", "PNG")
    inference.save(OUT_DIR / "inference-pipeline.png", "PNG")
    print(f"Wrote {OUT_DIR / 'project-overview.png'}")
    print(f"Wrote {OUT_DIR / 'training-pipeline.png'}")
    print(f"Wrote {OUT_DIR / 'inference-pipeline.png'}")


if __name__ == "__main__":
    main()
