"""generate an OpenCV contact sheet for training and test scroll patches."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CV_IO_MAX_IMAGE_PIXELS", str(2**34))

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGE_HEIGHT = 176
TEST_VIEW_WIDTH = 300
CARD_GAP = 10
GROUP_GAP = 14
GROUP_PAD = 10
GROUP_TITLE_HEIGHT = 38
CARD_LABEL_HEIGHT = 64
TEST_CARD_LABEL_HEIGHT = 96
SECTION_TITLE_HEIGHT = 54
CANVAS_WIDTH = 3_008
BACKGROUND = (246, 246, 242)
PANEL_BACKGROUND = (255, 255, 255)
TEXT = (25, 25, 25)
MUTED = (105, 105, 105)
ACCENT = (45, 70, 205)


@dataclass(frozen=True)
class Patch:
    patch_id: str
    name: str
    source: str = "labels"
    area: str = ""
    source_patch: str = ""


TRAINING_GROUPS = [
    ("PHerc0139", [
        Patch("20260115000000", "w044"),
        Patch("20250223000000", "w059"),
        Patch("20260206000001", "w047"),
        Patch("20260115000001", "w056"),
        Patch("20260210000000", "w058"),
        Patch("20260227000000", "w052"),
        Patch("20260318000000", "w049"),
        Patch("20260325000000", "w046"),
        Patch("20260108000000", "w041"),
        Patch("20250831000000", "w040"),
        Patch("20260302000000", "w039"),
        Patch("20260306000000", "w038"),
        Patch("20260310000000", "w037"),
        Patch("20260303000000", "w034"),
        Patch("20260317000000", "w035"),
    ]),
    ("PHerc0172", [
        Patch("20251111010954", "w068"),
        Patch("20251112000002", "w087"),
    ]),
    ("PHerc1667", [
        Patch("20240304144031", "w018"),
        Patch("20240304141531", "w013"),
        Patch("20231201215900", "Cr1 Fr3", "mask_overlay"),
    ]),
    ("PHerc0814", [Patch("20260226000000", "seg46527")]),
    ("PHerc0500P2", [Patch("20250628074500", "500P2 front")]),
    ("PHerc0009B", [Patch("20250919125754", "patch 487")]),
    ("PHercParis4", [Patch("20231210121321", "Paris4")]),
    ("PHercParis2", [Patch("20230301213755", "Fr143", "mask_overlay")]),
    ("PHerc51", [Patch("20231205222200", "Cr4 Fr8", "mask_overlay")]),
    ("PHercParis1", [Patch("20230301213423", "Fr34", "mask_overlay")]),
    ("PHerc0343P", [Patch("20250511003658", "tifxyz segment")]),
    ("PHerc0841", [Patch("20260221022814", "auto-grown 405")]),
]

TEST_GROUPS = [
    ("PHerc0813", [Patch(
        "20260814140748", "test surface", "mask", "33.31 cm^2",
        "auto_grown_20260814140748456",
    )]),
    ("PHerc0211", [Patch(
        "20260717193517", "test surface", "mask", "~28.60 cm^2",
        "five-patch merged surface",
    )]),
    ("PHerc1203", [Patch(
        "20260720090842", "test surface", "mask", "7.90 cm^2",
        "auto_grown_20260720090842117",
    )]),
    ("PHerc1447", [Patch(
        "20250703034159", "test surface", "mask", "51.27 cm^2",
        "20250703034159",
    )]),
    ("PHerc0826", [Patch(
        "20260723112922", "test surface", "mask", "~18.93 cm^2",
        "merged auto_grown_20260723112922652",
    )]),
    ("PHerc0846A", [Patch(
        "20260921094413", "test surface", "mask", "22.88 cm^2",
        "auto_grown_20260921094413486",
    )]),
    ("PHerc0175A", [Patch(
        "20260918132724", "test surface", "mask", "11.95 cm^2",
        "auto_grown_20260918132724424",
    )]),
    ("PHerc0306B", [Patch(
        "20260922073234", "test surface", "mask", "14.13 cm^2",
        "auto_grown_20260922073234974",
    )]),
    ("PHerc0800", [Patch(
        "20260922161631", "test surface", "mask", "~23.36 cm^2",
        "merged auto_grown_20260922161631422",
    )]),
]


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _fit_height(image: np.ndarray, height: int = IMAGE_HEIGHT) -> np.ndarray:
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    interpolation = cv2.INTER_AREA if image.shape[0] > height else cv2.INTER_NEAREST
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _label_preview(patch: Patch) -> tuple[np.ndarray, str]:
    base_path = ROOT / "inklabels" / "2_4um" / f"{patch.patch_id}.png"
    overlay_path = ROOT / "inklabels" / f"{patch.patch_id}.png"
    if patch.source == "native":
        base = _read_gray(overlay_path)
        overlay = None
        note = "native label only"
    else:
        base = _read_gray(base_path)
        overlay = _read_gray(overlay_path)
        note = "2.4um + aligned"

    base = _fit_height(base)
    preview = cv2.cvtColor(255 - base, cv2.COLOR_GRAY2BGR).astype(np.float32)
    if overlay is not None:
        overlay = cv2.resize(
            overlay,
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        alpha = (0.33 * overlay)[..., None]
        color = np.full_like(preview, ACCENT, dtype=np.float32)
        preview = preview * (1.0 - alpha) + color * alpha
    return np.clip(preview, 0, 255).astype(np.uint8), note


def _mask_overlay_preview(patch: Patch) -> tuple[np.ndarray, str]:
    mask = _fit_height(_read_gray(ROOT / "masks" / f"{patch.patch_id}.png"))
    inside = mask > 127
    preview = np.full((*mask.shape, 3), 255, np.uint8)
    preview[inside] = (232, 232, 227)
    kernel = np.ones((3, 3), np.uint8)
    outline = cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    preview[outline] = (35, 35, 35)

    overlay = _read_gray(ROOT / "inklabels" / f"{patch.patch_id}.png")
    overlay = cv2.resize(
        overlay,
        (preview.shape[1], preview.shape[0]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0
    alpha = (0.33 * overlay)[..., None]
    color = np.full_like(preview, ACCENT, dtype=np.float32)
    preview = preview.astype(np.float32) * (1.0 - alpha) + color * alpha
    return np.clip(preview, 0, 255).astype(np.uint8), "mask + aligned overlay"


def _test_area_value(patch: Patch) -> float:
    return float(patch.area.replace("~", "").replace("cm^2", "").strip())


def _test_mask(patch: Patch) -> np.ndarray:
    """rendered mask, or the cropped tifxyz valid grid for surfaces not yet assembled."""
    mask_path = ROOT / "masks" / f"{patch.patch_id}.png"
    if mask_path.exists():
        return _read_gray(mask_path)
    from assemble_test_segments import FRAGMENTS

    mesh = next(mesh for out_id, mesh, _, _ in FRAGMENTS if out_id == patch.patch_id)
    x = cv2.imread(str(ROOT / "tifxyz" / mesh / "x.tif"), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(mask_path)
    valid = x != -1
    ys, xs = np.where(valid)
    valid = valid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return valid.astype(np.uint8) * 255


def _test_preview_size(patch: Patch, shape: tuple[int, int]) -> tuple[int, int]:
    """preserve source aspect while making displayed bounds proportional to surface area."""
    height, width = shape
    aspect = width / max(height, 1)
    areas_and_aspects = []
    for _, patches in TEST_GROUPS:
        candidate = patches[0]
        candidate_mask = _test_mask(candidate)
        candidate_aspect = candidate_mask.shape[1] / max(candidate_mask.shape[0], 1)
        areas_and_aspects.append((_test_area_value(candidate), candidate_aspect))
    scale = min(
        min(
            TEST_VIEW_WIDTH / np.sqrt(area * candidate_aspect),
            IMAGE_HEIGHT / np.sqrt(area / candidate_aspect),
        )
        for area, candidate_aspect in areas_and_aspects
    )
    area = _test_area_value(patch)
    preview_width = max(1, int(round(scale * np.sqrt(area * aspect))))
    preview_height = max(1, int(round(scale * np.sqrt(area / aspect))))
    return preview_width, preview_height


def _mask_preview(patch: Patch) -> tuple[np.ndarray, str]:
    source = _test_mask(patch)
    preview_width, preview_height = _test_preview_size(patch, source.shape)
    mask = cv2.resize(source, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    inside = mask > 127
    silhouette = np.full((*mask.shape, 3), 255, np.uint8)
    silhouette[inside] = (235, 235, 230)
    kernel = np.ones((3, 3), np.uint8)
    outline = cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    silhouette[outline] = (20, 20, 20)
    preview = np.full((IMAGE_HEIGHT, TEST_VIEW_WIDTH, 3), 255, np.uint8)
    y = (IMAGE_HEIGHT - preview_height) // 2
    x = (TEST_VIEW_WIDTH - preview_width) // 2
    preview[y:y + preview_height, x:x + preview_width] = silhouette
    return preview, "area-scaled mask outline"


def _text_fit(image, text, origin, max_width, scale=0.55, color=TEXT, thickness=1):
    while scale > 0.3:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if width <= max_width:
            break
        scale -= 0.05
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _render_card(patch: Patch) -> np.ndarray:
    if patch.source == "mask":
        preview, note = _mask_preview(patch)
    elif patch.source == "mask_overlay":
        preview, note = _mask_overlay_preview(patch)
    else:
        preview, note = _label_preview(patch)

    height, preview_width = preview.shape[:2]
    label_height = TEST_CARD_LABEL_HEIGHT if patch.source == "mask" else CARD_LABEL_HEIGHT
    primary_label = f"{patch.name} | {patch.patch_id}"
    primary_width = cv2.getTextSize(
        primary_label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        2,
    )[0][0] + 14
    width = max(
        preview_width,
        TEST_VIEW_WIDTH if patch.source == "mask" else primary_width,
    )
    card = np.full((height + label_height, width, 3), PANEL_BACKGROUND, np.uint8)
    preview_x = (width - preview_width) // 2
    card[:height, preview_x:preview_x + preview_width] = preview
    cv2.line(card, (0, height), (width - 1, height), (220, 220, 216), 1)
    _text_fit(card, primary_label, (7, height + 26), width - 14,
              scale=0.58, thickness=2)
    if patch.source == "mask":
        _text_fit(card, f"area: {patch.area}", (7, height + 53), width - 14,
                  scale=0.54, color=TEXT, thickness=1)
        _text_fit(card, f"source: {patch.source_patch}", (7, height + 79), width - 14,
                  scale=0.48, color=MUTED, thickness=1)
    else:
        _text_fit(card, note, (7, height + 53), width - 14, scale=0.46,
                  color=MUTED, thickness=1)
    return card


def _render_group(domain: str, patches: list[Patch]) -> list[np.ndarray]:
    cards = [_render_card(patch) for patch in patches]
    chunks: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    current_width = 0
    max_inner_width = CANVAS_WIDTH - 2 * GROUP_PAD
    for card in cards:
        added = card.shape[1] + (CARD_GAP if current else 0)
        if current and current_width + added > max_inner_width:
            chunks.append(current)
            current = []
            current_width = 0
            added = card.shape[1]
        current.append(card)
        current_width += added
    if current:
        chunks.append(current)

    rendered = []
    for index, chunk in enumerate(chunks):
        width = sum(card.shape[1] for card in chunk) + CARD_GAP * (len(chunk) - 1)
        height = GROUP_TITLE_HEIGHT + chunk[0].shape[0] + 2 * GROUP_PAD
        group = np.full((height, width + 2 * GROUP_PAD, 3), PANEL_BACKGROUND, np.uint8)
        cv2.rectangle(group, (0, 0), (group.shape[1] - 1, group.shape[0] - 1), (205, 205, 198), 1)
        cv2.rectangle(group, (0, 0), (group.shape[1] - 1, GROUP_TITLE_HEIGHT), (35, 35, 33), -1)
        title = domain if len(chunks) == 1 else f"{domain}  {index + 1}/{len(chunks)}"
        cv2.putText(
            group,
            title,
            (GROUP_PAD, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        x = GROUP_PAD
        y = GROUP_TITLE_HEIGHT + GROUP_PAD
        for card in chunk:
            group[y:y + card.shape[0], x:x + card.shape[1]] = card
            x += card.shape[1] + CARD_GAP
        rendered.append(group)
    return rendered


def _pack_groups(groups: list[tuple[str, list[Patch]]]) -> list[np.ndarray]:
    rendered = [
        group
        for domain, patches in groups
        for group in _render_group(domain, patches)
    ]
    rows: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    current_width = 0
    for group in rendered:
        added = group.shape[1] + (GROUP_GAP if current else 0)
        if current and current_width + added > CANVAS_WIDTH:
            rows.append(current)
            current = []
            current_width = 0
            added = group.shape[1]
        current.append(group)
        current_width += added
    if current:
        rows.append(current)

    packed = []
    for row in rows:
        height = max(group.shape[0] for group in row)
        canvas = np.full((height, CANVAS_WIDTH, 3), BACKGROUND, np.uint8)
        x = 0
        for group in row:
            canvas[:group.shape[0], x:x + group.shape[1]] = group
            x += group.shape[1] + GROUP_GAP
        packed.append(canvas)
    return packed


def _section(title: str, subtitle: str, groups) -> list[np.ndarray]:
    header = np.full((SECTION_TITLE_HEIGHT, CANVAS_WIDTH, 3), BACKGROUND, np.uint8)
    cv2.putText(header, title, (0, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.82, TEXT, 2, cv2.LINE_AA)
    cv2.putText(header, subtitle, (210, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.47, MUTED, 1, cv2.LINE_AA)
    cv2.line(header, (0, 46), (CANVAS_WIDTH - 1, 46), (185, 185, 178), 1)
    return [header, *_pack_groups(groups)]


def _compact_section(title: str, subtitle: str, groups) -> np.ndarray:
    rendered = [
        group
        for domain, patches in groups
        for group in _render_group(domain, patches)
    ]
    width = sum(group.shape[1] for group in rendered) + GROUP_GAP * (len(rendered) - 1)
    body_height = max(group.shape[0] for group in rendered)
    panel = np.full((SECTION_TITLE_HEIGHT + body_height, width, 3), BACKGROUND, np.uint8)
    cv2.putText(panel, title, (0, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.82, TEXT, 2, cv2.LINE_AA)
    subtitle_x = min(width - 10, max(145, cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2
    )[0][0] + 34))
    _text_fit(
        panel,
        subtitle,
        (subtitle_x, 28),
        width - subtitle_x - 4,
        scale=0.47,
        color=MUTED,
        thickness=1,
    )
    cv2.line(panel, (0, 46), (width - 1, 46), (185, 185, 178), 1)
    x = 0
    for group in rendered:
        panel[
            SECTION_TITLE_HEIGHT:SECTION_TITLE_HEIGHT + group.shape[0],
            x:x + group.shape[1],
        ] = group
        x += group.shape[1] + GROUP_GAP
    return panel


def generate_guide(output_path: Path) -> Path:
    title = np.full((102, CANVAS_WIDTH, 3), BACKGROUND, np.uint8)
    cv2.putText(
        title,
        "VESUVIUS SCROLL PATCH GUIDE",
        (0, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.08,
        TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        title,
        "black: 2.4um label     red: aligned inklabel at 33% opacity     common preview height",
        (0, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(title, (0, 80), (32, 96), (0, 0, 0), -1)
    cv2.rectangle(title, (192, 80), (224, 96), ACCENT, -1)
    cv2.putText(title, "2.4um", (42, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA)
    cv2.putText(title, "aligned overlay", (234, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT, 1, cv2.LINE_AA)

    bands = [
        title,
        *_section("TRAINING", "29 labeled fragments grouped by physical scroll", TRAINING_GROUPS),
        *_section(
            "TEST",
            "nine unlabeled discovery surfaces; papyrus masks, areas, and source patches shown",
            TEST_GROUPS,
        ),
    ]
    divider = np.full((16, CANVAS_WIDTH, 3), BACKGROUND, np.uint8)
    canvas = np.vstack([
        band
        for index, item in enumerate(bands)
        for band in ((divider, item) if index else (item,))
    ])
    target_height = CANVAS_WIDTH * 9 // 16
    if canvas.shape[0] > target_height:
        raise RuntimeError(
            f"guide content is too tall for 16:9: {canvas.shape[1]}x{canvas.shape[0]}"
        )
    pad_top = (target_height - canvas.shape[0]) // 2
    pad_bottom = target_height - canvas.shape[0] - pad_top
    canvas = cv2.copyMakeBorder(
        canvas,
        pad_top,
        pad_bottom,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=BACKGROUND,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError(f"could not write {output_path}")
    print(f"wrote {output_path} ({canvas.shape[1]}x{canvas.shape[0]})")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "scroll_patch_guide.png",
    )
    args = parser.parse_args()
    generate_guide(args.output)


if __name__ == "__main__":
    main()