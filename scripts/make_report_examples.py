"""Rebuild the two montage example images of stage report I at readable sizes.

- figs/dead_missed.jpg: qualitative Dead-detection crops (CellViT-UNI, fold 3).
  Sources were lost, so tiles are re-sliced from the previous 8-column montage
  (committed before this change): 3 groups x (20px strip + 2 rows x 192px tiles),
  1536x1212. We keep 4 tiles per group (1 row), drop the tiny baked per-tile
  labels (top 18px), upscale x2, and redraw the group strips at ~11pt effective.
  Group strips verified against the decomposition table: Dead GT on fold 3 is
  1057 = 407/.385 (missed, no overlap) = 497/.470 (matched).
- figs/cf_arms.jpg: PanNuke-CF example grid. Cells are 516x516 PNGs with the GT
  outlines baked in; sources kept under figs/src_cf_cells/. Previous grid used
  6 rows x 3 cols (rows 5-6 contain failed/blank generations) -> keep rows 1-4,
  native resolution (no downscale, unlike the previous 259px cells).

Display sizes: dead_missed at 0.86\linewidth (~5.4in) -> 289 ppi, strips 44px
= 11pt; cf_arms at 0.7\linewidth (~4.4in) -> 355 ppi. Run from repo root:
~/.conda/envs/nuclei/bin/python3 scripts/make_report_examples.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "report" / "stage_report_1" / "figs"
BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)

GAP = 8
STRIP = 60  # group title strip height (px)


def build_dead_missed():
    old = Image.open(FIGS / "dead_missed.jpg")  # 1536x1212; fully read before overwrite
    # (group, row, col) of the 4 kept tiles per group, in display order
    picks = [
        [(1, 1, 3), (1, 2, 1), (1, 2, 8), (1, 2, 6)],   # Dead missed: 218/174/138/33 px
        [(2, 1, 4), (2, 1, 5), (2, 2, 2), (2, 1, 6)],   # Dead matched: 347/155/132/139 px
        [(3, 1, 2), (3, 2, 4), (3, 2, 6), (3, 2, 1)],   # other classes missed: Neop/Conn/Conn/Infl
    ]
    strips = ["Dead missed (n=407)", "Dead matched (n=497)", "other classes missed (n=6808)"]

    def tile(g, r, c):
        y = (g - 1) * 404 + 20 + (r - 1) * 192
        t = old.crop(((c - 1) * 192, y, c * 192, y + 192))
        t = t.crop((0, 18, 192, 192))  # drop baked per-tile label
        return t.resize((384, 348), Image.LANCZOS)

    tw = 4 * 384 + 3 * GAP
    out = Image.new("RGB", (tw, 3 * (STRIP + 348) + 2 * GAP), "white")
    dr = ImageDraw.Draw(out)
    y = 0
    for group, text in zip(picks, strips):
        dr.rectangle([0, y, tw, y + STRIP], fill="black")
        dr.text((12, y + 6), text, font=BOLD, fill="white")
        y += STRIP
        for i, (g, r, c) in enumerate(group):
            out.paste(tile(g, r, c), (i * (384 + GAP), y))
        y += 348 + GAP
    out.save(FIGS / "dead_missed.jpg", quality=92)
    print("dead_missed.jpg", out.size)


def build_cf_arms():
    cells = FIGS / "src_cf_cells"
    cell = Image.open(cells / "r1c1.png")
    s = cell.size[0]  # 516
    tw, th = 3 * s + 2 * GAP, 4 * s + 3 * GAP
    out = Image.new("RGB", (tw, th), "white")
    for r in range(1, 5):
        for c in range(1, 4):
            out.paste(Image.open(cells / f"r{r}c{c}.png"), ((c - 1) * (s + GAP), (r - 1) * (s + GAP)))
    out.save(FIGS / "cf_arms.jpg", quality=92)
    print("cf_arms.jpg", out.size)


if __name__ == "__main__":
    build_dead_missed()
    build_cf_arms()
