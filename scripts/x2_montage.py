"""B1 x2 split1 (TTA): 2-row base-vs-x2 overlay montages for the three error mechanisms.

Columns are cases; row 1 = baseline s19 predictions, row 2 = x2 predictions; GT nucleus
contour is green in both, predicted instances yellow (one contour per instance id, so
over-segmentation is visible as multiple yellow rings inside one green ring).
Groups: A large-nuclei damage (nonDead, area>=1600, base matched -> x2 split/merged/missed),
B Dead/small gains (Dead, area<400, base missed_bg -> x2 matched), C small FP fragments
(x2 fp_bg preds with area<80, no GT in the crop).
"""
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from nucseg.data.pannuke import PanNukeFold  # noqa: E402

BASE = ("runs/cellvit_uni/split1/eval_test_fold3_tta/gt_records.csv.gz",
        "runs/cellvit_uni/split1/pred_test_fold3_tta.npz")
X2 = ("runs/cellvit_uni_x2/split1/eval_test_fold3_tta/gt_records.csv.gz",
      "runs/cellvit_uni_x2/split1/pred_test_fold3_tta.npz")
HALF, S = 48, 3

f = PanNukeFold(3)
pb = np.load(BASE[1])["inst"]
px = np.load(X2[1])["inst"]
gb, gx = pd.read_csv(BASE[0]), pd.read_csv(X2[0])
for g in (gb, gx):
    g["idx"] = g.groupby("image").cumcount()
mm = gb[["image", "idx", "cls", "area", "status"]].merge(
    gx[["image", "idx", "status"]], on=["image", "idx"], suffixes=("_b", "_x"))
px2 = pd.read_csv(X2[0].replace("gt_records", "pred_records"))


def gt_mask(j, idx):
    lab = np.asarray(f.inst[j])
    return lab == idx + 1  # evaluator relabel order == cumcount order


def render(j, cy, cx, pred, gtm):
    y0, x0 = np.clip(cy - HALF, 0, 256 - 2 * HALF), np.clip(cx - HALF, 0, 256 - 2 * HALF)
    sl = (slice(y0, y0 + 2 * HALF), slice(x0, x0 + 2 * HALF))
    img = cv2.resize(np.asarray(f.images[j])[sl], None, fx=S, fy=S,
                     interpolation=cv2.INTER_CUBIC)[..., ::-1].copy()

    def draw(mask, color, w):
        up = cv2.resize(mask.astype(np.uint8), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, color, w)

    if gtm is not None and gtm[sl].any():
        draw(gtm[sl], (0, 255, 0), 2)
    for i in np.unique(pred[j][sl]):
        if i:
            draw((pred[j] == i)[sl], (0, 225, 255), 2)
    return img


def caption(txt, w):
    bar = np.full((34, w, 3), 24, np.uint8)
    cv2.putText(bar, txt, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def montage(cases, out, title):
    cols = [np.vstack([caption(f"img {j} [{cap}]", (2 * HALF) * S), render(j, cy, cx, pb, gtm),
                       caption("x2 (2x resolution)", (2 * HALF) * S), render(j, cy, cx, px, gtm)])
            for j, cy, cx, gtm, cap in cases]
    head = caption(title, cols[0].shape[1] * len(cols))
    cv2.imwrite(str(out), np.vstack([head, np.hstack(cols)]))
    print("wrote", out, len(cases), "cases")


rng = np.random.default_rng(0)
# A: large-nuclei damage
a = mm[(mm.cls != 4) & (mm.area >= 1600) & (mm.status_b == "matched")
       & (mm.status_x.isin(["split", "merged", "missed_shape"]))]
casesA = []
for _, r in a.sample(min(12, len(a)), random_state=0).iterrows():
    m = gt_mask(int(r.image), int(r.idx))
    ys, xs = np.nonzero(m)
    casesA.append((int(r.image), int(ys.mean()), int(xs.mean()), m, f"{r.status_x},A={int(r.area)}"))
montage(casesA, "runs/analysis/x2_montage_damage.png",
        "A: large nuclei (>=1600px) base-matched -> x2 split/merged/missed (green=GT, yellow=preds)")

# B: Dead gains
b = mm[(mm.cls == 4) & (mm.area < 400) & (mm.status_b == "missed_bg") & (mm.status_x == "matched")]
casesB = []
for _, r in b.sample(min(12, len(b)), random_state=0).iterrows():
    m = gt_mask(int(r.image), int(r.idx))
    ys, xs = np.nonzero(m)
    casesB.append((int(r.image), int(ys.mean()), int(xs.mean()), m, f"Dead A={int(r.area)}"))
montage(casesB, "runs/analysis/x2_montage_gain.png",
        "B: small Dead base-missed -> x2 matched (green=GT, yellow=preds)")

# C: tiny FP fragments in x2 (fp_bg, area<80): centre on the fragment, no GT expected
c = px2[(px2.status == "fp_bg") & (px2.area < 80)]
casesC = []
for _, r in c.sample(min(40, len(c)), random_state=0).iterrows():
    j = int(r.image)
    n, lab = cv2.connectedComponents((px[j] > 0).astype(np.uint8), 8)
    best, best_da = None, 1e9
    for i in range(1, n):
        m = lab == i
        da = abs(int(m.sum()) - int(r.area))
        if da < best_da:
            best, best_da = m, da
    if best is not None and best_da <= 40:  # record is in native px, same as component
        ys, xs = np.nonzero(best)
        casesC.append((j, int(ys.mean()), int(xs.mean()), None, f"frag A={int(r.area)}"))
    if len(casesC) >= 12:
        break
montage(casesC, "runs/analysis/x2_montage_fragments.png",
        "C: tiny fp_bg fragments only in x2 (yellow=preds, no GT in view)")
