# E3 候选质量学习（dev 轮）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练低维(+支持度)候选质量模型，在 split1_seed1 冻结 fold2 候选池上重放 A0/A1/A2/A3/A4 五臂菜单，按 spec §5 预注册门判定"候选效用是否可学习"。

**Architecture:** `src/nucseg/candidate_quality/` 四个纯逻辑模块（features/support/model/menu）+ 一个 CLI `scripts/run_e3_quality.py`（镜像 `run_existence_ef.py` 的 Pool-sweep→select_fusion→硬断言模式）。训练数据 = `existence_audit_20261005_v2/records.npz` 的 split2 三种子行（fold1 OOF）；应用数据 = `e1_input_check_20261010_split1_seed1/records.npz`（4,084 个 fold2 候选，与 v2 记录逐列绑定断言）。

**Tech Stack:** Python 3.10, numpy 1.23.5, scikit-learn（已在环境，T 轮用过）, torch 2.5.1（一致性头）, pytest。

**Spec:** `docs/superpowers/specs/2026-10-11-e3-candidate-quality-design.md`（特征菜单/臂/判定值以 spec 为准；A2 语义按 spec §8 修订为 (floor,cap) 网格）。

## Global Constraints

- 环境 `~/.conda/envs/nuclei`（numpy 1.23.5 / torch 2.5.1）；sklearn 仅用已装版本，不新增依赖。
- 训练行只允许 pair ∈ {split2_seed19, split2_seed1, split2_seed2}（fold1）；fold2/fold3 行进入
  拟合 = 中止。端点运行 `ensure_dev_fold(2, 1)`。
- NpzFile 先物化再索引（`with np.load(p) as z: arr = z["inst"]`）。
- A1 行必须与冻结 P2 selection.json 的 selected 行硬断言（mPQ/bPQ/strict_dead/interior_matched/
  added，≤1e-12）：.4830831946/.6516348815/.1344824731/499/1515。
- 无训练分割器、无新分割推理（唯一 GPU = HoVer-Net split2 fold1 val 推理，Task 6）。
- 全部输入 SHA256 + 代码指纹写入输出；`--out` 存在即拒绝。
- 代码进 `src/nucseg/candidate_quality/` + `scripts/`；工件进 `runs/analysis/e3_quality_20261011/`。
- git：research 分支 `research/e3-e1t`，逐任务提交。

## Review Focus

- fold2/fold3 行混入训练（records.npz 含全部 9 pair）→ `select_training_rows` 必须白名单过滤
  且对未知 pair 报错，测试构造含 split1/split3 行的 fixture。
- 特征列含非有限值或 records 列缺失 → 中止而非 NaN 传播；测试注入 NaN。
- 支持度来源 run 排除遗漏（候选来自 split2_seedS 时该 seed 的 x2 席必须排除）→ 测试构造
  同图两候选分属不同 pair 的 fixture。
- A1 的 allowed 集与 fuse 原生过滤不一致（双路径漂移）→ A1 走 fuse 原生参数
  （max_area/min_prob/interior_only）+ allowed 只补 area≥30 地板，并断言 chosen==1515。
- apply 侧候选与冻结记录列绑定失败（image/id/pmax 顺序错位）→ 载入即与 v2 记录的
  split1_seed1 子集逐列断言。
  - 对应测试分置 Task 1–5（见各任务 Step 1）。

---

### Task 1: 特征矩阵 `src/nucseg/candidate_quality/features.py`

**Files:**
- Create: `src/nucseg/candidate_quality/__init__.py`（空）
- Create: `src/nucseg/candidate_quality/features.py`
- Test: `tests/test_candidate_quality.py`

**Interfaces:**
- Produces:
  - `TRAIN_PAIRS = ("split2_seed19", "split2_seed1", "split2_seed2")`
  - `FEATURE_NAMES = ["log_area","extent","circularity","dist","pmax","margin","border",
    "class0",...,"class4"]`（12 维）
  - `select_training_rows(rec: dict) -> dict`：rec = np.load(records.npz) 的字典视图；返回仅
    TRAIN_PAIRS 行的子字典（保持列顺序），遇到其它 pair 行不报错但必须全部排除；返回值
    附带断言 `set(pairs) ⊆ TRAIN_PAIRS`。
  - `design_matrix(rec: dict) -> tuple[np.ndarray, np.ndarray]`：返回 (X (N,12) float64,
    y (N,) bool=matched)。非有限值 → `ValueError`（含列名与行号）。

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest
from nucseg.candidate_quality.features import (
    TRAIN_PAIRS, FEATURE_NAMES, select_training_rows, design_matrix)


def _rec(pairs):
    n = len(pairs)
    return {"pair": np.array(pairs), "area": np.full(n, 120.0),
            "extent": np.full(n, .7), "circularity": np.full(n, .8), "dist": np.full(n, 5.0),
            "pmax": np.full(n, .95), "margin": np.full(n, .2), "border": np.zeros(n, bool),
            "class": np.zeros(n, np.int64), "image": np.arange(n), "id": np.ones(n, np.int64),
            "matched": np.array([True, False] * ((n + 1) // 2))[:n], "typed": np.ones(n, bool),
            "gt_class": np.zeros(n, np.int64), "gt_already_by_base": np.zeros(n, bool),
            "added": np.zeros(n, bool), "iou_best": np.zeros(n)}


def test_select_training_rows_filters_and_keeps_order():
    rec = _rec(["split2_seed19", "split1_seed1", "split3_seed19", "split2_seed1"])
    sub = select_training_rows(rec)
    assert list(sub["pair"]) == ["split2_seed19", "split2_seed1"]
    assert len(sub["area"]) == 2


def test_design_matrix_shape_and_columns():
    rec = select_training_rows(_rec(list(TRAIN_PAIRS)))
    X, y = design_matrix(rec)
    assert X.shape == (3, 12) and y.dtype == bool and list(y) == [True, False, True]
    assert FEATURE_NAMES[0] == "log_area" and np.allclose(X[:, 0], np.log1p(120.0))
    assert X[:, 6] == 0 and X[:, 7] == 1  # border, class0 one-hot


def test_design_matrix_rejects_nonfinite():
    rec = select_training_rows(_rec(list(TRAIN_PAIRS)))
    rec["pmax"] = rec["pmax"].copy(); rec["pmax"][1] = np.nan
    with pytest.raises(ValueError, match="pmax"):
        design_matrix(rec)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data7/jinxinhao/bdd26 && ~/.conda/envs/nuclei/bin/python -m pytest tests/test_candidate_quality.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**（`select_training_rows` 用布尔掩码逐列索引；`design_matrix` 按
FEATURE_NAMES 顺序 hstack，`np.isfinite` 全检后 raise；class one-hot 用
`np.eye(5)[rec["class"]]`——class 越界（>4/<0）同样 ValueError）

- [ ] **Step 4: 跑测试确认通过** — `pytest tests/test_candidate_quality.py -q` → 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/nucseg/candidate_quality/ tests/test_candidate_quality.py
git commit -m "feat(e3): candidate feature matrix with fold whitelist and finite guards"
```

---

### Task 2: 支持度 `src/nucseg/candidate_quality/support.py`

**Files:**
- Create: `src/nucseg/candidate_quality/support.py`
- Modify: `tests/test_candidate_quality.py`（追加）

**Interfaces:**
- Consumes: `nucseg.metrics.instance.overlap`；records 的 pair/image/id 列；jury npz（fold 级
  堆叠 inst 数组，布局同 run_existence_ef）。
- Produces:
  - `jury_paths_fold1(round_root: Path, hn_fold1: Path | None) -> list[dict]`：6 席
    `round/p0/split2_seed{19,1,2}/val/{base,x2}/identity/pred.npz` +（可选）HN 1 席；
    每席 `{"name","path","source_pair"|"None","train_fold":2}`；x2 席带
    `source_pair="split2_seedS"`。
  - `support_matrix(rec: dict, jury: list[dict], thr=0.3) -> np.ndarray (N, J) uint8`：
    逐图（按 records 的 image 列分组）取候选 mask（来自该候选来源 pair 的 x2 val npz
    `p0/<pair>/val/x2/identity/pred.npz` 的 inst[img]==id）与各非排除席的 inst[img] 求
    最大 IoU ≥ thr → 1。候选来源席（同 pair 的 x2 席）记 0 并在返回的附带 mask 中标记
    `excluded`（实现为同时返回 `(matrix, excluded_bool)`，被排除席置 0）。
  - `effective_denominator(excluded: np.ndarray) -> np.ndarray (N,)`：每候选有效席数 =
    J − excluded 行和；`support_frac = support / effective_denominator`（调用方算）。

- [ ] **Step 1: 写失败测试**（合成 16×16 双图 fixture：候选来自 split2_seed19 时，
  split2_seed19 的 x2 席被排除、split2_seed1 的 x2 席计入；HN 席加入后 J=7；IoU 阈值
  边界 0.3 处的支持翻转——用重叠 9/16=0.5625 与 4/16=0.25 两个手工情形）

```python
def test_support_matrix_source_exclusion_and_thr(tmp_path):
    # 构造 records 2 行（同图，分属 split2_seed19 与 split2_seed1）+ 3 个 jury npz
    #（split2_seed19-x2 与候选 1 完全重合、split2_seed1-x2 与候选 2 完全重合、
    #  base 席与两候选各重叠 0.25）
    # 断言: matrix == [[0, 1, 0], [1, 0, 0]]（列序 [s19-x2, s1-x2, s19-base]）
    #       excluded == [[True, False, False], [False, True, False]]
    ...
```
（fixture 具体数组在实现时写全；另测 `jury_paths_fold1` 的路径枚举与 HN 可选席。）

- [ ] **Step 2: 确认失败** — `pytest tests/test_candidate_quality.py -q` → 新增用例 FAIL
- [ ] **Step 3: 实现**（每图一次性 `overlap` 候选×jury？不行——overlap 需要双标签图；
  改为对每候选 mask 单独对每个 jury inst 图调 overlap（与 e1_consensus.support_for_candidate
  同款循环）；先物化全部 npz 的 inst 堆叠数组）
- [ ] **Step 4: 确认通过** — 全文件 PASS
- [ ] **Step 5: Commit** — `git commit -am "feat(e3): fold1 jury support matrix with per-candidate source exclusion"`

---

### Task 3: 模型 `src/nucseg/candidate_quality/model.py`

**Files:**
- Create: `src/nucseg/candidate_quality/model.py`
- Modify: `tests/test_candidate_quality.py`（追加）

**Interfaces:**
- Produces:
  - `fit_logistic(X, y, groups, Cs=(0.1, 1.0, 10.0)) -> dict`：StandardScaler +
    LogisticRegression(max_iter=5000)；C 由 GroupKFold(5)（groups=image id）平均 AUC 选，
    平票取小 C；返回 {"scaler", "clf", "C", "cv_auc"}。
  - `predict(fit, X) -> np.ndarray (N,)`：概率。
  - `reliability(y_true, p, bins=10) -> list[dict]`（bin, n, mean_p, rate）。
  - `torch_head_directions(X, y, X_apply, seeds=(0, 1)) -> list[np.ndarray]`：标准化后
    torch 线性头各训练 200 epoch（Adam lr=0.05，BCE），返回 apply 概率；仅用于方向一致性
    报告，不进门。
- [ ] **Step 1: 失败测试**：合成可分数据（两类各 100 点，加噪）断言 `predict` AUC>0.9、
  `cv_auc` 有限、`reliability` 各 bin n 求和=N、`torch_head_directions` 两种子在 apply 上
  Spearman>0.8（scipy 已随 sklearn 在环境）。
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 确认通过**
- [ ] **Step 5: Commit** — `git commit -am "feat(e3): logistic quality model with grouped CV and torch direction check"`

---

### Task 4: 菜单与 A1 绑定 `src/nucseg/candidate_quality/menu.py`

**Files:**
- Create: `src/nucseg/candidate_quality/menu.py`
- Modify: `tests/test_candidate_quality.py`（追加）

**Interfaces:**
- Consumes: `run_existence_ef` 的 stage1 结构（p2 selection.json `selected`：
  `{"name","max_area","min_prob","interior_only",...}`）。
- Produces:
  - `a1_allowed(rec_apply: dict, p2cfg: dict, ef_floor=30) -> set[tuple[int,int]]`：
    (image,id) 集 = rows 满足 `area<=max_area and pmax>=min_prob and
    (not interior_only or not border) and area>=ef_floor`；返回集合与计数。
  - `build_menu(p_hat_low, p_hat_sup, rec_apply) -> list[dict]`：18 行
    `[{"name":"identity","arm":"A0"}, {"name":"a1_incumbent","arm":"A1"},
      {"name":f"a2_f{f}_c{c}","arm":"A2"} ×6 (f∈{0,30}, c∈{100,200,400}),
      {"name":f"a3_p{t}","arm":"A3"} ×5, {"name":f"a4_p{t}","arm":"A4"} ×5]`，
    每行带 `allowed: set[(image,id)]`（A3/A4 按 p̂ 阈值；A2 按 floor/cap）。
  - `selection_rows(sweep_rows) -> dict[arm, row]`：对 A2/A3/A4 分别调
    `scale_fusion.select_fusion([A0row] + arm_rows)`（identity 参照 = A0 行）；A1/A0 固定。
- [ ] **Step 1: 失败测试**：合成 rec_apply（含 area 20/50/150/250、pmax .85/.95、border
  True/False）手工推演 a1_allowed（max_area=200, min_prob=.9, floor=30 → 恰 1 个 id）；
  build_menu 行数=18、A3/A4 阈值边界（p̂ 恰等于阈值 → 保留，闭区间）；selection_rows 用
  伪造 sweep 行验证 select_fusion 复用语义（gate 不过回落 identity）。
- [ ] **Step 2–4: RED→实现→GREEN**（实现直接照 Interfaces；A1 的 allowed 与 fuse 原生
  参数一致性由 Task 5 的 1515 断言兜底）
- [ ] **Step 5: Commit** — `git commit -am "feat(e3): five-arm menu with incumbent binding"`

---

### Task 5: CLI `scripts/run_e3_quality.py`

**Files:**
- Create: `scripts/run_e3_quality.py`
- Test: `tests/test_e3_runner.py`

**Interfaces:**
- Consumes: Task 1–4 全部；`scale_fusion.fuse/probability_rows/select_fusion/strict_dead_pq`；
  `nucseg.metrics.light.summarize`；`matched_seed_stats.bootstrap`（sys.path 注入 scripts）；
  `fold_guard.ensure_dev_fold`；`run_p0_controls` 的 provenance 工具（同 EF runner 用法）。
- Produces: `runs/analysis/e3_quality_20261011/{results.json, tables.md, pred/}`；
  results.json 关键结构：`{"rows": [18 行 sweep 汇总], "selected": {arm: row},
  "gates": {"primary": bool per arm, "secondary": bool per arm}, "bootstrap": {...},
  "model": {"C","cv_auc","coef","torch_directions"}, "diagnostics":
  {"reliability_low","reliability_sup","utility_vs_q2": [...], "added_class_mix": {...}},
  "provenance": {...}}`。
- 命令行（冻结示例）：
```bash
~/.conda/envs/nuclei/bin/python scripts/run_e3_quality.py \
  --round runs/analysis/matched_seed_20261004 --pair split1_seed1 \
  --records runs/analysis/existence_audit_20261005_v2/records.npz \
  --apply-records runs/analysis/e1_input_check_20261010_split1_seed1/records.npz \
  --base-val runs/analysis/matched_seed_20261004/p0/split1_seed1/val/base/ms/pred.npz \
  --x2-val runs/analysis/matched_seed_20261004/p0/split1_seed1/val/x2/identity/pred.npz \
  --p2-selection runs/analysis/matched_seed_20261004/p2/split1_seed1/selection.json \
  --fold2-support runs/analysis/e1_consensus_20261010/per_candidate.npz \
  --hn-fold1 runs/analysis/e3_quality_20261011/hovernet_split2_fold1/pred.npz \
  --out runs/analysis/e3_quality_20261011 --workers 8
```
- 主流程：ensure_dev_fold(2,1) → 载入并逐列断言 apply-records == v2 records 的
  split1_seed1 子集 → 训练（fold1 行 + fold1 support）→ 两模型 p̂ → build_menu →
  物化 base/x2 npz → Pool 逐图 sweep 18 行（fuse 原生参数走 A1，其余 allowed_ids）→
  rows 汇总 → A1 硬断言（见 Global Constraints）→ selection_rows → gates（spec §5 数字）→
  bootstrap（ΔDead/ΔbPQ/ΔmPQ vs A1，seed 20261011，tissue 来自 PanNukeFold(2).tissue）→
  诊断（reliability、utility_vs_q2、类别构成）→ 写出。
- 关键实现细节：
  - per-image Dead PQ：由 light.image_stats 的每图 stats 数组按 `strict_dead_pq` 同款
    `_pq` 计算（数组列语义 = 该函数已有用法）。
  - `utility_vs_q2`：按类汇总 fold2 候选的 `p̂ × (iou_best if matched else 0)` 均值 vs
    该类 A0 行 PQ/2（诊断表，不进门）。
  - torch 一致性：两种子 p̂ 的符号方向（vs sklearn p̂ 的 Spearman）记录进 model 段。

- [ ] **Step 1: 失败测试**（合成微型 fold：4 图、每图 1-2 候选，伪造 jury/模型注入——
  通过 monkeypatch `candidate_quality.model.fit_logistic` 返回固定 p̂——断言：18 行 sweep
  计数正确、A1 断言失败路径 SystemExit、gates 数字代入（构造 ΔDead=+.003 → 主门 true；
  ΔDead=0 且 unmatched 80% → 次门 false）、fold-guard 拒绝 fold3）
- [ ] **Step 2: 确认失败**
- [ ] **Step 3: 实现**（照 EF runner 骨架：_G 全局 + `_sweep(j)` + Pool.map；先在微型
  fixture 上过测试再上真实数据）
- [ ] **Step 4: 确认通过 + 全量回归** — `python -m pytest -q` 无回归
- [ ] **Step 5: Commit** — `git commit -am "feat(e3): five-arm sweep runner with incumbent assert and pre-registered gates"`

---

### Task 6: HoVer-Net split2 fold1（val）推理（唯一 GPU）

- [ ] **Step 1: 运行**（fold1 是 split2 的验证折，合法；不触 test）：
```bash
~/.conda/envs/nuclei/bin/python scripts/predict_hovernet.py \
  --ckpt runs/hovernet/split2 --fold 1 \
  --out runs/analysis/e3_quality_20261011/hovernet_split2_fold1
```
- [ ] **Step 2: sanity**：val 上 mPQ 应与 split2 历史侧指标同量级（E1-C 的 hn fold2
  sanity .4509 为参照做法）；记录进 provenance。
- [ ] **Step 3: Commit**（仅记录，预测 npz 在 runs/ 不跟踪）

---

### Task 7: 真实执行 + 判定 + findings（执行任务）

- [ ] **Step 1: Task 6 完成后运行 Task 5 冻结命令**（预计 CPU 数小时；输出 results.json）
- [ ] **Step 2: 核验**
  - A1 断言通过（1515/.48308/.65163/499/.13448 全对上）；
  - 抽 1 图手工核对某 A3 行的 allowed 集 == 该图 p̂≥阈值的候选 id 集；
  - bootstrap 与点估计同脚本重跑 bit-exact（`--out` 换验证目录重放一次对比）。
- [ ] **Step 3: 按 spec §5 写判定**（主门/次门/停线三选一；E1-T 同日条目互引）；
  findings 新条目 + README/RESEARCH_PLAN 状态段更新（E1 人类评审不可得声明 + E3/E1-T 判定）。
- [ ] **Step 4: Commit + 汇报**
```bash
git add docs/findings.md README.md docs/RESEARCH_PLAN.md
git commit -m "docs: E3 candidate-quality dev verdict and E1-T closure; E1 human-review unavailability recorded"
```

## Self-Review 结论

- Spec 覆盖：§1 划分→Task 1/5；§2 特征→Task 1/2；§3 模型→Task 3；§4 臂→Task 4/5；§5 判定→Task 5/7；
  §6 护栏→Global Constraints + 各 assert；§7 成本→Task 6（唯一 GPU）。无缺口。
- 类型一致：`design_matrix→(X,y)`、`support_matrix→(matrix,excluded)`、`fit_logistic→dict`、
  `build_menu→rows(name,arm,allowed)`、`selection_rows→{arm:row}` 各任务间签名一致。
- Review Focus 五项均有对应测试（Task 1 Step1 ×2、Task 2 Step1 ×2、Task 4 Step1、Task 5 Step1）。
