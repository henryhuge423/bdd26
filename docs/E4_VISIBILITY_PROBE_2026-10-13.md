# E4 视野裁剪可见性机制探针结果（2026-10-13 spec，2026-10-14 执行）

[Spec](superpowers/specs/2026-10-13-visibility-probe-e4-design.md)（用户 2026-10-13 批准）｜
[Plan](superpowers/plans/2026-10-13-visibility-probe-e4.md)｜工件
`runs/analysis/visibility_probe_20261013/`（probe.json / tables.md / commands.md，
provenance：final.pth、pred_fold2.npz、eval_fold2 summary 的 SHA256 与代码指纹）。
冻结检查点 split1 seed19（x1、无 TTA、解码常量不动）上的只读机制探针；fold2（split1
验证折，2523 图）零训练零选择；门数字运行前冻结。

## 判定（预注册键）：三门全过 ⇒ 允许起草可见性监督训练轮 spec（另行立项）

| 门 | 数字（图像聚类 bootstrap 1000 次，组织内分层，95% CI） | 冻结线 | 判定 |
|---|---|---|---|
| H1 贴边−内部（Dead） | **+.2913 [+0.1997, +0.3847]**（贴边 .531 n=143 vs 内部 .240 n=741） | ≥+.05 且 lo>0 | **PASS** |
| H2 DiD(C2) | **+.1445 [+0.1367, +0.1532]**；DiD(C-int) −.0043；DiD(C3) +.1435 | ≥+.03、lo>0、\|C-int\|<C2 | **PASS** |
| H3 碎片面积 <30 vs ≥100 px | **+.8079 [+0.7848, +0.8298]**（.851 n=1134 vs .043 n=8643） | ≥+.05 且 lo>0 | **PASS** |

C0 绑定：恒等复评 == eval_fold2 官方 mPQ（逐位，.4835）。窗口 offsets 与
`sample_windows(seed=20261013)` 逐位一致（复核通过）。门判定从 probe.json 独立复算一致。

## 机制读数（次端点，描述性）

- **截断的因果效应**（H2 内部）：新截断核 miss .043→.171（C2）/→.160（C3），同窗保持完整核
  .060→.043/.037（对照甚至略降）；插值地板 C-int 的 DiD −.0043 ≈ 0——效应是 FOV 移除
  本身，不是插值/相位（C1 整体 miss .112 ≈ C0 .116，mPQ .4616 vs .4835）。
- **剂量响应**：DiD(C3 1.6×) +.1435 ≈ DiD(C2 1.333×) +.1445（更大的窗移除更多上下文但
  每核截断概率下降，两者相抵）。
- **面积梯度**（H3）：截断碎片 miss <30px .851 / 30–100px .367 / ≥100px .043——
  与训练管线 `hv_targets` 对 <30 工作像素清零 HV 监督的机制假设一致（碎片越小越不可见）。
- crop-frame mPQ/bPQ：C2 .4460/.6052、C3 .4211/.5545（vs C0 .4835/.6531；zoom 使解码
  常量相对组织缩小，如实披露，DiD 内部对照共享同一解码）。HV 诊断：截断片段内
  |pred−ideal| 均值 C2 .175 / C3 .185（仅诊断）。matched 类型正确率 C0 .856 → C2 .809。

## 限制（spec §8 预注册）

- 截断与窗边位置不可分离；配对设计 + C-int 校准插值/相位，不校准位置本身。
- C2/C3 的 zoom 与 FOV 移除共变，DiD（同窗保持完整对照）分离放大效应；C3 为剂量点。
- 单检查点（split1 seed19）机制证据；种子/训练随机性不在本轮。
- fold2 同时是 T 轮验证折——本轮只读分析、无训练无选择，披露即可。
- 过门 ≠ 训练轮会成功；训练轮 spec 另行预注册（对照至少含 crop-augmentation 加强版
  与原基线），含成本与停止规则。

## 执行补记

首轮运行暴露三个实现 bug（均已修复并全量重跑）：C1 secondary 的 `shift_pad` 对
(n,256,256) 栈移错轴（图像轴而非行轴）；`crop_group` 输出 `full` 而 H2 过滤器用
`stays_full`（保持完整组曾整组丢失）；tables 打印旧键名。`--limit N` 为调试 shim
（跳过绑定，仅 scratch 目录）。
