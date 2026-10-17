# E15：单点条件掩码读出未通过拟合继续条件

2026-10-17。该准确配方关闭，不加步数、宽度或阈值菜单；没有方法质量或扩量通过。它使用GT内部点与实例身份，是掩码读出诊断，不是检测器。

冻结原E7Q模型及掩码特征，使用合法点处64通道特征+点xy，通过共享66→256→256→133的MLP预测掩码向量和相对二次空间项。沿用E12的固定16张训练图；仅8fit图优化300步，8audit图本轮不优化，但E7见过两组，均非独立验证/测试。每步一fit图循环，AdamW3e−4、wd0，原5BCE+5Dice，GT对象分块16。无分类器或预测点输入，无保存权重。

| paired raw-mask覆盖（IoU>.5，%） | seed19 fit | seed1 fit | seed19 audit | seed1 audit |
|---|---:|---:|---:|---:|
| 全体 | 79.08 | 82.14 | 68.77 | 73.12 |
| Dead | 77.63 | 81.58 | 31.58 | 42.11 |

fit为196实例/76Dead，audit为253实例/19Dead。预注册继续条件要求两种子fit全体与Dead均至少80%，seed19失败；不将接近门槛解释为通过。下一项E16是预先记录的新输入表示：固定3×3邻域局部描述子，原几何公式和损失不变；输入层参数量随之增加，不能单独归因为上下文。

独立脚本重算每个对象的初始/最终覆盖与均值，核对GT身份、原检查点、图组和E12冻结特征基哈希，全部通过。归一化只用fit点描述子，保存的均值/std维度与下界通过检查；没有声称独立重建未保存的原始描述子统计。模型梯度为空、输入与特征哈希训练后不变。TDD3RED→3GREEN，LM2同样3项通过；E14/E15便携测试7通过。未改指标/HoVer-Net，未重跑完整套件。

GPU exit0，42.45秒，PyTorch峰值reserved1.4102GiB，LM2GPU5。无LM1GPU、新SSH、依赖升级、Git提交/发布。历史完成控制PID908484、startticks1066282564，仅供溯源。主分支`research/set-mask-e6`/`fd98c5f0a7618ed2e1cd454bfbc4dda36a336a3b`；隔离实现源`research/set-mask-e7-schedule`/`c6d9b58ff080a3686ea760c05c533a2f62f51ac8`。

结果SHA256=`837d04404cc07465b494ee7b168bb3b37ab87243a37d2d47f93950aedcc9a06d`，driver=`cf8cb1802c9dbf464d778782bef4e1e71249a27284f8cf836474a2354f6c6448`，设计=`1cc303c9222dcb68c0ef4f7d1a876545b41170a4860b243f3e1ae0c5e90552fd`。LM2原始`/data6/jinxinhao/bdd26/ops/e15_point_20261017`，LM1回收同名ops目录。配套：[数据](results.json)、[独立核算](independent_verification.json)。

方法动机参考 [Objects as Points](https://arxiv.org/abs/1904.07850) 与 [CondInst](https://arxiv.org/abs/2003.05664)，本原型不是其完整复现。
