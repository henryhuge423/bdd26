# Stage report 8 / 第八阶段报告

本报告从 P27 开始，覆盖截至 2026-11-17 已完成的 P31。三个交付文件沿用第七阶段报告命名：`stage_report_8_en.pdf`（详细英文）、`stage_report_8_zh.pdf`（详细中文）、`summary_plain_zh.pdf`（中文导读）。同名 `.tex` 是可编辑的 LaTeX 源码；`assets/` 保存真实结果图，`figs/` 保存随文编译的 TikZ/PGFPlots 图表，`evidence/` 保存数值、坐标和校验记录。

On Windows, use existing XeLaTeX, CTeX/Fandol and PGFPlots and run `powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1` from this directory. The script compiles each included `.tex` file twice. Alternatively, run XeLaTeX twice on each of the three `.tex` files in this directory. Run `python verify_numbers.py` to check P27–P31 paired counts, the retained P31 inputs and object matches. `NUMBER_SOURCES.md` identifies the experiment records and figure provenance.

三份 PDF 已逐页渲染和审阅；校验结果见 `QA.json`，交付文件哈希见 `SHA256SUMS.txt`。实验的原始模型检查点和完整历史结果保留在既有项目结果目录。
