# CDSM 英文论文稿

论文题目为 **CDSM: Budgeted Multi-Entry Search for Reducing Severe Recall Failures**。交付稿限制为 **12 页，包含正文、附录和参考文献**，作者留空。主贡献仍是 **CDSM 搜索方法**：备用入口先进行有限局部探索，再将未完成搜索状态反馈到主前沿，改变剩余预算下的展开方向。各阶段共享距离缓存、访问状态和邻接扫描进度，并遵守同一个查询预算。最新补强及完成证据见 `R2-REVISION-RESPONSE.zh-CN.md` 与 `QA.json`。

主方法使用原始四 scout 策略 **F**。付费入口选择与后续调度变体作为扩展单独报告，不将扩展结果归到原始 F 名下。论文同时保留 T2I 上的正面结果，以及 LAION、WebVid 上的迁移边界；后两者不支持“跨数据集普遍提高平均召回率”的结论。直接续搜 C、共享队列多入口搜索和同图移植的 FANNG 搜索各有明确身份，不能与原生数据库或原生索引的完整系统性能混为一谈。

本轮新增 **服务器 ANN 补测**：1/4/8/16 入口 MEP、开发选择的 MEP、从现有前沿启动的 scout，以及每路 200/800 的邻近深度。原 F 冻结，C/F 与新策略共同计时。历史查询不算新的独立验证。严格同计算量、C 多给 10% 计算的工作点和原生质量—时间背景分别报告；QPS 按各自波次解释，1.5% 是实用容差。旧的 41 次救援阈值分析和停止规则消融继续保留，未为本次论文重建在桌面端运行 ANN。

交付文件包括：

- `paper/main.tex`：英文论文入口；`paper/sections/` 为正文与附录，`paper/references.bib` 为参考文献。
- `paper/tables/`、`paper/figures/`：论文使用的表格和图。
- `build.ps1`：PowerShell 构建入口，不需要 `latexmk` 或 Perl。
- `build_tables.py`、`TABLE-SOURCES.json`：从冻结汇总生成表格的脚本，以及表格来源文件的 SHA-256 记录。
- `derive_revision_data.py`、`REVISION-DATA.json`：从已有记录复算命中分布、阈值敏感性、追加计算工作点及ABS成本。只用Python标准库，不执行ANN。
- `derive_r2_depth.py`、`R2-DEPTH-EVIDENCE.json`：整理旧 Java 深度开发记录；`derive_r2_native_workpoints.py`、`R2-NATIVE-WORKPOINTS.json`：保留原生与 F 的完整双向工作点匹配，包括未匹配情况。
- `source-data/cdsm-review-strengthening-20260922/`：最新服务器协议、查询身份、源代码、选参锁定、结果汇总、审计及本地独立复核。`RUNBOOK.zh-CN.md` 解释术语和复现条件。大体积逐查询原始记录另存于实验目录的 `server-evidence.tar.gz`，哈希和文件清单在 `SERVER-BUNDLE.json`；论文 ZIP 不重复装入这份原始归档。
- `source-data/`：可追溯材料副本，保留原 `research/` 目录下的相对路径。例如，原 `research/fanng-search-comparison-20260922/SUMMARY.json` 对应 `source-data/fanng-search-comparison-20260922/SUMMARY.json`。这些材料用于追溯源码、算法版本、冻结汇总与审计口径，不代表附带了全部向量数据或完整运行环境。
- `output/pdf/`：生成的论文 PDF；本地 `qa/` 保存排版检查材料。`QA.json` 记录最终 PDF 的文件名、实际页数和检查结果；源码 ZIP 不重复附带逐页预览图片和编译临时文件。

在本目录的 PowerShell 中运行：

```powershell
.\build.ps1
```

构建需要已安装的 `pdflatex`、`bibtex` 及其 LaTeX 依赖；重建论文不会启动 ANN 测评。默认使用附带图表。从冻结记录重新生成图表可运行 `.\build.ps1 -RegenerateAssets -Python python`；脚本只使用 Python 标准库，矢量图另需 PGFPlots/TikZ 和 standalone 宏包。`SOURCE-DATA.json` 为全部随包证据提供哈希，表格及修订分析另有来源记录。新增逐事件命中文件支持分布复算，但源码包仍不是完整向量、逐查询执行轨迹或可运行测评环境。

用户要求正文与附录合计不超过 12 页；本交付进一步将**参考文献也计入 12 页总上限**。最终页数由 `QA.json` 记录，本文不以中间编译版本的页数替代最终核验。随包的官方 `acmart.cls` 与 `pvldb.sty` 保持原样。

`package_delivery.py` 是可选的交付核验与打包脚本，另需 `pypdf`。它要求先重建PDF、将全部页面渲染为 `qa/final-page-XX.png` 并完成逐页目视检查，再传入 `--visual-review-confirmed`；该参数是人工检查声明，脚本本身不替代视觉审阅。`BASELINE-EDITION.json` 保留修订前的源数据与样式身份，便于检查历史材料未被改写。

作者按用户要求留空，出版元数据也有意留空，没有虚构作者、单位、卷期、DOI 或正式发表信息。这是一份可审阅、可继续修改的论文稿，**不是 camera-ready 版本**；采用官方样式并控制页数，不等于已经完成特定投稿轮次的全部合规检查。

GitHub 当前目录已清理早期代码包、旧审稿意见与已执行完的修订计划。当前论文 PDF、论文与图表源码、110 份冻结实验材料及其哈希均保持不变。源码 ZIP 按清理后的内容重新打包，新的成员清单见 `PACKAGE-MANIFEST.json`；旧内容仍可通过 Git 历史恢复。
