# 最终统一版：经皮皮肤—肝脏二维加权蒙特卡洛研究

本文件夹是**源码包**，不是模拟结果包。所有数值必须在你自己的电脑上重新生成。不要把 `results_smoke` 的数字放入论文。

## 0. 安装

推荐 Python 3.10 / 3.11，Windows 下新建或激活你现有的 Python 环境。在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

解压时保持以下结构（整个 `MC_final_unified` 文件夹可以放到任意位置，不需要修改硬编码的 D: 路径）：

```text
MC_final_unified/
├── src/
│   ├── run_final_paper.py                  # 唯一必运行入口
│   ├── research_delivery_pipeline_v2.py   # 数据、特征、E11、评估函数
│   ├── skin_liver_fibrosis_mc.py           # 光学参数；连续 severity 修正
│   ├── mc_liver_fast.py                    # numba 主模拟，count+weighted 双评分
│   ├── validate_fast_core.py              # 物理及实现检查
│   └── make_paper_figures_tables.py        # 五图 + Table2 + S2，只后处理
├── requirements.txt
└── README_运行说明.md
```

## 1. 先运行一次快速检查（建议）

在 `src` 目录的终端里：

```powershell
python run_final_paper.py --quick
```

生成到 `../results_smoke/`：24 synthetic subjects，每波长 3,000 packets，3 次 train/test split，E11 仅 1 次。**不能用于论文**。

如果完成，你会看到 `COMPLETE`，并能在 `results_smoke/figures/` 看到 Figure 1–5。

## 2. 正式重跑（唯一必运行命令）

```powershell
python run_final_paper.py
```

这条命令会依次：

1. 进行 4 项 MC 检查：快/慢核心一致性、重复性、能量守恒、weighted-scorer 一致性，失败立即终止。
2. **只运行一次** 240 个合成个体 × 5 个波长 × 每波长每人 200,000 photons 的主 weighted MC；保存唯一 `main_cohort.npz`。
3. 使用同一数据推导 feature；做 30 次分层 70/30 主分类比较、统计对照。
4. 用同一特征和 cohort 做 30 次 E11（训练集内穷举 31 个波长子集 + SDS forward selection + one-SE）；无需第二次 Monte Carlo。
5. 自动出 Figure 1–5、weighted Table 2、Supplementary S2A / S2B 和 50 行的 `Figure_5_values.csv`。

正式结果位于 `../results_final/`。运行时间与 CPU、Numba 首次编译、E11 的嵌套交叉验证次数有关，可能比最初预估的单次 MC 时间长。

## 3. 所有输出在一个目录下

```text
results_final/
├── tables/
│   ├── main_cohort.npz                    # 唯一原始 weighted cohort
│   ├── cohort_features.csv
│   ├── main_comparison.csv                # 原始完整特征组比较
│   ├── statistical_tests.csv
│   ├── channel_optimization_splits.csv    # E11 每次选中波长/SDS 和测试结果
│   ├── channel_optimization_summary.csv
│   ├── channel_selection_frequency.csv
│   ├── Table_2_Weighted_Final.csv         # 正文 Table 2
│   ├── Table_2_Weighted_Final.md
│   ├── Table_S2A_E11_Optimization.csv
│   ├── Table_S2B_Selection_Frequency.csv
│   └── Figure_5_values.csv               # Q、raw weighted R/L 的 pooled Cohen's d
├── figures/
│   ├── Figure_1_Workflow.png / .svg
│   ├── Figure_2_Bilateral_Geometry.png / .svg
│   ├── Figure_3_Weighted_R_L.png / .svg
│   ├── Figure_4_Feature_Performance.png / .svg
│   ├── Figure_5_Q_and_Pooled_Cohen_d.png / .svg
│   └── supp_roc_curves.png                # 额外诊断图，非五张正文图
├── reports/
│   ├── validation_report.json
│   ├── run_config.json                   # 运行参数和数据代码的 SHA256
│   ├── data_fingerprints.json            # 主 NPZ/CSV 的校验哈希
│   └── final_run_summary.json
└── supplementary/                       # 预留其他补充材料
```

## 4. 如果中途报错怎么办

- **MC 已完成并保存 `main_cohort.npz`**，后续 E11 或出图报错：修好**不涉及物理和分析逻辑**的错误后，用

  ```powershell
  python run_final_paper.py --resume
  ```

  这只复用该输出目录中的完整 weighted cohort，重新进行特征评价和 E11，**不重新运行 MC**。如果物理/主评估代码改变，它会拒绝使用旧 cohort，应换新的 `--out` 目录完整重跑。

- **只修改字体、标题、Figure 排版和表格格式**：

  ```powershell
  python run_final_paper.py --figures-only
  ```

  只使用经过 SHA256 校验的当前 NPZ/CSV 出图，不再执行 MC、特征训练或 E11。此路径允许仅修改 `make_paper_figures_tables.py` 中的绘图样式。

- 已有 `results_final/`，需要**从头再跑**：不要覆盖旧目录，请手动备份/更名旧 `results_final`，或使用

  ```powershell
  python run_final_paper.py --out ..\results_final_rerun02
  ```

- 如果在 PyCharm 里运行，`Script path` 指向 `src/run_final_paper.py`，`Parameters` 留空即可；你不需要逐个启动其他五个 Python 文件。

## 5. 和旧代码相比保留/更正了什么

- `skin_liver_fibrosis_mc.optical_properties` 里的明确 severity 可以用于 **low- 和 high- 两组连续插值**，不再把 low 组强制设为 0；但未指定 severity 的正常/纤维化端点 API 不变。
- 光子数 `radial_count_hist` 用于轨迹概率；剩余权重 `radial_weight_hist` 用于论文中的 **R、L**。`channel_counts` 保留浮点精度。
- 正负方向 `0.25–4.75 mm`0 两侧对称位置按 `abs(x)` 合为 10 个 bilateral scoring channels；宽度 0.5 mm/每侧。
- Fig. 5 的 **Q = L/R、d_R、d_L** 同样来自新 NPZ 中 *未经仪器噪声扰动* 的原始加权信号；d 用 pooled sample SD (`ddof=1`)；与分类阶段经过特征噪声的 Table 2 基准不同，图注须注明。
- Fig. 4 的分类实验使用相同 logistic regression / 30 次 stratified 70/30 splits；E11 另行 nested-CV 内层选择，最终选择测试阈值固定 0.5；不要认为 E11 与 Figure 4 用了完全相同的阈值方法。
- 所有图表数据都从本次新 `results_final/` 的表和 NPZ 生成；不会扫描旧的 `results_weighted/final_reanalysis` 等目录。
- 省略了不会进入主文的额外 MC 运行（photon budget、staging）。**这并不表示已做临床或 3D 模型验证。**

## 6. 注意

- 这里所有实验是二维三层仿体合成研究，参数是模拟端点，不代表患者真实采样；不要从 `L` 或 `Q` 推出常规连续波仪器可直接读取其值。
- 原始 cohort 是一次合成队列；30 次重复划分是 repeated train/test splits，不是 30 个独立队列。
- 不允许用旧 count-scoring 数据或其他种子的 E11 结果替换最终主表。论文中所有旧数字需以后续完整正式运行 CSV 为准。
