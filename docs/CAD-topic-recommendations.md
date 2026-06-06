# 计划：基于 CAD 深度学习研究报告，为 AutoResearchClaw 推荐合规 topic

## Context（背景与目的）

用户希望我以资深 AI 专家身份：① 搞清楚 AutoResearchClaw 项目运行所需的 "topic"（研究主题）到底有什么要求；② 通读
`/root/Deep-research/deep_research_report_20260605_090117.md`（主题：**深度学习在 CAD 领域的最新研究方向**）；
③ 基于报告内容，给出若干**符合项目硬性约束、可直接投喂流水线**的英文 topic 候选供选择。

这是一个**研究主题推荐**任务，不涉及改代码。最终交付物 = 一组可直接粘贴到 `researchclaw run --topic "..."` 的英文 topic，
以及每个 topic 的新颖性、数据集、baseline、可证伪假设、领域配置与可行性评级。

**已确认的用户偏好**：算力为「多 GPU / 较强」；推荐重心放在「跨模态 / 逆向工程（CADCL 路线）」，但仍给出多方向候选。

---

## 一、项目对 topic 的硬性要求（调研结论）

来源：`researchclaw/cli.py`、`researchclaw/pipeline/stage_impls/_topic.py`、`researchclaw/prompts/ml.py`、
`docs/config-research-fields.md`、`researchclaw/config.py`。

| 维度 | 要求 |
|---|---|
| 语言 | **必须英文**（Stage 4 用 topic 直接构造 OpenAlex/Semantic Scholar/arXiv 检索，中文会失败）|
| 形式 | 字符串，1–3 句，约 ≤200 字符；`research.topic` 为必填项 |
| 新颖性 | 必须填补 gap，不能让 reviewer 说 "this is already well-known"（prompts/ml.py: topic_init）|
| 聚焦度 | 单篇会议论文粒度；"machine learning" 这类过宽会被判低分 |
| 趋势验证（强制）| topic 要能讲清 gap + **指定 benchmark/数据集** + 是否有 SOTA |
| 可证伪 | 至少能落地 **≥2 个带明确"失败条件"的假设**（Stage 8），其中 ≥1 个"反直觉" |
| 算力可行 | 默认面向单 GPU/小时级；本次用户有多 GPU，可适度放宽，但仍须单篇粒度 |
| 数据可得 | 数据集需公开可检索；Stage 4 须能召回 ≥8 篇真实论文 |
| 领域配置 | `research.domains` 可声明（24 个 ID，如 `ml_generative`/`ml_graph`/`ml_vision`），否则自动检测 |
| 质量自评 | Stage 2 (IMP-35) 对 topic 打分 novelty/specificity/feasibility/overall（<5 会建议改写）|

**关键启示**：报告四大方向里，B-rep 扩散（BrepGen/GraphBrep，10M 形状）与端到端 3DGS 逆向（训练 3DGS 前端）整管线太重；
要符合"单篇 + 可行"，必须裁成**消融 / 评测 / 在已有几何上做下游任务**的形态。跨模态对齐与 GNN 任务天然轻量，最契合。

---

## 二、推荐 topic 候选（英文，可直接投喂）

> 排序按用户偏好（跨模态/逆向优先）；每条给出报告出处、gap、数据集（均公开）、baseline、≥2 个可证伪假设、领域、可行性。

### ⭐ 候选 1（首推，跨模态/逆向）— 对比学习对齐 B-rep 与命令序列
- **--topic**: `Hard-negative contrastive alignment between B-rep graph embeddings and CAD command sequences for reverse-engineering editable parametric models, evaluated on the DeepCAD benchmark`
- 报告出处：§1.4 CADCL（InfoNCE 跨模态对齐）；趋势 §4.3 语义化逆向。
- gap / 新颖角度：CADCL 用**随机负样本**做 InfoNCE；本课题验证**拓扑相似但操作不同的难负样本挖掘 + 按操作复杂度的课程式对齐**能否提升逆向序列的检索与重建保真度。
- 数据集：**DeepCAD**（178k 模型，含 sketch-extrude 命令序列 + OpenCASCADE 生成的 B-rep，公开）。
- baseline：① 监督式 seq2seq（DeepCAD autoencoder）；② 随机负样本 InfoNCE（CADCL 复现）；③ 仅 UV-Net 编码器无对比。
- 可证伪假设：
  - H1：难负样本对比预训练使 B-rep→序列 **top-1 检索准确率**较随机负样本 ≥ +5%。失败：≤ baseline。
  - H2（反直觉）：对比预训练对**长序列（>20 步）**重建的相对增益高于短序列。失败：增益随长度不变或反向。
- 领域：`ml_generative` + `ml_graph`。可行性：**高**（小图 GNN + 小 Transformer，多 GPU 充裕）。

### ⭐ 候选 2（跨模态/逆向）— 点云→B-rep 的基元拟合：经典 vs 学习
- **--topic**: `Comparing RANSAC analytic primitive fitting against a lightweight learned segmentation network for reverse-engineering B-rep surfaces from CAD point clouds on the Fusion 360 Gallery dataset`
- 报告出处：§1.3 BrepGaussian / Point2CAD（RANSAC 拟合解析曲面）；规避了重型 3DGS 前端，直接用**已有点云**。
- gap / 新颖角度：报告指出 RANSAC 在"高频复杂交接处（复杂圆角）拟合精度欠佳"；本课题量化经典 RANSAC 与轻量 PointNet 风格分割在**平面/圆柱/圆锥/球 vs 自由曲面**上的精度断点。
- 数据集：**Fusion 360 Gallery (Segmentation)** 或 **ABC dataset**（均公开，自带点云/网格 + 曲面标签）。
- baseline：① 纯 RANSAC（Point2CAD 风格）；② 监督 PointNet++ 分割。
- 可证伪假设：
  - H1：学习式分割在含噪/低纹理金属件上的曲面分类 F1 ≥ RANSAC + 8%。失败：无显著差异。
  - H2：RANSAC 在解析曲面（平面/圆柱）上精度反超学习法（说明二者互补）。失败：学习法全面碾压。
- 领域：`ml_vision` + `ml_graph`。可行性：**中高**（用现成点云，避开 3DGS 训练）。

### ⭐ 候选 3（跨模态/逆向，偏趋势）— 从 B-rep 反推加工工序历史
- **--topic**: `Graph-based reasoning to recover subtractive machining operation history (design intent) from B-rep solids, going beyond surface fitting on the Fusion 360 Gallery reconstruction dataset`
- 报告出处：趋势 §4.3「语义化高阶历史重建」+ §1.4。
- gap / 新颖角度：现有逆向止于"外观曲面拟合"；本课题预测**操作序列（钻孔/铣槽/倒角的先后与类型）**，即设计意图，而非仅几何。
- 数据集：**Fusion 360 Gallery (Reconstruction)**（含构造序列，公开）；可辅以 MFCAD++ 特征标签。
- baseline：① 仅几何特征分类（无序列）；② DeepCAD 序列预测。
- 可证伪假设：
  - H1：图推理网络对操作**类型 + 偏序**的联合准确率 ≥ 仅几何 baseline + 10%。失败：≤ baseline。
  - H2（反直觉）：加入拓扑邻接信息对"相交特征"的工序还原增益最大、对孤立特征几乎无增益。失败：增益均匀。
- 领域：`ml_graph` + `ml_generative`。可行性：**中**（序列+图，多 GPU 适配；标注需对齐）。

### 候选 4（GNN 特征识别，最易跑通的稳妥项）— 合成→真实的域适应
- **--topic**: `Closing the synthetic-to-real domain gap in B-rep machining-feature recognition via adversarial domain adaptation on the MFCAD++ benchmark`
- 报告出处：§1.4 BrepMFR（两阶段迁移学习 + 域适应，24 种加工特征）。
- gap / 新颖角度：系统对比无适应 / 简单微调 / 对抗域适应在合成→真实零件上的精度断崖修复幅度。
- 数据集：**MFCAD / MFCAD++**（公开，~59k 模型、24+ 特征类型，与报告"24 种特征"吻合）。
- baseline：① 仅合成训练直测；② 朴素微调；③ UV-Net/Graph Transformer 无适应。
- 可证伪假设：
  - H1：对抗域适应使真实零件 mIoU/准确率较"仅合成"≥ +12%。失败：< +3%。
  - H2：高度相交特征的相对增益高于孤立特征。失败：无差异。
- 领域：`ml_graph`。可行性：**最高**（小图分类，CPU 都能勉强跑，适合做基线/兜底）。

### 候选 5（命令序列生成）— 坐标量化误差与指针吸附消融
- **--topic**: `Quantifying coordinate quantization error accumulation in autoregressive CAD command generation and testing pointer-based entity snapping as a mitigation on the DeepCAD benchmark`
- 报告出处：§1.1 Text2CAD/Pointer-CAD（量化误差、指针网络、SegE）；§3.2 瓶颈②参数漂移。
- gap / 新颖角度：系统量化**不同量化位宽（8 vs 16-bit）/ 方案（均匀 vs 对数）**下漂移随序列长度的累积，并检验指针吸附的缓解上限。
- 数据集：**DeepCAD**。
- baseline：① 8-bit token 回归；② 16-bit；③ 指针吸附（Pointer-CAD 风格复现）。
- 可证伪假设：
  - H1：指针吸附使段级误差 SegE 较纯 token 回归降低 ≥1 个数量级。失败：< 2×。
  - H2（反直觉）：单纯提升量化位宽（8→16）对长序列末端漂移的改善小于指针机制。失败：位宽足以解决。
- 领域：`ml_generative`。可行性：**高**。

---

## 三、推荐落地方式（如何把选中的 topic 跑起来）

1. 复制配置：`config.researchclaw.example.yaml` → `config.arc.yaml`（已 gitignore），填 `llm.base_url/api_key/primary_model`，
   建议设独立 `llm.reviewer_model`（破除 generator==judge 自偏好，影响 Stage 18/20）。
2. 在 `research.domains` 写入对应领域 ID（见各候选）；多 GPU 下可把 `experiment.mode` 设 `sandbox` 或 `docker`，
   并适当上调 `experiment.time_budget_sec` / `max_iterations`。
3. 先做**低成本 topic 体检**：以 co-pilot 模式跑到 Stage 2，读取 `artifacts/rc-<id>/stage-02/topic_evaluation.json`
   （IMP-35 的 novelty/specificity/feasibility/overall 评分），overall ≥6 再 `--auto-approve` 全量跑。
   ```bash
   researchclaw run --topic "<选中的英文 topic>" --mode co-pilot --config config.arc.yaml
   ```

## 四、验证（如何确认推荐是对的）

- **合规性验证**：对最终选定的 1 个 topic，运行到 Stage 2，确认 `topic_evaluation.json` 的 overall ≥ 5（最好 ≥6）、
  且 feasibility 不低分；同时确认 Stage 4 召回 ≥8 篇真实文献（说明 benchmark 在文献库可检索）。
- **可证伪性验证**：确认 Stage 8 能从该 topic 生成 ≥2 个带"Failure condition"的假设（查看 stage-08 输出）。
- **领域路由验证**：确认 Stage 9 域检测把 topic 路由到声明的 `domains`（查看 `_experiment_design.py` 日志/产物）。

> 注：本计划阶段不改动任何代码，仅产出 topic 推荐与跑通方式；选定后由用户决定是否实际启动流水线。
