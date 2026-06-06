# `research` 配置项说明

## 配置示例

```yaml
research:
  topic: "Your research topic here"
  domains:
    - "ml_generic"       # 使用下方表格中的 domain_id（不要写 "machine-learning" 旧别名）
  daily_paper_count: 10
  quality_threshold: 4.0
```

---

## 字段详解

### `topic`（必须）

研究主题，直接传入所有 23 个阶段的 LLM Prompt。**必须用英文**，因为文献检索（OpenAlex/Semantic Scholar/arXiv）基于此构造搜索关键词，中文 topic 会导致检索质量下降。

```yaml
topic: "Reinforcement learning for drug discovery under distribution shift"
```

---

### `domains`（可选，但影响实验代码质量）

该字段是**提示（hint）**，不是严格枚举。真正的领域识别发生在 `researchclaw/domains/detector.py::detect_domain()`，它结合 `topic` + 假设 + 文献文本做三级判定：(1) 关键词匹配 → (2) LLM 分类 → (3) 回落 `generic`。配置中写的 `domains` 会以 `Configured domains: ...` 的形式拼进分类上下文，在 topic 含糊时偏向目标领域；命中不到任何 profile 时降级为 `generic`（使用通用模板）。

识别到的 `DomainProfile`（定义于 `researchclaw/domains/profiles/*.yaml`）会影响：
1. **代码生成模板**：文件结构、入口文件、`pip_packages`
2. **执行环境**：Docker 镜像、是否需要 GPU
3. **评测与呈现**：度量类型、统计检验、图表类型、标准基线
4. **Prompt 专家角色与 `code_generation_hints`**

#### 推荐写法

直接使用下表的 **`domain_id`**（profile 文件名），例如：

```yaml
research:
  domains:
    - "ml_vision"        # 而不是 "computer-vision"
    - "ml_rl"            # 而不是 "machine-learning"（旧别名，会走回落）
```

> 可以写多个值。形如 `machine-learning`、`computer-vision` 的旧式短横线别名**不是** profile ID——它们在关键词规则里匹配不到，只作为 LLM 分类的上下文文本起作用，命中不到就回落 `generic`。建议统一用下划线形式的 `domain_id`。

#### 当前支持的 `domain_id`（24 个，来自 `researchclaw/domains/profiles/`）

**机器学习 / AI**（`parent: ml`）

| `domain_id` | 适用场景 |
|---|---|
| `ml_generic` | 通用 ML/AI、神经网络、深度学习 |
| `ml_vision` | 图像分类、检测、分割、ViT/CNN |
| `ml_nlp` | 文本、语言模型、Transformer、LLM |
| `ml_rl` | 强化学习、策略梯度、Gymnasium、SB3 |
| `ml_graph` | 图神经网络、节点/链接预测 |
| `ml_tabular` | 表格数据、XGBoost / LightGBM / CatBoost |
| `ml_generative` | GAN、Diffusion、VAE、图像/文本生成 |
| `ml_compression` | 知识蒸馏、剪枝、量化、模型压缩 |

**物理**（`parent: physics`）

| `domain_id` | 适用场景 |
|---|---|
| `physics_simulation` | 分子动力学、N-body、JAX-MD、ASE、OpenMM |
| `physics_pde` | PDE 求解器（FEM、FDM、谱方法）、FEniCS、Navier–Stokes |
| `physics_quantum` | 量子力学、Schrödinger、哈密顿量、波函数 |

**化学**（`parent: chemistry`）

| `domain_id` | 适用场景 |
|---|---|
| `chemistry_qm` | 量子化学（DFT、Hartree-Fock、CCSD、PySCF） |
| `chemistry_molprop` | 分子性质预测（SMILES、RDKit、fingerprint、ADMET） |

**生物**（`parent: biology`）

| `domain_id` | 适用场景 |
|---|---|
| `biology_singlecell` | 单细胞分析（scRNA-seq、scanpy、AnnData、Leiden） |
| `biology_genomics` | 基因组学（测序、变异检测、Biopython） |
| `biology_protein` | 蛋白质科学（AlphaFold、ESM、折叠/性质预测） |

**神经科学**（`parent: neuroscience`）

| `domain_id` | 适用场景 |
|---|---|
| `neuroscience_computational` | 脉冲神经网络、Brian2、Hodgkin-Huxley、神经动力学 |
| `neuroscience_imaging` | fMRI / EEG / MEG、Nilearn、MNE-Python、功能连接 |

**数学**（`parent: mathematics`）

| `domain_id` | 适用场景 |
|---|---|
| `mathematics_numerical` | 数值方法、收敛阶、ODE 求解器、Runge-Kutta、SymPy |
| `mathematics_optimization` | 凸优化、线性规划、演化算法、无梯度优化 |

**其它**

| `domain_id` | parent | 适用场景 |
|---|---|---|
| `economics_empirical` | `economics` | 计量经济学、因果推断、面板数据、DiD、IV |
| `robotics_control` | `robotics` | 机器人、控制、操纵、MuJoCo、PyBullet |
| `security_detection` | `security` | 入侵检测、恶意软件、异常检测、Scapy |
| `generic` | — | 兜底/跨领域（通用 numpy+scipy 模板） |

#### 如何新增一个领域

若现有列表不够用，可按下列步骤扩展：

1. **新增 profile YAML**：在 `researchclaw/domains/profiles/<new_id>.yaml` 参考 `ml_vision.yaml` 的字段编写（`experiment_paradigm`、`core_libraries`、`pip_packages`、`docker_image`、`metric_types`、`code_generation_hints` 等）。
2. **在 `detector.py::_KEYWORD_RULES` 加关键词规则**（该列表顺序敏感，具体规则要放到通用规则之前）。
3. **在 `detector.py::_LLM_CLASSIFY_PROMPT` 的候选列表里追加一行**，让 LLM 路径也能返回该 ID。
4. 若引入了**全新的一级类别**（不在 `ml_ / physics_ / chemistry_ / biology_ / economics_ / mathematics_ / security_ / neuroscience_ / robotics_` 之列），还需在 `pipeline/_domain.py` 的 `_COARSE_DOMAIN_ALIASES` / `_DISPLAY_NAME_BY_PARENT` / `_TOP_VENUES_BY_PARENT` 补映射，否则默认期刊推荐会落到通用 arXiv。

合理的扩展候选：`earth_climate`、`materials_science`、`astro_observational`、`quantum_computing`（区别于 `physics_quantum`）、`engineering_mechanical` / `engineering_electrical`（`_domain.py` 已为 `engineering` 预留 venues）、`hci_social`。

---

### `daily_paper_count`（控制文献收集数量）

Stage 4（LITERATURE_COLLECT）每次搜索目标收集的论文候选数量。设为 `0` 时内部默认取 `20`。

```yaml
daily_paper_count: 10   # 收集约 10 篇候选论文
```

数量越大，文献综述越全面，但 LLM token 消耗和运行时间也相应增加。

---

### `quality_threshold`（论文质量门槛，范围 0~10）

用于两个阶段：

1. **Stage 5（LITERATURE_SCREEN）筛选文献**：LLM 给每篇候选论文打分，低于此值的论文被过滤掉
2. **Stage 20（QUALITY_GATE）论文质量门**：论文整体评分需达到此值，否则自动触发重写（Stage 16-19 重跑）

```yaml
quality_threshold: 4.0
```

建议范围：

| 值 | 效果 |
|----|------|
| `3.0 ~ 5.0` | 宽松，快速完成，适合测试和调试 |
| `6.0 ~ 7.0` | 正常，会触发 1~2 次论文修改迭代 |
| `8.0+` | 严格，可能多次重写，消耗大量 token |
