# SAE 特征单义性的信息论评估框架

本文档总结我们提出的基于香农熵 H 与 KL 散度的 SAE 特征单义性评估方法、相关实验结果以及与已有工作的对比。

---

## 1. 方法论：基于信息论的单义性度量 H 与 KL

### 1.1 动机

评估一个 SAE 特征是否"单义"，本质上是在问：**这个特征在激活的时候，是不是集中在某一个（或极少数）语义概念上？** 如果一个特征只在"电子邮件地址"这一类 token 上被点亮，那它就是高度单义的；如果它在"电子邮件"、"人名"、"城市"、"普通句法 token"上都会以不同程度被点亮，那它就是多义的。

要把这个直觉变成可计算的指标，自然的选择是把"特征的激活"视作一个在类别上的概率分布，然后用信息论的度量来刻画这个分布的形状：

- **分布集中在单一类别**：熵极低，接近 0；
- **分布均匀分散在所有类别**：熵极大，接近 $\log_2 C$。

在此基础上我们再引入 KL 散度作为辅助——它度量的是特征的类别分布相对于"数据本身的类别先验"的偏离程度，在类别分布严重不均衡的数据集上尤其重要（见 [1.6](#16-kl-散度作为辅助指标)）。

### 1.2 符号与记号

记号约定：

- $F$：SAE 特征总数（gemma-scope-2b-pt-res 16k width 下 $F = 16384$）；
- $C$：类别总数（pii-masking-300k 含非实体类时 $C = 26$，不含时 $C = 25$）；
- $N$：参与统计的有效 token 总数（`total_valid`）；
- $a_{f}(t) \ge 0$：SAE 特征 $f$ 在 token $t$ 上的激活值（JumpReLU 之后，严格非负）；
- $y(t) \in \{0, 1, \dots, C-1\} \cup \{-1\}$：token $t$ 的类别标签，$-1$ 表示被忽略的 PAD/特殊 token。

"特征在类别上的分布"的核心量是：

$$
S_f(c) = \sum_{t:\, y(t) = c} a_f(t), \qquad T_f = \sum_{c=0}^{C-1} S_f(c) = \sum_{t:\, y(t)\ge 0} a_f(t).
$$

$S_f(c)$ 是特征 $f$ 在类别 $c$ 所有 token 上的**累积激活强度**，$T_f$ 是它在所有有效 token 上的累积激活强度。对应代码见 [main_token.py:267-268](sae_bench/evals/info_theory/main_token.py#L267-L268) 的流式累加 `class_acts[c] += v_acts[cmask].sum(axis=0)`，以及 [main_token.py:332](sae_bench/evals/info_theory/main_token.py#L332) 的 `total_activation = class_acts.sum(axis=0)`。

### 1.3 条件概率 $P(c \mid f)$

给定特征 $f$，我们定义"在该特征被激活的条件下，token 属于类别 $c$ 的概率"为：

$$
P(c \mid f) = \frac{S_f(c)}{T_f}.
$$

注意两点：

1. **分母是激活强度之和而非 fire 次数**。这意味着 $P(c \mid f)$ 是一个**激活值加权**的分布，而不是"fire 过的 token 中各类占比"。在 JumpReLU SAE 下，强激活和弱激活对单义性判断的贡献不同——强激活更能代表"特征真正关心什么"，用激活值加权天然就给了强激活更大的权重。代码对应 [main_token.py:332-333](sae_bench/evals/info_theory/main_token.py#L332-L333)：

   ```python
   total_activation = class_acts.sum(axis=0)
   P = class_acts[:, alive_mask] / total_activation[alive_mask]
   ```

2. **只在 alive 特征上计算**。我们把 $T_f$ 极小的特征（`total_activation > 1e-5`，[main_token.py:323](sae_bench/evals/info_theory/main_token.py#L323)）视为死特征，排除在外；死特征的 $P(c \mid f)$ 会因分母过小而数值上不可靠，纳入统计只会引入噪声。

### 1.4 类别先验 $Q$

类别先验 $Q$ 取数据本身的 token 频次：

$$
Q(c) = \frac{N_c}{N}, \qquad N_c = \#\{t : y(t) = c\}.
$$

对应代码 [main_token.py:311](sae_bench/evals/info_theory/main_token.py#L311)：

```python
Q = class_token_counts.astype(np.float64) / class_token_counts.sum()
Q = np.clip(Q, 1e-10, 1.0)
```

我们**没有**采用均匀先验 $Q(c) = 1/C$，原因有两条：

1. **实际数据中的类别分布本身就是高度不均衡的**。以 pii-masking-300k 为例，"O"（非实体 token）占绝大多数，"GIVENNAME"、"CITY" 等常见实体有几千到几万 token，而 "PASSPORT"、"IDCARD" 等冷门类别只有几百。如果用均匀先验，KL 会被"凡是偏离均匀的分布"一概视为信息量高，把"只是跟着数据分布走"的平庸特征也奖励掉。
2. **token 频次先验让 KL 真正度量"相对于数据背景的偏离"**。一个特征如果其 $P(c \mid f)$ 恰好等于数据的类别频率 $Q(c)$，说明它对类别完全没有选择性——这正是我们希望 KL = 0 的情形。token 频次先验让这个直觉成立。

clip 到 $[10^{-10}, 1]$ 是为了避免 $\log(0)$ 的数值问题，不影响语义。

### 1.5 归一化香农熵 $H$

特征 $f$ 的（归一化）香农熵定义为：

$$
H(f) = \frac{-\sum_{c=0}^{C-1} P(c \mid f) \log_2 P(c \mid f)}{\log_2 C} \in [0, 1].
$$

代码见 [main_token.py:308, 335](sae_bench/evals/info_theory/main_token.py#L308)：

```python
log2_C = np.log2(num_classes)
...
h_alive = -np.sum(P * np.log2(P), axis=0) / log2_C
```

**归一化**（除以 $\log_2 C$）的作用是让 $H$ 总落在 $[0,1]$，从而可以跨类别数不同的数据集横向比较：

- $H(f) \to 0$：$P(c \mid f)$ 集中在某一类，特征**高度单义**；
- $H(f) \to 1$：$P(c \mid f)$ 均匀分散在所有类，特征**完全多义**；
- 中间值则量化了两端之间的连续谱。

**H 是本工作的主指标**。它直接刻画了"激活被多少个类别分享"这件事，且不依赖任何外部先验——只要特征的类别分布本身集中，H 就低。这正是单义性的朴素数学翻译。

### 1.6 KL 散度（作为辅助指标）

特征 $f$ 的 KL 散度定义为：

$$
\mathrm{KL}(f) = \sum_{c=0}^{C-1} P(c \mid f) \log_2 \frac{P(c \mid f)}{Q(c)}.
$$

对应代码 [main_token.py:336](sae_bench/evals/info_theory/main_token.py#L336)：

```python
kl_alive = np.sum(P * np.log2(P / Q[:, None]), axis=0)
```

**KL 和 H 关心的不是同一件事**：

- $H$ 关心"分布有多集中"；
- $\mathrm{KL}$ 关心"分布相对于数据背景偏离多少"。

两者在**类别分布高度不均衡**的数据集上会产生实质分歧。考虑一个极端例子：数据中 90% 的 token 属于"O"，只有 10% 是各种实体类。如果一个特征把 99% 的激活都打在"O"上，它的 H 仍然很低（集中在一个类），但它其实只是跟随数据背景、没有任何选择性。KL 会正确地把它判为低信息量（因为它的 $P$ 和 $Q$ 都集中在"O"）。

换句话说：

- **H 偏爱"集中"**，不管集中在哪；
- **KL 偏爱"与背景不同的集中"**。

在类别分布接近均衡时，$Q(c) \approx 1/C$，此时 $\mathrm{KL}(f) = \log_2 C - H(f) \cdot \log_2 C$，两者近似线性反相关，信息等价；在类别分布严重不均衡时，两者解耦，KL 能补上 H 忽略的维度。

因此我们的定位是：**H 作为主指标**承担单义性量化的主要工作；**KL 作为辅助指标**在类别先验严重不均衡的数据集上作为补救和交叉验证。后续 P/R 验证（第 3、4 节）中我们把 H 和 KL 作为两套独立的排序信号分别考察，正是为了让 H 的主角地位经得起对照。

### 1.7 为什么采用 token-level 设计

这套 H/KL 定义在 document-level 和 token-level 上都能形式化，区别只在于 $S_f(c)$ 的累加范围是 document 级还是 token 级。我们选择以 **token-level** 作为主战场，有两个原因：

**1）粒度决定信号强度**。我们最早是在 document-level 上做的（对应 [main.py](sae_bench/evals/info_theory/main.py)），在 ag_news（4 类）和 dbpedia14（14 类）这类粗粒度、类数少的数据集上，$H$ 普遍偏高——因为"一篇新闻"这种粒度天然会同时涉及多个概念，特征很难只在单一 document 类上激活。粒度太粗导致 $H$ 的动态范围被压缩、单义性信号被稀释。转到 token-level 之后，类别粒度从"文档主题"细化到"PII 实体类型"，类数也从十几个扩展到 25 个，$H$ 的分布才显著拉开、低 $H$ 特征才成规模地出现。这部分的对比数据将在 [第 2 节](#2-hkl-结果分析) 展开。

**2）细粒度标签本身更贴近"概念"的定义**。一个理想的单义特征应该和人类可识别的**最小语义单位**绑定，而"邮箱地址"、"人名姓"这种实体类型就是这样的最小单位。把它们作为 document 级主题的一部分（如"个人信息类新闻"）会丢失绝大多数结构。token-level NER 标签是天然的细粒度概念标注。

**方法论上的诚实观察**：即使转到 token-level，也很少有单个特征能独占一个 PII 类别——我们在实验中反复看到"一个 label 需要多个特征联合才能解释"的现象。这并不是方法论的缺陷，而是 SAE 特征本身的组合性：一个"人名姓"的概念可能分散在 3–5 个子特征上（大写字母开头 / 常见姓氏 / 西方人名 / ...），各自只覆盖一部分样本。这一观察直接促成了第 3、4 节 P/R 验证中采用 $k \in \{1, 5, 10, 20\}$ 的 joint union 评估——单特征 precision/recall 无法公平反映"top-k 特征联合覆盖"的真实情况。

### 1.8 有效特征筛选（alive mask）

上述所有公式只对 alive 特征计算，定义见 [main_token.py:323](sae_bench/evals/info_theory/main_token.py#L323)：

```python
total_activation = class_acts.sum(axis=0)  # [F]
alive_mask = total_activation > 1e-5
```

这是纯粹的数值稳健性考虑：$T_f$ 过小的特征会让 $P(c \mid f)$ 的分母接近 0，数值不可靠。被标记为非 alive 的特征在结果中记为 $H = \mathrm{KL} = -1$（[main_token.py:328-329](sae_bench/evals/info_theory/main_token.py#L328-L329)），在聚合统计时排除。

与此不同的是 **density band-pass 过滤**（`min_feature_density` / `max_feature_density`），那是对 alive 特征的进一步筛选，用于排除"句式/语法类通用特征"和"激活频次极低的噪声特征"——这属于**实验设置**而非方法定义本身，我们把它留到 [第 7 节](#7-实验设置) 统一说明。

---

## 2. H/KL 结果分析

*（待第 1 节 review 后展开）*

---
