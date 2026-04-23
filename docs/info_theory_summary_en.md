# An Information-Theoretic Framework for Evaluating SAE Feature Monosemanticity

This document summarizes our proposed method for evaluating SAE feature monosemanticity using Shannon entropy H and KL divergence, together with the corresponding empirical results and evidence for the validity of the approach.

---

## 1. Information-Theoretic Measures of Monosemanticity: H and KL

### 1.1 Motivation

The core question in evaluating whether an SAE feature is “monosemantic” is: **are its activations concentrated primarily on a single semantic concept, or at most a very small number of concepts?** In this work, those “semantic concepts” are approximated by the dataset labels. For example, a feature that fires consistently only on tokens of the “email address” class can be regarded as highly monosemantic; by contrast, a feature that activates on “email,” “person name,” “city,” and “non-entity tokens” at the same time is closer to a polysemantic feature.

Strictly speaking, what we measure directly is not “monosemanticity” itself, but rather **the concentration of the activation distribution under a predefined class partition**, which we refer to below as **class alignment / class selectivity**. Class alignment is a **necessary but not sufficient** condition for monosemanticity. A truly monosemantic feature should remain stably aligned under meaningful class partitions; conversely, a feature that aligns only with one particular label partition may still mix multiple subconcepts within a class. We therefore treat class alignment as an **operational proxy** for monosemanticity, and test this proxy via the reverse P/R validation introduced in Chapters 3 and 4.

Under this view, we treat the allocation of a feature’s activations across classes as a probability distribution, and use its concentration to measure class alignment. It is not enough to look only at **which class receives the most activation**; what matters is whether **the full class distribution is sufficiently concentrated**.

Shannon entropy is exactly suited to quantify this concentration. If a feature’s activations are entirely concentrated in a single class, entropy takes its minimum value, 0; if its activations are uniformly distributed across all classes, entropy takes its maximum value, $\log_2 C$. Entropy therefore formalizes the following intuition naturally: **the more concentrated the activations, the more monosemantic the feature; the more diffuse the activations, the more polysemantic the feature.** On top of this, we introduce KL divergence as an **auxiliary metric** to handle the bias that can arise when entropy is used alone under highly imbalanced class priors.

### 1.2 Symbols and Notation

To write “which classes a feature mainly responds to” in a computable form, we first define the following notation:

- $F$: total number of SAE features;
- $C$: total number of classes;
- $N$: total number of valid samples included in the statistics;
- $a_f(t) \ge 0$: activation value of feature $f$ on sample $t$;
- $y(t) \in \{0, 1, \dots, C-1\} \cup \{-1\}$: class label of sample $t$, where $-1$ means that the sample is excluded from the statistics.

Based on this notation, define the **accumulated activation strength** of feature $f$ on class $c$ as

$$
S_f(c) = \sum_{t:\, y(t) = c} a_f(t),
$$

that is, the sum of the activation values of feature $f$ over all samples in class $c$. We further define

$$
T_f = \sum_{c=0}^{C-1} S_f(c) = \sum_{t:\, y(t)\ge 0} a_f(t),
$$

which represents the total activation strength of feature $f$ across all labeled samples.

$S_f(c)$ tells us how much activation strength feature $f$ accumulates on class $c$, while $T_f$ gives its total activation strength. All subsequent metrics are built on these two quantities. In other words, we are not concerned with how many samples a feature fires on, but with **how its total activation strength is distributed across classes**.

### 1.3 Framework

Based on $S_f(c)$ and $T_f$ defined in Section 1.2, we first normalize feature activations into a class distribution, and then construct two feature-level metrics. **Normalized Shannon entropy $H$** serves as the primary metric for measuring how concentrated activations are across classes, while **KL divergence** serves as an auxiliary metric for identifying spurious low-entropy cases caused by class-imbalance priors.

The first step is to normalize the accumulated activations in Section 1.2 into a probability distribution. We define the conditional probability that a sample belongs to class $c$ given that feature $f$ is activated as

$$
P(c \mid f) = \frac{S_f(c)}{T_f}.
$$

$P(c \mid f)$ satisfies the basic properties of a probability distribution: $\sum_c P(c \mid f) = 1$ and $P(c \mid f) \ge 0$. It describes **what fraction of all activations of feature $f$ fall on class $c$**. Thus, $P(c \mid f)$ turns feature activation into a probability distribution over the class set, which can then be characterized directly by information-theoretic metrics. The central intuition from Section 1.1 is that the more a feature’s activations are concentrated on a small number of classes, the more monosemantic it is. Shannon entropy captures exactly this intuition. We define the normalized entropy of feature $f$ as

$$
H(f) = \frac{-\sum_{c=0}^{C-1} P(c \mid f) \log_2 P(c \mid f)}{\log_2 C} \in [0, 1].
$$

with the convention that $0 \log 0 = 0$. The denominator $\log_2 C$ is the maximum entropy of the uniform distribution over $C$ classes; it normalizes $H$ into the interval $[0,1]$, enabling cross-dataset comparison when the number of classes differs. When $P(c \mid f)$ is fully concentrated on a single class, $H(f)=0$; when $P(c \mid f)$ is uniform over all classes, $H(f)=1$. Therefore, the smaller $H(f)$ is, the stronger the feature’s **class alignment** under the current class partition; under the proxy assumption introduced in Section 1.1, this also means that the feature is closer to being monosemantic.

$H$ is the primary metric in this work because it answers the question we care about most directly: **are a feature’s activations concentrated on only a few classes?** It depends only on the feature’s own class distribution and introduces no external prior.

The limitation of $H$ follows directly from its strength: it only measures how concentrated a distribution is, but not why it is concentrated or where it is concentrated. On datasets whose class distributions are approximately balanced, this is usually not a problem. Under **severe class imbalance**, however, it introduces a systematic bias.

Consider a typical named-entity setting: 90% of tokens belong to “O” (non-entity), while only 10% belong to all entity classes combined. If a feature places 99% of its activations on the “O” class, its $H$ can still be very low, making it look like a “highly monosemantic” feature. In reality, however, it may simply be tracking the most common background class in the data, rather than encoding a genuinely discriminative concept.

This shows that $H$ cannot distinguish between two different sources of “concentration”: one in which a feature is **truly locked onto a concept**, and another in which it is **merely following the data prior**. The latter produces a feature that appears monosemantic on the surface but in fact provides almost no additional discriminative power with respect to the classes. We refer to this phenomenon as **spurious low entropy**.

To identify such spurious low-entropy cases, we need a reference distribution for $P(c \mid f)$. The natural reference is not the uniform distribution, but the **class prior of the data itself**:

$$
Q(c) = \frac{N_c}{N}, \qquad N_c = \#\{t : y(t) = c\}.
$$

Here we use the empirical class distribution of the data as the prior, rather than the uniform prior $Q(c)=1/C$. The reason is that real datasets are usually class-imbalanced. If the uniform distribution were used as the baseline, then any feature deviating from uniformity could be misread as having additional class-discriminative power, including background features that merely track the data prior. Under the empirical class prior, by contrast, $\mathrm{KL}=0$ corresponds exactly to the case where **the feature’s class distribution is identical to the data prior, and therefore contributes no additional discriminative information about the classes**.

On this basis, we define the KL divergence of feature $f$ relative to the class prior as

$$
\mathrm{KL}(f) = \sum_{c=0}^{C-1} P(c \mid f) \log_2 \frac{P(c \mid f)}{Q(c)}.
$$

Returning to the example above: if a feature merely follows the background “O” distribution, then its $P(c \mid f)$ will be very close to $Q(c)$, and hence $\mathrm{KL}(f) \approx 0$. If another low-$H$ feature instead concentrates its activations on a rare entity class, then $P(c \mid f)$ will differ substantially from $Q(c)$, and $\mathrm{KL}(f)$ will be correspondingly larger. The former has low entropy but does not depart from the background distribution; the latter is more plausibly a genuinely class-discriminative feature, and under the proxy interpretation, closer to monosemanticity.

When the class distribution is perfectly balanced, i.e. $Q(c) = 1/C$, we have

$$
\mathrm{KL}(f) = \log_2 C \cdot (1 - H(f)).
$$

In this case, $H$ and $\mathrm{KL}$ carry the same information. When class distributions are approximately balanced, they remain approximately equivalent, so $H$ alone is sufficient. Under strong class imbalance, however, the two decouple: $H$ still measures the absolute concentration of the distribution, whereas $\mathrm{KL}$ measures its deviation from the data prior, and can therefore identify the spurious low-entropy cases described above. In summary, we use **$H$ as the primary metric** to quantify how concentrated a feature’s activations are across classes, and **$\mathrm{KL}$ as the auxiliary metric** to identify spurious low-entropy cases under class imbalance. Chapters 3 and 4 will further test whether this “primary metric + auxiliary metric” division of labor is justified.

### 1.4 Implementation Details

This section discusses how the H/KL framework above is instantiated and implemented in this work. Broadly, the relevant details fall into three levels: the choice of label granularity, the choice of aggregation scheme, and the concrete computation and filtering procedure. See Chapter 6 for the SAE configuration, datasets, and the remaining experimental settings.

First, at the level of label granularity, the H/KL framework can be defined either at the document level or at the token level; the only difference is whether $S_f(c)$ is accumulated over documents or over tokens. In this work, **token-level** is the primary instantiation. The direct reason is that label granularity determines the effective resolution of the metric. On document-level datasets such as ag_news (4 classes) and dbpedia14 (14 classes), a single document often involves multiple concepts, so a feature rarely aligns cleanly with one document class, and the dynamic range of $H$ becomes compressed into the high-entropy region. By contrast, token-level labels refine the class units into PII entity types and increase the number of classes, making low-$H$, highly selective features much easier to observe.

For `pii-masking-300k`, the main results in this work use the **noO variant that excludes the non-entity class `O`** (denoted below as **PII-noO**). This does not change the definition of H/KL itself; rather, it avoids having a dominant background class compress the readability of the metrics in the main experiment. Its necessity and consequences are examined further in Section 2.5.

Second, in terms of aggregation, we use **activation-weighted statistics** rather than activation counts. In other words, $S_f(c)$ and $T_f$ accumulate activation strengths rather than the number of times activation occurs. The reason is that, in JumpReLU SAEs, activation magnitude usually reflects how well the feature matches the input: strong activations are more representative of the semantic content encoded by the feature. If activation counts were used instead, all nonzero activations would be treated equally, and weak activations would receive the same weight as strong ones, diluting the semantic signal.

Finally, in the concrete computation procedure, all metrics are computed only on alive features. If the total accumulated activation strength $T_f$ of a feature across all labeled samples is below a threshold (set to $10^{-5}$ in this work), the feature is treated as dead and removed from the statistics, to avoid numerical instability caused by an excessively small denominator. Excluded features are marked as invalid in the output and do not participate in later aggregation. This should be distinguished from density filtering, which is applied on top of alive features and belongs to the experimental setup rather than to the metric definition itself. For the H/KL aggregation statistics in Chapter 2, we impose an upper density bound of $\le 10^{-2}$ in order to exclude syntactic / generic high-frequency features from the aggregated means; otherwise they would systematically inflate the aggregated $\text{mean }H$.

Implementation-wise, computation only requires accumulating feature activation strength for each class to obtain $S_f(c)$ and $T_f$, normalizing them into $P(c \mid f)$, and then computing $H$, the class prior $Q(c)$, and $\mathrm{KL}$. The prior $Q(c)$ is clipped from below to avoid numerical instability in the logarithm when zero-frequency classes appear. The entire procedure can be implemented in streaming form, with time complexity $O(N \cdot F)$ and no need to train any additional model.

---

## 2. H/KL Results and Analysis

This chapter presents the behavior of H/KL across different datasets, layers, and sparsity levels. The experiments are based on 15 `gemma-scope-2b-pt-res` SAEs, formed by combining 3 layers (layer 5 / 12 / 19) with 5 sparsity levels (from ultra-sparse to very-dense), yielding $3 \times 5$ configurations in total. Results are reported on three datasets: ag_news (4 classes, document-level), dbpedia14 (14 classes, document-level), and the PII-noO variant of `pii-masking-300k` (25 classes, token-level). See Chapter 6 for the full model and SAE configurations.

### 2.1 Effect of Label Granularity on the Dynamic Range of H/KL

The values of H lie in $[0,1]$, but whether its distribution can open up meaningfully depends on label granularity. To study this, we run the same set of 15 SAEs on ag_news (4 classes, document-level), dbpedia14 (14 classes, document-level), and `pii-masking-300k` (25 classes, token-level), and compare whether the H/KL distributions can be effectively separated under different label granularities. The aggregated results are shown in Table 2.1.

**Table 2.1 Aggregated statistics across the three datasets (mean over 15 SAEs)**

| Dataset | # Classes | Label Granularity | Mean H | Median H | Mean KL | frac(H<0.3) | frac(H<0.5) |
| :-----: | :-------: | :---------------: | :----: | :------: | :-----: | :---------: | :---------: |
| ag_news | 4 | document | 0.8843 | 0.9748 | 0.2313 | 3.7% | 7.3% |
| dbpedia14 | 14 | document | 0.8451 | 0.9338 | 0.5908 | 4.2% | 8.4% |
| **PII-noO** | **25** | **token** | **0.4782** | **0.4996** | **2.4167** | **23.5%** | **49.0%** |

Table 2.1 summarizes the central tendency and threshold fractions of the H/KL distributions, while Figure 2.1 visualizes the full distribution of H. As the figure shows, in both document-level panels most of the mass is concentrated near the high-value end close to $H=1$, yielding a clear right-edge pile-up; under token-level PII-noO, by contrast, the distribution opens up substantially and develops a clear left tail in the low-H region.

![H distribution across datasets](figs/fig_h_dist_crossdataset.png)

**Figure 2.1** Distribution of normalized Shannon entropy $H$ across the three datasets. Dashed lines indicate means and dotted lines indicate medians. On the document-level tasks (ag_news, dbpedia14), the $H$ distributions are strongly right-skewed and compressed into the high-H regime; under token-level PII-noO, the distribution opens up substantially, with much more mass appearing in the low-H region.

Table 2.1 and Figure 2.1 show that the median H exceeds 0.93 on both document-level datasets. More than half of the alive features have class distributions that are **close to uniform** on these two datasets, while the fraction of “clearly monosemantic” features (H<0.3) is only 3.7% / 4.2%. In other words, under the document-level setting, **the dynamic range of H is compressed into the narrow interval [0.7, 1.0]**, and most features appear highly polysemantic under the metric.

The picture changes substantially under token-level PII-noO. The mean H drops from 0.88/0.85 to 0.48, the median H drops to 0.50, the fraction of “clearly monosemantic” features rises from roughly 4% to **23.5%**, and nearly half of all features satisfy H<0.5. KL changes in the opposite direction: the mean KL rises from 0.23 on ag_news to 2.42 on PII-noO, an increase of about 10×. In other words, **the dynamic range of H and KL only opens up clearly on fine-grained token-level tasks**.

These differences show that the usefulness of the measurement depends on whether it matches the structural granularity of the object being measured. Document-level tasks contain only 4–14 coarse classes, and a single news article often covers multiple themes, so an individual feature is naturally more likely to spread over multiple document classes. Once the labels are refined into 25 token-level PII entity classes, the situation “one feature activates for one entity class” becomes much closer to something statistically observable, and monosemanticity signals become easier for H to capture. Thus, the cross-dataset comparison here primarily reflects **the effect of label granularity on the dynamic range of the metric**, and should not be read as showing that “the same group of SAEs performs absolutely better on one dataset.”

For this reason, the remaining analysis takes **PII-noO** as the main analysis object, because only under this setting is the dynamic range of H/KL fully opened up. ag_news and dbpedia14 serve as **control objects**, helping distinguish which phenomena come from the SAE itself and which come from label granularity.

### 2.2 Feature Monosemanticity Across SAEs with Different Sparsity Levels

Having established PII-noO as the main analysis object, this subsection turns to the effect of **SAE sparsity on feature monosemanticity**. Concretely, we first aggregate over the 15 SAEs by average L0 tier in order to isolate the main effect of sparsity.

**Table 2.2 Aggregation by average L0 tier (feature pools from 3 layers combined within each tier, PII-noO)**

| Average L0 tier | Mean H | Mean KL | frac(H<0.1) | frac(H<0.3) | frac(H<0.5) |
| :-------------: | :----: | :-----: | :---------: | :---------: | :---------: |
| ultra-sparse (L0<25) | 0.352 | 3.134 | 20.1% | 40.1% | 69.0% |
| sparse (25–50) | 0.419 | 2.798 | 12.8% | 29.4% | 57.7% |
| mid (50–100) | 0.481 | 2.448 | 7.1% | 20.6% | 47.0% |
| dense (100–200) | 0.537 | 2.095 | 4.0% | 14.3% | 37.1% |
| very-dense (>200) | 0.556 | 1.890 | 3.4% | 12.6% | 33.4% |

The most important conclusion from Table 2.2 is immediate: as the SAE moves from ultra-sparse to very-dense, the mean H rises monotonically while the mean KL falls monotonically. This means that **denser SAEs tend to produce more polysemantic features overall, whereas sparser SAEs tend to produce more monosemantic ones.** This trend remains monotonic across all five L0 tiers.

More informative than the means themselves is the behavior of the low-H tail. As L0 increases, the fraction of features below all three thresholds—H<0.1, H<0.3, and H<0.5—falls systematically. The strictest bucket, H<0.1, drops from 20.1% to 3.4%. This suggests that dense SAEs lose the most monosemantic part of the distribution first, rather than merely shifting the entire distribution slightly to the right.

Looking more closely along the L0 axis, the monotonic trend is not uniform in speed, but shows clear **saturation**. From ultra-sparse to dense, both the mean H and the low-H fractions change substantially. From dense to very-dense, however, the marginal effect of further increasing L0 becomes much smaller. In other words, **the main effect of sparsity on monosemanticity is concentrated in the low-to-mid L0 regime; beyond that, the marginal impact of larger L0 diminishes.**

If we inspect the absolute number of low-H features in representative SAEs, the size of the low-H tail becomes even clearer. On PII-noO, the three ultra-sparse SAEs (L0 ≈ 18–23) each independently contribute **2100–2650 features with H<0.1 and 4300–5300 features with H<0.3**; by contrast, the most extreme very-dense model (layer 19, L0=279) has only 41 features with H<0.1, a difference of about 65×. That is, a single sparse SAE already contains thousands of usable monosemantic candidates—more than enough to support the P/R validation of low-H features carried out in Chapters 3 and 4.

This result is consistent with the familiar “sparsity–capacity tradeoff” in SAE training. A low-L0 constraint limits the number of features that can be called upon simultaneously for each token, making each feature more likely to carry a narrower and purer semantic direction. A high L0, by contrast, allows more features to participate jointly in reconstruction, making broader and more mixed activation patterns more likely.

### 2.3 Feature Monosemanticity Across Layers

Section 2.2 showed that L0 determines the overall direction of variation in monosemanticity. This subsection asks a further question: **on top of that overall trend, do different layers also change the structure of feature monosemanticity?** If we only look at the marginal means aggregated by layer on PII-noO, the mean H of layers 5 / 12 / 19 all fall in the narrow range 0.474–0.485, differing by less than 3%, which seems to suggest that the role of layer is weak. But this is only after marginal averaging. Once we expand all three datasets, three layers, and five L0 levels together, the interaction structure between layer, L0, and task becomes visible:

![mean H vs L0 across 3 layers, 3 datasets](figs/fig_h_vs_l0_3panel.png)

**Figure 2.3** Mean H as a function of L0 on the three datasets, color-coded by layer (blue: layer 5, green: layer 12, red: layer 19). The three panels share the same y-axis range 0.3–1.0 for direct comparison, corresponding from left to right to ag_news / dbpedia14 / PII-noO.

The patterns in the figure naturally split into two types: document-level and token-level.

**(a) document-level (ag_news, dbpedia14): stable advantage of layer 12.** In both document-level panels, layer 5 consistently has the highest H, layer 12 is almost always the lowest, and layer 19 stays in the middle but closer to layer 12. More importantly, this ordering is essentially unchanged from low L0 to high L0, so the layer effect here can be summarized as: **stable ranking, consistent direction, and little modulation by L0**.

One interpretation consistent with this pattern is that concepts such as news topics or encyclopedia categories lie at an intermediate level of abstraction, and therefore are more likely to form relatively stable directions in the middle residual stream. Shallow-layer representations retain more character-, morphology-, and local-collocation-level information, which is often still insufficient to summarize the topic of an entire document; deep-layer representations, by contrast, are more strongly shaped by next-token prediction and therefore mix in more local context and output-oriented information. Middle layers sit between these two regimes: they have moved beyond shallow local-form features, but have not yet become fully dominated by deep predictive organization, making them more likely to align with document-level “topic-class” concepts.

**(b) token-level (PII-noO): the layer effect flips with L0.** Under PII-noO, the picture changes completely. At low L0, layer 19 has the lowest H and layer 5 the highest; as L0 increases, layer 5 gradually overtakes layer 19 and becomes the layer with the lowest H, while at high L0 layer 5 is best and layer 19 worst. In other words, **PII-noO does not lack a layer effect; rather, the layer effect itself changes direction with L0.** This also explains why the marginal means across layers are so close: the layer ordering in different L0 regimes cancels out after aggregation.

One possible explanation is that PII classes can be identified both from **character/form cues** (e.g., `@` → email, long digit strings → phone number, date formats → birthday) and from **contextual semantic cues** (e.g., a person name following “my name is ...”), and that these two kinds of information are more strongly represented in shallow and deep layers respectively. At low L0, each feature is forced to carry only one cue; in this regime, deeper layers (layer 19), having undergone stronger semantic organization, are more likely to align directly with a specific PII class, and thus have lower H. At high L0, however, capacity is greater, and shallow layers (layer 5) can allocate many one-to-one form-level cues to different features; collectively, this can produce a more concentrated distribution over PII classes, allowing shallow layers to overtake deep ones. What the data directly support, however, is only the interaction phenomenon that “the layer effect depends on L0”; determining which semantic cues are actually carried by each layer still requires feature-level evidence.

Taken together, patterns (a) and (b) show that the question “which layer is best?” does not exist independently of task, but depends on **the match between the conceptual unit implied by the labels and the hierarchy of the residual stream**. In the current experiments, document-level topic tasks exhibit a relatively stable middle-layer advantage, while token-level PII tasks exhibit a strong layer × L0 interaction.

### 2.4 Joint Distribution of H and density

Sections 2.2 and 2.3 compared average behavior across different L0 tiers and layers at the SAE level, but did not explain the distributional structure of monosemantic features in feature space. This section therefore moves down to the **feature level**, aggregating all alive features from the 15 SAEs on PII-noO by density interval. Here, density refers to the activation density of a feature on the evaluation set, which can be understood approximately as “the fraction of tokens on which this feature activates.”

**Table 2.4 Variation of $H$ with density on PII-noO** (aggregated over 15 SAEs)

| density interval | Median $H$ | Mean $H$ | frac($H$<0.3) | frac($H$>0.7) |
| :--------------: | :--------: | :------: | :-----------: | :-----------: |
| $[0, 10^{-4})$ | 0.333 | 0.312 | 43.5% | 0.7% |
| $[10^{-4}, 10^{-3})$ | 0.588 | 0.549 | 11.1% | 19.6% |
| $[10^{-3}, 10^{-2})$ | 0.645 | 0.595 | 9.9% | 36.5% |
| $[10^{-2}, 1]$ | 0.730 | 0.672 | 5.9% | 57.2% |

Table 2.4 reveals a clear three-part structure, which in turn provides the empirical basis for the main analysis range of this chapter and for the two density thresholds.

**The ultra-low-density regime $[0, 10^{-4})$ is a small-sample artifact band.** Its median H is only 0.33 and frac($H$<0.3) is as high as 43.5%, which at first glance looks like a “monosemantic feature belt.” But features in this interval typically activate only 1–2 times over 10,000 evaluation samples, so $P(c\mid f)$ almost inevitably collapses to a single class under finite sampling. The low H here is therefore a statistical artifact rather than real monosemanticity—this is also why the later validation stage imposes a lower density bound at $10^{-3}$.

**The high-density regime $[10^{-2}, 1]$ is dominated by high-frequency polysemantic features.** Here the median H rises to 0.73, frac($H$>0.7) reaches 57.2%, and frac($H$<0.3) drops to only 5.9%. Such features often correspond to syntactic or otherwise generic high-frequency patterns. Including them in the aggregation would systematically raise $\text{mean }H$ and hide the low-H tail that sparse SAEs would otherwise exhibit; for this reason, the chapter imposes an upper density bound of $10^{-2}$ for aggregation.

**The middle range $[10^{-4}, 10^{-2})$ is the main study region of this chapter.** Within this interval, H increases only mildly with density (median H rises from 0.59 to 0.65), but the low-H tail remains stable: frac($H$<0.3) stays consistently around 10%–11%. This means that **truly low-H features are not confined to the extreme low-frequency end, but are distributed across the entire middle band**. Taken together, the three-part structure implies that low H in $[0,10^{-4})$ is not trustworthy, the $[10^{-2},1]$ region mostly contains generic patterns such as lexical or syntactic ones, and only the middle band preserves reliable monosemantic candidates. This is precisely the empirical basis for the later use of both H and density in candidate filtering.

### 2.5 The Spurious Low-Entropy Trap

Beyond the main results above, there is a separate methodological trap that directly affects the interpretation of the metrics: if the `"O"` (non-entity) class is **not removed**, and one instead treats the results from the withO version of `pii-masking-300k` (denoted below as **PII-withO**) as evidence of monosemanticity, one obtains a set of numbers that look extremely strong but are in fact misleading:

**Table 2.5 PII-withO vs. PII-noO: comparison illustrating spurious low entropy**

| Dataset | Includes O? | # Classes | Mean H | Median H | Mean KL | H<0.3 fraction | H<0.5 fraction |
| :-----: | :---------: | :-------: | :----: | :------: | :-----: | :------------: | :------------: |
| PII-noO | No | 25 | 0.4782 | 0.4996 | 2.4167 | 0.235 | 0.490 |
| PII-withO | Yes | 26 | 0.1962 | 0.1087 | 0.8205 | 0.743 | 0.892 |

Table 2.5 shows that on PII-withO, the mean H drops from 0.48 to 0.20, the median H from 0.50 to 0.11, and the fraction of features with `H<0.3` rises from 24% to 74%. Looking only at H, one might almost conclude that “three quarters of the SAE features are monosemantic.” But this is exactly a canonical spurious low-entropy case.

The source of the trap is precisely the issue already discussed in Chapter 1: **when `"O"` accounts for the overwhelming majority of tokens (80%+), a mediocre feature that merely follows the background distribution will also concentrate its $P(c \mid f)$ on the `"O"` class, and therefore receive a very low H.** But H measures only how concentrated the distribution is; it cannot distinguish between concentration on a meaningful entity class and concentration on a trivial background class.

**KL provides the necessary correction signal here.** The mean KL drops from 2.42 on PII-noO to 0.82 on PII-withO, less than one third of the PII-noO value. This indicates that those features which appear “low-H” under the withO setting have class distributions very close to the data prior $Q$, with little deviation from it, and are therefore more plausibly tracking the background distribution than expressing truly discriminative concepts. In other words, under such strongly imbalanced class priors, **low H must be interpreted together with KL**.

For this reason, the main experiments in this work use PII-noO, which removes the `"O"` class from evaluation and lets H carry the primary judgment in a setting without a dominant background class, while retaining KL as the necessary correction signal for background effects.

### 2.6 Chapter Summary

The empirical results of this chapter can be summarized by two main observations: first, **in fine-grained token-level settings, H/KL can effectively open up the structure of monosemanticity**; second, **this structure is shaped jointly by label granularity, SAE sparsity, and the interaction between layer and L0**. The main conclusions and limitations are summarized below.

Main conclusions:

- **Label granularity is the primary factor determining whether H/KL has discriminative power.** In fine-grained token-level settings such as PII-noO, the dynamic range of H/KL can be effectively opened up: the mean H is around 0.48, and the fraction of features with `H<0.3` reaches 23.5%.
- **SAE sparsity is the dominant hyperparameter affecting monosemanticity.** The sparser the SAE, the more monosemantic the features. This trend appears not only in the means of H/KL, but also in the fraction and absolute number of features in the low-H tail.
- **The role of layer depends on the match between the conceptual unit of the labels and the hierarchy of the residual stream.** On document-level topic tasks, layer 12 consistently dominates; on token-level PII tasks, the layer effect is modulated by L0 and exhibits a clear interaction-driven flip.
- **At the feature level, $H$ exhibits a three-part structure as a function of density.** The interval $[0,10^{-4})$ is a small-sample artifact band, $[10^{-2},1]$ is dominated by high-frequency generic features, and only the middle range $[10^{-4},10^{-2})$ is the main region in which the low-H tail is concentrated.

Limitations:

- **Resolution remains limited on document-level tasks.** On `ag_news` and `dbpedia14`, the H distributions remain compressed in the high-value regime.
- **H/KL is more sensitive to class-specific monosemantic candidates.** For high-frequency generic features that are shared across classes, H/KL may still produce weak class-alignment signals even when their internal semantics are relatively stable; therefore, the conclusions of this chapter primarily characterize the structure of high-class-alignment features.
- **The layer × L0 interaction still lacks direct mechanistic evidence.** For example, the apparent reversal in which shallow layers overtake deep ones under high L0 on PII-noO remains largely a qualitative interpretation at present.
- **The `withO` setting induces a canonical spurious low-entropy phenomenon.** When a dominant background class is present, low H cannot be treated directly as evidence of monosemanticity and must be interpreted together with KL. This is the direct reason why the main experiments use PII-noO.

---

## 3. Reverse Cross-Validation of H/KL

### 3.1 Motivation

Chapters 1 and 2 use H/KL to describe how concentrated a feature’s class distribution is, and how far it deviates from the data prior. Both metrics are computed in the same direction: **Feature → Concept**. They answer the question: “given one activation of a feature, which class is it most likely to come from?” Chapter 3 therefore introduces a reverse external validation direction: **Concept → Feature**. Given all tokens of a class, the question is no longer “which class do these activations belong to?”, but rather “how many instances of this class are covered by the top-k features selected for it?”

This corresponds directly to Recall, which is the core evidence for whether H/KL is genuinely useful: if the features selected by low H / high KL are indeed more monosemantic, then they should activate stably on instances of the corresponding class and achieve higher coverage.

Precision and Recall are complementary here. Recall measures coverage, while Precision measures whether those features truly concentrate their activations on the target class. Ideally, a highly class-aligned feature should achieve both high Recall and high Precision. By contrast, a feature whose class alignment is spuriously inflated because it hits a class only on a few samples will usually be exposed first by Recall.

Accordingly, Chapter 4 focuses on two comparisons: the Recall gain relative to the random baseline answers whether “H/KL ranking itself is effective,” while the Precision advantage relative to `density` / `mi` answers whether “this effectiveness is merely a side effect of picking high-frequency features.”

### 3.2 P/R Evaluation Framework

Let $\mathcal{F}$ denote the set of features participating in evaluation, and $\mathcal{C}$ the class space. For each feature $f \in \mathcal{F}$, we first assign it to a unique main class using the existing $P(c \mid f)$ from Chapter 1:

$$
c_f \;=\; \arg\max_{c \in \mathcal{C}} P(c \mid f)
\;=\; \arg\max_{c} \frac{\sum_{t:\, y_t = c} a_f(t)}{\sum_{t} a_f(t)} .
$$

Here $a_f(t)$ is the activation value of feature $f$ on token $t$, and $y_t$ is the class label of that token. Both numerator and denominator are activation-weighted, exactly matching the definition of $P(c \mid f)$ in Chapter 1. We then proceed as follows: for each evaluation class $c$, we first collect the candidate features whose main class is $c$. For each ranking group $g$, we then select the top-$k$ features from this pool according to its score function $s_g$:

$$
\mathcal{T}_{g,c,k} \;=\; \text{top-}k\bigl(\{f : c_f = c\},\; s_g\bigr)
$$

where $s_g$ may be KL, H, density, and so on; the concrete ranking groups are defined in Section 3.3.

This immediately gives the token-level frequency precision. We view the $k$ features in $\mathcal{T}_{g,c,k}$ as an OR-union classifier: a token $t$ is predicted as class $c$ if and only if at least one feature in the set activates on $t$. Let $\mathbb{1}_{\text{hit}}(t) = \mathbb{1}[\exists f \in \mathcal{T}_{g,c,k}: a_f(t) > 0]$, and let $N_c$ be the total number of tokens of class $c$. Then:

$$
\text{TP}(c) = \sum_t \mathbb{1}_{\text{hit}}(t)\,\mathbb{1}[y_t = c],\quad
\text{FP}(c) = \sum_t \mathbb{1}_{\text{hit}}(t)\,\mathbb{1}[y_t \ne c],\quad
\text{FN}(c) = N_c - \text{TP}(c)
$$

$$
P_{\text{tok}}(g,c,k) = \frac{\text{TP}(c)}{\text{TP}(c)+\text{FP}(c)},\quad
R_{\text{tok}}(g,c,k) = \frac{\text{TP}(c)}{\text{TP}(c)+\text{FN}(c)} .
$$

Beyond frequency precision, we also define amplitude-weighted precision. Let the summed activation over the $k$ features on token $t$ be $w(t) = \sum_{f \in \mathcal{T}_{g,c,k}} a_f(t)$. Then:

$$
P_{\text{amp}}(g,c,k) \;=\; \frac{\sum_t w(t)\,\mathbb{1}_{\text{hit}}(t)\,\mathbb{1}[y_t = c]}{\sum_t w(t)\,\mathbb{1}_{\text{hit}}(t)} .
$$

For token-level entity tasks such as PII, span-level evaluation is also necessary. We merge each contiguous sequence of tokens with the same label into a span instance (indexed by $s$, with class $y_s$). A span is counted as hit if any token inside it is hit by the OR-union classifier. Let $\mathbb{1}^{\text{spn}}_{\text{hit}}(s) = \mathbb{1}[\exists t \in s: \mathbb{1}_{\text{hit}}(t)=1]$, and let $N^{\text{spn}}_c$ be the total number of spans of class $c$. Then:

$$
\text{TP}_{\text{spn}}(c) = \sum_s \mathbb{1}^{\text{spn}}_{\text{hit}}(s)\,\mathbb{1}[y_s = c], \quad
\text{FP}_{\text{spn}}(c) = \sum_s \mathbb{1}^{\text{spn}}_{\text{hit}}(s)\,\mathbb{1}[y_s \ne c], \quad
\text{FN}_{\text{spn}}(c) = N^{\text{spn}}_c - \text{TP}_{\text{spn}}(c)
$$

$$
P_{\text{spn}}(g,c,k) = \frac{\text{TP}_{\text{spn}}(c)}{\text{TP}_{\text{spn}}(c)+\text{FP}_{\text{spn}}(c)}, \quad
R_{\text{spn}}(g,c,k) = \frac{\text{TP}_{\text{spn}}(c)}{N^{\text{spn}}_c} .
$$

Finally, for each group/k pair, we compute a macro average over classes.

### 3.3 Candidate Features and Ranking Groups

Section 3.2 introduced the formal P/R definitions. The next question is: **how is the candidate pool for each class formed, and how are candidate features ranked?** This section focuses on the construction of candidate pools, ranking groups, and baselines.

We use six ranking groups: `h` / `kl` test the effectiveness of H and KL as ranking signals; `h_f` / `kl_f` test their more robust versions with a density floor; and `density` / `mi` serve as frequency-dominated control baselines. In addition, a random baseline is constructed separately within the same candidate pool in order to test whether these rankings actually outperform random sampling.

All ranking groups share the same candidate pool: the **alive filter** defined in Chapter 1 ($T_f \ge 10^{-5}$, removing dead features). Unlike the aggregation statistics in Chapter 2, the P/R stage **does not impose an upper density bound**, because `density` and `mi` themselves are control baselines defined in terms of density. Excluding high-frequency features beforehand would destroy the comparability of those baselines and would no longer answer the central question “does H/KL outperform simply ranking by frequency?” Before ranking, only dead features are removed; high-frequency generic features remain in the pool and are scored by each ranking rule. On top of this shared candidate pool, each group selects top-k according to its own score function:

| Group | Ranking rule | Additional density floor | Role |
|---|---|---|---|
| `kl` | KL descending | none | raw KL ranking — deviation from prior only |
| `h` | H ascending | none | raw H ranking — concentration only |
| `density` | density descending | none | frequency ranking — a counterexample baseline for “high-frequency features are good features” |
| `mi` | density × KL descending | none | MI-like ranking — a tradeoff between frequency and deviation |
| `kl_f` | KL descending | 0.001 | KL ranking + removal of extremely low-frequency noise |
| `h_f` | H ascending | 0.001 | H ranking + the same low-frequency floor |

Here “additional density floor” means an extra constraint layered on top of the shared candidate pool: `kl_f` / `h_f` require `density ≥ 0.001`, while the raw groups impose no density constraint and rank directly over the entire alive pool. For H, lower values correspond to more concentrated class distributions, and thus are ranked in ascending order.

`density` and `mi` are counterfactual baselines for the question “can the metric select truly monosemantic features?” If “high frequency implies good,” then density should dominate KL; Chapter 4 will show that this does not happen. The density floor in `kl_f` / `h_f` is included only to prevent ultra-rare features with 1–2 activations from entering top-k by accident; such features are not meaningful under P/R regardless of whether their scores are high or low. The floor is therefore best understood as a **noise floor**, not a core methodological constraint. The paired presence of raw `kl` / `h` and floored `kl_f` / `h_f` is precisely intended to isolate the effect of this floor.

Accompanying these ranking groups is the **random baseline**: for each class $c$ and each $k$, we uniformly sample $k$ features from the main-class candidate pool $\{f : c_f = c\}$ of that class, and average over multiple repetitions. Because it uses the same candidate pool as $\mathcal{T}_{g,c,k}$, its difference from the other ranking groups is reduced essentially to one factor—whether or not sorting by score is used—without mixing in differences in candidate-pool size or class sample size. If the Recall of some ranking group is close to random, then the ranking itself is not providing an effective signal.

In the experiments, $k$ ranges over $\{1, 5, 10, 20\}$, covering scales from the single-feature regime to multi-feature combinations. Once candidate features are selected, the remaining question is how these $k$ features should be combined into a class-decision rule, and on what units Precision and Recall should be computed.

### 3.4 Evaluation Setup

Section 3.2 introduced the formal P/R definitions, and Section 3.3 explained how candidate features are produced. This section specifies four evaluation choices: how top-k features are combined, the unit of Recall, the computation of Precision, and the aggregation rule over classes.

**Feature combination.** We combine top-k features using OR-union: if any one of them activates, the event counts as a positive prediction for the class. This lets us directly test whether multiple candidate features for the same class improve coverage when combined, without introducing extra ensemble mechanisms such as AND rules or voting. $k=1$ corresponds to a single-feature view, while $k=20$ corresponds to a more permissive multi-feature ensemble view.

**Unit of Recall.** Since a single entity often spans multiple tokens, token-level Recall underestimates features that reliably hit only part of an entity. Span-level Recall is therefore closer to the question “was an entity instance found?”, whereas token-level Recall more strictly evaluates coverage of the tokens inside each instance.

**Computation of Precision.** Each top-k feature set gives rise to two versions of Precision: frequency Precision, which counts the fraction of target-class tokens, and amplitude Precision, which weights by activation magnitude. We use amplitude Precision as the primary version, because FP activations are typically systematically weaker than TP activations, making frequency Precision prone to underestimating the true monosemanticity of the feature group. At the same time, Recall remains a strictly non-circular external validation signal, so we always place Recall first and treat amplitude Precision as auxiliary interpretation.

Under these settings, Chapter 4 focuses on two main comparisons: first, the Recall gain of `h_f` / `kl_f` relative to `random`, which answers directly whether “H/KL ranking itself is effective”; second, the amplitude-Precision gap between `h_f` / `kl_f` and `density` / `mi`, which answers whether “that effectiveness is merely a byproduct of selecting high-frequency features.”

---

## 4. P/R Results and Analysis

This chapter uses the P/R validation framework defined in Chapter 3 to test whether H/KL ranking selects SAE features with **stronger class alignment**; under the proxy assumption in Section 1.1, this is also necessary evidence for monosemanticity. We first present the core results, and then examine the evidence for the validity of H/KL, the rationale for the primary evaluation metrics, and the extent to which the sparsity and layer conclusions from Chapter 2 are confirmed. All data in this chapter are macro-averaged over the 15 SAEs (3 layers × 5 L0 levels) on PII-noO.

### 4.1 Core Results

To keep the main line clear, the main text reports only the primary metric pair `(ampP, spnR)`, where `ampP` denotes amplitude Precision and `spnR` denotes span Recall, and focuses on five key comparison groups: `h_f` / `kl_f` / `density` / `mi` / `random`. The complete result table is given in Appendix A.

Table 4.1 Results of the five key comparison groups under the primary metric pair `(ampP, spnR)` at different values of `k`.

| k | `h_f` | `kl_f` | `density` | `mi` | `random` |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.858 / 0.404 | 0.836 / 0.386 | 0.251 / 0.830 | 0.438 / 0.887 | 0.448 / 0.105 |
| 5  | 0.772 / 0.799 | 0.763 / 0.782 | 0.295 / 0.973 | 0.406 / 0.974 | 0.389 / 0.391 |
| 10 | 0.709 / 0.902 | 0.705 / 0.892 | 0.315 / 0.985 | 0.398 / 0.983 | 0.379 / 0.573 |
| 20 | 0.631 / 0.955 | 0.631 / 0.949 | 0.332 / 0.989 | 0.388 / 0.988 | 0.372 / 0.748 |

Two facts can be read off immediately from Table 4.1. First, `h_f` / `kl_f` maintain relatively high coverage while achieving substantially higher amplitude Precision than `density` / `mi`, showing that H/KL ranking does not simply pick high-frequency features. Second, `h_f` and `kl_f` are broadly similar, but `h_f` holds a slight edge. The difference between the raw `h` / `kl` groups and the floored groups, as well as the complete token/span results, are deferred to the later discussion and Appendix A.

### 4.2 Evidence for the Validity of H/KL

First, the density floor is a necessary prerequisite. Compared with `kl_f` and `h_f`, the raw `kl` and `h` groups retain high Precision but have Recall that is lower by more than an order of magnitude. At k=1, for example, `kl` has spnR of only **0.028**, while `kl_f` rises to **0.386**; `h` has spnR of only **0.026**, while `h_f` rises to **0.404**.

The top-1 features selected by raw KL/H from the full candidate pool are often features that activated only 1–2 times, with those few activations happening to fall in a single class. Their $P(c \mid f)$ looks almost perfect, but across the full evaluation set they barely fire at all, yielding a degenerate solution with “high Precision, near-zero Recall.” The `min_density` floor (0.001 by default) is therefore not an optional tweak, but a necessary condition for KL/H ranking to generate nontrivial coverage. The main discussion below is based on the four groups that do have coverage: `h_f` / `kl_f` / `density` / `mi`.

Second, H-ranking is slightly better than KL-ranking. For every value of $k$, the spnR of `h_f` is slightly higher than that of `kl_f` (k=5: 0.799 vs 0.782; k=10: 0.902 vs 0.892; k=20: 0.955 vs 0.949). The same direction holds for ampP, though the difference is smaller.

The underlying reason is that after assignment by $\arg\max_c P(c \mid f)$, the ranking problem has already reduced from “does this feature contain signal?” to “is this feature a good representative of class $c_f$?” At that point, what matters is how concentrated $P(\cdot \mid f)$ is on $c_f$. H measures this directly, whereas KL also rewards cases where “the remaining probability mass is concentrated on rare classes.” Because $\log(1/Q(c'))$ grows explosively for rare classes, KL can rank too highly a feature that is not pure enough for the target class but also responds strongly to another rare class.

For example, suppose feature A satisfies $P(c_f \mid A)=0.85$ and spreads the remaining mass uniformly across common classes, while feature B satisfies $P(c_f \mid B)=0.75$ and concentrates the remaining mass on another rare class $c'$. Under the goal of selecting a representative feature for class $c_f$, A is clearly more concentrated and purer. H therefore favors A, while KL may rank B higher because of the extra reward induced by the rare class. For this reason, the remainder of the chapter uses **`h_f` as the default reference group**.

Finally, one still has to rule out a key alternative explanation: is the high Recall of `h_f` merely because it incidentally picked high-frequency features? To answer this, `density` (pure frequency), `mi` (a KL variant with frequency bias), and `random` must all be included in the comparison. The key numbers at k=5 are:

Table 4.2 Main-metric comparison of `h_f`, `density`, `mi`, and `random` at k=5.

| Group | ampP | spnR |
|---|---:|---:|
| `h_f` | **0.772** | 0.799 |
| `density` | 0.295 | 0.973 |
| `mi` | 0.406 | 0.974 |
| `random` | 0.389 | 0.391 |

This comparison establishes three points. First, the spnR of `h_f` is substantially higher than `random`: 0.799 vs 0.391 at k=5, and 0.902 vs 0.573 at k=10. This shows that H-ranking is not something that would hold under random selection from the candidate pool. Second, although `density` / `mi` achieve near-perfect spnR, their ampP is only around 0.30 / 0.41. They are selecting generic high-frequency features that activate easily across many classes, rather than good class representatives. Third, the ampP of `h_f` is substantially higher than that of `density` / `mi`, so its Recall cannot be explained as merely “it happened to pick high-frequency features.” In short, `h_f` achieves both coverage and purity, and this is the core evidence that H/KL ranking is valid.

### 4.3 Choice of the Primary Evaluation Metrics

This section explains from the Precision and Recall sides, respectively, why the main text uses `(ampP, spnR)` as the primary metric pair.

Table 4.3 Comparison between tokP and ampP for `h_f` under different values of `k`.

| k | tokP | ampP | Δ |
|---:|---:|---:|---:|
| 1  | 0.821 | 0.858 | +0.037 |
| 5  | 0.632 | 0.772 | +0.140 |
| 10 | 0.492 | 0.709 | +0.217 |
| 20 | 0.356 | 0.631 | +0.275 |

The larger k is, the larger the gap becomes. At k=20, tokP is only 0.356, while ampP still reaches 0.631. This is not a contradiction, but directly reflects the point emphasized in Section 3.4: **FP activations are systematically weaker in magnitude than TP activations.**

Frequency Precision treats every FP activation as an equally weighted error, but in SAE systems many FP activations have amplitudes only a fraction of TP amplitudes and are therefore closer to noise. Amplitude Precision weights by activation magnitude, so it better answers the question “how much of the total activation budget of this feature group is actually spent on the target class?” This becomes especially clear in comparison with the random baseline: at k=20, `h_f` has ampP=0.631, while random has only 0.372; the gap under tokP is smaller. Therefore, **ampP is more suitable than tokP as the primary Precision metric in this work**.

On the Recall side, at k=5 `h_f` has tokR=0.497 and spnR=0.799; at k=10 it has tokR=0.666 and spnR=0.902. After tokenization, multi-token entities often span 3–8 subwords. Span-level Recall collapses the tokens within the same instance into one evaluation unit: as long as any top-k feature activates on any token within that instance, the instance is counted as found. The fact that spnR is substantially higher than tokR shows that these features usually do fire somewhere on PII instances, even if they do not cover every subword within the instance.

The fact that tokR is substantially lower does not mean, in all cases, that “the feature really missed the token.” The gap between tokR and spnR has two sources. One is a “false miss”: some tokens appear in the contexts of multiple PII classes at once (typical examples include spaces, punctuation, and cross-class digit strings). Features that activate on such tokens have their $P(c \mid f)$ spread over multiple classes, which raises H and pushes them down in H-ascending ranking, so they rarely enter top-k. The features that do get selected are often highly monosemantic ones that deliberately avoid such ambiguous tokens, and tokR penalizes them as if they had “failed to cover” those ambiguous tokens. The other source is a “true miss”: some subwords that appear only inside spans of the target class are indeed not hit by any top-k feature, and this part constitutes the true Recall gap.

spnR collapses all tokens inside an instance into a single evaluation unit: as long as any top-k feature activates on any token in the instance, the whole instance is counted as hit. spnR therefore reflects only the “true misses,” and is not dragged down by these “false misses.” From the downstream perspective of “can a full entity instance be identified?”, **spnR is closer to the actual PII task and is therefore the more appropriate primary Recall metric in this work**.

### 4.4 Validation of Sparsity and Layer Effects

We begin with sparsity. Chapter 2 showed the pattern “lower L0 → lower H (more monosemantic).” For ease of comparison, we define within-layer L0 ranks: for each layer, we sort its five SAEs by L0 from smallest to largest and merge equal ranks across layers into the same tier. Thus, tier 0 means “the sparsest SAE in each layer,” and tier 4 means “the densest SAE in each layer,” with each tier containing exactly three SAEs (one per layer). On this basis, we aggregate `h_f` at k=5:

Table 4.4 `h_f` ampP and spnR across L0 tiers (k=5).

| L0 tier | h_f ampP | h_f spnR |
|:---:|:---:|:---:|
| 0 (sparse) | **0.816** | 0.787 |
| 1 | 0.802 | 0.791 |
| 2 | 0.778 | 0.813 |
| 3 | 0.753 | 0.803 |
| 4 (dense) | 0.710 | 0.802 |

The table shows that ampP decreases monotonically from 0.82 at tier 0 to 0.71 at tier 4, indicating that the top-5 features selected from sparse SAEs have substantially higher amplitude purity. This is a direct **class-alignment** signal, and under the proxy assumption of Section 1.1, also supports stronger monosemanticity.

By contrast, spnR is roughly flat or even slightly increasing (tier 0: 0.787 → tier 4: 0.802), but this does not mean that higher-L0 SAEs are more monosemantic. A more plausible explanation is that dense SAEs have more alive features (Chapter 2 showed about 72% alive at L0=22 versus about 99% alive at L0=445), so the same-class candidate pools are larger and the k=5 top-k more easily forms a feature set that covers most spans of the class. In other words, dense SAEs exchange “multi-feature redundancy” for a small gain in OR-union coverage, rather than making each individual feature more monosemantic.

What truly reflects class alignment here is still ampP, because it measures “how much of the total activation budget of the selected feature set is actually spent on the target class,” and is not affected by candidate-pool size. Therefore, the H curves in Chapter 2 and the ampP curve here point in the same direction: **sparsity constraints improve class alignment**, which under the proxy assumption also implies stronger monosemanticity.

We next consider layer effects. Relative to L0 tiers, the differences across layers are much smaller (five SAEs per layer across five L0 values): in ampP, layer 19 exceeds layer 5 by only 5.3 points (0.741 vs 0.688), and in spnR, layer 12 exceeds layer 5 by 3.0 points (0.919 vs 0.889), though the directions are still clear.

Table 4.5 `h_f` ampP and spnR across layers (k=10).

| layer | h_f ampP (k=10) | h_f spnR (k=10) |
|:---:|:---:|:---:|
| 5  | 0.688 | 0.889 |
| 12 | 0.699 | **0.919** |
| 19 | **0.741** | 0.897 |

Layer 19 has the highest ampP, while layer 12 has the highest spnR, corresponding to a division of labor in which deeper-layer single-feature candidates are purer and middle-layer candidate sets are more complete within class. Section 2.3 showed that the H ordering across layers on PII-noO **flips with L0** (layer 19 is best at low L0, but layer 5 overtakes it at high L0). When split by L0 tier, however, **this flip is not transmitted to ampP / spnR**: the ampP of layer 19 and the spnR of layer 12 remain ranked first across all five L0 tiers. One interpretation is that the H flip is a property of the **distribution center** of the features, whereas `h_f` top-k uses only the **tail of a small number of features** in each layer; this tail remains stably stronger for layers 19 / 12 than for layer 5, effectively filtering out the interaction. Since the cross-layer gap is also clearly smaller than the L0-tier gap, layer is treated only as a secondary observation in this section.

### 4.5 Chapter Summary

The core conclusions are as follows. First, the P/R results validate H/KL as an effective ranking signal for **class alignment**, and under the proxy assumption in Section 1.1 support its use as an operational indicator of monosemanticity. The following three pieces of evidence support this conclusion jointly:

1. **Non-degenerate**: once the density floor is added, the spnR of `h_f` rises from 0.026 to 0.40 (k=1), 0.80 (k=5), and 0.90 (k=10), showing that the features selected by H ranking do activate stably on the target class rather than forming a degenerate solution that never really fires.
2. **Non-random**: `h_f` exceeds the random baseline by 30–40 points in spnR (0.799 vs 0.391 at k=5), and by 26 points in ampP at k=20. The H ranking itself is therefore providing a class-alignment signal; if H did not distinguish aligned from non-aligned features, these numbers should be close.
3. **Not merely high-frequency**: using `density` / `mi` as counterfactual baselines for the claim “if high frequency ≈ good,” their ampP fluctuates only between 0.25 and 0.44, while `h_f` reaches 0.63–0.86, about 2–3× larger than `density`. The H ranking is not selecting high-frequency features, but genuinely concentrated ones.

Thus, H/KL is effective as a **class-alignment ranking signal** for SAE features, and under the proxy assumption can be used as an **operational indicator of monosemanticity**. P/R provides an independent non-circular external confirmation and rules out alternative explanations based on random baselines, high-frequency baselines, and low-frequency degeneracy. The metric pair `(ampP, spnR)` can therefore be used directly in later comparisons across SAE families.

Limitations:
- **There is still a ceiling in the absolute numbers**: even in the best configuration, the highest `h_f` ampP reaches only 0.86 (k=1), leaving a substantial gap from the ideal value of 1.0 for a perfectly monosemantic feature. This may come from (a) intrinsic similarity between PII classes (e.g. FIRSTNAME and LASTNAME are themselves hard to distinguish), and (b) limitations intrinsic to this particular gemma-scope SAE family. Disentangling these requires comparison across multiple SAE families, which is outside the scope of this work.
- **The jump in Recall from k=1 to k=5** (`h_f` spnR: 0.404 → 0.799) suggests that most classes have 2–4 “synonymous subfeatures,” and that genuine “one class, one feature” behavior is uncommon. This is a form of redundancy under wide dictionaries and is consistent with the “feature splitting” phenomenon observed in Templeton et al. 2024.

---

## 5. Related Work

Several technical routes have emerged for evaluating SAE feature monosemanticity. Among them, auto-interp (Section 5.1) most directly targets “one feature, one concept” style monosemanticity; probing (Section 5.2) measures **concept separability**; intervention-based methods (Section 5.3) measure **causal behavioral effects**; and FMS (Section 5.4) attempts to quantify monosemanticity with a feature-level mathematical score. In addition, there is a **reconstruction-fidelity** route (Section 5.5). Although it does not directly measure monosemanticity, it uses information-theoretic quantities, as this work also does, to provide a model-level usability baseline, and is therefore included in the comparison here. By contrast, the H/KL framework proposed in this work more directly measures **class alignment**, i.e., an operational proxy for monosemanticity. For a consistent comparison, this chapter evaluates these methods along seven dimensions: **measurement target, whether supervision is required, whether monosemanticity is directly evaluated, cost and scalability, support for full-dictionary evaluation, alignment with downstream tasks, and whether independent external validation is available**. We discuss auto-interp, probing, intervention, FMS, and reconstruction fidelity in turn, and then summarize the position and contributions of this work.

### 5.1 Manual / Automatic Interpretability Annotation (manual & auto-interp)

This route is represented by Bricken et al. 2023, Templeton et al. 2024, and Bills et al. 2023. Its basic pipeline is to generate a natural-language description from high-activation examples of a single feature, and then ask a human or another LLM to evaluate the description in terms of **specificity** and **sensitivity** with respect to subsequent activations. The measurement target of auto-interp is therefore **the semantic description of a single feature and the quality of that description**, making it the route in this chapter that most directly corresponds to “strict monosemanticity.”

From the comparison perspective, the strength of auto-interp is that its output is naturally readable and can capture semantic patterns that remain invisible under category-label-based settings. Its cost is equally clear: it depends on external raters, is the most expensive route, usually covers only top-N candidates rather than the full dictionary, and is the least economical for large-scale comparisons across many SAEs. Its relation to downstream tasks is not fixed either; it depends on whether the feature interpretation happens to match the semantic unit that the task actually cares about.

Its relation to H/KL is better understood as a division of labor rather than substitution. Auto-interp answers “what is this feature saying?”, whereas H/KL answers “how cleanly is this feature aligned with predefined categories?” This matches the hierarchy established in Section 1.1: H/KL provides class alignment as a proxy for monosemanticity, and auto-interp then supplies positive evidence at the “one feature, one concept” level. When the two are chained together, one can first use `h_f` to select top-N class-aligned candidates out of 16k features, and then run auto-interp only on those candidates, reducing the LLM budget from $O(F)$ to $O(N)$.

### 5.2 Probing-Based Methods (supervised probe)

This line includes sparse probing, neuron-level linear probes, and SCR / TPP in SAEBench. Its basic idea is to train a supervised probe on SAE activations and use accuracy to test whether a concept can be read out from that activation set. Its measurement target is therefore **concept separability**: whether information is present and sufficiently linearly readable, rather than whether a single feature is already aligned with one category.

This is also the most fundamental difference between probing and monosemanticity. Probe accuracy is a dictionary-level set property: a category may be readable by a combination of many features, each contributing only part of the signal, in which case accuracy can be high without implying that there exists any feature-level “clean representative.” Probing is therefore better suited to answering “does the SAE expose this concept?” rather than “which feature monosemantically represents this concept?” Its strengths are clear supervision, a single scalar output, and direct relevance to downstream task separability; its limitations are that it cannot naturally localize a single latent and is also affected by one-vs-rest design choices and covariate redundancy.

Compared with the H/KL framework in this work, the key difference lies in measurement granularity and the role played by labels. Probing uses task labels as supervision and asks whether the activation set can read out a concept; H/KL also uses labels, but does not train a classifier, and instead directly examines whether the activation distribution of a single feature is concentrated on one category within a fixed class space. The former can show that “the information exists somewhere in this representation set,” whereas the latter tries to answer “which features look more like representatives of this category.” The two are therefore better understood as complementary measurements rather than mutually substitutable evaluation criteria.

### 5.3 Causal / Intervention-Based Methods (causal intervention, steering, attribution)

This family includes steering, ablation, activation patching, and sparse feature circuits. Its core idea is to directly manipulate the activation of candidate features and then observe the resulting change in output distributions or downstream task behavior, so its measurement target is **the causal behavioral effect of candidate features**. Relative to the previous routes, it is closest to the real control objective and most directly corresponds to the question “if this feature is changed, does model behavior change?”

This strength also determines its boundary. Intervention methods do not directly evaluate monosemanticity and are not well suited to scanning the full dictionary. In practice, the expensive step is often not the intervention itself but the candidate selection before it. If candidate selection is poor, no amount of detailed forward analysis afterwards will help. Behavioral change itself also cannot be equated with monosemanticity: if clamping a feature hurts performance, the feature may simply be one redundant component in a distributed representation rather than a monosemantic carrier of a single concept.

Accordingly, the most suitable role of H/KL here is as a candidate-pool generator. Using `h_f` top-k to first select a small set of high-class-alignment candidates for each category from 16k features turns the question “which features are worth intervening on?” from a manual heuristic into a reproducible unsupervised ranking, and concentrates expensive causal experiments on a subset that is more likely to matter.

### 5.4 FMS (Feature Monosemanticity Score)

The Feature Monosemanticity Score (FMS) proposed by Bussmann et al. 2024 represents a feature-level mathematical evaluation route. Instead of relying on expensive natural-language interpretation and second-stage scoring as in auto-interp, it assigns each feature a monosemanticity score based on activation purity on a specific concept dataset. In terms of positioning, this route provides a representative low-cost and batch-computable way to quantify SAE monosemanticity.

Its advantages are that the output remains a feature-level scalar and is therefore suitable for large-scale comparison, while also supporting offline evaluation after training. Its limitations are also clear: FMS depends on concept-labeled datasets to construct positive and negative samples, and uses tree classifiers to estimate single-feature capacity together with local and global disentanglement. As a result, the score depends simultaneously on concept granularity, label partitioning, and the specific classifier implementation. In other words, it measures whether “under a given concept set and a given classifier, a feature cleanly carries that concept,” rather than a universal notion of monosemanticity independent of task definition and annotation structure.

Compared with this work, the key difference between FMS and H/KL lies in the role played by labeled data. FMS uses concept labels: it constructs positive and negative examples around a target concept and then asks whether a feature cleanly carries that concept. H/KL uses predefined class labels: within a fixed class space, it asks whether a feature’s activation distribution is concentrated on one category and interprets that concentration as a proxy for class alignment. Both aim at low-cost, feature-level, batch-computable evaluation, but the former is more concept-centric direct scoring, whereas the latter is more class-centric characterization of distributional structure.

### 5.5 Reconstruction Fidelity / Model Behavior Preservation

This line is represented by Bricken et al. 2023, Cunningham et al. 2023, and Rajamanoharan et al. 2024, and is also one of the main metrics in the SAEBench `core` module. Its basic procedure is to replace the original activation at some model layer with the SAE reconstruction $\text{Dec}(\text{Enc}(x))$, continue the forward pass to the final logits, and then measure the output difference before and after replacement using information-theoretic quantities. Two common readouts are **$\mathrm{KL}_{\mathrm{logits}}$** (the KL divergence between the original-model logits and the SAE-reconstructed-model logits) and **Δ CE loss** (the increase in next-token-prediction cross-entropy).

What this method measures is **the overall reconstruction fidelity of the SAE**, not the quality of individual features. Although it also uses information-theoretic quantities, the question it addresses is orthogonal to H/KL: reconstruction fidelity asks whether “the model behavior is preserved after replacing the layer with the SAE,” which is a **usability baseline**; H/KL asks whether “a feature is aligned to only a few categories,” which is a **proxy for monosemanticity**.

There may even be directional tension between the two. Pursuing lower $\mathrm{KL}_{\mathrm{logits}}$ in isolation tends to favor denser SAEs (higher L0), whereas Section 2 showed that **sparser SAEs tend to be more monosemantic**. In other words, on the tradeoff axis of “fidelity vs. monosemanticity,” the two kinds of metrics may point to different optima. They are therefore better understood as **complementary orthogonal dimensions**: reconstruction fidelity serves as a usability threshold at the SAE level, while H/KL serves as a feature-level proxy for monosemanticity.

### 5.6 Positioning and Contributions of This Work

The position of this work on this map can be summarized as follows: **it also seeks to evaluate SAE feature quality with low-cost feature-level scalars, but it does not define the score as monosemanticity itself; instead, it fixes the output as a feature-level proxy for class alignment and further adds independent external validation.** In Table 5.1, the term “proxy (class alignment)” is consistent with Section 1.1, while “independent external validation” corresponds to the P/R framework with `(ampP, spnR)` ultimately adopted in Chapters 3 and 4.

Table 5.1 Positioning comparison of the method families discussed in Chapter 5.

| Dimension | auto-interp | probing | intervention | FMS | reconstruction fidelity | H/KL (this work) |
|---|---|---|---|---|---|---|
| Measurement target | semantic description of a single feature and its description quality | dictionary-level concept separability | causal behavioral effect of candidate features | feature carrying purity with respect to a given concept | overall SAE reconstruction fidelity at the model-output level | feature class alignment under predefined categories |
| Requires supervision ¹ | no (but needs human/LLM scoring) | yes | depends on the task objective | yes (concept labels) | no (only model outputs) | yes (category labels) |
| Directly evaluates monosemanticity | most directly | no | no | relatively directly | no (does not measure monosemanticity) | no (proxy) |
| Cost and scalability | high, usually only top-N | medium, requires training a probe | high, usually requires candidate narrowing first | low, batch-computable | low, batch-computable | low, batch-computable |
| Supports full-dictionary evaluation | no | yes | no | yes | not applicable (SAE-level) | yes |
| Alignment with downstream tasks | depends on the interpretation and task semantics | high | strong | depends on the concept definition | medium (depends on the language-modeling objective) | depends on the category definition |
| Independent external validation | interpretation quality must be evaluated separately | no | behavior change itself is the readout | no | yes (model behavior itself is the ground truth) | yes |

¹ Here, “supervision” refers to the external labels/targets needed to train or compute the metric: auto-interp does not train a classifier but relies on human or LLM scoring; FMS relies on concept labels; H/KL relies on category labels; intervention methods can themselves be unsupervised, but the choice of target behavior usually depends on task definition.

Overall, the contribution of this work is not to propose yet another method that “directly defines monosemanticity,” but rather to provide a **low-cost, feature-level, externally validated** evaluation path. More specifically, the contributions of this work can be summarized in three points:

1. **A feature-level evaluation framework based on H/KL.** This framework formalizes class alignment under predefined categories as an operational proxy for monosemanticity, thereby providing a batch-computable structural metric for SAE features.
2. **A paired reverse-validation framework based on P/R.** This framework uses the independent metric pair `(ampP, spnR)` to test whether H/KL ranking is effective, filling a common gap in information-theoretic evaluation, namely the lack of non-circular external validation.
3. **A set of structural empirical conclusions derived from this framework.** The results show that label granularity determines whether H/KL has discriminative power; sparser SAEs are more likely to produce higher-purity candidate features; and layer effects exist, but are generally weaker than the main effect of sparsity.

Therefore, the relationship between this work and auto-interp, probing, intervention, FMS, and reconstruction fidelity is better understood as one of complementary division of labor rather than mutual substitution.

---

## 6. Experimental Setup

This chapter summarizes the model, data, label-processing rules, and the concrete H/KL settings used in the experiments above, to facilitate reproduction.

### 6.1 Model and SAE

All experiments use `google/gemma-2-2b` in bfloat16 precision. The SAE family is `gemma-scope-2b-pt-res`, with width fixed at 16k ($F = 16384$). We evaluate 15 configurations in total: the layers are `blocks.{5,12,19}.hook_resid_post`, and each layer contains 5 L0 levels, namely layer 5 ∈ {18, 34, 68, 143, 309}, layer 12 ∈ {22, 41, 82, 176, 445}, and layer 19 ∈ {23, 40, 73, 137, 279}.

All experiments can be run on a single A100 40GB GPU. The main memory cost during execution comes from the streaming accumulation of token-level activations, which is most evident in the H/KL setup described below.

### 6.2 Datasets and Label Settings

The three datasets cover two granularities (document-level vs. token-level):

| Dataset | Granularity | Number of Classes $C$ | Samples | Notes |
|---|---|---|---|---|
| `fancyzhx/ag_news` | doc-level | 4 | all test samples | single-label classification |
| `dbpedia14` | doc-level | 14 | all test samples | single-label classification |
| `ai4privacy/pii-masking-300k` | token-level | 25 / 26 | first 10,000 samples from the validation split | one BIO label per token |

The test sets of `fancyzhx/ag_news` and `dbpedia14` contain fewer than 10,000 examples, so we use all of them. For `pii-masking-300k`, the original labels include the non-entity class `O` by default ($C = 26$); most H/KL figures in Chapter 2 use the PII-noO setting, which removes `O` from the label set and yields $C = 25$. In addition, the `CARDISSUER` class has extremely few total samples in the validation split (< 20 tokens), which does not support stable P/R estimation, so it is excluded from macro-averaged P/R. As a result, the P/R results in Chapter 4 are macro-averaged over 24 classes.

### 6.3 H/KL Computation Settings

Section 1.4 already defined H/KL, the alive rule, the `> 0` activation criterion, and the density filtering rules. In the token-level H/KL experiments, we use the first 10,000 samples from the `validation` split with `context_length = 128`. Under this setup, for each SAE we accumulate total activation strength conditioned on token labels, and then compute $H_f$ and KL feature-wise from $P(c \mid f) = S_f(c) / T_f$.

Defining whether “a feature is active” as “its activation value is greater than 0” requires the SAE activations themselves to exhibit clear zero-valued sparsity. To verify this, we additionally examined the activation distribution of the gemma-scope family. The results show a clear gap between nonzero activations and zero values, so the `> 0` criterion is safe for the SAEs used in this work. If the setup is extended to other SAE families, especially non-JumpReLU variants, this point should be rechecked.

---

## Appendix A. Full P/R Results Table

### A.1 Full P/R Results for 15 SAEs on PII-noO

This appendix provides the complete result table corresponding to Section `4.1` of Chapter 4, retaining all ranking groups (`h` / `kl` / `h_f` / `kl_f` / `mi` / `density` / `random`) and all evaluation combinations (`tokP / tokR`, `spnP / spnR`, and `ampP / spnR`). The main text keeps only the primary metric pair `(ampP, spnR)` in order to preserve the main line of argument.

```
 k | metric      |       h       |      kl       |      h_f      |     kl_f      |       mi      |    density    |    random
───┼─────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────
 1 | tokP / tokR | 0.978 / 0.013 | 0.802 / 0.015 | 0.821 / 0.168 | 0.807 / 0.159 | 0.312 / 0.687 | 0.179 / 0.630 | 0.419 / 0.043
 1 | spnP / spnR | 0.974 / 0.026 | 0.798 / 0.028 | 0.785 / 0.404 | 0.771 / 0.386 | 0.228 / 0.887 | 0.133 / 0.830 | 0.402 / 0.105
 1 | ampP / spnR | 0.985 / 0.026 | 0.812 / 0.028 | 0.858 / 0.404 | 0.836 / 0.386 | 0.438 / 0.887 | 0.251 / 0.830 | 0.448 / 0.105
───┼─────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────
 5 | tokP / tokR | 0.942 / 0.073 | 0.823 / 0.070 | 0.632 / 0.497 | 0.632 / 0.473 | 0.184 / 0.896 | 0.137 / 0.897 | 0.337 / 0.185
 5 | spnP / spnR | 0.926 / 0.110 | 0.807 / 0.109 | 0.517 / 0.799 | 0.514 / 0.782 | 0.117 / 0.974 | 0.089 / 0.973 | 0.296 / 0.391
 5 | ampP / spnR | 0.966 / 0.110 | 0.846 / 0.109 | 0.772 / 0.799 | 0.763 / 0.782 | 0.406 / 0.974 | 0.295 / 0.973 | 0.389 / 0.391
───┼─────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────
10 | tokP / tokR | 0.897 / 0.124 | 0.813 / 0.126 | 0.492 / 0.666 | 0.498 / 0.642 | 0.148 / 0.933 | 0.121 / 0.938 | 0.304 / 0.305
10 | spnP / spnR | 0.874 / 0.181 | 0.790 / 0.190 | 0.357 / 0.902 | 0.362 / 0.892 | 0.091 / 0.983 | 0.077 / 0.985 | 0.246 / 0.573
10 | ampP / spnR | 0.940 / 0.181 | 0.853 / 0.190 | 0.709 / 0.902 | 0.705 / 0.892 | 0.398 / 0.983 | 0.315 / 0.985 | 0.379 / 0.573
───┼─────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────┼───────────────
20 | tokP / tokR | 0.821 / 0.210 | 0.764 / 0.212 | 0.356 / 0.801 | 0.365 / 0.774 | 0.122 / 0.953 | 0.109 / 0.959 | 0.269 / 0.459
20 | spnP / spnR | 0.784 / 0.316 | 0.726 / 0.314 | 0.231 / 0.955 | 0.233 / 0.949 | 0.076 / 0.988 | 0.070 / 0.989 | 0.196 / 0.748
20 | ampP / spnR | 0.900 / 0.316 | 0.839 / 0.314 | 0.631 / 0.955 | 0.631 / 0.949 | 0.388 / 0.988 | 0.332 / 0.989 | 0.372 / 0.748
```
