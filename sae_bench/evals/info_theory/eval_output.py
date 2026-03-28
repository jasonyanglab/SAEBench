from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.evals.info_theory.eval_config import InfoTheoryEvalConfig

EVAL_TYPE_ID_INFO_THEORY = "info_theory"

@dataclass
class InfoTheoryMeanMetrics(BaseMetrics):
    mean_kl_divergence: float = Field(
        title="Mean KL Divergence",
        description="Average KL divergence across filtered features",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    mean_normalized_entropy: float = Field(
        title="Mean Normalized Shannon Entropy",
        description="Average H/log2(C) across filtered features, range [0,1], lower is purer",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    alive_features_ratio: float = Field(
        title="Alive Features Ratio",
        description="Ratio of features that activated at least once during evaluation",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    filtered_features_ratio: float = Field(
        title="Filtered Features Ratio",
        description="Ratio of features passing density band-pass filter (min_density <= d <= max_density)",
        json_schema_extra=DEFAULT_DISPLAY,
    )

@dataclass
class InfoTheoryMetricCategories(BaseMetricCategories):
    mean: InfoTheoryMeanMetrics = Field(
        title="Mean Metrics",
        description="Averaged information theory metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )

@dataclass
class InfoTheoryResultDetail(BaseResultDetail):
    feature_index: int = Field(
        title="Feature Index",
        description="The latent index of the SAE feature",
    )
    density: float = Field(
        title="Activation Density",
        description="Token-level activation density (fraction of tokens where this feature fires)",
    )
    normalized_entropy: float = Field(
        title="Normalized Shannon Entropy",
        description="H/log2(C), range [0,1]. -1 for dead features",
    )
    kl_divergence: float = Field(
        title="KL Divergence",
        description="KL divergence relative to the natural prior distribution. -1 for dead features",
    )

@dataclass(config=ConfigDict(title="Information Theory Alignment"))
class InfoTheoryEvalOutput(
    BaseEvalOutput[
        InfoTheoryEvalConfig,
        InfoTheoryMetricCategories,
        InfoTheoryResultDetail,
    ]
):
    """
    Evaluates monosemanticity using information theory metrics:
    normalized Shannon Entropy and KL divergence, with density band-pass filtering.
    """
    eval_config: InfoTheoryEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: InfoTheoryMetricCategories
    eval_result_details: list[InfoTheoryResultDetail] = Field(
        default_factory=list,
        title="Per-Feature Info Theory Results",
        description="Each object contains metrics for a specific SAE feature.",
    )
    eval_type_id: str = Field(default=EVAL_TYPE_ID_INFO_THEORY)
