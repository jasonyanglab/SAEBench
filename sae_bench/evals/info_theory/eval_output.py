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
        description="Average KL divergence across all alive features",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    mean_shannon_entropy: float = Field(
        title="Mean Shannon Entropy",
        description="Average Shannon Entropy across all alive features",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    mean_fdn: float = Field(
        title="Mean FDN",
        description="Average Feature Distribution Non-Uniformity",
    )
    alive_features_ratio: float = Field(
        title="Alive Features Ratio",
        description="Ratio of features that activated at least once during evaluation",
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
        description="Approximated L0 global density over tokens",
    )
    shannon_entropy: float = Field(
        title="Shannon Entropy",
        description="Shannon entropy of feature activation across classes",
    )
    kl_divergence: float = Field(
        title="KL Divergence",
        description="KL divergence relative to the natural prior distribution",
    )
    fdn: float = Field(
        title="FDN",
        description="Feature Distribution Non-Uniformity",
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
    Evaluates monosemanticity using information theory metrics: Shannon Entropy, KL divergence, and FDN.
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