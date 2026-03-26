from pydantic import Field
from pydantic.dataclasses import dataclass
from sae_bench.evals.base_eval_output import BaseEvalConfig

@dataclass
class InfoTheoryEvalConfig(BaseEvalConfig):
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed",
    )
    dataset_name: str = Field(
        default="fancyzhx/ag_news",
        title="Dataset Name",
        description="The dataset used for conceptual alignment verification. Default is AG News.",
    )
    num_samples: int = Field(
        default=2000,
        title="Number of Samples",
        description="Number of dataset samples to evaluate.",
    )
    context_length: int = Field(
        default=128,
        title="LLM Context Length",
        description="The maximum length of each input to the LLM.",
    )
    model_name: str = Field(
        default="",
        title="Model Name",
        description="Model name. Must be set with a command line argument.",
    )
    llm_batch_size: int = Field(
        default=32,
        title="LLM Batch Size",
        description="LLM batch size. Can be overridden via command line or activation_collection defaults.",
    )
    sae_batch_size: int = Field(
        default=125,
        title="SAE Batch Size",
        description="SAE batch size, inference only",
    )
    llm_dtype: str = Field(
        default="",
        title="LLM Data Type",
        description="LLM data type.",
    )
    lower_vram_usage: bool = Field(
        default=False,
        title="Lower Memory Usage",
        description="Lower GPU memory usage by moving model to CPU when not required.",
    )