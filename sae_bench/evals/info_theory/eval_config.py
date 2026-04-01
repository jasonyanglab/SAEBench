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
    dataset_split: str = Field(
        default="train",
        title="Dataset Split",
        description="Which split to use (train, test, validation).",
    )
    text_column: str = Field(
        default="text",
        title="Text Column",
        description="Name of the text column in the dataset.",
    )
    label_column: str = Field(
        default="label",
        title="Label Column",
        description="Name of the label column in the dataset.",
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
    min_feature_density: float = Field(
        default=1e-4,
        title="Min Feature Density",
        description="Lower density threshold. Features below this are excluded from aggregate metrics "
                    "due to insufficient activations for reliable distribution estimation.",
    )
    max_feature_density: float = Field(
        default=1e-2,
        title="Max Feature Density",
        description="Upper density threshold. Features above this are excluded from aggregate metrics "
                    "as high-frequency features likely encode syntax/position rather than semantics.",
    )
    label_type: str = Field(
        default="document",
        title="Label Type",
        description="Label granularity: 'document' for document-level labels (AG News, DBpedia), "
                    "'token' for token-level labels (PII, NER).",
    )
    include_non_entity: bool = Field(
        default=True,
        title="Include Non-Entity Tokens",
        description="For token-level labels: whether to include 'O' (non-entity) tokens as a class. "
                    "Ignored for document-level labels.",
    )