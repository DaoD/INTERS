import os
from dataclasses import dataclass, field
from typing import Optional, List, Union
    

@dataclass
class ModelArgs:
    model_cache_dir: Optional[str] = field(
        # default=None,
        default="/share/LMs",
        metadata={'help': 'Default path to save language models.'}
    )
    dataset_cache_dir: Optional[str] = field(
        # default=None,
        default="/share/peitian/Data/Datasets/huggingface",
        metadata={'help': 'Default path to save huggingface datasets.'}
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={'help': 'Evaluation json file.'},
    )
    
    model_name_or_path: str = field(
        default='meta-llama/Llama-2-7b-chat-hf',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    tokenizer_name_or_path: str = field(
        default='meta-llama/Llama-2-7b-chat-hf',
        metadata={'help': 'Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side.'}
    )
    access_token: Optional[str] = field(
        default=None,
        metadata={'help': 'Huggingface access token.'}
    )
    max_length: int = field(
        default=2048,
        metadata={'help': 'How many tokens at maximum for each input?'},
    )
    
    lora: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA ID.'},
    )

    dtype: str = field(
        default="bf16",
        metadata={'help': 'Data type for embeddings.'}
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={'help': 'Use flash attention?'}
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )

    metrics: List[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'List of metrics. {rouge, acc}'}
    )
