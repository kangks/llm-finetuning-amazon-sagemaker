from dataclasses import dataclass, field, asdict
from typing import List, Optional
import logging
import os

@dataclass
class ModelConfig:
    model_name: str = "unsloth/DeepSeek-R1-Distill-Llama-8B"
    max_seq_length: int = 2048
    lora_rank: int = 32
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.6
    dtype: Optional[str] = None

@dataclass
class PromptConfig:
    system_prompt: str = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
    
    instruction_prompt: str = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an automotive expert with extensive knowledge of cars, their specifications, features, and market values.
Please answer the following question about cars.

### Question:
{}

### Response:
<think>{}"""

    training_prompt: str = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an automotive expert with extensive knowledge of cars, their specifications, features, and market values.
Please answer the following question about cars.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

@dataclass
class TrainingConfig:
    output_dir: str = os.environ.get("SM_OUTPUT_DIR", "output")
    training_job_name: str = os.environ.get("TRAINING_JOB_NAME", "local-training")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    logging_steps: int = 10
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 0.9
    beta: float = 0.04
    use_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.9

@dataclass
class DataConfig:
    dataset_path: str = os.environ.get("SM_CHANNEL_TRAINING", "../training_data/cars.json")
    split: str = "train"
    max_samples: Optional[int] = None

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """Log configuration after initialization."""
        logger = logging.getLogger(__name__)
        logger.info("Initializing configuration")
        
        # Log each configuration section
        logger.info("Model configuration:")
        for key, value in asdict(self.model).items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Training configuration:")
        for key, value in asdict(self.training).items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Data configuration:")
        for key, value in asdict(self.data).items():
            logger.info(f"  {key}: {value}")
        
        # Don't log full prompt templates as they're verbose
        logger.info("Prompt configuration loaded")
        
        logger.info("Configuration initialization complete")
