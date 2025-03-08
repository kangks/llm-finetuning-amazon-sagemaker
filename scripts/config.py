from dataclasses import dataclass
from typing import List, Optional

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
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>{}"""

    training_prompt: str = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

@dataclass
class TrainingConfig:
    output_dir: str = "output"
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
    dataset_path: str = "cars_training_data.json"
    split: str = "train"
    max_samples: Optional[int] = None

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    prompt: PromptConfig = PromptConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()