import os
import sys
import logging
import argparse
from typing import Optional
from dataclasses import asdict

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config.logging_config import setup_logging
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from config.config import Config, ModelConfig, TrainingConfig, DataConfig
from preprocessing.dataset import load_training_dataset, format_training_prompt
from training.rewards import get_default_reward_functions

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Training script with configuration options')
    
    # Get default config for help messages
    default_config = Config()

    # Model configuration arguments
    parser.add_argument('--model-name', type=str, default=default_config.model.model_name,
                       help=f'Name of the model to use (default: {default_config.model.model_name})')
    parser.add_argument('--max-seq-length', type=int, default=default_config.model.max_seq_length,
                       help=f'Maximum sequence length (default: {default_config.model.max_seq_length})')
    parser.add_argument('--lora-rank', type=int, default=default_config.model.lora_rank,
                       help=f'LoRA rank (default: {default_config.model.lora_rank})')
    parser.add_argument('--load-in-4bit', type=bool, default=default_config.model.load_in_4bit,
                       help=f'Whether to load in 4-bit precision (default: {default_config.model.load_in_4bit})')
    parser.add_argument('--fast-inference', type=bool, default=default_config.model.fast_inference,
                       help=f'Whether to use fast inference (default: {default_config.model.fast_inference})')
    parser.add_argument('--gpu-memory-utilization', type=float, default=default_config.model.gpu_memory_utilization,
                       help=f'GPU memory utilization (default: {default_config.model.gpu_memory_utilization})')
    parser.add_argument('--dtype', type=str, default=default_config.model.dtype,
                       help=f'Data type for model (default: {default_config.model.dtype})')

    # Training configuration arguments
    parser.add_argument('--output-dir', type=str, default=default_config.training.output_dir,
                       help=f'Output directory (default: {default_config.training.output_dir})')
    parser.add_argument('--training-job-name', type=str, default=default_config.training.training_job_name,
                       help=f'Training job name (default: {default_config.training.training_job_name})')
    parser.add_argument('--num-train-epochs', type=int, default=default_config.training.num_train_epochs,
                       help=f'Number of training epochs (default: {default_config.training.num_train_epochs})')
    parser.add_argument('--per-device-train-batch-size', type=int, default=default_config.training.per_device_train_batch_size,
                       help=f'Per device training batch size (default: {default_config.training.per_device_train_batch_size})')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=default_config.training.gradient_accumulation_steps,
                       help=f'Gradient accumulation steps (default: {default_config.training.gradient_accumulation_steps})')
    parser.add_argument('--learning-rate', type=float, default=default_config.training.learning_rate,
                       help=f'Learning rate (default: {default_config.training.learning_rate})')
    parser.add_argument('--logging-steps', type=int, default=default_config.training.logging_steps,
                       help=f'Logging steps (default: {default_config.training.logging_steps})')
    parser.add_argument('--num-generations', type=int, default=default_config.training.num_generations,
                       help=f'Number of generations (default: {default_config.training.num_generations})')
    parser.add_argument('--max-prompt-length', type=int, default=default_config.training.max_prompt_length,
                       help=f'Maximum prompt length (default: {default_config.training.max_prompt_length})')
    parser.add_argument('--max-completion-length', type=int, default=default_config.training.max_completion_length,
                       help=f'Maximum completion length (default: {default_config.training.max_completion_length})')
    parser.add_argument('--temperature', type=float, default=default_config.training.temperature,
                       help=f'Temperature (default: {default_config.training.temperature})')
    parser.add_argument('--beta', type=float, default=default_config.training.beta,
                       help=f'Beta parameter (default: {default_config.training.beta})')
    parser.add_argument('--use-vllm', type=bool, default=default_config.training.use_vllm,
                       help=f'Whether to use VLLM (default: {default_config.training.use_vllm})')
    parser.add_argument('--vllm-gpu-memory-utilization', type=float, default=default_config.training.vllm_gpu_memory_utilization,
                       help=f'VLLM GPU memory utilization (default: {default_config.training.vllm_gpu_memory_utilization})')
    parser.add_argument('--evaluation-steps', type=int, default=default_config.training.evaluation_steps,
                       help=f'Evaluation step (default: {default_config.training.evaluation_steps})')


    # Data configuration arguments
    parser.add_argument('--dataset-path', type=str, default=default_config.data.dataset_path,
                       help=f'Path to dataset (default: {default_config.data.dataset_path})')
    parser.add_argument('--dataset-filename', type=str, default=default_config.data.dataset_filename,
                       help=f'Dataset filename (default: {default_config.data.dataset_filename})')
    parser.add_argument('--split', type=str, default=default_config.data.split,
                       help=f'Dataset split (default: {default_config.data.split})')
    parser.add_argument('--max-samples', type=int, default=default_config.data.max_samples,
                       help=f'Maximum number of samples (default: {default_config.data.max_samples})')

    # Other arguments
    parser.add_argument('--config-path', type=str, help='Path to configuration file')
    parser.add_argument('--log-file', type=str, default='logs/training.log', help='Path to log file')

    return parser.parse_args()

def update_config_from_args(config: Config, args) -> Config:
    """Update configuration with command line arguments."""
    # Update ModelConfig
    model_args = {
        'model_name': args.model_name,
        'max_seq_length': args.max_seq_length,
        'lora_rank': args.lora_rank,
        'load_in_4bit': args.load_in_4bit,
        'fast_inference': args.fast_inference,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'dtype': None if args.dtype == "None" else args.dtype
    }
    model_args = {k: v for k, v in model_args.items() if v is not None}
    if model_args:
        config.model = ModelConfig(**{**asdict(config.model), **model_args})

    # Update TrainingConfig
    training_args = {
        'output_dir': args.output_dir,
        'training_job_name': args.training_job_name,
        'num_train_epochs': args.num_train_epochs,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'logging_steps': args.logging_steps,
        'num_generations': args.num_generations,
        'max_prompt_length': args.max_prompt_length,
        'max_completion_length': args.max_completion_length,
        'temperature': args.temperature,
        'beta': args.beta,
        'use_vllm': args.use_vllm,
        'vllm_gpu_memory_utilization': args.vllm_gpu_memory_utilization
    }
    training_args = {k: v for k, v in training_args.items() if v is not None}
    if training_args:
        config.training = TrainingConfig(**{**asdict(config.training), **training_args})

    # Update DataConfig
    data_args = {
        'dataset_path': args.dataset_path,
        'dataset_filename': args.dataset_filename,
        'split': args.split,
        'max_samples': args.max_samples
    }
    data_args = {k: v for k, v in data_args.items() if v is not None}
    if data_args:
        config.data = DataConfig(**{**asdict(config.data), **data_args})

    return config

def setup_model(config: Config):
    """Initialize and configure the model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up model: {config.model.model_name}")
    PatchFastRL("GRPO", FastLanguageModel)
    
    logger.info(f"config.model.dtype:{config.model.dtype}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        fast_inference=config.model.fast_inference,
        max_lora_rank=config.model.lora_rank,
        dtype=config.model.dtype,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
    )
    
    logger.info("Model and tokenizer setup complete")
    return model, tokenizer

def prepare_model_for_training(model, tokenizer, config: Config):
    """Configure model for training with LoRA."""
    logger = logging.getLogger(__name__)
    logger.info("Preparing model for training with LoRA")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
            "gate_proj", 
            "up_proj", 
            "down_proj",
        ],
        lora_alpha=config.model.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    logger.info("Model prepared for training")
    return model

def get_training_args(config: Config) -> GRPOConfig:
    """Get training configuration."""
    return GRPOConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        logging_steps=config.training.logging_steps,
        num_generations=config.training.num_generations,
        max_prompt_length=config.training.max_prompt_length,
        max_completion_length=config.training.max_completion_length,
        temperature=config.training.temperature,
        beta=config.training.beta,
        use_vllm=config.training.use_vllm,
        vllm_gpu_memory_utilization=config.training.vllm_gpu_memory_utilization,
        evaluation_steps=config.training.evaluation_steps
    )

def test_model(model, tokenizer, question: str, prompt_template: str):
    """Test model with a sample question."""
    logger = logging.getLogger(__name__)
    logger.info("Testing model with sample question")
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_template.format(question, "")], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    response_text = response[0].split("### Response:")[1]
    logger.info("Model test complete")
    return response_text

def save_model(model, tokenizer, output_dir: str, training_job_name: str):
    """Save the trained model."""
    logger = logging.getLogger(__name__)
    
    # Create model output directory using training job name
    model_output_dir = os.path.join(output_dir, training_job_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    logger.info(f"Saving model to {model_output_dir}")
    model.save_lora(model_output_dir)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    model.save_pretrained_merged(model_output_dir, tokenizer, save_method="merged_16bit")
    logger.info("Model saved successfully")

def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file)
    logger = logging.getLogger(__name__)

    logger.info(os.environ)
    logger.info(f"""SM_CHANNELS:{os.getenv("SM_CHANNELS")}""")
    logger.info(f"""SM_NUM_GPUS:{os.getenv("SM_NUM_GPUS")}""")

    logger.info("Starting training process")
    
    # Load base configuration
    config = Config()
    logger.info("Base configuration loaded")
    
    # Update configuration with command line arguments
    config = update_config_from_args(config, args)
    logger.info("Configuration updated with command line arguments")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model(config)
    
    # Test model before training
    logger.info("Testing model before training")
    question = "I'm looking for a car from 2017 that has relatively low mileage. Any suggestions?"
    response = test_model(model, tokenizer, question, config.prompt.instruction_prompt)
    logger.info(f"Pre-training test response: {response}")
    
    # Prepare model for training
    model = prepare_model_for_training(model, tokenizer, config)
    
    # Load dataset
    logger.info("Loading training dataset")
    dataset_fullpath = os.path.join(config.data.dataset_path, config.data.dataset_filename)
    training_dataset = load_training_dataset(
        data_path=dataset_fullpath,
        split=config.data.split,
        system_prompt=config.prompt.system_prompt,
        max_samples=config.data.max_samples
    )
    logger.info(f"Training dataset loaded with {len(training_dataset)} samples")
    
    # Configure trainer
    training_args = get_training_args(config)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=get_default_reward_functions(),
        args=training_args,
        train_dataset=training_dataset,
    )
    
    # Train model
    logger.info("Starting model training")
    trainer.train()
    logger.info("Model training completed")
    
    # Test model after training
    logger.info("Testing model after training")
    response = test_model(model, tokenizer, question, config.prompt.instruction_prompt)
    logger.info(f"Post-training test response: {response}")
    
    # Save model with training job name
    save_model(model, tokenizer, config.training.output_dir, config.training.training_job_name)
    logger.info("Training process completed successfully")

if __name__ == "__main__":
    main()
