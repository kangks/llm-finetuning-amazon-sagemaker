import os
import sys
import logging

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config.logging_config import setup_logging
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from config.config import Config
from preprocessing.dataset import load_training_dataset, format_training_prompt
from training.rewards import get_default_reward_functions

def setup_model(config: Config):
    """Initialize and configure the model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up model: {config.model.model_name}")
    PatchFastRL("GRPO", FastLanguageModel)
    
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

def main(config_path: str = None, log_file: str = "logs/training.log"):
    """Main training function."""
    # Setup logging
    setup_logging(log_file=log_file)
    logger = logging.getLogger(__name__)

    logger.info(os.environ)
    logger.info(f"""SM_CHANNELS:{os.getenv("SM_CHANNELS")}""")
    logger.info(f"""SM_NUM_GPUS:{os.getenv("SM_NUM_GPUS")}""")

    logger.info("Starting training process")
    # Load configuration
    config = Config()
    logger.info("Configuration loaded")
    if config_path:
        # TODO: Add config loading from file if needed
        pass
    
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
    dataset_fullpath=os.path.join(config.data.dataset_path, config.data.dataset_filename)
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
