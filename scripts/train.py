import torch
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from .config import Config
from .dataset import load_training_dataset, format_training_prompt
from .rewards import get_default_reward_functions

def setup_model(config: Config):
    """Initialize and configure the model."""
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
    
    return model, tokenizer

def prepare_model_for_training(model, tokenizer, config: Config):
    """Configure model for training with LoRA."""
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
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_template.format(question, "")], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    return response[0].split("### Response:")[1]

def save_model(model, tokenizer, output_dir: str):
    """Save the trained model."""
    model.save_lora(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

def main(config_path: str = None):
    """Main training function."""
    # Load configuration
    config = Config()
    if config_path:
        # TODO: Add config loading from file if needed
        pass
    
    # Setup model and tokenizer
    model, tokenizer = setup_model(config)
    
    # Test model before training
    print("\nBEFORE TRAINING:")
    question = "I'm looking for a car from 2017 that has relatively low mileage. Any suggestions?"
    response = test_model(model, tokenizer, question, config.prompt.instruction_prompt)
    print(response)
    
    # Prepare model for training
    model = prepare_model_for_training(model, tokenizer, config)
    
    # Load dataset
    training_dataset = load_training_dataset(
        config.data.dataset_path,
        config.data.split,
        config.prompt.system_prompt,
        config.data.max_samples
    )
    
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
    trainer.train()
    
    # Test model after training
    print("\nAFTER TRAINING:")
    response = test_model(model, tokenizer, question, config.prompt.instruction_prompt)
    print(response)
    
    # Save model
    save_model(model, tokenizer, config.training.output_dir)

if __name__ == "__main__":
    main()