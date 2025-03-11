import os
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

# MODEL="unsloth/DeepSeek-R1-Distill-Llama-8B"
MODEL="unsloth/deepseek-r1-distill-llama-8b-unsloth-bnb-4bit"

MODEL="/mnt/efs/fs1/vllm/cache/hub/models--unsloth--deepseek-r1-distill-llama-8b-unsloth-bnb-4bit/snapshots/5e0ac06dc3e90b3e84ce2c4b6bd3257974b1bb0a"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    revision="5e0ac06dc3e90b3e84ce2c4b6bd3257974b1bb0a",
    load_in_4bit=True,
    # load_in_8bit=True,
    dtype=None,
    fast_inference=True
)

# from trl import GRPOConfig, GRPOTrainer
# GRPOConfig(
#     output_dir="out"
# )

# trainer = GRPOTrainer(
#     model=model,
#     processing_class=tokenizer,
#     reward_funcs=get_default_reward_functions(),
#     args=training_args,
#     train_dataset=training_dataset,
# )
