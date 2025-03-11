from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute

# Step 1: Download to S3
# huggingface-cli download unsloth/DeepSeek-R1-Distill-Llama-8B --local-dir-use-symlinks False 

# Image URI for the training job, https://github.com/aws/deep-learning-containers/blob/master/available_images.md
pytorch_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-ec2"

# Define the script to be run
source_code = SourceCode(
    source_dir="src",
    requirements="requirements.txt",
    entry_script="train.py",
)

training_compute = Compute(
    instance_type="ml.g6e.4xlarge", # "ml.g4dn.xlarge", #"ml.c5.xlarge", # "ml.g6e.4xlarge",
    instance_count=1
)

# Hyperparameters organized by configuration type
hyperparameters = {
    # Model configuration
    # "model-name": "unsloth/DeepSeek-R1-Distill-Llama-8B",
    "model-name": "/opt/ml/input/data/models/models--unsloth--DeepSeek-R1-Distill-Llama-8B/snapshots/736e4a6391593d33a1f019f23bc23f91cf56b830/",
    "max-seq-length": 2048,
    "lora-rank": 32,
    "load-in-4bit": True,
    "fast-inference": True,
    "gpu-memory-utilization": 0.9,
    "dtype": "None",  # Optional, can be removed if not needed
    
    # Training configuration, https://docs.unsloth.ai/get-started/fine-tuning-guide
    "num-train-epochs": 3,
    "per-device-train-batch-size": 1,
    "gradient-accumulation-steps": 8,
    "learning-rate": 5e-3,
    "logging-steps": 10,
    "evaluation-steps": 100,
    "num-generations": 8,
    "max-prompt-length": 512,
    "max-completion-length": 256,
    "temperature": 0.9,
    "beta": 0.04,
    "use-vllm": True,
    "vllm-gpu-memory-utilization": 0.9,
    
    # Data configuration
    "dataset-filename": "cars.json",
    "split": "train",
    # max_samples is optional, remove if not needed
    # "max_samples": None
}

environment_variables: dict[str, str] = {
        "HF_HUB": "/opt/ml/input/data/models",
        "HF_HUB_OFFLINE": "1"
    }

# Define the ModelTrainer (https://sagemaker.readthedocs.io/en/stable/api/training/model_trainer.html#modeltrainer)
model_trainer = ModelTrainer(
    training_image=pytorch_image,
    source_code=source_code,
    compute=training_compute,
    role="arn:aws:iam::654654616949:role/SageMaker-job-finetuning",
    hyperparameters=hyperparameters,
    base_job_name="deepseek-finetuning-job",
    environment=environment_variables
)

# Pass the input data
input_data = InputData(
    channel_name="training",
    data_source="s3://654654616949-use1-finetuning/training_data/cars.json", # S3 path where training data is stored
)

model_data = InputData(
    channel_name="models",
    data_source="s3://654654616949-use1-finetuning/models/"
)

# Start the training job
model_trainer.train(input_data_config=[input_data, model_data], wait=False)
