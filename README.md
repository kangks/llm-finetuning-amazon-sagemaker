# LLM Fine-tuning with Amazon SageMaker

This repository contains scripts for fine-tuning Large Language Models (LLMs) using Amazon SageMaker with GRPO (Generative Reward Proximal Policy Optimization).

## Getting Started

### Prerequisites

1. AWS credentials configured with appropriate permissions
2. SageMaker execution role with necessary permissions
3. Python 3.10 or later
4. Required packages installed from `requirements.txt`

### Setup

1. Install the required packages:
```bash
pip install -r src/requirements.txt
```

2. Set up your AWS credentials and SageMaker role:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region
export SAGEMAKER_ROLE_ARN=arn:aws:iam::your_account_id:role/your_sagemaker_role
```

### Running the Project

1. Upload the required data to S3:
   - Training data: Upload your training data JSON files to S3 (e.g., `s3://your-bucket/training_data/cars.json`)
   - Model: Upload the pre-trained model files to S3 (e.g., `s3://your-bucket/models/`)
   - Inference test data: Upload test cases to S3 (e.g., `s3://your-bucket/inference_test/`)

2. Update the parameters in `scripts/submit-sm-trainingjob.py`:
   - Set the S3 paths for your data sources in `InputData` configurations
   - Configure the compute resources in `training_compute`
   - Adjust hyperparameters as needed:
     ```python
     hyperparameters = {
         "model-name": "your-model-path",
         "max-seq-length": 2048,
         "lora-rank": 32,
         "num-train-epochs": 3,
         "learning-rate": 5e-3,
         # ... other parameters
     }
     ```

3. Submit the training job:
```bash
python scripts/submit-sm-trainingjob.py
```

## AWS Infrastructure Overview

The project utilizes the following AWS components:

1. **Amazon SageMaker Training Job**:
   - Uses PyTorch deep learning container
   - Runs on GPU instances (e.g., ml.g6e.4xlarge)
   - Handles distributed training and resource management

2. **S3 Storage**:
   - Training data bucket
   - Model artifacts storage
   - Inference test data
   - Training logs and outputs

3. **IAM Roles**:
   - SageMaker execution role with permissions for:
     - S3 access
     - CloudWatch logging
     - ECR image pulling

## SageMaker Model Training

The training infrastructure is managed through the `ModelTrainer` class, which handles:

1. **Source Code Configuration**:
   - Entry point: `train.py`
   - Dependencies: `requirements.txt`
   - Source directory packaging

2. **Input Channels**:
   - training: Training dataset
   - models: Pre-trained model files
   - inference_testing: Test cases for evaluation

3. **Compute Resources**:
   - Instance selection
   - Resource allocation
   - Multi-GPU support

## Training Script Overview

The training process in `train.py` follows these key steps:

1. **Preparation**:
   - Configuration loading and validation
   - Model and tokenizer initialization
   - Dataset preparation and loading

2. **GRPO Configuration**:
   - Policy setup
   - Reward function configuration
   - Training hyperparameters

3. **Training Process**:
   - LoRA (Low-Rank Adaptation) setup
   - GRPO training loop
   - Model evaluation
   - Checkpoint saving

4. **Model Saving**:
   - LoRA weights saving
   - Model merging
   - Tokenizer configuration

## Key Fine-tuning Parameters

Important parameters that affect the training process:

1. **Model Configuration**:
   - `lora-rank`: Rank of LoRA adaptation (default: 32)
   - `max-seq-length`: Maximum sequence length (default: 2048)
   - `load-in-4bit`: Enable 4-bit quantization

2. **Training Configuration**:
   - `learning-rate`: Learning rate for optimization (default: 5e-3)
   - `num-train-epochs`: Number of training epochs (default: 3)
   - `gradient-accumulation-steps`: Steps for gradient accumulation (default: 8)
   - `per-device-train-batch-size`: Batch size per device (default: 1)

3. **Generation Parameters**:
   - `temperature`: Sampling temperature (default: 0.9)
   - `max-prompt-length`: Maximum prompt length (default: 512)
   - `max-completion-length`: Maximum completion length (default: 256)