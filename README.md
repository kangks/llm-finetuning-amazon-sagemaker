# LLM Fine-tuning with Amazon SageMaker

This repository contains scripts for fine-tuning Large Language Models (LLMs) using Amazon SageMaker with GRPO (Generative Reward Proximal Policy Optimization) and Low-Rank Adaptation (LoRA).

## Overview of LoRA in LLM Fine-tuning

Fine-tuning large-scale pre-trained models can be resource-intensive due to their massive number of parameters. Low-Rank Adaptation (LoRA) addresses this challenge by introducing trainable low-rank matrices into each layer of the Transformer architecture, effectively reducing the number of trainable parameters while maintaining performance. This approach freezes the original model weights and injects these low-rank matrices, leading to more efficient fine-tuning processes. 
## Description of GRPO and Reward Functions

Generative Reward Proximal Policy Optimization (GRPO) is a reinforcement learning technique tailored for training generative models, such as LLMs. It extends the Proximal Policy Optimization (PPO) algorithm by incorporating reward functions that guide the model toward generating desired outputs. In the context of LLM fine-tuning, reward functions are designed to encourage the model to produce text that aligns with specific objectives, such as coherence, relevance, or adherence to certain guidelines. By optimizing these reward functions, GRPO enables the model to learn from feedback and improve its generative capabilities in a controlled manner.

## Amazon SageMaker's ModelTrainer Module and Its Benefits

Amazon SageMaker's `ModelTrainer` class is part of the SageMaker Python SDK, providing a high-level interface for training machine learning models on SageMaker. It simplifies the configuration and submission of training jobs by abstracting the underlying infrastructure details. Key benefits include:

- **Simplified Configuration**: `ModelTrainer` reduces the complexity of setting up training jobs by requiring only essential parameters, making it more accessible for users without deep AWS expertise. 

- **Seamless Integration**: It integrates smoothly with other SageMaker components, allowing for straightforward transitions from training to deployment phases. 

- **Scalability**: `ModelTrainer` leverages SageMaker's infrastructure to automatically scale resources based on the training workload, ensuring efficient use of computational resources. 

- **Flexibility**: Users can customize training configurations and incorporate advanced features as needed, providing a balance between simplicity and control. 

By utilizing `ModelTrainer`, users can streamline the process of fine-tuning LLMs, benefiting from SageMaker's robust infrastructure and services.

## Getting Started

### Prerequisites

1. AWS credentials configured with appropriate permissions
2. SageMaker execution role with necessary permissions
3. Python 3.10 or later
4. Required packages installed from `requirements.txt`

### Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
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
python scripts/submit-sagemaker-training.py
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
   - `learning-rate`: Learning rate for optimization (default 