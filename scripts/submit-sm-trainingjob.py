from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, InputData, Compute

# Image URI for the training job, https://github.com/aws/deep-learning-containers/blob/master/available_images.md
pytorch_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-ec2"

# Define the script to be run
source_code = SourceCode(
    source_dir="src",
    requirements="requirements.txt",
    entry_script="train.py",
)

training_compute = Compute(
    instance_type="ml.g6e.4xlarge",
    instance_count=1
)

hyperparameters = {
    "num_train_epochs": 3,
    "lora_rank": 32,
    "max_seq_length": 2048
}

# Define the ModelTrainer (https://sagemaker.readthedocs.io/en/stable/api/training/model_trainer.html#modeltrainer)
model_trainer = ModelTrainer(
    training_image=pytorch_image,
    source_code=source_code,
    compute=training_compute,
    role="arn:aws:iam::654654616949:role/SageMaker-job-finetuning",
    hyperparameters=hyperparameters,
    base_job_name="deepseek-finetuning-job",
)

# Pass the input data
input_data = InputData(
    channel_name="training",
    data_source="s3://654654616949-use1-finetuning/training_data/cars.json", # S3 path where training data is stored
)

# Start the training job
model_trainer.train(input_data_config=[input_data], wait=False)