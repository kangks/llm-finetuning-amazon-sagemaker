import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from config import Config

def create_sagemaker_training_job():
    # Initialize session
    session = boto3.Session()
    sagemaker_session = sagemaker.Session(boto_session=session)
    
    # Get default bucket
    bucket = sagemaker_session.default_bucket()
    role = os.environ.get('SAGEMAKER_ROLE_ARN')
    
    if not role:
        raise ValueError("SAGEMAKER_ROLE_ARN environment variable must be set")
    
    # Load configuration
    config = Config()
    
    # Define hyperparameters
    hyperparameters = {
        'model_name': config.model.model_name,
        'max_seq_length': config.model.max_seq_length,
        'lora_rank': config.model.lora_rank,
        'num_train_epochs': config.training.num_train_epochs,
        'per_device_train_batch_size': config.training.per_device_train_batch_size,
        'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
        'learning_rate': config.training.learning_rate,
        'logging_steps': config.training.logging_steps,
    }
    
    # Configure metrics
    metric_definitions = [
        {'Name': 'train:loss', 'Regex': 'train_loss: ([0-9\\.]+)'},
        {'Name': 'eval:loss', 'Regex': 'eval_loss: ([0-9\\.]+)'}
    ]
    
    # Create HuggingFace estimator
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='scripts',
        instance_type='ml.g5.12xlarge',  # Using G5 instance with NVIDIA A10G GPUs
        instance_count=1,
        role=role,
        transformers_version='4.49.0',
        pytorch_version='2.5.1',
        py_version='py310',
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
        max_run=72*3600,  # 72 hours max run time
        keep_alive_period_in_seconds=1800,  # 30 minutes
        volume_size=100,  # 100 GB storage
        environment={
            'TRANSFORMERS_CACHE': '/tmp/transformers_cache',
            'HF_HOME': '/tmp/huggingface',
        }
    )
    
    # Start training job
    training_input = sagemaker.TrainingInput(
        s3_data=f's3://{bucket}/training_data',
        content_type='application/json'
    )
    
    huggingface_estimator.fit({'training': training_input}, wait=False)
    
    print(f"Training job started: {huggingface_estimator.latest_training_job.name}")
    print(f"You can monitor the training job in the SageMaker console")

if __name__ == '__main__':
    create_sagemaker_training_job()