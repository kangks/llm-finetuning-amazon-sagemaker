from typing import Dict, List, Optional, Any
from datasets import load_dataset, Dataset
import logging
import re

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Extracting answer from XML text: {text[:100]}...")
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    logger.debug(f"Extracted answer: {answer[:100]}...")
    return answer.strip()

def extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from hash-formatted text."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Extracting answer from hash-formatted text: {text[:100]}...")
    if "####" not in text:
        logger.warning("No hash separator found in text")
        return None
    answer = text.split("####")[1].strip()
    logger.debug(f"Extracted answer: {answer[:100]}...")
    return answer

def format_training_prompt(question: str, prompt_template: str) -> str:
    """Format a single training prompt using the template."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Formatting prompt for question: {question[:100]}...")
    formatted = prompt_template.format(question, "")
    logger.debug(f"Formatted prompt: {formatted[:100]}...")
    return formatted

def load_training_dataset(
    data_path: str,
    split: str = "train",
    system_prompt: str = "",
    max_samples: Optional[int] = None
) -> Dataset:
    """Load and prepare the training dataset."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset from {data_path} with split {split}")
    
    data = load_dataset('json', data_files=data_path)[split]
    logger.info(f"Loaded {len(data)} examples from dataset")
    
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
        logger.info(f"Selected {len(data)} samples for training")

    def prepare_example(example):
        logger.debug(f"Preparing example with question: {example['Question'][:100]}...")
        return { # type: ignore
            'prompt': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': example['Question']}
            ],
            "reasoning": example["Reasoning"],
            'answer': example['Answer']
        }
    
    logger.info("Preparing training examples with prompts and answers")
    prepared_data = data.map(prepare_example)
    logger.info(f"Dataset preparation complete. Final size: {len(prepared_data)} examples")
    return prepared_data

def count_xml_tags(text: str) -> float:
    """Count XML tags and calculate format score."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Counting XML tags in text: {text[:100]}...")
    
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    
    logger.debug(f"XML tag score: {count}")
    return count

def check_strict_format(text: str) -> bool:
    """Check if text follows strict XML format."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Checking strict format for text: {text[:100]}...")
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    result = bool(re.match(pattern, text))
    logger.debug(f"Strict format check result: {result}")
    return result

def check_soft_format(text: str) -> bool:
    """Check if text follows soft XML format."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Checking soft format for text: {text[:100]}...")
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    result = bool(re.match(pattern, text))
    logger.debug(f"Soft format check result: {result}")
    return result
