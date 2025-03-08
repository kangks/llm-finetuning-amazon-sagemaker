from typing import Dict, List, Optional, Any
from datasets import load_dataset, Dataset
import re

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from hash-formatted text."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_training_prompt(question: str, prompt_template: str) -> str:
    """Format a single training prompt using the template."""
    return prompt_template.format(question, "")

def load_training_dataset(
    data_path: str,
    split: str = "train",
    system_prompt: str = "",
    max_samples: Optional[int] = None
) -> Dataset:
    """Load and prepare the training dataset."""
    data = load_dataset('json', data_files=data_path)[split]
    
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))

    def prepare_example(example):
        return { # type: ignore
            'prompt': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': x['Question']}
            ],
            "reasoning": x["Reasoning"],
            'answer': x['Answer']
        }
    
    return data.map(prepare_example)

def count_xml_tags(text: str) -> float:
    """Count XML tags and calculate format score."""
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
    return count

def check_strict_format(text: str) -> bool:
    """Check if text follows strict XML format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    return bool(re.match(pattern, text))

def check_soft_format(text: str) -> bool:
    """Check if text follows soft XML format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return bool(re.match(pattern, text))