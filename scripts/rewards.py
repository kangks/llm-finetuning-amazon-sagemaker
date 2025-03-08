from typing import List, Dict, Any
from .dataset import extract_xml_answer, count_xml_tags, check_soft_format, check_strict_format

def correctness_reward_func(prompts: List[Dict[str, Any]], completions: List[Dict[str, Any]], 
                          answer: List[str], **kwargs) -> List[float]:
    """Reward function for answer correctness."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Reward function for integer responses."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Reward function that checks if the completion has a strict format."""
    responses = [completion[0]["content"] for completion in completions]
    matches = [check_strict_format(r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Reward function that checks if the completion has a soft format."""
    responses = [completion[0]["content"] for completion in completions]
    matches = [check_soft_format(r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Reward function that evaluates XML tag structure."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml_tags(c) for c in contents]

def get_default_reward_functions():
    """Get the default list of reward functions."""
    return [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]