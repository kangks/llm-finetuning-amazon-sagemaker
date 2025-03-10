from typing import List, Dict, Any
from preprocessing.dataset import extract_xml_answer, count_xml_tags, check_soft_format, check_strict_format
import logging
import re

def correctness_reward_func(prompts: List[Dict[str, Any]], completions: List[Dict[str, Any]], 
                          answer: List[str], **kwargs) -> List[float]:
    """Reward function for answer correctness."""
    logger = logging.getLogger(__name__)
    logger.debug("Calculating correctness rewards")
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    
    logger.debug(f"Correctness rewards calculated: {rewards}")
    return rewards

def int_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Reward function for integer responses."""
    logger = logging.getLogger(__name__)
    logger.debug("Calculating integer rewards")
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    
    logger.debug(f"Integer rewards calculated: {rewards}")
    return rewards

def strict_format_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Reward function that checks if the completion has a strict format."""
    logger = logging.getLogger(__name__)
    logger.debug("Checking strict format compliance")
    
    responses = [completion[0]["content"] for completion in completions]
    matches = [check_strict_format(r) for r in responses]
    rewards = [0.5 if match else 0.0 for match in matches]
    
    logger.debug(f"Strict format rewards calculated: {rewards}")
    return rewards

def soft_format_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """Reward function that checks if the completion has a soft format."""
    logger = logging.getLogger(__name__)
    logger.debug("Checking soft format compliance")
    
    responses = [completion[0]["content"] for completion in completions]
    matches = [check_soft_format(r) for r in responses]
    rewards = [0.5 if match else 0.0 for match in matches]
    
    logger.debug(f"Soft format rewards calculated: {rewards}")
    return rewards

def count_xml(text) -> float:
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
    
    logger.debug(f"XML tag count score: {count}")
    return count

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    logger = logging.getLogger(__name__)
    logger.debug("Checking think/answer format compliance")
    
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    
    logger.debug(f"Think/answer format rewards calculated: {rewards}")
    return rewards

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Calculate XML tag count rewards."""
    logger = logging.getLogger(__name__)
    logger.debug("Calculating XML tag count rewards")
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]
    
    logger.debug(f"XML count rewards calculated: {rewards}")
    return rewards

def answer_relevance_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Calculate relevance rewards based on term overlap."""
    logger = logging.getLogger(__name__)
    logger.debug("Calculating answer relevance rewards")
    
    responses = [completion[0]["content"] for completion in completions]
    questions = [prompt[-1]["content"] for prompt in prompts]

    def check_relevance(response, question, reference):
        score = 0.0
        # Extract key terms
        question_terms = set(question.lower().split())
        response_terms = set(response.lower().split())
        reference_terms = set(reference.lower().split())

        # Check question term overlap
        if len(question_terms) > 0:
            common_qr = question_terms.intersection(response_terms)
            overlap_ratio = len(common_qr) / len(question_terms)
            logger.debug(f"Question-response term overlap ratio: {overlap_ratio:.2f}")
            if overlap_ratio > 0.3:
                score += 0.5

        # Check reference term overlap
        if len(reference_terms) > 0:
            common_rr = response_terms.intersection(reference_terms)
            overlap_ratio = len(common_rr) / len(reference_terms)
            logger.debug(f"Reference-response term overlap ratio: {overlap_ratio:.2f}")
            if overlap_ratio > 0.3:
                score += 0.5

        return score

    rewards = [check_relevance(r, q, a) for r, q, a in zip(responses, questions, answer)]
    logger.debug(f"Answer relevance rewards calculated: {rewards}")
    return rewards

def get_default_reward_functions():
    """Get the default list of reward functions."""
    logger = logging.getLogger(__name__)
    logger.info("Loading default reward functions")
    reward_funcs = [
        format_reward_func,
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        answer_relevance_reward
    ]
    logger.info(f"Loaded {len(reward_funcs)} reward functions")
    return reward_funcs
