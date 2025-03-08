from typing import List, Dict, Any
from preprocessing.dataset import extract_xml_answer, count_xml_tags, check_soft_format, check_strict_format
import re

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

# def xmlcount_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
#     """Reward function that evaluates XML tag structure."""
#     contents = [completion[0]["content"] for completion in completions]
#     return [count_xml_tags(c) for c in contents]

def count_xml(text) -> float:
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

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def answer_relevance_reward(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    questions = [prompt[-1]["content"] for prompt in prompts]

    def check_relevance(response, question, reference):
        score = 0.0
        # Extract key terms from question
        question_terms = set(question.lower().split())
        response_terms = set(response.lower().split())
        reference_terms = set(reference.lower().split())

        # 1) Check if response addresses key terms from question
        if len(question_terms) > 0:
            common_qr = question_terms.intersection(response_terms)
            if len(common_qr) / len(question_terms) > 0.3:
                score += 0.5

        # 2) Check if response uses similar key terms as reference
        if len(reference_terms) > 0:
            common_rr = response_terms.intersection(reference_terms)
            if len(common_rr) / len(reference_terms) > 0.3:
                score += 0.5

        return score

    return [check_relevance(r, q, a) for r, q, a in zip(responses, questions, answer)]

def get_default_reward_functions():
    """Get the default list of reward functions."""
    return [
        format_reward_func,
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        answer_relevance_reward
    ]