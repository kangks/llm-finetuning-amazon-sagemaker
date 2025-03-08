from datasets import load_dataset
# data = load_dataset('openai/gsm8k', 'main')["train"] # type: ignore
# data = load_dataset("csv", data_dir="data", 
#                     data_files="cars_details.csv", sep=",")

# data = load_dataset("json", data_files="test_data.json")
# print(dir(data))
# data = data.map(lambda x: { print(x) }) # type: ignore
# data.to_json("car.json")
# print(data)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

data = load_dataset('json', data_dir="../training_data" data_files='cars.json')["train"] # type: ignore
data = data.map(lambda x: { # type: ignore
    'prompt': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': x['Question']}
    ],
    'answer': extract_hash_answer(x['answer'])
}) # type: ignore
print(data)
