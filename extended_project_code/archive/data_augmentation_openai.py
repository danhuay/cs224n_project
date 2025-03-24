import csv
import json
import copy

data_holder = {
    "custom_id": "<TO BE FILLED>",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {"role": "system", "content": "<TO BE FILLED>"},
            {
                "role": "user",
                "content": "<TO BE FILLED>",
            },
        ],
        "max_tokens": 1000,
    },
}

random_mask_completion_msg = """You are a robot that only outputs JSON.
Please fill in the blank user provided indicated by '[MASK]' using about 1-4 words.
Return your response in the format suggested in below example.
Example input: I am [MASK] that I was [MASK] movie.
Example JSON return: {"output": "I am happy that I was able to watch the movie."}"""

rephrase_msg = """You are a robot that only outputs JSON.
Rephrase the sentence user provided, and using about the same amount of words to this sentence.
Return your response in the format suggested in below example.
Example input: `The team must complete the project by next week.`
Example JSON return: {"output": "the project deadline is next week."}"""

back_translation_msg = """You are a robot that only outputs JSON.
Please follow the steps to provide the output.
Step 1: Translate the sentence user provided from English to French.
Step 2: Rephrase the French sentence to be more natural.
Step 3: Translate the rephrased French to English.
Step 4: Report the output in steps {"output": <Step 3 output>}
Do not include outputs from previous steps.
Example input: `I'm glad that I watched this movie.`
Example JSON return: {"output": "I'm delighted I watched this movie."}"""

SYS_PROMPTS = {
    "rnd_mask": random_mask_completion_msg,
    "rephrase": rephrase_msg,
    "back_trans": back_translation_msg,
}


def generate_jsonl_file(input_file, output_file, pairwise=False):
    # Open the CSV
    for key, sys_msg in SYS_PROMPTS.items():
        with open(input_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter="\t")
            # Open the JSONL file
            with open(f"{output_file}_{key}.jsonl", "w+") as jsonl_file:
                for row in csv_reader:
                    if pairwise:
                        cols = ["sentence1", "sentence2"]
                    else:
                        cols = ["sentence"]
                    for _col in cols:
                        # Create the JSON structure
                        data = copy.deepcopy(data_holder)
                        # custom_id
                        suffix = "|" + _col[8:] if pairwise else ""
                        data["custom_id"] = row["id"] + suffix
                        # system message
                        data["body"]["messages"][0]["content"] = sys_msg
                        # user message
                        data["body"]["messages"][1]["content"] = row[_col]

                        # Write the JSON structure to the JSONL file
                        jsonl_file.write(json.dumps(data) + "\n")


generate_jsonl_file("../data/ids-sst-train.csv", "data/sst-train", pairwise=False)
