import requests
import json
import csv
from decouple import config
import sys
import logging
import time
from tqdm import tqdm

# logging setup
# Set up logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    # filemode="w+",vim
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def completion_augmentation(text, hint_phrase="To put it differently,"):
    """
    According to
    In other words
    Essentially
    That is to say
    To put it differently
    """

    prompt = """
    You are a robot that only outputs JSON.
    Complete the sentence quoted in ``, and using about the same amount of words to this sentence,
    but only return the completion part of the sentence, not the part quoted in ``.
    Return your response in the format suggested in below example.

    Example input: `The team must complete the project by next week. {hint_phrase}`
    Example JSON return: "{{"output": "the project deadline is next week."}}"

    Now here is the sentence to complete:
    `{text} {hint_phrase}`
    """.format(
        text=text, hint_phrase=hint_phrase
    )

    return prompt


def back_translation_augmentation(text):
    # back-translated prompt
    prompt = """
    You are a robot that only outputs JSON.
    Please follow the steps to provide the output.
    Step 1: Translate the sentence quoted in `` from English to French.
    Step 2: Rephrase the French sentence to be more natural.
    Step 3: Translate the rephrased French to English.
    Step 4: Report the output in steps 
    "{{
        "step1": <Step 1 output>,
        "step2": <Step 2 output>,
        "step3": <Step 3 output>,
        "output": <Step 3 output>,
    }}"
    Do not include outputs from previous steps.

    Example input: `I'm glad that I watched this movie.`
    Example JSON return: "{{
        "step1": "Je suis content que j'ai vu ce film.",
        "step2": "Je suis ravi d'avoir vu ce film.",
        "step3": "I'm delighted I watched this movie.",
        "output": "I'm delighted I watched this movie."
    }}"

    Now here is the sentence to translate:
    `{text}`
    """.format(
        text=text
    )

    return prompt


def random_mask_completion_augmentation(text):
    import random

    def split_sum(x):
        parts = []
        while x > 0:
            part = random.choice([i for i in [1, 2, 3, 4] if i <= x])
            parts.append(part)
            x -= part
        return parts

    def mask_elements(Y):
        # Calculate x as 25% of the length of Y with a minimum value of 1
        x = max(1, round(random.gauss(0.25, 0.01) * len(Y)))

        # Generate the parts for masking
        parts = split_sum(x)

        # Pick random starting indices such that each replacement fits within the list
        available_indices = set(range(len(Y)))
        # Replace segments with 'MASK'
        for part in parts:
            if len(Y) < part:
                break
            start_index = random.randint(0, len(Y) - part)
            Y = Y[:start_index] + ["[MASK]"] + Y[start_index + part :]

        return Y

    words = text.split(" ")
    masked_words = mask_elements(words)
    masked_sentence = " ".join(masked_words)

    prompt = """
    You are a robot that only outputs JSON.
    Please fill in the blank in the sentence quoted in `` using about 1-4 words. Blank is indicated by `[MASK]`.
    Return your response in the format suggested in below example.

    Example input: `I am [MASK] that I was [MASK] movie.`
    Example JSON return: "{{"output": "I am happy that I was able to watch the movie."}}"

    Now here is the sentence to fill in the blank:
    `{text}`
    """.format(
        text=masked_sentence
    )

    return prompt


def send_request(prompt, **kwargs):
    url = "https://api.together.xyz/v1/chat/completions"
    model = "meta-llama/Llama-3-8b-chat-hf"
    TGAI_KEY = config("TGAI_KEY")

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {TGAI_KEY}",
    }

    payload = {
        "model": model,
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "echo": False,
        "stop": ["<|eot_id|>"],
        "messages": [{"content": prompt, "role": "user"}],
    }

    payload.update(kwargs)

    try:
        response = requests.post(url, json=payload, headers=headers)
    except:
        logger.info(f"Failed to send request.")
        return None

    return response


def get_data_augmentation(text, augmentation_func, **kwargs):
    prompt = augmentation_func(text)
    response = send_request(prompt, **kwargs)

    # if not successful, return the original text
    if response is None or response.status_code != 200:
        logger.info(f"Failed to get response, returning original text.")
        return text
    try:
        output = json.loads(response.json()["choices"][0]["message"]["content"])[
            "output"
        ]
        return output
    except Exception as e:
        logger.info(f"Failed to get output, returning original text. Error: {e}")
        return text


def process_and_write_file(
    source_filename, dest_filename, augmentation_func, pairwise=True
):
    with open(source_filename, "r") as source_file, open(
        dest_filename, "a", newline=""
    ) as dest_file:
        reader = csv.DictReader(source_file, delimiter="\t")
        writer = csv.DictWriter(dest_file, fieldnames=reader.fieldnames, delimiter="\t")

        writer.writeheader()

        for i, row in tqdm(enumerate(reader)):
            if i <= 6831:
                continue
            
            time.sleep(0.1)  # 10 QPS limit
            if pairwise:
                row["sentence1"] = get_data_augmentation(
                    row["sentence1"], augmentation_func
                )
                row["sentence2"] = get_data_augmentation(
                    row["sentence2"], augmentation_func
                )
                writer.writerow(row)
            else:
                row["sentence"] = get_data_augmentation(
                    row["sentence"], augmentation_func
                )
                writer.writerow(row)


approach_dict = {
    "rnd_mask_completion": random_mask_completion_augmentation,
    # "completion": completion_augmentation,
    # "back_translation": back_translation_augmentation,
}


# source_filename = "data/sts-train.csv"
# dest_filename = "data/sts-train-aug-back_translation.csv"
# process_and_write_file(source_filename, dest_filename, back_translation_augmentation)


for key, func in approach_dict.items():
    source_filename = "data/quora-train-sample.csv"
    dest_filename = f"data/quora-train-sample-aug-{key}.csv"
    process_and_write_file(source_filename, dest_filename, func, pairwise=True)
