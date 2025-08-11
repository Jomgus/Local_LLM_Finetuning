
import json
import os
from datasets import load_dataset

#config
NUM_TRAIN_SAMPLES = 1000
NUM_VALID_SAMPLES = 50
NUM_TEST_SAMPLES = 100
OUTPUT_DIR = "summarization_data"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    dataset = load_dataset("xsum", "default")

    # special format Phi-3 uses for conversations.
    template = "<|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>"

    # preparing training data
    print(f"Preparing {NUM_TRAIN_SAMPLES} training samples...")
    train_sample = dataset["train"].shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))
    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w") as f:
        for item in train_sample:
            prompt = f"Summarize this article in a single sentence:\n\n{item['document']}"
            response = item['summary']
            text = template.format(prompt=prompt, response=response)
            f.write(json.dumps({"text": text}) + "\n")

    # prepare validation data
    print(f"Preparing {NUM_VALID_SAMPLES} validation samples...")
    valid_sample = dataset["validation"].shuffle(seed=42).select(range(NUM_VALID_SAMPLES))
    with open(os.path.join(OUTPUT_DIR, "valid.jsonl"), "w") as f:
        for item in valid_sample:
            prompt = f"Summarize this article in a single sentence:\n\n{item['document']}"
            response = item['summary']
            text = template.format(prompt=prompt, response=response)
            f.write(json.dumps({"text": text}) + "\n")



# The training script requires a `test.jsonl` file in this exact format to run without error, so we create it here as a required placeholder. according to 
# StrathWeb's "Fine tuning Phi models with MLX"
    print(f"Preparing {NUM_TEST_SAMPLES} test samples for lora script compatibility...")
    test_sample = dataset["test"].shuffle(seed=42).select(range(NUM_TEST_SAMPLES))
    with open(os.path.join(OUTPUT_DIR, "test.jsonl"), "w") as f:
        for item in test_sample:
            prompt = f"Summarize this article in a single sentence:\n\n{item['document']}"
            response = item['summary']
            text = template.format(prompt=prompt, response=response)
            f.write(json.dumps({"text": text}) + "\n")

    print(f"Data successfully saved to the '{OUTPUT_DIR}' with a consistent format.")

if __name__ == "__main__":
    main()