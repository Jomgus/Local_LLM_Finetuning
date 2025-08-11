
import json
import os
from datasets import load_dataset

NUM_TEST_SAMPLES = 100
OUTPUT_FILE = "benchmark_test.jsonl"

def main():
    print(f"Preparing {NUM_TEST_SAMPLES} samples for final benchmark...")
    dataset = load_dataset("xsum", "default")
    test_sample = dataset["test"].shuffle(seed=42).select(range(NUM_TEST_SAMPLES))
    
    with open(OUTPUT_FILE, "w") as f:
        for item in test_sample:
            formatted_item = {
                "prompt": f"Summarize this article in a single sentence:\n\n{item['document']}\n\nSummary:",
                "reference": item['summary']
            }
            f.write(json.dumps(formatted_item) + "\n")
            
    print(f"Benchmark data successfully saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()