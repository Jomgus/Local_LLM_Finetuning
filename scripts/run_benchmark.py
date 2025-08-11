import argparse
import json
import time
from mlx_lm import load, generate
import evaluate

def main():
    # defines and uses the command-line arguments 
    parser = argparse.ArgumentParser(description="Benchmark a fine-tuned model.")
    parser.add_argument("--model-path", required=True, help="Path to the model directory.")
    parser.add_argument("--data-path", required=True, help="Path to the benchmark JSONL data file.")
    parser.add_argument("--output-file", default="benchmark_results.jsonl", help="Path to save the generated summaries.")
    # save final scores
    parser.add_argument("--score-file", default="final_benchmark_scores.txt", help="Path to save the final ROUGE scores.")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}...")
    model, tokenizer = load(args.model_path)

    print(f"Loading benchmark data from {args.data_path}...")
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]

    results = []
    print(f"Running inference on {len(data)} samples...")
    for i, item in enumerate(data):
        print(f"Benchmarking: {i+1}/{len(data)}", end="\r")
        prompt = item["prompt"]
        reference_summary = item["reference"]
        
        response = generate(model, tokenizer, prompt, max_tokens=100, verbose=False)
        
        results.append({
            "generated": response,
            "reference": reference_summary
        })
    print("\nBenchmark complete.")


#saves to file

    with open(args.output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Results saved to {args.output_file}")

    print("Calculating ROUGE score...")
    rouge = evaluate.load("rouge")
    

    predictions = [res["generated"] for res in results]
    references = [res["reference"] for res in results]
    
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    # print results to console
    print("\n--- Final Summarization Benchmark Results ---")
    print(f"ROUGE-1 Score: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-L Score: {rouge_scores['rougeL']:.4f}")
    print("---------------------------------------------")

    # save to file
    with open(args.score_file, "w") as f:
        f.write("--- Final Summarization Benchmark Results ---\n")
        f.write(f"ROUGE-1 Score: {rouge_scores['rouge1']:.4f}\n")
        f.write(f"ROUGE-L Score: {rouge_scores['rougeL']:.4f}\n")
        f.write("---------------------------------------------\n")
    print(f"Final scores saved to {args.score_file}")


if __name__ == "__main__":
    main()
