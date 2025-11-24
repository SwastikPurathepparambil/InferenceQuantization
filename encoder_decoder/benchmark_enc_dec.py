import argparse
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig


# Default to a seq2seq model like Flan-T5
DEFAULT_MODEL_ID = "google/flan-t5-base"
DEFAULT_PROMPT = "Explain quantization in simple terms."


# model_id and precision are both strings
def load_model_and_tokenizer(model_id, precision):
    """
    The goal of this function is to load the tokenizer and model
    for an *encoder-decoder* (seq2seq) architecture, e.g. Flan-T5, T5, BART.
    """

    print(f"\n[LOAD] Model: {model_id} | Precision: {precision}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Many seq2seq models (like T5) already have a pad_token.
    # This is a safe fallback in case it's missing.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if precision == "fp16":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="cuda",
        )

    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(
            # loads the weights as 8 bits
            load_in_8bit=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
        )

    elif precision == "4bit":
        bnb_config = BitsAndBytesConfig(
            # loads the weights as 4 bits
            load_in_4bit=True,
            # after dequantizing, we choose
            # which dtype we want to run computations in
            bnb_4bit_compute_dtype=torch.float16,
            # normalized float 4, very high accuracy
            bnb_4bit_quant_type="nf4",
            # quantizes twice
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
        )

    else:
        raise ValueError(f"Unknown precision: {precision}")

    # model.eval() tells the model we are running inference now.
    # the options are model.train() and model.eval()
    # for example, in model.train(), you might have dropout enabled
    # model.eval() will disable dropout
    model.eval()
    print("[LOAD] Done.")
    return model, tokenizer


def run_benchmark(model, tokenizer, prompt, max_new_tokens: int = 50):
    """
    This function runs benchmarking once, used as
    a helper to be called multiple times in main().

    For seq2seq models, we still use model.generate(), but
    under the hood they use an encoder-decoder architecture
    (encoder reads the input; decoder generates the output).
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark expects a GPU.")

    device = "cuda"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def vram_gb():
        return torch.cuda.memory_allocated() / (1024 ** 3)

    start_mem = vram_gb()

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    end_mem = vram_gb()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    input_len = inputs["input_ids"].shape[-1]
    output_len = output[0].shape[-1]
    generated_tokens = max(output_len - input_len, 1)
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    per_token_latency_sec = elapsed / generated_tokens

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    metrics = {
        "total_latency_sec": elapsed,
        "per_token_latency_sec": per_token_latency_sec,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": tokens_per_sec,
        "start_mem_gb": start_mem,
        "end_mem_gb": end_mem,
        "peak_mem_gb": peak_mem,
    }

    return metrics, generated_text

# --------- ARGUMENTS ----------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Benchmark FP16, INT8, and 4-bit for a single encoder-decoder (seq2seq) model."
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id (e.g. google/flan-t5-base).",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt / input text to benchmark.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of new tokens to generate.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available; run this on your GPU machine.")
        return

    precisions = ["fp16", "int8", "4bit"]
    all_results = {}

    for prec in precisions:
        try:
            model, tokenizer = load_model_and_tokenizer(args.model_id, prec)
            print(f"[RUN] Precision: {prec}")
            metrics, generated_text = run_benchmark(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
            all_results[prec] = {
                "metrics": metrics,
                "generated_text": generated_text,
            }

            print("\n--- Generated Text ---")
            print(generated_text[:500], "...\n")

            print("--- Metrics ---")
            for k, v in metrics.items():
                if "mem" in k:
                    print(f"{k}: {v:.3f} GB")
                elif "sec" in k:
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
            print("\n" + "=" * 60 + "\n")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Failed for precision '{prec}': {e}")
            all_results[prec] = {"error": str(e)}

    gpu_name = torch.cuda.get_device_name(0)

    print("\n===== SEQ2SEQ SUMMARY (FP16 vs INT8 vs 4-bit) =====\n")
    header = f"{'precision':<8} | {'latency(s)':>10} | {'tokens/s':>10} | {'peak_mem(GB)':>12}"
    print(header)
    print("-" * len(header))

    text_lines = []
    text_lines.append("===== SEQ2SEQ SUMMARY (FP16 vs INT8 vs 4-bit) =====\n")
    text_lines.append(f"GPU: {gpu_name}\n\n")
    text_lines.append(header + "\n")
    text_lines.append("-" * len(header) + "\n")

    for prec in precisions:
        res = all_results.get(prec, {})
        metrics = res.get("metrics")

        if metrics is None:
            row = f"{prec:<8} | {'ERROR':>10} | {'ERROR':>10} | {'ERROR':>12}"
            print(row)
            text_lines.append(row + "\n")
        else:
            row = (
                f"{prec:<8} | "
                f"{metrics['total_latency_sec']:>10.4f} | "
                f"{metrics['tokens_per_sec']:>10.2f} | "
                f"{metrics['peak_mem_gb']:>12.3f}"
            )
            print(row)
            text_lines.append(row + "\n")

    # ---------- NEW: CSV output (latency, tokens/s, peak mem only) ----------
    import csv
    csv_path = "seq2seq_benchmark_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "latency_sec", "tokens_per_sec", "peak_mem_gb", "error"])

        for prec in precisions:
            res = all_results.get(prec, {})
            metrics = res.get("metrics")
            error = res.get("error")

            if metrics is None:
                writer.writerow([prec, "", "", "", error or "ERROR"])
            else:
                writer.writerow([
                    prec,
                    metrics["total_latency_sec"],
                    metrics["tokens_per_sec"],
                    metrics["peak_mem_gb"],
                    ""
                ])

    print(f"[INFO] Saved CSV summary to {csv_path}")

    # ---------- NEW: append response section ----------
    label_map = {
        "fp16": "fp 16",
        "int8": "int8",
        "4bit": "nf4",
    }

    text_lines.append("\n===== RESPONSES BY PRECISION =====\n\n")
    text_lines.append(f"Question: {args.prompt}\n\n")

    for prec in precisions:
        res = all_results.get(prec, {})
        answer = res.get("generated_text")
        error = res.get("error")
        label = label_map.get(prec, prec)

        if answer is None:
            text_lines.append(f"{label} answer: [ERROR: {error}]\n\n")
        else:
            text_lines.append(f"{label} answer: {answer}\n\n")

    # Write updated text file
    text_path = "seq2seq_benchmark_summary.txt"
    with open(text_path, "w") as f:
        f.writelines(text_lines)

    print(f"[INFO] Saved formatted summary text to {text_path}")


if __name__ == "__main__":
    main()
