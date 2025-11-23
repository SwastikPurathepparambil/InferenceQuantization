import argparse
import time

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
)


DEFAULT_MODEL_ID = "google/paligemma-3b-pt-224"
# For the pretrained checkpoints, the task is controlled via a text prefix.
# "<image>caption en" = describe the image in English.
DEFAULT_PROMPT = "<image>caption en"
DEFAULT_IMAGE_PATH = "person_and_dog.jpg"  # change this via --image-path


# model_id and precision are both strings
def load_model_and_processor(model_id, precision):
    """ 
    The goal of this function is to load the processor and model
    for a multimodal PaLI-Gemma model (image + text).
    """

    print(f"\n[LOAD] Model: {model_id} | Precision: {precision}")
    processor = AutoProcessor.from_pretrained(model_id)

    if precision == "fp16":
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        )

    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(
            # loads the weights as 8 bits 
            load_in_8bit=True
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
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
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
        )

    else:
        raise ValueError(f"Unknown precision: {precision}")

    # model.eval() tells the model we are running inference now.
    # the options are model.train() and model.eval()
    # for example, in model.train(), could have dropout enabled
    # model.eval() will disable dropout
    model.eval()
    print("[LOAD] Done.")
    return model, processor


def run_benchmark(model, processor, prompt, image_path, max_new_tokens: int = 50):
    """
    this function runs benchmarking once used as
    a helper to be called multiple times in main()

    For PaLI-Gemma, the input is (image + text) and we still use model.generate().
    """
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark expects a GPU.")

    # The model may be on cuda:0 or spread across devices; processor.to(model.device) is the safe pattern.
    device = model.device if hasattr(model, "device") else "cuda"

    # Load and prepare the image (must be RGB)
    image = Image.open(image_path).convert("RGB")

    # 🔑 IMPORTANT: Paligemma processor expects (images, text), not (text, images).
    # Also: don't force dtype here, just move to the model.device.
    inputs = processor(
        image,
        prompt,
        return_tensors="pt",
    ).to(device)

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

    # Decode the generated text.
    full_text = processor.decode(output[0], skip_special_tokens=True)

    metrics = {
        "total_latency_sec": elapsed,
        "per_token_latency_sec": per_token_latency_sec,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": tokens_per_sec,
        "start_mem_gb": start_mem,
        "end_mem_gb": end_mem,
        "peak_mem_gb": peak_mem,
    }

    return metrics, full_text

# --------- ARGUMENTS ----------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Benchmark FP16, INT8, and 4-bit for a single PaLI-Gemma multimodal model."
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id (e.g. google/paligemma-3b-pt-224).",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt to benchmark (e.g. 'caption en').",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the image to use for benchmarking.",
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
            model, processor = load_model_and_processor(args.model_id, prec)
            print(f"[RUN] Precision: {prec}")
            metrics, generated_text = run_benchmark(
                model=model,
                processor=processor,
                prompt=args.prompt,
                image_path=args.image_path,
                max_new_tokens=args.max_new_tokens,
            )
            all_results[prec] = {
                "metrics": metrics,
                "generated_text": generated_text,
            }

            print("\n--- Generated Text ---")
            print(generated_text[:500], "...\n")  # truncate for sanity

            print("--- Metrics ---")
            for k, v in metrics.items():
                if "mem" in k:
                    print(f"{k}: {v:.3f} GB")
                elif "sec" in k:
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
            print("\n" + "=" * 60 + "\n")

            # Free GPU memory for the next precision
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Failed for precision '{prec}': {e}")
            all_results[prec] = {"error": str(e)}

    # Summary table
    gpu_name = torch.cuda.get_device_name(0)

    print("\n===== PaLI-Gemma MULTIMODAL SUMMARY (FP16 vs INT8 vs 4-bit) =====\n")
    header = f"{'precision':<8} | {'latency(s)':>10} | {'tokens/s':>10} | {'peak_mem(GB)':>12}"
    print(header)
    print("-" * len(header))

    # Build text file content
    text_lines = []
    text_lines.append("===== PaLI-Gemma MULTIMODAL SUMMARY (FP16 vs INT8 vs 4-bit) =====\n")
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

    # Write to text file
    text_path = "paligemma_benchmark_summary.txt"
    with open(text_path, "w") as f:
        f.writelines(text_lines)

    print(f"\n[INFO] Saved formatted summary text to {text_path}")


if __name__ == "__main__":
    main()
