from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# now, we add in the timing
import time

MODEL_ID = "microsoft/phi-2"  # or whatever the real id is

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
    )

    prompt = "Explain quantization in simple terms."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # torch.cuda.synchronize makes it so that the
    # CPU and GPU work in parallel
    # The CPU schedules work → GPU starts executing, then
    # → CPU continues running Python code immediately.
    # torch.cuda.synchronize() forces the CPU to wait 
    # until all queued GPU operations in the current CUDA 
    # stream have completed execution
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )
    
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = end - start
    print(f"Elapsed: {elapsed:.4f} s")

    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()