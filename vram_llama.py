from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def vram_gb():
    return torch.cuda.memory_allocated() / (1024 ** 3)

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


    start_mem = vram_gb()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )
    
    end_mem = vram_gb()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    print("Start: ", start_mem, "GB")
    print("End: ", end_mem, "GB")
    print("Peak: ", peak_mem, "GB")



    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()