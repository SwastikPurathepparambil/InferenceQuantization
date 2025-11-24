# Inference Quantization Benchmarking #

## Introduction

Quantization turns parameters or weights that are represented in 32 bit or 16 bit into 8 or 4 or 2 or even 1 bit representations. This is done to save VRAM. Quantizing from 16 to 8 bits, for example, can significantly decrease the amount of VRAM usage depending on the model.

**What is VRAM?** VRAM is the GPU's memory. It contains model weights (which makes up ~40-70% of the VRAM usage of a model during inference), and then other things like activations, KV cache, etc.

**Activations** are the intermediate matrices produced from each layer of the model during the forward pass.

**KV Cache** is the cache that stores previous Key Value vectors (from QKV) that are computed during attention. This is only a feature of models that use attention (which is basically everything nowadays).

So when we quantize the model weights, we are saving on **the portion of the VRAM that holds the weights**, not the whole VRAM. This is why the save in GPU usage depends on each model and **does NOT necessarily mean** that quantizing from 16 to 8 or 8 to 4 bits means we halve our VRAM usage. During inference, however, the **model's weights make up the majority of the VRAM.**

**What is memory bandwidth?** This is a measure of how much data can move in and out of the VRAM (measured in GB/s or TB/s or PB/s). This is important because if it's low then computation step needs to wait for all the data to get passed to it first and then it needs to send it back which is also slow (memory-bound).

Inference is often bottlenecked by:

- memory bandwidth
- VRAM capacity
- cache sizes
- precision type

**The bottlenecks in each GPU are different and even depend on the ML model being used.**

Quantizing the weights allows us to **reduce the pressure on the memory bandwidth**. For example, let's say we have a model that has 2 GBs in model weights in **16 bit representation**. If our bandwidth is 1 GB/s and if we have to load all our weights, then that process will take **2 seconds**. If, however, we quantize to a **4 bit representation**, then our model weights will take up 0.5 GB. Thus the process will take **0.5 seconds**! *This is the usefulness of quantization.*

## Key Coding Libraries/Classes Used

### Transformers

This is the Hugging Face library that **provides pretrained models** (LLMs, vision, audio, multimodal) so you can generate results from them directly.

**AutoModelForCausalLM**

Loads GPT-style decoder-only models.

**AutoTokenizer**

Makes sure the model tokenizes properly with the model provided.

### BitsAndBytesConfig

If there is no config for 8 bit or 4 bit, then nothing.

If 8 bit, then the weights are now stored as 8 bits instead of 16 bits.

Quantization is done by applying a **scale** to an original weight. Let's say our scale for 8-bit is 0.00390625 which is around ≈ 1/256

Then for:

```
float_val = 0.425612
```

Quantization:

```
int_val = round(0.425612 / 0.00390625)
int_val = 109
```

Now we have our 16 bit floating point into an 8 bit value. Then the kernel dequantizes the weight as such:

```
float_val ≈ int_val × scale
float_val ≈ 109 × 0.00390625
float_val ≈ 0.42578125
```

Which is ≈ 0.425612

For the 4-bit quantization, we also set the type to be **Normalized Float 4 (nf4)**, which is a **learned distribution** of floating point values.

While int4 looks like `{−8, −7, ..., +7}`

NF4 uses a set like: `{−1.0, −0.83, −0.67, −0.55, ..., +0.65, +0.78, +1.0}`

This is done because **spacing is denser around 0 because Transformer weights cluster heavily near zero.**

The scales that quantize and dequantize to 4 bit and into 16-bit respectively ARE ALSO QUANTIZED (this is what **double quantization** is).

This saves about 20-30% more in terms of memory and very minimal change in accuracy.

### Torch

**torch.cuda_is_available()** → Checks if PyTorch can see a CUDA-capable GPU.

**torch.cuda.empty_cache()** → Releases unused memory from cache so we can measure VRAM accurately.

**torch.cuda.reset_peak_memory_stats()** → We reset every time we run a new model so that the peak_memory_gb is accurate and doesn't have any lingering results from the previous model's peak memory.

**torch.cuda.memory_allocated()** → Returns the **current VRAM usage** (in bytes) by PyTorch tensors on the default GPU.

**torch.cuda.max_memory_allocated()** → Returns the **peak VRAM usage** PyTorch has measured since the last reset.

**torch.cuda.synchronize()** → Forces the CPU to wait until **all asynchronous GPU work is finished.** This is important because CUDA runs asynchronously, meaning CPU tasks run while GPU tasks are running. Thus our timing of the latency will be off if we don't call torch.cuda.synchronize().

## Project Specifications ##

This project benchmarks various lightweight models across three precision modes:
- FP16
- INT8 
- 4-bit (NF4)

It measures:

- Latency (total & per-token)
- Throughput (tokens/sec)
- VRAM usage (start/end/peak)

Across:
- NVIDIA GeForce RTX 2080 Ti
- NVIDIA L40S
- Quadro RTX 8000

Models Tested:
- Meta Llama3.2 1B
- Microsoft Phi-2
- Google Flan T5-Base
- Google PaLI Gemma

## Environment Setup (Recommended) ##

Create environment

conda create -n quantbench python=3.10 -y

conda activate quantbench

## Install dependencies ##
3.1 Install compatible NumPy

Old packages crash on NumPy ≥1.25, so pin a stable version:

python -m pip install "numpy==1.23.5"

3.2 Install PyTorch (GPU, CUDA 12.1)

For NVIDIA L40S / Ampere / Ada GPUs:

python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

3.3 Install Hugging Face tooling
python -m pip install "transformers>=4.39.0" "accelerate" "safetensors" "huggingface_hub" -U

3.4 Install bitsandbytes
python -m pip install -U bitsandbytes

pip install Pillow

## Running the scripts ##
Each main folder has a benchmark script with argument specs provided 

Run FP16:
python benchmark_decoder.py \
  --precision fp16 \
  --model-id microsoft/phi-2 \
  --max-new-tokens 64

Run INT8 (LLM.int8):
python benchmark_decoder.py \
  --precision int8 \
  --model-id microsoft/phi-2 \
  --max-new-tokens 64

Run 4-bit (NF4):
python benchmark_decoder.py \
  --precision 4bit \
  --model-id microsoft/phi-2 \
  --max-new-tokens 64

## Key Conclusions

### **4-bit quantization works very well on all three GPUs**
- Fastest throughput
- Lowest VRAM usage
- Usually faster than FP16 and INT8

### **L40S performs the best out of the three**
- 2×–4× faster than 2080 Ti and Quadro 8000
- Best INT8 and 4-bit performance
- Best VRAM efficiency and memory bandwidth

### **INT8 is inconsistent**
- **Fast on Ada (L40S)**  
- **Slower than FP16 on 2080 Ti and Quadro 8000**  


## Per-Model Behavior

### **Decoder-only models (Phi-2, LLaMA-3.2-1B)**
- Quantized very well
- 4-bit gives largest speedup

### **Multimodal models (PaLI-Gemma)**
- Very bandwidth-bound  
- 4-bit produces very large performance gains

### **Encoder–decoder models (Flan-T5)**
- Less benefit from INT8
- 4-bit still improves speed and VRAM
