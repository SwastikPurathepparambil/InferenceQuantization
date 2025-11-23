# Inference Quantization Benchmarking #
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