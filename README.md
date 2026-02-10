# Cosmos-Reason2 Quick Start


## Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM
- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.10+

## Installation

### 1. Clone the official Cosmos-Reason2 repo

```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2
```

### 2. Install uv package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3. Install dependencies

```bash
uv sync
source .venv/bin/activate
```

### 4. Install torchvision 

```bash
uv pip install torchvision
```

### 5. (Optional) Login to Hugging Face

```bash
uv tool install -U huggingface_hub
hf auth login
```

## Run the Demo

### 1. Copy the test script to the cosmos-reason2 folder

```bash
cp test_video_analysis.py cosmos-reason2/scripts/
```

### 2. Place your video in the assets folder

```bash
cp your_video.mp4 cosmos-reason2/assets/
```

### 3. Edit the script to point to your video

Open `scripts/test_video_analysis.py` and change line 20:
```python
VIDEO_PATH = "assets/your_video.mp4"
```

### 4. Run the script

```bash
cd cosmos-reason2
source .venv/bin/activate
python scripts/test_video_analysis.py
```


## Gradio Web UI (Video + Image Upload)

A browser-based UI for uploading videos/images and chatting with the model. Uses vLLM for fast inference.

### 1. Install vLLM and Gradio

```bash
cd cosmos-reason2
source .venv/bin/activate
pip install vllm gradio
```

### 2. Start the vLLM server (Terminal 1)

```bash
cd cosmos-reason2
source .venv/bin/activate
vllm serve nvidia/Cosmos-Reason2-8B \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 16384 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --port 8000
```

Wait until you see `Application startup complete.` (takes 2-3 minutes on first run).

### 3. Launch the Gradio UI (Terminal 2)

```bash
cd cosmos-reason2
source .venv/bin/activate
python ../ui.py
```

### 4. Open in browser

If using **VS Code / Cursor Remote SSH**:
- Press `Ctrl+Shift+P` → type "Forward a Port" → enter `7860`
- Repeat for port `8000`

Open **http://localhost:7860** in your browser. Upload a video or image, type a prompt, and click "Run Inference".

### 5. Stopping

- Stop the Gradio UI: `Ctrl+C` in Terminal 2
- Stop the vLLM server: `Ctrl+C` in Terminal 1


## Troubleshooting

### Out of Memory (OOM)
Reduce `max_vision_tokens` and `fps` in the script. For vLLM, reduce `--max-model-len` (e.g. `8192`).

### Model download issues
Run `hf auth login` and accept the model license at:
https://huggingface.co/nvidia/Cosmos-Reason2-8B

### Video too long
The model can handle ~10-15 seconds of video at `--max-model-len 16384` with FPS=4.
For longer videos, increase `--max-model-len` (e.g. `32768`) if you have enough VRAM.


## Links

- [Cosmos-Reason2 GitHub](https://github.com/nvidia-cosmos/cosmos-reason2)
- [Cosmos-Reason2 Docs](https://docs.nvidia.com/cosmos/latest/reason2/index.html)
- [Model on HuggingFace](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
