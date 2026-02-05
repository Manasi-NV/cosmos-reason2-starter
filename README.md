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


## Troubleshooting

### Out of Memory (OOM)
Reduce `max_vision_tokens` and `fps` in the script.

### Model download issues
Run `hf auth login` and accept the model license at:
https://huggingface.co/nvidia/Cosmos-Reason2-2B


## Links

- [Cosmos-Reason2 GitHub](https://github.com/nvidia-cosmos/cosmos-reason2)
- [Cosmos-Reason2 Docs](https://docs.nvidia.com/cosmos/latest/reason2/index.html)
- [Model on HuggingFace](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
