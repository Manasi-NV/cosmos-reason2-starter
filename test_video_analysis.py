#!/usr/bin/env python
"""
Cosmos-Reason2 
==================================

Usage:
    1. Place this script in cosmos-reason2/scripts/
    2. Edit VIDEO_PATH below to point to your video
    3. Run: python scripts/test_video_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import gc
from pathlib import Path
import torch
import transformers

# ============================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================

VIDEO_PATH = "assets/sample.mp4"  # ← Change this to your video path

MAX_VISION_TOKENS = 8192   # Image/video resolution
MIN_VISION_TOKENS = 256    # Minimum resolution
FPS = 4                    # Frames per second
MAX_NEW_TOKENS = 4096      # Max output length

# Model selection
MODEL_NAME = "nvidia/Cosmos-Reason2-8B"  # or "nvidia/Cosmos-Reason2-8B"


ROOT = Path(__file__).parents[1]
PIXELS_PER_TOKEN = 32 * 32  
SEPARATOR = "=" * 60


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print(SEPARATOR)
    print("COSMOS-REASON2 VIDEO ANALYSIS")
    print(SEPARATOR)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected. Cosmos-Reason2 requires a CUDA GPU.")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    video_path = VIDEO_PATH if Path(VIDEO_PATH).is_absolute() else f"{ROOT}/{VIDEO_PATH}"
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        return
    print(f"Video: {video_path}")
    
    print(f"\nLoading model: {MODEL_NAME}")
    print("(This may take a few minutes on first run...)")
    
    transformers.set_seed(42)
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(MODEL_NAME)
    
    # Configure vision processing
    processor.image_processor.size = {
        "shortest_edge": MIN_VISION_TOKENS * PIXELS_PER_TOKEN,
        "longest_edge": MAX_VISION_TOKENS * PIXELS_PER_TOKEN,
    }
    processor.video_processor.size = {
        "shortest_edge": MIN_VISION_TOKENS * PIXELS_PER_TOKEN,
        "longest_edge": MAX_VISION_TOKENS * PIXELS_PER_TOKEN,
    }
    
    print(f"Settings: max_vision_tokens={MAX_VISION_TOKENS}, fps={FPS}")
    print(SEPARATOR)
    
    # ========================================
    # TEST 1: Video Caption
    # ========================================
    print("\n[TEST 1] Video Caption")
    print("-" * 40)
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that analyzes videos."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Describe what is happening in this video in detail."},
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", fps=FPS
    ).to(model.device)
    
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print("Generating response...")
    
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print("\nResponse:")
    print(output_text[0])
    
    clear_gpu()
    
    # ========================================
    # TEST 2: Object Detection
    # ========================================
    print("\n" + SEPARATOR)
    print("[TEST 2] Object Detection with Bounding Boxes")
    print("-" * 40)
    
    conversation2 = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that detects objects in videos."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Detect all people and objects in this video. For each, provide: label, bounding box [x1,y1,x2,y2] normalized 0-1000, and brief description. Return as JSON."},
            ],
        },
    ]
    
    inputs2 = processor.apply_chat_template(
        conversation2, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", fps=FPS
    ).to(model.device)
    
    print(f"Input tokens: {inputs2['input_ids'].shape[1]}")
    print("Generating response...")
    
    generated_ids2 = model.generate(**inputs2, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids_trimmed2 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs2.input_ids, generated_ids2)]
    output_text2 = processor.batch_decode(generated_ids_trimmed2, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print("\nResponse:")
    print(output_text2[0])
    
    clear_gpu()
    
    # ========================================
    # TEST 3: Reasoning
    # ========================================
    print("\n" + SEPARATOR)
    print("[TEST 3] Scene Reasoning")
    print("-" * 40)
    
    conversation3 = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an AI assistant that reasons about video content."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "What is likely to happen next in this video? Think step by step about the current situation and predict the next actions."},
            ],
        },
    ]
    
    inputs3 = processor.apply_chat_template(
        conversation3, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", fps=FPS
    ).to(model.device)
    
    print(f"Input tokens: {inputs3['input_ids'].shape[1]}")
    print("Generating response...")
    
    generated_ids3 = model.generate(**inputs3, max_new_tokens=MAX_NEW_TOKENS)
    generated_ids_trimmed3 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs3.input_ids, generated_ids3)]
    output_text3 = processor.batch_decode(generated_ids_trimmed3, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    print("\nResponse:")
    print(output_text3[0])
    
    print("\n" + SEPARATOR)
    print("All tests completed!")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
