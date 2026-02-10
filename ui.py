"""Gradio UI for Cosmos-Reason2 via vLLM server."""

import base64
import json
import gradio as gr
import requests

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "nvidia/Cosmos-Reason2-8B"


def encode_video(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def infer(prompt: str, video: str | None, image: str | None, enable_reasoning: bool):
    if not prompt.strip():
        prompt = "Caption the video in detail." if video else "Describe the image in detail."

    system_text = "You are a helpful assistant."
    if enable_reasoning:
        system_text += (
            "\n\nAnswer the question using the following format:\n\n"
            "<think>\nYour reasoning.\n</think>\n\n"
            "Write your final answer immediately after the </think> tag."
        )

    user_content = []

    if video:
        video_b64 = encode_video(video)
        user_content.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
        })
    elif image:
        image_b64 = encode_image(image)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })

    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.6,
        "stream": True,
    }

    response = requests.post(VLLM_URL, json=payload, stream=True, timeout=300)
    response.raise_for_status()

    reasoning = ""
    answer = ""

    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]

            if "reasoning_content" in delta and delta["reasoning_content"]:
                reasoning += delta["reasoning_content"]
                yield reasoning, answer

            if "content" in delta and delta["content"]:
                answer += delta["content"]
                yield reasoning, answer

        except (json.JSONDecodeError, KeyError):
            continue

    yield reasoning, answer


with gr.Blocks(title="Cosmos-Reason2") as demo:
    gr.Markdown("# Cosmos-Reason2 — Physical AI Reasoning")
    gr.Markdown("Upload a video or image and ask questions. The model reasons about physical common sense and embodied scenarios.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            image_input = gr.Image(label="Or Upload Image", type="filepath", sources=["upload"])
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Caption the video in detail...",
                lines=3,
            )
            reasoning_toggle = gr.Checkbox(label="Enable chain-of-thought reasoning", value=True)
            submit_btn = gr.Button("Run Inference", variant="primary", size="lg")

        with gr.Column(scale=1):
            reasoning_output = gr.Textbox(label="Reasoning (thinking)", lines=12, interactive=False)
            answer_output = gr.Textbox(label="Answer", lines=12, interactive=False)

    submit_btn.click(
        fn=infer,
        inputs=[prompt_input, video_input, image_input, reasoning_toggle],
        outputs=[reasoning_output, answer_output],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
