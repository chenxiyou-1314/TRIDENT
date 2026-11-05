"""
Debug tool: test multi-image input with a GPT-4o-compatible vision API.

Moved from root test.py. This is a development/debug script, NOT part of
the TRIDENT pipeline.

Security notes:
- Never hardcode API keys here; always use environment variables.
- Set OPENAI_BASE_URL only if you use a custom/proxy endpoint.
"""
import base64
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# TODO: replace with your own image paths
image_paths = [
    "/path/to/ship/image1.jpg",
    "/path/to/ship/image2.jpg",
]


def encode_image_to_data_url(path):
    with open(path, "rb") as f:
        img_bytes = f.read()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_img}"


def main():
    content = [{
        "type": "text",
        "text": (
            "Before my question, I will give you an example; "
            "please strictly follow my answer format (just use '-' before every answer) "
            "and just give me the answer (the answer starts with '-'), no other words!!! "
            "Do not include my questions and examples in your answer.\n"
            "What type of ships are shown in these images?"
        ),
    }]
    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_to_data_url(path)},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=500,
        temperature=0,
    )
    print("LLM Response:\n", response.choices[0].message.content)


if __name__ == "__main__":
    main()
