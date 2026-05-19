import base64
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://xiaoai.plus/v1"
)

# === 多张图片路径 ===
image_paths = [
    "/data/ccy/ccy-factory/datasets/id_data/boat-29/images/aircraft_carrier/127657.jpg",
    "/data/ccy/ccy-factory/datasets/id_data/boat-29/images/firefighting/109443.jpg",
    # 可以继续添加更多图片
]

def encode_image_to_data_url(path):
    with open(path, "rb") as f:
        img_bytes = f.read()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_img}"

# === 构建 prompt + 多图 ===
content = [{
    "type": "text",
    "text": (
        "Before my question, I will give you an example; "
        "please strictly follow my answer format (just use '-' before every answer) "
        "and just give me the answer (the answer starts with '-'), no other words!!! "
        "Do not include my questions and examples in your answer.\n"
        "What type of ships are shown in these images?"
    )
}]

# === 添加每张图像 ===
for path in image_paths:
    image_url = encode_image_to_data_url(path)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": image_url
        }
    })

# === 发送请求 ===
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": content
        }
    ],
    max_tokens=500,
    temperature=0
)

# === 输出响应 ===
print("LLM Response:\n", response.choices[0].message.content)
