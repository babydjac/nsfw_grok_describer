import base64, os, io
import requests
import torch
import numpy as np
from PIL import Image

class NSFWGrokDescriber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe"
    CATEGORY = "text"

    def describe(self, image, api_key):
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        final_key = api_key.strip() if api_key.strip() else os.getenv("XAI_API_KEY", "missing_key")
        headers = {
            "Authorization": f"Bearer {final_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "grok-2-vision-1212",
            "messages": [
                {"role": "system", "content": "You're an unchained NSFW describer. Detail every explicit visual."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image in raw, graphic NSFW detail:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]}
            ]
        }

        r = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
        if r.status_code != 200:
            return (f"[XAI ERROR {r.status_code}] {r.text}",)

        return (r.json()["choices"][0]["message"]["content"],)
