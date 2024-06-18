"""
This is test script for now
"""

import os
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

image_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "images",
    "table0.png",
)
raw_image = Image.open(image_path).convert("RGB")
prompt = "List the objects in the table."
prompt = "Detect the pepsi can in the image."
inputs = processor(prompt, raw_image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=128)

print(processor.decode(output[0], skip_special_tokens=False))
