import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class OwlViT:
    def __init__(self, model_id="google/owlvit-base-patch32", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = OwlViTProcessor.from_pretrained(model_id)
        self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)

    def load_image_from_url(self, image_url):
        return Image.open(requests.get(image_url, stream=True).raw)

    def detect_objects(self, image, text, threshold=0.1, clean_output: bool = False):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(text=[text], images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )

        if clean_output:
            results = [
                {
                    "labels": result["labels"],
                    "boxes": result["boxes"].detach().cpu().tolist(),
                    "scores": result["scores"].detach().cpu().tolist(),
                }
                for result in results
            ]

        return results

    def draw_boxes(self, image, boxes, labels=None):
        max_axis = max(image.size)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=max_axis // 50)

        for i, box in enumerate(boxes):
            draw.rectangle(box.tolist(), outline="red", width=2)
            if labels:
                label_text = labels[i]
                label_size = draw.textbbox((0, 0), label_text, font=font)
                label_background = [
                    box[0],
                    box[1] - label_size[3],
                    box[0] + (label_size[2] - label_size[0]),
                    box[1],
                ]
                draw.rectangle(label_background, fill="red")
                draw.text(
                    (box[0], box[1] - label_size[3]),
                    label_text,
                    fill="white",
                    font=font,
                )

        return image
