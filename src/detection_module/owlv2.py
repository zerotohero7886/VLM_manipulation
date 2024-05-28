import requests
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class Owlv2:
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)

    def load_image_from_url(self, image_url):
        return Image.open(requests.get(image_url, stream=True).raw)

    def detect_objects(
        self, image, text, box_threshold=0.1, clean_output: bool = False
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(text=[text], images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=torch.Tensor([image.size[::-1]]),
            threshold=box_threshold,
        )
        # NOTE : https://github.com/huggingface/transformers/blob/573565e35a5cc68f6cfb6337f5a93753ab16c65b/src/transformers/models/owlv2/image_processing_owlv2.py#L484
        # I guess post_process_object_detection has no nms (non-maximum suppression) and it returns all the boxes.

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
