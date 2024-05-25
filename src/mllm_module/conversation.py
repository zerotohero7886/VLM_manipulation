import os
import io, base64
from dataclasses import dataclass, field
from PIL import Image
import PIL
from typing import List, Dict, Optional, Union


@dataclass
class Message:
    role: str
    text: Optional[str] = None
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[str, Dict[str, str]]]:
        """Converts the message to a dictionary format expected by OpenAI API."""
        content = []
        if self.text:
            content.append({"type": "text", "text": self.text})
        if self.image_path:
            if isinstance(self.image_path, list):
                encoded_images = [self.encode_image(imgp) for imgp in self.image_path]
                for encoded_image in encoded_images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        }
                    )
            else:
                encoded_image = self.encode_image(self.image_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    }
                )
        return {"role": self.role, "content": content}

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encodes the image at the given path to base64. Compresses the image if it's larger than 20 MB."""

        if isinstance(image_path, PIL.Image.Image):
            # resize to 500 x 500
            image_path = image_path.resize((500, 500))
            return base64.b64encode(image_path.tobytes()).decode("utf-8")

        initial_size = os.path.getsize(image_path)

        if initial_size > 20 * 1024 * 1024:
            with Image.open(image_path) as img:
                # Resize the image to 500x500 pixels
                img_resized = img.resize((500, 500))

                # Save the resized image to a buffer
                buffer = io.BytesIO()
                img_resized.save(buffer, format="JPEG")

                # Check if the resized image's size is still over 20 MB
                if buffer.getbuffer().nbytes > 20 * 1024 * 1024:
                    raise ValueError(
                        f"Unable to reduce the image size below 20 MB. [{image_path}]"
                    )
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")


@dataclass
class Conversation:
    messages: List[Message]

    def get_llama_style_prompt_string(self):
        prompt = ""
        for message in self.messages:
            if message.role.lower() == "system":
                prompt += f"<s>[INST] <<SYS>>\n{message.text}\n<</SYS>>\n\n"
            else:
                prompt += f"{message.text} [/INST]\n"
        return prompt

    def to_openai_format(self) -> List[Dict[str, str]]:
        formatted_messages = []
        for message in self.messages:
            formatted_messages.append({"role": message.role, "content": message.text})
        return formatted_messages


@dataclass
class ImageTextConversation:
    messages: List[Message] = field(default_factory=list)

    def to_openai_format(
        self,
    ) -> List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]:
        """Converts the conversation to a format suitable for OpenAI API."""
        return [message.to_dict() for message in self.messages]