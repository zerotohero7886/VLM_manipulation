import sys
import os
import openai
import unittest
from termcolor import colored
import pytest
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from mllm_module.gpt import gpt4v_completion_async, gpt4v_completion
from mllm_module.conversation import Message, ImageTextConversation


class TestGPTFunctions(unittest.TestCase):

    def setUp(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise EnvironmentError(
                "Please set the OPENAI_API_KEY environment variable."
            )

    def test_gpt4v_completion_with_image(self):
        models = ["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview"]
        console = Console()

        # Create a test conversation with an image
        image_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "images", "table0.png"
        )
        conversation = ImageTextConversation(
            messages=[
                Message(role="user", text="Describe the image", image_path=image_path),
            ]
        )

        for model in models:
            # Call the function
            response = gpt4v_completion(conversation, model=model)
            # Create a panel with the response
            panel = Panel(
                Text(response, style="bold"),
                title=colored(model, "green"),
                expand=False,
            )
            # Print the panel
            print("\n")
            console.print(panel)

        # Check that the response is a non-empty string for each model
        for model in models:
            response = gpt4v_completion(conversation, model=model)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)


@pytest.mark.asyncio
async def test_gpt4v_completion_async_with_image():
    models = ["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview"]
    console = Console()

    # Create a test conversation with an image
    image_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "images", "table0.png"
    )
    conversation = ImageTextConversation(
        messages=[
            Message(role="user", text="Describe the image", image_path=image_path),
        ]
    )

    for model in models:
        # Call the function
        response = await gpt4v_completion_async(conversation, model=model)
        # Create a panel with the response
        panel = Panel(
            Text(response, style="bold"), title=colored(model, "green"), expand=False
        )
        # Print the panel
        print("\n")
        console.print(panel)

    # Check that the response is a non-empty string for each model
    for model in models:
        response = await gpt4v_completion_async(conversation, model=model)
        assert isinstance(response, str)
        assert len(response) > 0


if __name__ == "__main__":
    unittest.main()

    # pytest -s tests/test_gpt.py

    # test only test_gpt4v_completion_with_image
    # pytest -k test_gpt4v_completion_with_image -s
