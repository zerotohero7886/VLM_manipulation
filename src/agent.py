import os
import sys
import base64
from typing import List, Optional, Literal, Any
from io import BytesIO
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import openai
import instructor
from PIL import Image
from pydantic import BaseModel, Field
from rich import print
from rich.console import Console
from rich.text import Text

from gdino_module.gdino import GroundingDINO
from mllm_module.gpt import gpt4v_completion_async, gpt4v_completion
from mllm_module.conversation import ImageTextConversation, Message
from prompts import AGENT_SYSTEM_PROMPT


class QueryToDetectionModel(BaseModel):
    reasoning: Optional[str] = Field(
        None, description="Detailed reasoning for the detection queries."
    )
    queries: str

    def to_prompt(self):
        prompt = f"queries: {self.queries}\nreasoning: {self.reasoning}"
        return prompt


class DetectionOutput(BaseModel):
    labels: List[str]
    boxes: List[List[float]]

    def to_prompt(self):
        prompt = ""
        for label, box in zip(self.labels, self.boxes):
            prompt += f"{label} : {list(map(int, box))}\n"
        return prompt


class RobotAction(BaseModel):
    rationale: Optional[str] = Field(
        None, description="Detailed reasoning for the action to achieve the goal."
    )
    action: Literal["Pick", "Place"]
    param: Any = Field(
        ...,
        description=(
            "- For 'Pick', provide the bounding box coordinates of the object to pick. "
            "The coordinates will be automatically projected to real-world coordinates.\n"
            "- For 'Place', provide the target bounding box coordinates where the object should be placed.\n"
            "- Coordinates should be in the format [x1, y1, x2, y2]."
        ),
    )

    class Config:
        schema_extra = {
            "example": {
                "rationale": "The object needs to be moved to the target location for further processing.",
                "action": "Pick",
                "param": [100, 200, 150, 250],
            }
        }


class Agent:
    def __init__(self, gpt_model_name="gpt-4o"):
        self.console = Console()
        self.gpt_model_name = gpt_model_name
        self.gdino = GroundingDINO()

        # set base conversation
        self.conversation = ImageTextConversation(
            messages=[
                Message(role="system", text=AGENT_SYSTEM_PROMPT),
                # Message(role="user", text="Describe the image", image_path=image_path),
            ]
        )

    def run(self, image_path: str, instruction: str, VERBOSE: bool = False):
        ########################################
        # step1 : Generate Query for Detection #
        ########################################

        self.conversation.messages.append(
            Message(role="user", text=instruction, image_path=image_path)
        )
        detection_queries = gpt4v_completion(
            self.conversation,
            model=self.gpt_model_name,
            response_model=QueryToDetectionModel,
        )
        self.conversation.messages.append(
            Message(role="assistant", text=detection_queries.to_prompt())
        )

        if VERBOSE:
            self.console.print("\n########################")
            self.console.print("#  [bold red]Detection Queries[/bold red]   #")
            self.console.print("########################")
            self.console.print(f"[cyan]Instruction:[/cyan] {instruction}")
            self.console.print(detection_queries)

        #################################
        # step2 : Detection using GDINO #
        #################################
        _detection_output = self.gdino.detect_objects(
            image_path, detection_queries.queries
        )[0]
        detection_output: DetectionOutput = DetectionOutput(**_detection_output)

        self.conversation.messages.append(
            Message(
                role="user",
                text=f"Here are detection outputs : {detection_output.to_prompt()}\nGenerate the robot action for this image",
            )
        )
        if VERBOSE:
            self.console.print("\n########################")
            self.console.print("#   [bold red]Detection Output[/bold red]   #")
            self.console.print("########################")
            self.console.print(detection_output)

        ##################################
        # step3 : Robotic Action Calling #
        ##################################

        robot_action = gpt4v_completion(
            self.conversation,
            model=self.gpt_model_name,
            response_model=RobotAction,
        )

        if VERBOSE:
            self.console.print("\n########################")
            self.console.print("# [bold red]Final Robotic Action[/bold red] #")
            self.console.print("########################")
            self.console.print(robot_action)

        # return response


if __name__ == "__main__":
    # cd src; python3 -m agent

    agent = Agent()
    image_path = "../data/images/table0.png"  # 예제 이미지 경로
    instruction = "Pick up the apple and place next to the lemon"

    response = agent.run(image_path, instruction, VERBOSE=True)
