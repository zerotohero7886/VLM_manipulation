import os
import sys
import base64
from typing import List, Optional, Literal, Any
from io import BytesIO
import torch

import openai
import instructor
from PIL import Image
from pydantic import BaseModel, Field

from gdino_module.gdino import GroundingDINO
from mllm_module.gpt import gpt4v_completion_async, gpt4v_completion
from mllm_module.conversation import ImageTextConversation, Message
from prompts import AGENT_SYSTEM_PROMPT


class QueryToDetectionModel(BaseModel):
    queries: str
    reasoning: str

    def to_prompt(self):
        prompt = f"queries: {self.queries}\reasoning: {self.reasoning}"
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

        self.gpt_model_name = gpt_model_name
        self.gdino = GroundingDINO()

        # set base conversation
        self.conversation = ImageTextConversation(
            messages=[
                Message(role="system", text=AGENT_SYSTEM_PROMPT),
                # Message(role="user", text="Describe the image", image_path=image_path),
            ]
        )

    def run(self, image_path: str, instruction: str):
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

        print(detection_queries)
        #################################
        # step2 : Detection using GDINO #
        #################################

        _detection_output = self.gdino.detect_objects(
            image_path, detection_queries.queries
        )[0]
        detection_output: DetectionOutput = DetectionOutput(**_detection_output)

        print(detection_output.to_prompt())
        self.conversation.messages.append(
            Message(
                role="user",
                text=f"Here are detection outputs : {detection_output.to_prompt()}\nGenerate the robot action for this image",
            )
        )

        ##################################
        # step3 : Robotic Action Calling #
        ##################################

        robot_action = gpt4v_completion(
            self.conversation,
            model=self.gpt_model_name,
            response_model=RobotAction,
        )
        print("##################################")
        print("# step3 : Robotic Action Calling #")
        print("##################################")
        print(robot_action)

        # return response


if __name__ == "__main__":
    # cd src; python3 -m agent

    agent = Agent()
    image_path = "../data/images/table0.png"  # 예제 이미지 경로
    instruction = "Pick up the apple and place next to the lemon"

    response = agent.run(image_path, instruction)
    print(response)
