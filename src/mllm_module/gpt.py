import os
import openai
from openai import AsyncOpenAI
from typing import Optional, Literal
from pydantic import BaseModel

from mllm_module.conversation import Conversation, ImageTextConversation, Message
import instructor


async def gpt4v_completion_async(
    conversation: ImageTextConversation,
    model: str = "gpt-4-vision-preview",
    max_tokens: int = 1024,
    top_p: float = 0.9,
    temperature: float = 0.1,
    VERBOSE: bool = False,
):
    # Asynchronous call for gpt4v
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages_for_api = conversation.to_openai_format()
    async_client = AsyncOpenAI(api_key=openai.api_key)

    if VERBOSE:
        print(f"User : ")
        print(conversation.messages[-1])

    chat_completion = await async_client.chat.completions.create(
        model=model,
        messages=messages_for_api,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
    )

    if VERBOSE:
        print(f"Response : ")
        print(chat_completion.choices[0].message.content)

    return chat_completion.choices[0].message.content


def gpt4v_completion(
    conversation: ImageTextConversation,
    model: str = "gpt-4-vision-preview",
    max_tokens: int = 1024,
    top_p: float = 0.9,
    temperature: float = 0.1,
    response_model: Optional[BaseModel] = None,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages_for_api = conversation.to_openai_format()

    if response_model is not None:
        client = instructor.from_openai(openai.OpenAI())
        pydantic_output = client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            response_model=response_model,
        )
        return pydantic_output
    else:
        client = openai.OpenAI()
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content
