import os
import openai
from openai import AsyncOpenAI
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from mllm_module.conversation import ImageTextConversation
import instructor

# Load environment variables from .env file
load_dotenv()


async def gpt4v_completion_async(
    conversation: ImageTextConversation,
    model: str = "gpt-4-vision-preview",
    max_tokens: int = 1024,
    top_p: float = 0.9,
    temperature: float = 0.1,
    VERBOSE: bool = False,
    response_model: Optional[BaseModel] = None,
):
    # Asynchronous call for gpt4v
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages_for_api = conversation.to_openai_format()
    async_client = AsyncOpenAI(api_key=openai.api_key)

    if VERBOSE:
        print("User :")
        print(conversation.messages[-1])
        print('------')
        print(messages_for_api)

    if response_model is not None:
        client = instructor.from_openai(async_client)
        pydantic_output = await client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            response_model=response_model,
        )
        if VERBOSE:
            print("Response :")
            print(pydantic_output)
        return pydantic_output
    else:
        chat_completion = await async_client.chat.completions.create(
            model=model,
            messages=messages_for_api,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        if VERBOSE:
            print("Response :")
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
    # Synchronous call for gpt4v
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
