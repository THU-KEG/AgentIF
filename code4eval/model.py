import os

from openai import OpenAI
import requests
import httpx
from cache import Cache

from openai.lib.azure import AzureOpenAI


def get_azure_openai_client():
    """Return an AzureOpenAI client."""
    endpoint = os.getenv("EVALUATOR_URL")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=os.getenv("EVALUATOR_API_KEY"),
        api_version=os.getenv("EVALUATOR_API_VERSION", "2024-02-15-preview")
    )


class APIModel:
    def __init__(self, cache, base_url, model_name, api_key="EMPTY"):
        if "azure" in base_url:
            self.client = get_azure_openai_client()
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=httpx.Client(
                    base_url=base_url,
                    follow_redirects=True,
                ),
            )
        self.base_url = base_url
        self.model_name = model_name
        self.cache = cache

    def generate(self, query, max_tokens=8192, temperature=0.0):
        # import pdb; pdb.set_trace()

        if temperature == 0.0:
            response = self.cache.check_prompt(query)
        else:
            response = None
        if response is None:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": query}],
                        model=self.model_name,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    response = chat_completion.choices[0].message.content
                    self.cache.save_prompt(query, response)
                    break
                except Exception as e:
                    print(f"Attempt {retry_count + 1} failed: {e}")
                    retry_count += 1
                    if retry_count == max_retries:
                        print("All retries failed")
                        response = None
        return response

    def generate_chat(self, messages, max_tokens=8192, temperature=0.0):
        if temperature == 0.0:
            response = self.cache.check_prompt(messages[-1]["content"])
        else:
            response = None
        if response is None:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response = chat_completion.choices[0].message.content
                self.cache.save_prompt(messages[-1]["content"], response)
            except Exception as e:
                print(e)
                response = None
        return response
