from openai import OpenAI
import requests
import httpx
from cache import Cache
class APIModel:
    def __init__(self, cache, base_url, model_name, api_key="EMPTY"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True,
            ),
        )
        self.model_name = model_name
        self.cache = cache

    def generate(self, query, max_tokens=32000, temperature=0.0):
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
                        messages=[
                            {
                                "role": "user", 
                                "content": query
                            }
                        ],
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

    def generate_chat(self, messages, max_tokens=32000, temperature=0.0):
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

# test
# cache = Cache("/data1/qyj/RealIF/data/.cache/new_test.pkl")
# base_url = "https://svip-hk.xty.app/v1"
# model_name = "o1-preview-2024-09-12"
# api_key = "sk-vP85CSMGSmkzxyoDN4zIxzzExvWjoKcDW8o15kJVftnaFQHO"
# model = APIModel(cache, base_url, model_name, api_key)
# messages = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant."
#     },
#     {
#         "role": "user",
#         "content": "Hello, world!"
#     }
# ]
# response = model.generate_chat(messages)
# print(response)
