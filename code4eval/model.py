from openai import OpenAI
import requests
import httpx
import re
import time
from cache import Cache
class APIModel:
    def __init__(self, cache, base_url, model_name, api_key="EMPTY", reasoning_effort=None, timeout=3000):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True,
            ),
            timeout=timeout   # per-request timeout in seconds (the judge uses 600 so that a hung request does not block a worker for long)
        )
        self.api_key = api_key
        self.base_url = base_url
        if "glm" in model_name: # big model
            self.headers = {
                "Authorization": "Bearer " + self.api_key ,
                "Content-Type": "application/json"
            }

        self.model_name = model_name
        self.cache = cache
        # reasoning effort for gpt-5 family models (minimal/low/medium/high); None = not sent
        self.reasoning_effort = reasoning_effort
        # details of the last generate_chat call (reasoning content split off by the vLLM reasoning parser / finish reason / max_tokens actually used)
        self.last_reasoning = None
        self.last_finish_reason = None
        self.last_max_tokens = None
        self.last_error = None
        # set when the request was rejected by a server-side content filter 3 times in a row; the caller scores the constraint as judge_null
        self.last_filtered = None

    _FILTER_MARKERS = ("content management policy", "content_filter", "ResponsibleAIPolicyViolation", "content filtering")

    @classmethod
    def _is_content_filtered(cls, msg):
        return any(m in msg for m in cls._FILTER_MARKERS)

    # vLLM answers 400 when prompt + max_tokens exceeds max-model-len:
    # "This model's maximum context length is 65536 tokens. However, you requested 70000 tokens
    #  (38000 in the messages, 32000 in the completion). ..."
    _CTX_RE = re.compile(r"maximum context length is (\d+) tokens.*?requested (\d+) tokens.*?\((\d+) in the messages", re.S)

    def _shrink_max_tokens_from_error(self, msg, cur_max_tokens):
        m = self._CTX_RE.search(msg)
        if not m:
            return None
        ctx, _, prompt_len = (int(x) for x in m.groups())
        new = ctx - prompt_len - 16
        if new <= 0 or new >= cur_max_tokens:
            return None
        return new

    def _build_kwargs(self, query, max_tokens, temperature, with_temp=True, with_effort=True):
        """Build request kwargs per model family.
        gpt-5 family: max_completion_tokens / reasoning_effort go through extra_body (older SDKs reject the keywords);
                      gpt-5.6-luna only accepts the default temperature, `with_temp` controls whether it is sent.
        others:       plain max_tokens + temperature.
        """
        kwargs = dict(
            messages=[{"role": "user", "content": query}],
            model=self.model_name,
        )
        if self.model_name.startswith("gpt-5"):
            extra = {"max_completion_tokens": max_tokens}
            if with_effort and self.reasoning_effort:
                extra["reasoning_effort"] = self.reasoning_effort
            kwargs["extra_body"] = extra
            if with_temp:
                kwargs["temperature"] = temperature
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        return kwargs

    def generate(self, query, max_tokens=8192, temperature=0.0):
        self.last_error = None   # last error after all retries failed; the caller decides whether to abort
        self.last_filtered = None
        if temperature == 0.0:
            response = self.cache.check_prompt(query)
        else:
            response = None
        if response is None:
            # judge retries: up to 10 attempts with linear back-off 3s, 6s, 9s, ...
            max_retries = 10
            retry_sleep_base = 3
            filter_max = 3          # give up on this prompt after 3 consecutive content-filter rejections; other errors use all 10 attempts and then abort the evaluation
            n_filtered = 0
            with_temp = True
            with_effort = True
            for attempt in range(max_retries):
                try:
                    chat_completion = self.client.chat.completions.create(
                        **self._build_kwargs(query, max_tokens, temperature, with_temp, with_effort))
                    response = chat_completion.choices[0].message.content
                    self.cache.save_prompt(query, response)
                    break
                except Exception as e:
                    msg = str(e)
                    # temperature rejected (gpt-5.6-luna only supports the default) -> retry without it
                    if with_temp and "temperature" in msg:
                        with_temp = False
                        continue
                    # reasoning_effort not supported -> retry without it
                    if with_effort and "reasoning_effort" in msg:
                        with_effort = False
                        continue
                    # content filter (400): retried as well; only after 3 consecutive rejections the prompt is skipped (an unavailable API uses all 10 attempts and aborts instead)
                    if self._is_content_filtered(msg):
                        n_filtered += 1
                        print(f"judge content filtered ({n_filtered}/{filter_max}): {msg[:160]}", flush=True)
                        if n_filtered >= filter_max:
                            print("content filtered 3 times, skip this prompt", flush=True)
                            self.last_filtered = msg
                            response = None
                            break
                        time.sleep(retry_sleep_base * n_filtered)
                        continue
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {msg[:160]}", flush=True)
                    if attempt == max_retries - 1:
                        print("All retries failed", flush=True)
                        self.last_error = msg
                        response = None
                    else:
                        wait = retry_sleep_base * (attempt + 1)
                        print(f"retry in {wait}s", flush=True)
                        time.sleep(wait)
        return response

    def generate_chat(self, messages, max_tokens=8192, temperature=0.0,
                      top_p=None, presence_penalty=None, extra_body=None, use_cache=True):
        """Chat completion.
        Optional parameters (None => not sent):
          top_p / presence_penalty : standard OpenAI sampling parameters
          extra_body               : extra fields passed to vLLM, e.g.
                                     {"top_k": 20, "min_p": 0, "repetition_penalty": 1.0,
                                      "chat_template_kwargs": {"enable_thinking": True}}
          use_cache                : False = neither read nor write the cache (independent repeated runs)
        """
        self.last_reasoning = None
        self.last_finish_reason = None
        self.last_max_tokens = max_tokens
        self.last_error = None   # set when all retries failed; last_error None with response None means the API succeeded but returned no final answer
        if temperature == 0.0 and use_cache:
            response = self.cache.check_prompt(messages[-1]["content"])
            # response = None
        else:
            response = None
        if response is None:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if "glm" in self.model_name:
                        # import pdb;pdb.set_trace()
                        chat_completion = self.client.chat.completions.create(
                            messages=messages,
                            model=self.model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            extra_body={
                                "thinking": {
                                    "type": "enabled",
                                },
                            }
                        )
                        response = chat_completion.choices[0].message.content
                        self.cache.save_prompt(messages[-1]["content"], response)
                        break
                        # url = self.base_url
                        # payload = {
                        #     "model": self.model_name,
                        #     "messages": messages,
                        #     "thinking": { "type": "enabled" },
                        #     "stream": False,
                        #     "do_sample": True,
                        #     "temperature": temperature,
                        #     "top_p": 0.95,
                        #     "tool_stream": False,
                        #     "response_format": { "type": "text" }
                        # }
                        # chat_completion = requests.post(url, json=payload, headers=self.headers)
                        # response = chat_completion.json()["choices"][0]["message"]["content"]
                        # self.cache.save_prompt(messages[-1]["content"], response)
                        break
                    else:
                        kwargs = dict(
                            messages=messages,
                            model=self.model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        if top_p is not None:
                            kwargs["top_p"] = top_p
                        if presence_penalty is not None:
                            kwargs["presence_penalty"] = presence_penalty
                        if extra_body:
                            kwargs["extra_body"] = extra_body
                        chat_completion = self.client.chat.completions.create(**kwargs)
                        choice = chat_completion.choices[0]
                        response = choice.message.content
                        self.last_finish_reason = getattr(choice, "finish_reason", None)
                        # with --reasoning-parser vLLM returns the thinking part in reasoning_content (or reasoning)
                        self.last_reasoning = getattr(choice.message, "reasoning_content", None) \
                            or getattr(choice.message, "reasoning", None) \
                            or (getattr(choice.message, "model_extra", None) or {}).get("reasoning_content")
                        self.last_max_tokens = max_tokens
                        if use_cache:
                            self.cache.save_prompt(messages[-1]["content"], response)
                        break
                except Exception as e:
                    msg = str(e)
                    # prompt + max_tokens exceeds max-model-len: shrink max_tokens to the number reported by the server and retry (not counted as a retry)
                    new_max = self._shrink_max_tokens_from_error(msg, max_tokens)
                    if new_max is not None:
                        print(f"max_tokens {max_tokens} exceeds context, shrink to {new_max} and retry")
                        max_tokens = new_max
                        continue
                    print(f"Attempt {retry_count + 1} failed: {e}")
                    retry_count += 1
                    if retry_count == max_retries:
                        # import pdb;pdb.set_trace()
                        print("All retries failed")
                        self.last_error = msg
                        response = None
                    else:
                        time.sleep(2 * retry_count)
        return response
