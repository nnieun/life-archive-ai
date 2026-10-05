"""Ollama native structured QA with explicit generation and residency limits."""

from types import SimpleNamespace

import httpx
from pydantic import BaseModel


class OllamaQAModel:
    def __init__(self, model: str, base_url: str, schema: type[BaseModel], *,
                 max_tokens: int = 768, keep_alive: str = '15m', think: bool = False,
                 client: httpx.Client | None = None):
        self.model = model
        self.url = base_url.removesuffix('/').removesuffix('/v1') + '/api/chat'
        self.schema = schema
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self.think = think
        self.client = client or httpx.Client(timeout=300, trust_env=False)

    def invoke(self, messages):
        roles = {'system': 'system', 'human': 'user', 'ai': 'assistant'}
        response = self.client.post(self.url, json={
            'model': self.model, 'messages': [{'role': roles[message.type], 'content': message.content} for message in messages],
            'format': self.schema.model_json_schema(), 'stream': False, 'think': self.think,
            'keep_alive': self.keep_alive, 'options': {'temperature': 0, 'num_predict': self.max_tokens},
        })
        response.raise_for_status()
        payload = response.json()
        metadata = {'finish_reason': 'length' if payload.get('done_reason') == 'length' else 'stop'}
        metadata.update({key: payload.get(key, 0) for key in ('load_duration', 'prompt_eval_duration', 'eval_duration', 'prompt_eval_count', 'eval_count')})
        raw = SimpleNamespace(response_metadata=metadata)
        if payload.get('done_reason') == 'length':
            return {'raw': raw, 'parsed': None, 'parsing_error': None}
        try:
            parsed = self.schema.model_validate_json(payload['message']['content'])
        except (ValueError, KeyError) as error:
            return {'raw': raw, 'parsed': None, 'parsing_error': error}
        return {'raw': raw, 'parsed': parsed, 'parsing_error': None}
