
from typing import Dict, Any, Optional
from .base import LLMBackend


class AsyncQwenLLMBackend(LLMBackend):

    CHINESE_NAME = "通义千问异步LLM后端"

    def _build_api_url(self, configs: Dict[str, Any]) -> str:
        return configs.get("api_url") or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def _build_headers(self, api_key: str) -> dict:
        headers = super()._build_headers(api_key)
        headers["X-DashScope-Synchronous"] = "true"
        return headers

    def _build_json_payload(
            self,
            prompt: str,
            model: str,
            params: dict,
            *,
            system_prompt: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": model,
            "input": {"messages": messages},
            "parameters": params
        }

    def _build_text_payload(self, prompt: str, model: str, params: dict, **kwargs) -> Dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        adjusted = self._normalize_params_for_free_text(params)
        return {
            "model": model,
            "messages": messages,
            **adjusted
        }

    def _extract_content_from_response(self, data: Dict) -> Optional[str]:
        try:
            if 'output' in data and 'choices' in data['output']:
                return data['output']['choices'][0]['message']['content']
            elif 'output' in data and 'text' in data['output']:
                return data['output']['text']
        except (KeyError, IndexError, TypeError):
            pass
        return None

    @staticmethod
    def _normalize_params_for_free_text(params: dict) -> dict:
        """清理不适合自由文本生成的参数"""
        adjusted = params.copy()
        # 移除可能引发异常的字段
        adjusted.pop("result_format", None)
        return adjusted
