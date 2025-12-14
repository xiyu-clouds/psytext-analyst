
from typing import Dict, Any, Optional
from .base import LLMBackend
from src.state_of_mind.utils.logger import LoggerManager as logger


class AsyncDeepSeekBackend(LLMBackend):

    CHINESE_NAME = "DeepSeek 异步 LLM 后端"

    def _build_api_url(self, configs: dict) -> str:
        base_url = configs.get("api_url") or "https://api.deepseek.com"
        return f"{base_url.rstrip('/')}/chat/completions"

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

        adjusted = self._normalize_params_for_json(params)

        return {
            "model": model,
            "messages": messages,
            **adjusted
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
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            pass
        return None

    @staticmethod
    def _normalize_params_for_json(params: dict) -> dict:
        """
        将通用参数转换为特定 LLM 后端所需的格式
        仅在需要结构化输出时才启用 response_format
        """
        adjusted = params.copy()
        if "max_output_tokens" in adjusted:
            adjusted["max_tokens"] = adjusted.pop("max_output_tokens")

        if "result_format" in adjusted:
            fmt = adjusted.pop("result_format")
            if fmt == "json_object" or fmt == "message":
                adjusted["response_format"] = {"type": "json_object"}
            else:
                logger.warning("不支持的 result_format，已忽略", extra={"format": fmt})

        return adjusted

    @staticmethod
    def _normalize_params_for_free_text(params: dict) -> dict:
        """清理不适合自由文本生成的参数"""
        adjusted = params.copy()
        # 移除可能引发异常的字段
        adjusted.pop("result_format", None)
        # 兼容 max_output_tokens → max_tokens
        if "max_output_tokens" in adjusted:
            adjusted["max_tokens"] = adjusted.pop("max_output_tokens")
        return adjusted
