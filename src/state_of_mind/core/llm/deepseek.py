import json
import re
from typing import Dict, Any, Optional
import httpx
from aiohttp import ClientResponseError, RequestInfo
from yarl import URL
from .base import LLMBackend
from ..validator import validate_with_diagnosis
from ...config import config
from ...utils.retry_util import retry_decorator
from src.state_of_mind.utils.logger import LoggerManager as logger


class AsyncDeepSeekBackend(LLMBackend):
    CHINESE_NAME = "DeepSeek 异步 LLM 后端"

    def __init__(self, client: httpx.AsyncClient = None):
        super().__init__()
        self.client = client
        self.api_url = None

    async def init(self, configs: dict = None):
        await super().init(configs)
        if self.client is None:
            api_key = self.configs.get("api_key") or configs.get("api_key") or config.LLM_API_KEY
            base_url = self.configs.get("base_url") or configs.get("base_url") or config.LLM_API_URL

            if not api_key:
                raise EnvironmentError("LLM_API_KEY 未设置")
            if not base_url:
                raise EnvironmentError("LLM_API_URL 未设置")

            self.api_url = f"{base_url.rstrip('/')}/chat/completions"

            timeout = self.configs.get("timeout", 60.0)
            self.client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=timeout
            )
        return self

    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def async_call(self, prompt: str, model: str, template_name: str,
                         step_name: str, params: dict, prompt_type: str) -> Dict[str, Any]:
        api_error = None
        # 启用结构化输出
        adjusted_params = self._normalize_params_for_backend(params)

        try:
            messages = [
                {"role": "system", "content": "只输出严格 JSON 格式，不要添加任何解释、前缀或 Markdown 代码块。"},
                {"role": "user", "content": prompt}
            ]

            # 构建 payload
            payload = {
                "model": model,
                "messages": messages,
                **adjusted_params
            }

            logger.info("调用 DeepSeek 异步 API", extra={
                "model": model,
                "template_name": template_name,
                "step_name": step_name,
                "prompt_length": len(prompt)
            })

            response = await self.client.post(self.api_url, json=payload)
            logger.info(f"DeepSeek 原始响应: {response.json()}")

            if response.status_code != 200:
                try:
                    resp_json = response.json()
                    error_msg = resp_json.get("error", {}).get("message", f"HTTP {response.status_code}")
                    code = resp_json.get("error", {}).get("type", "Unknown")
                except Exception:
                    error_msg = f"HTTP {response.status_code} (无法解析响应体)"
                    code = "ParseError"

                if response.status_code >= 500 or response.status_code == 429:
                    fake_request_info = RequestInfo(
                        url=URL(self.api_url),
                        method="POST",
                        headers={},
                        real_url=URL(self.api_url)
                    )
                    raise ClientResponseError(
                        request_info=fake_request_info,
                        history=(),
                        status=response.status_code,
                        message=error_msg,
                        headers=response.headers
                    )

                api_error = f"[{code}] {error_msg}"
                logger.warning("DeepSeek API 返回错误", extra={
                    "status_code": response.status_code,
                    "error": api_error,
                    "template_name": template_name,
                    "step_name": step_name
                })
                parsed_json = {"__api_error": api_error, "__status_code": response.status_code}
                raw_content = None
            else:
                content = self._extract_content_from_response(response.json())
                if not content or not content.strip():
                    api_error = "模型返回内容为空"
                    logger.warning("模型返回空内容", extra={
                        "template_name": template_name,
                        "step_name": step_name
                    })
                    parsed_json = {"__api_error": api_error}
                    raw_content = None
                else:
                    content = self.remove_check(content.strip())
                    raw_content = content
                    parsed_json = self.extract_json_safely(content)

        except Exception as e:
            system_error = str(e)
            logger.exception("async_call 系统级异常", extra={
                "model": model,
                "template_name": template_name,
                "step_name": step_name,
                "error": system_error
            })
            return {
                "__success": False,
                "__valid_structure": False,
                "data": {},
                "__raw_response": None,
                "__validation_errors": [],
                "__api_error": None,
                "__system_error": system_error,
                "model": model,
                "template_name": template_name,
                "step_name": step_name,
                "prompt_type": prompt_type
            }

        # 结构校验
        validation_result = validate_with_diagnosis(
            data=parsed_json,
            template_name=template_name,
            step_name=step_name
        )

        return {
            "__success": True,
            "__valid_structure": validation_result["is_valid"],
            "data": validation_result["cleaned_data"] or {},
            "__raw_response": raw_content,
            "__validation_errors": validation_result["errors"],
            "__api_error": api_error,
            "__system_error": None,
            "model": model,
            "template_name": template_name,
            "step_name": step_name,
            "prompt_type": prompt_type
        }

    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def generate_text(self, prompt: str, model: str, params: dict) -> str:
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            adjusted_params = params.copy()
            if "max_output_tokens" in adjusted_params:
                adjusted_params["max_tokens"] = adjusted_params.pop("max_output_tokens")

            if "result_format" in adjusted_params:
                fmt = adjusted_params.pop("result_format")
                logger.info(
                    "generate_text 用于生成自由文本，忽略 result_format 参数",
                    extra={"format": fmt}
                )

            payload = {
                "model": model,
                "messages": messages,
                **adjusted_params
            }
            response = await self.client.post(self.api_url, json=payload)
            if response.status_code != 200:
                error_detail = response.text[:200] if response.text else ""
                return f"生成失败: HTTP {response.status_code} - {error_detail}"

            content = self._extract_content_from_response(response.json())
            return content.strip() if content else "生成失败：无内容返回"
        except Exception as e:
            logger.exception("generate_text 失败", exc_info=True)
            return f"生成失败: {str(e)}"

    @staticmethod
    def _extract_content_from_response(data: Dict) -> Optional[str]:
        try:
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            pass
        return None

    @staticmethod
    def extract_json_safely(content: str) -> dict:
        if not content or not content.strip():
            return {"__error": "空响应内容"}

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r'^```(?:json|text|markdown)?\s*', '', content, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace('\\\\', '\\').replace('\\\'', '\'').replace('\\"', '"')

        try:
            match = re.search(r'{.*}', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

        return {"__error": "无法提取有效 JSON", "__raw": content[:200]}

    @staticmethod
    def remove_check(text: str) -> str:
        text = re.sub(r'^```(?:json|text|markdown)?\s*\n?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n?```$', '', text)
        return text

    @staticmethod
    def _normalize_params_for_backend(params: dict) -> dict:
        """
        将通用参数转换为特定 LLM 后端所需的格式
        仅在需要结构化输出时才启用 response_format
        """
        adjusted = params.copy()

        # max_output_tokens → max_tokens (通用)
        if "max_output_tokens" in adjusted:
            adjusted["max_tokens"] = adjusted.pop("max_output_tokens")

        # result_format → response_format (仅当需要结构化输出时才处理)
        # 注意：这个转换应由调用方决定是否需要结构化，而不是在 generate_text 中默认做
        if "result_format" in adjusted:
            fmt = adjusted.pop("result_format")
            if fmt == "json_object":
                adjusted["response_format"] = {"type": "json_object"}
            else:
                logger.warning("不支持的 result_format，已忽略", extra={"format": fmt})

        return adjusted

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None
