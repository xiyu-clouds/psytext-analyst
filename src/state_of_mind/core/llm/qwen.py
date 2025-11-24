import json
import re
import httpx
from typing import Dict, Any, Optional

from aiohttp import ClientResponseError, RequestInfo
from yarl import URL

from .base import LLMBackend
from ..validator import validate_with_diagnosis
from ...config import config
from ...utils.retry_util import retry_decorator
from src.state_of_mind.utils.logger import LoggerManager as logger


class AsyncQwenLLMBackend(LLMBackend):
    CHINESE_NAME = "通义千问异步LLM后端"

    def __init__(self, client: httpx.AsyncClient = None):
        super().__init__()
        self.client = client
        self.api_url = None

    async def init(self, configs: dict = None):
        await super().init(configs)
        if self.client is None:
            api_key = self.configs.get("api_key") or configs.get("api_key") or config.LLM_API_KEY
            if not api_key:
                raise EnvironmentError("LLM_API_KEY 未设置")

            timeout = self.configs.get("timeout", 60.0)
            self.api_url = config.LLM_API_URL
            if not self.api_url:
                # 默认 fallback
                self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

            self.client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Synchronous": "true"
                },
                timeout=timeout
            )
        return self

    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def async_call(self, prompt: str, model: str, template_name: str,
                         step_name: str, params: dict, prompt_type: str) -> Dict[str, Any]:
        """
        调用大模型并进行结构化校验。
        返回结构始终一致，便于上层统一处理。

        字段说明：
        - __success: True 表示 HTTP 请求成功（200），False 表示网络/认证等系统级失败
        - __valid_structure: True 表示响应内容通过 JSON 解析 + Schema 校验
        - __raw_response: LLM 返回的原始字符串（清理后），可能为 None
        - __validation_errors: 结构校验错误列表（若 __valid_structure=False）
        - __api_error: DashScope API 返回的业务错误（如 quota 耗尽）
        - __system_error: 系统级异常（如超时、连接失败）
        - data: 最终提取的数据（若结构有效则为 clean dict，否则可能为空或含错误占位）
        """
        api_error = None

        try:
            messages = [
                {"role": "system", "content": "只输出严格 JSON 格式，不要添加任何解释、前缀或 Markdown 代码块。"},
                {"role": "user", "content": prompt}
            ]

            payload = {
                "model": model,
                "input": {"messages": messages},
                "parameters": params
            }

            logger.info("[Qwen - 原始数据处理] 调用 LLM 异步 API", extra={
                "model": model,
                "template_name": template_name,
                "step_name": step_name,
                "prompt_length": len(prompt)
            })

            response = await self.client.post(self.api_url, json=payload)
            logger.info(f"[Qwen - 原始数据处理]  原始响应：{response.json()}")

            # === HTTP 非 200：API 业务错误 ===
            if response.status_code != 200:
                try:
                    resp_json = response.json()
                    error_msg = resp_json.get("message", f"HTTP {response.status_code}")
                    code = resp_json.get("code", "Unknown")
                except Exception:
                    error_msg = f"HTTP {response.status_code} (无法解析响应体)"
                    code = "ParseError"

                if response.status_code >= 500 or response.status_code == 429:
                    fake_request_info = RequestInfo(
                        url=URL(self.api_url),
                        method="POST",
                        headers={},  # 可为空
                        real_url=URL(self.api_url)
                    )
                    raise ClientResponseError(
                        request_info=fake_request_info,  # 允许为 None
                        history=(),  # 允许为空 tuple
                        status=response.status_code,
                        message=error_msg,
                        headers=response.headers
                    )

                api_error = f"[{code}] {error_msg}"
                logger.warning("[Qwen - 原始数据处理] API 返回错误", extra={
                    "status_code": response.status_code,
                    "error": api_error,
                    "template_name": template_name,
                    "step_name": step_name
                })
                # 不中断，继续走结构校验流程（但 data 为空）
                parsed_json = {"__api_error": api_error, "__status_code": response.status_code}
                raw_content = None
            else:
                # === HTTP 200：尝试提取内容 ===
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
            # === 系统级异常：网络、超时、配置错误等 ===
            system_error = str(e)
            logger.exception("[Qwen - 原始数据处理] async_call 系统级异常", extra={
                "model": model,
                "template_name": template_name,
                "step_name": step_name,
                "error": system_error
            })
            # 构造失败结构，不继续后续逻辑
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

        # === 无论哪种情况，都进行结构诊断 ===
        validation_result = validate_with_diagnosis(
            data=parsed_json,
            template_name=template_name,
            step_name=step_name
        )

        # === 统一返回结构 ===
        result = {
            "__success": True,  # 只要没进 except，就算 API 调用成功（即使返回空或错误）
            "__valid_structure": validation_result["is_valid"],
            "data": validation_result["cleaned_data"] or {},
            "__raw_response": raw_content,
            "__validation_errors": validation_result["errors"],
            "__api_error": api_error,
            "__system_error": None,  # 系统错误已在 except 中返回，此处为 None
            "model": model,
            "template_name": template_name,
            "step_name": step_name,
            "prompt_type": prompt_type
        }

        return result

    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def generate_text(self, prompt: str, model: str, params: dict) -> str:
        # 安全日志用的关键参数（Qwen parameters 字段）
        log_params = {
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p"),
            "seed": params.get("seed"),
        }

        try:
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": model,
                "input": {"messages": messages},
                "parameters": params
            }

            logger.info(
                "📝 [Qwen - 生成建议] 调用大模型 API",
                extra={
                    "module_name": self.CHINESE_NAME,
                    "model": model,
                    "params": log_params,
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:100].replace("\n", "\\n")
                }
            )

            response = await self.client.post(self.api_url, json=payload)

            if response.status_code != 200:
                error_detail = response.text[:300] if response.text else ""
                logger.error(
                    "❌ [Qwen - 生成建议] API 调用失败",
                    extra={
                        "module_name": self.CHINESE_NAME,
                        "status_code": response.status_code,
                        "error_snippet": error_detail.replace("\n", "\\n"),
                        "model": model
                    }
                )
                return f"生成失败: HTTP {response.status_code} - {error_detail[:100]}"

            result = response.json()
            content = self._extract_content_from_response(result)

            if not content:
                logger.warning(
                    "⚠️ [Qwen - 生成建议] 返回空内容",
                    extra={"module_name": self.CHINESE_NAME, "model": model}
                )
                return "生成失败"

            stripped = content.strip()
            logger.info(
                "✅ [Qwen - 生成建议] 调用成功",
                extra={
                    "module_name": self.CHINESE_NAME,
                    "model": model,
                    "output_length": len(stripped),
                    "output_preview": stripped[:100].replace("\n", "\\n")
                }
            )
            return stripped

        except Exception as e:
            logger.exception(
                "💥 [Qwen - 生成建议] 调用异常",
                extra={
                    "module_name": self.CHINESE_NAME,
                    "model": model,
                    "error": str(e)
                }
            )
            return f"生成失败: {str(e)}"

    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def bottom_dissolving_pronouns(self, prompt: str, model: str, params: dict) -> Dict[int, str]:
        log_params = {
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "top_p": params.get("top_p")
        }

        try:
            messages = [
                {"role": "system", "content": "只输出严格 JSON 格式，不要添加任何解释、前缀或 Markdown 代码块。"},
                {"role": "user", "content": prompt}
            ]
            payload = {
                "model": model,
                "input": {"messages": messages},
                "parameters": params
            }

            logger.info(
                "🧠 [Qwen - 指代消解] 调用大模型 API",
                extra={
                    "module_name": self.CHINESE_NAME,
                    "model": model,
                    "params": log_params,
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:100].replace("\n", "\\n")
                }
            )

            response = await self.client.post(self.api_url, json=payload)

            if response.status_code != 200:
                error_detail = response.text[:300] if response.text else ""
                logger.error(
                    "❌ [Qwen - 指代消解] API 调用失败",
                    extra={
                        "module_name": self.CHINESE_NAME,
                        "status_code": response.status_code,
                        "error_snippet": error_detail.replace("\n", "\\n"),
                        "model": model
                    }
                )
                return {}  # ❌ 失败返回空 dict，不是字符串！

            result = response.json()
            content = self._extract_content_from_response(result)

            if not content:
                logger.warning("⚠️ [Qwen - 指代消解] 返回空内容", extra={"module_name": self.CHINESE_NAME})
                return {}

            stripped = content.strip()

            # === 关键：在这里解析 JSON ===
            start = stripped.find("{")
            end = stripped.rfind("}") + 1
            if start == -1 or end <= start:
                logger.warning(
                    "⚠️ [Qwen - 指代消解] 无有效 JSON",
                    extra={"output": stripped[:200], "module_name": self.CHINESE_NAME}
                )
                return {}

            try:
                parsed = json.loads(stripped[start:end])
            except json.JSONDecodeError as e:
                logger.warning(
                    "⚠️ [Qwen - 指代消解] JSON 解析失败",
                    extra={"error": str(e), "output": stripped[:200], "module_name": self.CHINESE_NAME}
                )
                return {}

            # 转换 key 为 int（如果可能）
            result_dict = {}
            for k, v in parsed.items():
                if isinstance(k, str) and isinstance(v, str):
                    try:
                        idx = int(k)
                        result_dict[idx] = v
                    except ValueError:
                        continue  # 忽略非法 key
            return result_dict

        except Exception as e:
            logger.exception(
                "💥 [Qwen - 指代消解] 调用异常",
                extra={"module_name": self.CHINESE_NAME, "model": model, "error": str(e)}
            )
            return {}  # ❌ 任何异常都返回空 dict

    @staticmethod
    def _extract_content_from_response(data: Dict) -> Optional[str]:
        try:
            if 'output' in data and 'choices' in data['output']:
                return data['output']['choices'][0]['message']['content']
            elif 'output' in data and 'text' in data['output']:
                return data['output']['text']
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

    async def close(self):
        """必须显式调用以关闭连接池"""
        if self.client:
            await self.client.aclose()
            self.client = None
