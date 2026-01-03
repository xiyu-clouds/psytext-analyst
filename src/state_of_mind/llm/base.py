import json
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from src.state_of_mind.common.llm_response import LLMResponse
from src.state_of_mind.stages.perception.data_validator import DataValidator
from src.state_of_mind.utils.async_decorators import async_timed
from src.state_of_mind.utils.llm_helpers import remove_check, extract_json_safely
from src.state_of_mind.utils.logger import LoggerManager as logger
import httpx
from src.state_of_mind.utils.retry_util import retry_decorator


class LLMBackend(ABC):
    """
    抽象基类：所有 LLM 后端必须继承
    """
    CHINESE_NAME = "抽象基类LLM后端"

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.api_url: Optional[str] = None
        self._initialized = False
        self.data_validator = DataValidator()

    async def init(self, configs: Dict[str, Any]) -> 'LLMBackend':
        if self._initialized:
            return self

        api_key = configs.get("api_key")
        if not api_key:
            raise ValueError(f"{self.CHINESE_NAME} 缺少 api_key 配置")

        timeout = int(configs.get("timeout", 60.0))
        self.api_url = self._build_api_url(configs)
        self.client = httpx.AsyncClient(
            headers=self._build_headers(api_key),
            timeout=timeout
        )
        self._initialized = True
        logger.info(f"✅ {self.CHINESE_NAME} 初始化完成" + (f"，API URL: {self.api_url}" if self.api_url else ""))
        return self

    @abstractmethod
    def _build_api_url(self, configs: Dict[str, Any]) -> str:
        """子类提供 API 地址构建逻辑"""
        pass

    def _build_headers(self, api_key: str) -> dict:
        """默认 header，子类可 override"""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    @abstractmethod
    def _build_json_payload(
            self,
            prompt: str,
            model: str,
            params: dict,
            *,
            system_prompt: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        子类实现：构建符合该 LLM 协议的请求体
        """
        pass

    @abstractmethod
    def _build_text_payload(
            self,
            prompt: str,
            model: str,
            params: dict,
            **kwargs
    ) -> Dict[str, Any]:
        """
        子类实现：构建符合该 LLM 协议的请求体
        """
        pass

    @abstractmethod
    def _extract_content_from_response(self, data: Dict) -> Optional[str]:
        pass

    # ========================
    # 统一调用入口（模板方法）
    # ========================
    @async_timed
    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def async_call(
            self,
            prompt: str,
            model: str,
            params: dict,
            template_name: str,
            step_name: str,
            prompt_type: str
    ) -> Dict[str, Any]:
        """
        统一入口：调用 LLM 并返回标准化 LLMResponse 结构。
        子类无需重写此方法，只需实现抽象方法。
        """
        start_time = time.time()
        latency_ms: Optional[float] = None
        system_prompt = "你必须输出一个严格的 JSON 对象。不要有任何额外文字解释、前缀、后缀或 Markdown 代码块。确保输出是有效的 json 格式。"

        try:
            payload = self._build_json_payload(
                prompt=prompt,
                model=model,
                params=params,
                system_prompt=system_prompt
            )

            logger.info(f"[{self.CHINESE_NAME} - 原始数据处理] 调用 LLM 异步 API", extra={
                "model": model,
                "template_name": template_name,
                "step_name": step_name,
                "prompt_length": len(prompt)
            })

            assert self.client is not None and self.api_url is not None, "未初始化 client 或 api_url"
            response = await self.client.post(self.api_url, json=payload)
            latency_ms = (time.time() - start_time) * 1000

            # 记录原始响应（用于调试）
            try:
                resp_debug = response.json()
            except Exception:
                resp_debug = response.text[:200]
            logger.info(f"[{self.CHINESE_NAME} - 原始数据处理] 原始响应: {resp_debug}")

            # --- 处理非 200 响应 ---
            if response.status_code != 200:
                error_msg = self._parse_api_error(response)

                if response.status_code >= 500 or response.status_code == 429:
                    # 触发重试（由 retry_decorator 捕获）
                    fake_request = httpx.Request(method="POST", url=self.api_url)
                    raise httpx.HTTPStatusError(
                        message=error_msg,
                        request=fake_request,
                        response=response
                    )
                else:
                    # 客户端错误（400/401/403/404）：不重试，返回 API 错误
                    logger.warning(
                        f"[{self.CHINESE_NAME} - 原始数据处理] API 返回客户端错误",
                        extra={
                            "status_code": response.status_code,
                            "error": error_msg,
                            "template_name": template_name,
                            "step_name": step_name
                        }
                    )
                    return LLMResponse.from_api_error(
                        status_code=response.status_code,
                        error_message=error_msg,
                        model=model,
                        template_name=template_name,
                        step_name=step_name,
                        prompt_type=prompt_type,
                        latency_ms=latency_ms
                    ).to_dict()

            # --- 处理 200 响应 ---
            content = self._extract_content_from_response(response.json())
            if not content or not content.strip():
                api_error = "模型返回内容为空"
                logger.warning("模型返回空内容", extra={
                    "template_name": template_name,
                    "step_name": step_name
                })
                validation_result = {"is_valid": False, "cleaned_data": {}, "errors": [api_error]}
                raw_content = None
            else:
                content = remove_check(content.strip())
                raw_content = content
                parsed_json = extract_json_safely(content)
                validation_result = self.data_validator.validate(
                    data=parsed_json,
                    template_name=template_name,
                    step_name=step_name
                )
            # 构造成功调用的响应（无论结构是否有效）
            return LLMResponse.from_successful_call(
                valid_structure=validation_result["is_valid"],
                data=validation_result["cleaned_data"] or {},
                raw_response=raw_content,
                validation_errors=validation_result["errors"],
                api_error="模型返回内容为空" if not raw_content else None,
                model=model,
                template_name=template_name,
                step_name=step_name,
                prompt_type=prompt_type,
                latency_ms=latency_ms
            ).to_dict()
        except Exception as e:
            # 系统级异常：网络、超时、JSON 解析崩溃等
            system_error = str(e)
            logger.exception(
                f"[{self.CHINESE_NAME} - 原始数据处理] async_call 系统级异常",
                extra={
                    "model": model,
                    "template_name": template_name,
                    "step_name": step_name,
                    "error": system_error
                }
            )
            return LLMResponse.from_system_error(
                system_error=system_error,
                model=model,
                template_name=template_name,
                step_name=step_name,
                prompt_type=prompt_type,
                include_traceback=True
            ).to_dict()

    @async_timed
    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def generate_text(
            self,
            prompt: str,
            model: str,
            params: dict,
            step_name: str,
            prompt_type: str
    ) -> Dict[str, Any]:
        return await self._call_text_mode(
            prompt=prompt,
            model=model,
            params=params,
            step_name=step_name,
            prompt_type=prompt_type,
            payload_fn=self._build_text_payload
        )

    @async_timed
    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def guided_global_semantic_signature(
            self,
            prompt: str,
            model: str,
            params: dict,
            step_name: str,
            prompt_type: str
    ) -> Dict[str, Any]:
        return await self._call_text_mode(
            prompt=prompt,
            model=model,
            params=params,
            step_name=step_name,
            prompt_type=prompt_type,
            payload_fn=self._build_text_payload
        )

    @async_timed
    @retry_decorator(max_retries=3, enable_exp_backoff=True)
    async def bottom_dissolving_pronouns(
            self,
            prompt: str,
            model: str,
            params: dict,
            step_name: str,
            prompt_type: str
    ) -> Dict[str, Any]:
        return await self._call_json_coref_mode(
            prompt=prompt,
            model=model,
            params=params,
            step_name=step_name,
            prompt_type=prompt_type,
            payload_fn=self._build_json_payload
        )

    @staticmethod
    def _parse_api_error(response) -> str:
        """可被子类 override 以定制错误解析"""
        try:
            resp_json = response.json()
            msg = resp_json.get("error", {}).get("message") or resp_json.get("message") or ""
            code = resp_json.get("error", {}).get("type") or resp_json.get("code") or "Unknown"
            return f"[{code}] {msg}"
        except Exception:
            return f"HTTP {response.status_code} (无法解析响应)"

    # ========================
    # 通用调用方法（仅用于 text 类型）
    # ========================
    async def _call_text_mode(
            self,
            prompt: str,
            model: str,
            params: dict,
            step_name: str,
            prompt_type: str,
            payload_fn
    ) -> Dict[str, Any]:
        start_time = time.time()
        try:
            payload = payload_fn(prompt=prompt, model=model, params=params)
            response = await self.client.post(self.api_url, json=payload)
            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                f"[{self.CHINESE_NAME}] 调用 LLM (text mode)",
                extra={
                    "step_name": step_name,
                    "model": model,
                    "latency_ms": round(latency_ms, 1),
                    "prompt_length": len(prompt)
                }
            )

            if response.status_code != 200:
                raw_resp = response.text
                result = ""
                api_error = f"HTTP {response.status_code}"
                success = False
                logger.warning(
                    f"[{self.CHINESE_NAME}] HTTP 错误",
                    extra={"step_name": step_name, "status_code": response.status_code}
                )
            else:
                result = self._extract_content_from_response(response.json()) or ""
                raw_resp = result
                api_error = None
                success = bool(result.strip())
                logger.info(
                    f"[{self.CHINESE_NAME}] 成功返回文本",
                    extra={"step_name": step_name, "result_length": len(result)}
                )

            return {
                "data": result.strip(),
                "step_name": step_name,
                "prompt_type": prompt_type,
                "__raw_response": raw_resp,
                "__success": success,
                "__valid_structure": True,
                "__system_error": None,
                "__api_error": api_error,
                "__validation_errors": []
            }

        except Exception as e:
            logger.exception(
                f"[{self.CHINESE_NAME}] 文本调用异常",
                extra={"step_name": step_name, "error": str(e)}
            )
            return {
                "data": "",
                "step_name": step_name,
                "prompt_type": prompt_type,
                "__raw_response": "",
                "__success": False,
                "__valid_structure": True,
                "__system_error": str(e),
                "__api_error": None,
                "__validation_errors": []
            }

    # ========================
    # 通用调用方法（仅用于 JSON coref 模式）
    # ========================
    async def _call_json_coref_mode(
            self,
            prompt: str,
            model: str,
            params: dict,
            step_name: str,
            prompt_type: str,
            payload_fn
    ) -> Dict[str, Any]:
        start_time = time.time()
        system_prompt = "你必须输出一个严格的 JSON 对象。不要有任何额外文字解释、前缀、后缀或 Markdown 代码块。确保输出是有效的 json 格式。"
        try:
            payload = payload_fn(
                prompt=prompt,
                model=model,
                params=params,
                system_prompt=system_prompt
            )
            response = await self.client.post(self.api_url, json=payload)
            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                f"[{self.CHINESE_NAME}] 调用 LLM (JSON coref mode)",
                extra={
                    "step_name": step_name,
                    "model": model,
                    "latency_ms": round(latency_ms, 1),
                    "prompt_length": len(prompt)
                }
            )

            if response.status_code != 200:
                raw_resp = response.text
                result = {}
                api_error = f"HTTP {response.status_code}"
                success = False
                logger.warning(
                    f"[{self.CHINESE_NAME}] HTTP 错误 (JSON mode)",
                    extra={"step_name": step_name, "status_code": response.status_code}
                )
            else:
                content = self._extract_content_from_response(response.json())
                if not content:
                    raw_resp = ""
                    result = {}
                    api_error = "空响应"
                    success = False
                    logger.warning(f"[{self.CHINESE_NAME}] 模型返回空内容", extra={"step_name": step_name})
                else:
                    stripped = content.strip()
                    raw_resp = stripped
                    start = stripped.find("{")
                    end = stripped.rfind("}") + 1
                    if start == -1 or end <= start:
                        result = {}
                        api_error = "无有效 JSON"
                        success = False
                        logger.warning(f"[{self.CHINESE_NAME}] 未找到 JSON 块", extra={"step_name": step_name})
                    else:
                        try:
                            parsed = json.loads(stripped[start:end])
                            result = {}
                            for k, v in parsed.items():
                                if isinstance(k, str) and isinstance(v, str):
                                    try:
                                        idx = int(k)
                                        result[idx] = v
                                    except ValueError:
                                        pass
                            api_error = None
                            success = True
                            logger.info(f"[{self.CHINESE_NAME}] 成功解析 JSON coref", extra={"step_name": step_name})
                        except json.JSONDecodeError as je:
                            result = {}
                            api_error = f"JSON 解析失败: {str(je)}"
                            success = False
                            logger.warning(
                                f"[{self.CHINESE_NAME}] JSON 解析失败",
                                extra={"step_name": step_name, "error": str(je)}
                            )

            return {
                "data": result,
                "step_name": step_name,
                "prompt_type": prompt_type,
                "__raw_response": raw_resp,
                "__success": success,
                "__valid_structure": True,
                "__system_error": None,
                "__api_error": api_error,
                "__validation_errors": []
            }

        except Exception as e:
            logger.exception(
                f"[{self.CHINESE_NAME}] JSON coref 调用异常",
                extra={"step_name": step_name, "error": str(e)}
            )
            return {
                "data": {},
                "step_name": step_name,
                "prompt_type": prompt_type,
                "__raw_response": "",
                "__success": False,
                "__valid_structure": True,
                "__system_error": str(e),
                "__api_error": None,
                "__validation_errors": []
            }

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None
