import asyncio
from typing import Dict, Any, Set
from src.state_of_mind.cache.base import BaseCache
from src.state_of_mind.common.llm_response import LLMResponse
from src.state_of_mind.utils.registry import GlobalSingletonRegistry
from src.state_of_mind.utils.logger import LoggerManager as logger
from src.state_of_mind.stages.perception.prompt_builder import PromptBuilder


class StepExecutor:
    CHINESE_NAME = "全息感知基底：通用LLM执行器"

    def __init__(
            self,
            backend_name: str,
            llm_model: str,
            recommended_params: Dict[str, Any],
            llm_cache: BaseCache,
            prompt_builder: PromptBuilder
    ):
        self.backend_name = backend_name
        self.llm_model = llm_model
        self.recommended_params = recommended_params
        self.llm_cache = llm_cache
        self.prompt_builder = prompt_builder
        self._backend = None
        self._init_lock = asyncio.Lock()

    async def get_backend(self):
        if self._backend is None:
            async with self._init_lock:
                if self._backend is None:
                    logger.info("🔄 首次获取LLM后端，正在异步初始化...")
                    self._backend = await GlobalSingletonRegistry.get_backend_async(self.backend_name)
                    if self._backend is None:
                        raise RuntimeError(f"无法获取 backend: {self.backend_name}")
        return self._backend

    """异步执行单个 LLM 调用，支持缓存"""
    async def execute_step(
            self,
            prompt_template: str,
            template_name: str,
            step_name: str,
            cache_key: str,
            prompt_type: str
    ) -> Dict[str, Any]:
        cache_response = await self.llm_cache.get(cache_key)
        if cache_response.get("success"):
            cached_data = cache_response.get("data")
            if cached_data is not None:
                logger.info("🔁 使用缓存结果", extra={
                    "template": template_name,
                    "step": step_name,
                    "cache_key": cache_key
                })
                return cached_data

        try:
            backend = await self.get_backend()
            result = await backend.async_call(
                prompt=prompt_template,
                model=self.llm_model,
                params=self.recommended_params,
                template_name=template_name,
                step_name=step_name,
                prompt_type=prompt_type
            )
            return result
        except Exception as e:
            # 系统级异常：网络、超时、JSON 解析崩溃等
            system_error = str(e)
            logger.error(f"[{step_name}] LLM 调用异常 - {str(e)}")
            return LLMResponse.from_system_error(
                system_error=system_error,
                model=self.llm_model,
                template_name=template_name,
                step_name=step_name,
                prompt_type=prompt_type,
                include_traceback=True
            ).to_dict()

    """异步执行生成原始文本解读"""
    async def execute_suggestion(self, prompt: str) -> str:
        logger.info("🧠 开始生成 LLM 建议内容", module_name=self.CHINESE_NAME)
        try:
            backend = await self.get_backend()
            result = await backend.generate_text(
                prompt=prompt,
                model=self.llm_model,
                params=self.recommended_params,
            )
            if result.startswith("生成失败"):
                logger.warning("⚠️ LLM 建议生成失败", extra={"error": result})
            else:
                logger.info("✅ LLM 建议生成成功", module_name=self.CHINESE_NAME)
            return result
        except Exception as e:
            error_msg = f"LLM 建议生成异常: {type(e).__name__}: {str(e)}"
            logger.exception(error_msg, module_name=self.CHINESE_NAME)
            return error_msg

    """
    异步执行批量指代消解
    输入：{原始事件索引 -> 代词}
    输出：{原始事件索引 -> 确定的合法参与者名}（不确定的不返回）
    """
    async def perform_coreference_resolution(
            self,
            user_input: str,
            index_to_pronoun: Dict[int, str],
            legitimate_participants: Set[str]
    ) -> Dict[int, str]:
        logger.info(f"→ 启动 LLM 指代消解（待解析代词: {list(index_to_pronoun.values())}）",
                    extra={"module_name": self.CHINESE_NAME})

        if not index_to_pronoun or not legitimate_participants:
            return {}

        try:
            prompt = self.prompt_builder.build_coref_prompt(
                user_input=user_input,
                legitimate_participants=legitimate_participants,
                index_to_pronoun=index_to_pronoun
            )
        except Exception as e:
            logger.warning(
                "构建指代消解 prompt 异常",
                extra={"error": str(e), "module_name": self.CHINESE_NAME}
            )
            return {}

        try:
            backend = await self.get_backend()
            resolved_from_llm: Dict[int, str] = await backend.bottom_dissolving_pronouns(
                prompt=prompt,
                model=self.llm_model,
                params=self.recommended_params
            )
        except Exception as e:
            logger.exception(
                "调用 bottom_dissolving_pronouns 异常",
                extra={"error": str(e), "module_name": self.CHINESE_NAME}
            )
            return {}

        resolved_map: Dict[int, str] = {}
        for raw_idx, name in resolved_from_llm.items():
            try:
                idx = int(raw_idx)
            except (ValueError, TypeError):
                continue
            if idx in index_to_pronoun and name in legitimate_participants:
                resolved_map[idx] = name

        logger.info(f"← LLM 消解结果: {resolved_map}", extra={"module_name": self.CHINESE_NAME})
        return resolved_map
