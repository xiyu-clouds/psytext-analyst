import asyncio
import time
import uuid
from typing import List, Any, Tuple, Dict, Optional
from src.state_of_mind.cache.base import BaseCache
from src.state_of_mind.cache.redis import RedisLLMCache
from src.state_of_mind.stages.perception.prompt_builder import PromptBuilder
from src.state_of_mind.cache.llm_cache import LLMCache
from src.state_of_mind.config import config
from src.state_of_mind.utils.async_decorators import async_timed
from .constants import REQUIRED_FIELDS_BY_CATEGORY, LLM_PARTICIPANTS_EXTRACTION, \
    CATEGORY_RAW, PARALLEL_PREPROCESSING, PARALLEL_PERCEPTION, PARALLEL_HIGH_ORDER, \
    SERIAL_SUGGESTION, OTHER, ALLOWED_PARALLEL_PERCEPTION_MARKERS, ALLOWED_SERIAL_SUGGESTION_MARKERS, \
    ALLOWED_PARALLEL_HIGH_ORDER_MARKERS, PARALLEL_PERCEPTION_KEYS
from src.state_of_mind.utils.file_util import FileUtil
from src.state_of_mind.utils.logger import LoggerManager as logger
from .context_builder import ContextBuilder
from .executor import StepExecutor
from .participant_filter import ParticipantFilter
from .report_generator import ReportGenerator
from .result_assembler import ResultAssembler
from ...common.llm_response import LLMResponse
from ...common.raw_data_factory import create_raw_basic_data
from ...utils.concurrency_manager import ConcurrencyManager
from src.state_of_mind.core.types import StageProtocol


class PerceptionPipeline(StageProtocol):
    CHINESE_NAME = "第一阶段：全息感知基底"
    REPORT_URL_PREFIX = "/reports/"
    RAW_DATA_DIR = config.DATA_YUAN_RAW_DIR
    DYE_VAT_DIR = config.DATA_YUAN_DYE_VAT_DIR

    def __init__(
            self,
            backend_name: Optional[str] = None,
            llm_model: Optional[str] = None,
            recommended_params: Optional[dict] = None
    ):
        self.backend_name = backend_name or config.LLM_BACKEND
        self.llm_model = llm_model or config.LLM_MODEL
        self.recommended_params = recommended_params or config.LLM_RECOMMENDED_PARAMS or {}
        current_parallel_concurrency = config.get("CURRENT_PARALLEL_CONCURRENCY", 3)
        self.concurrency_manager = ConcurrencyManager(current_parallel_concurrency)
        self.prompt_builder = PromptBuilder()
        self.prompt_result = None
        self.llm_cache = self._create_cache_backend(config)
        self.file_util = FileUtil()
        self.report_generator = ReportGenerator(self.file_util)
        self.step_executor = StepExecutor(self.backend_name, self.llm_model, self.recommended_params, self.llm_cache,
                                          self.prompt_builder)
        self.result_assembler = ResultAssembler(self.llm_model, self.prompt_builder, self.step_executor)
        self._top_field_to_step_types = self._build_top_field_to_step_types()
        self._step_type_to_config = self._build_step_type_to_config()
        self._participant_filter = None
        self._context_builder = None
        self._participant_filter_lock = asyncio.Lock()
        self._context_builder_lock = asyncio.Lock()
        logger.info(f"PerceptionPipeline 初始化成功，使用 backend: {self.backend_name}, model: {self.llm_model}")

    @staticmethod
    def _create_cache_backend(c) -> BaseCache:
        storage = c.STORAGE_BACKEND
        if storage == c.STORAGE_LOCAL:
            return LLMCache(
                max_size=c.LLM_CACHE_MAX_SIZE,
                ttl_seconds=c.LLM_CACHE_TTL
            )
        elif storage == c.STORAGE_REDIS:
            return RedisLLMCache(config=c, default_ttl=c.LLM_CACHE_TTL)
        else:
            raise ValueError(f"Unsupported storage backend: {storage}")

    async def _get_participant_filter(self):
        if self._participant_filter is None:
            async with self._participant_filter_lock:
                if self._participant_filter is None:
                    backend = await self.step_executor.get_backend()
                    self._participant_filter = ParticipantFilter(self.prompt_builder, backend)
        return self._participant_filter

    async def _get_context_builder(self):
        if self._context_builder is None:
            async with self._context_builder_lock:
                if self._context_builder is None:
                    participant_filter = await self._get_participant_filter()
                    self._context_builder = ContextBuilder(
                        self.prompt_builder,
                        participant_filter,
                        self._step_type_to_config,
                        self._top_field_to_step_types
                    )
        return self._context_builder

    async def run(self, user_input: str, category: str = CATEGORY_RAW, **kwargs) -> Dict[str, Any]:
        return await self.async_extract(
            user_input=user_input,
            template_name=category,
            suggestion_type=config.SUGGESTION_TYPE,
            title=config.REPORT_TITLE,
            **kwargs
        )

    async def run_batch(self, user_inputs: List[str], category: str = CATEGORY_RAW, **kwargs) -> List[Dict[str, Any]]:
        if not user_inputs:
            return []
        # 并发执行，保持顺序，任一失败则抛出
        results = await asyncio.gather(
            *(self.run(inp, category, **kwargs) for inp in user_inputs)
        )
        return list(results)

    @async_timed
    async def async_extract(self, template_name: str, user_input: str, suggestion_type: str,
                            title: str = "全息感知基底", **template_vars) -> Dict[str, Any]:
        """异步核心流程"""
        trace_id = str(uuid.uuid4())
        logger.set_trace_id(trace_id)
        context = template_vars.copy()
        context["user_input"] = user_input
        context["llm_model"] = self.llm_model

        if not template_name or not isinstance(template_name, str):
            raise ValueError("template_name 必须是非空字符串")
        if user_input is not None and not isinstance(user_input, str):
            raise TypeError("user_input 必须是非空字符串")

        cache_key = self.llm_cache.make_key(template_name, **context)
        logger.info(f"整体缓存 key: {cache_key[:8]}...")
        cache_response = await self.llm_cache.get(cache_key)
        if cache_response.get("success"):
            cached_data = cache_response.get("data")
            if cached_data is not None:
                report_url = cached_data.get("meta", {}).get("report_url", "")
                res = {"report_url": report_url}
                logger.info("🔁 使用缓存结果", extra={"template": template_name, "report_url": report_url})
                return res

        self.prompt_result = self.prompt_builder.build_raw()
        preprocessing_prompts = self.prompt_result["preprocessing_prompts"]
        perception_prompts = self.prompt_result["perception_prompts"]
        high_order_prompts = self.prompt_result["high_order_prompts"]
        suggestion_prompts = self.prompt_result["suggestion_prompts"]
        basic_data = create_raw_basic_data(user_input, self.llm_model)

        all_step_results = []
        prompt_records = {PARALLEL_PREPROCESSING: [], PARALLEL_PERCEPTION: [], PARALLEL_HIGH_ORDER: [], SERIAL_SUGGESTION: [], OTHER: []}
        raw_response_records = {PARALLEL_PREPROCESSING: [], PARALLEL_PERCEPTION: [], PARALLEL_HIGH_ORDER: [], SERIAL_SUGGESTION: [], OTHER: []}
        context_desc_info = []

        await self._run_preprocessing_parallel_async(
            preprocessing_prompts, context, template_name, cache_key, all_step_results, prompt_records,
            context_desc_info
        )

        # === 动态过滤：仅使用 context ===
        filtered_parallel_prompts = [
            (step_name, driven_by, prompt)
            for (step_name, driven_by, prompt) in perception_prompts
            if context.get("pre_screening", {}).get(driven_by, False)
        ]

        await self._run_perception_parallel_async(
            filtered_parallel_prompts, context, template_name, cache_key, all_step_results, prompt_records,
            context_desc_info
        )

        # 判断是否启用高阶推理
        eligible = context.get("eligibility", {}).get("eligible", False)
        if eligible:
            has_valid_perception = any(
                key in context and bool(context[key])
                for key in PARALLEL_PERCEPTION_KEYS
            )
            if has_valid_perception:
                await self._run_high_order_parallel_async(
                    high_order_prompts, context, template_name, cache_key,
                    all_step_results, prompt_records, context_desc_info
                )
                await self._run_suggestion_serial_async(
                    suggestion_prompts, context, template_name, cache_key,
                    all_step_results, prompt_records, context_desc_info
                )
            else:
                logger.info("⏭️ eligible=true 但无有效并行感知数据，跳过高阶策略、矛盾、操控、建议四步链")
        else:
            logger.info("⏭️ eligible=false，跳过高阶策略、矛盾、操控、建议四步链")

        result = self.result_assembler.assemble_final_data(context, basic_data)
        valid_result = self.result_assembler.validate_final_result(result)
        is_success = bool(valid_result.get("__success"))
        if is_success:
            # 注入原始文本解读内容
            await self.result_assembler.inject_suggestion_into_result(result, user_input, suggestion_type, all_step_results, prompt_records, title)
            # 注入全局语义标识
            await self.result_assembler.inject_global_semantic_signature(result, user_input, all_step_results, prompt_records)

        aggregation = self.result_assembler.aggregate_step_results(all_step_results, raw_response_records)
        aggregation["__errors_summary"]["final_validation_errors"] = [
            {"step": "final_validation", "errors": valid_result["__final_validation_errors"]}
        ] if valid_result["__final_validation_errors"] else []
        result["meta"]["validity_level"] = valid_result["__validity_level"]

        # 注意：即使失败，也要持久化 dye_vat 诊断数据
        report_url = await self._persist_extraction_artifacts(
            result=result,
            aggregation=aggregation,
            template_name=template_name,
            user_input=user_input,
            prompt_records=prompt_records,
            raw_response_records=raw_response_records,
            is_success=is_success
        )

        if is_success:
            await self.llm_cache.set(cache_key, result)
            logger.info("✅ 最终结果已缓存", extra={"cache_key": cache_key})
        else:
            logger.warning("🟡 提取流程未完全成功，跳过缓存", extra={
                "cache_key": cache_key,
                "validity_level": valid_result.get("__validity_level"),
                "final_errors": valid_result.get("__final_validation_errors")
            })
        return {"report_url": report_url}

    @async_timed
    async def _run_preprocessing_parallel_async(
            self,
            prompts: List[Tuple[str, str, str]],
            context: Dict[str, Any],
            template_name: str,
            cache_key_base: str,
            all_step_results: List[Dict],
            prompt_records: Dict,
            context_desc_info: List
    ):
        if not prompts:
            logger.info("⏭️ 无预处理任务")
            return

        logger.info("⚡ 并发执行预处理任务", extra={"count": len(prompts)})
        context_builder = await self._get_context_builder()

        async def _task(idx: int, step_name: str, driven_by: str, prompt_template: str) -> Dict[str, Any]:
            try:
                async with self.concurrency_manager.semaphore:
                    cache_key = f"{cache_key_base}:{step_name}:{idx}"
                    logger.info(f"⚡ [{step_name}] 缓存 key: ...{cache_key[-10:]}")
                    rendered_prompt = context_builder.build_user_input_context(
                        prompt_template, context["user_input"], context_desc_info
                    )

                    prompt_records[PARALLEL_PREPROCESSING].append({
                        "step_name": step_name,
                        "prompt": rendered_prompt
                    })

                    result = await self.step_executor.execute_step(
                        prompt_template=rendered_prompt,
                        template_name=template_name,
                        step_name=step_name,
                        cache_key=cache_key,
                        prompt_type=PARALLEL_PREPROCESSING
                    )

                    if result.get("__success") is True:
                        try:
                            await self.llm_cache.set(cache_key, result)
                        except Exception as cache_err:
                            logger.warning(
                                f"⚠️ 预处理任务缓存写入失败 [{step_name}]: {type(cache_err).__name__}: {cache_err}",
                                extra={"step": step_name}
                            )

                    logger.info(f"✅ 预处理任务 [{step_name}] 执行完成")
                    return result
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[{step_name}] 预处理任务异常: {error_msg}")
                failure_resp = LLMResponse.from_system_error(
                    system_error=error_msg,
                    model=self.llm_model,
                    template_name=template_name,
                    step_name=step_name,
                    prompt_type=PARALLEL_PREPROCESSING,
                    include_traceback=True
                )
                return failure_resp.to_dict()

        tasks = [
            _task(idx, step_name, driven_by, prompt)
            for idx, (step_name, driven_by, prompt) in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for idx, result in enumerate(results):
            try:
                step_name = result.get("step_name", f"unknown_preprocessing_{idx}")
                all_step_results.append(result)
                context_builder.update_context_from_result(result, context, step_name)
                if step_name == LLM_PARTICIPANTS_EXTRACTION:
                    context_builder.build_common_context(
                        step_name=step_name,
                        context=context,
                        context_desc_info=context_desc_info
                    )

            except Exception as e:
                system_error = str(e)
                logger.error(
                    f"⚠️ 预处理后处理失败 [idx={idx}, step={result.get('step_name', 'unknown')}]: {system_error}"
                )
                fallback_result = LLMResponse.from_system_error(
                    system_error=system_error,
                    model=self.llm_model,
                    template_name=template_name,
                    step_name=result.get("step_name", f"unknown_{idx}"),
                    prompt_type=PARALLEL_PREPROCESSING,
                    include_traceback=True
                )
                all_step_results.append(fallback_result.to_dict())

        success_count = sum(1 for r in results if r.get("__success", False))
        logger.info(
            f"并行预处理任务完成: {len(results)} 个任务, 成功 {success_count} 个",
            extra={"total": len(results), "success": success_count}
        )

    @async_timed
    async def _run_perception_parallel_async(
            self,
            prompts: List[Tuple[str, str, str]],
            context: Dict[str, Any],
            template_name: str,
            cache_key_base: str,
            all_step_results: List[Dict],
            prompt_records: Dict,
            context_desc_info: List,
    ):
        """并发执行感知任务"""
        if not prompts:
            logger.info("⏭️ 无并行感知任务")
            return

        logger.info("⚡ 执行并行感知任务", extra={"count": len(prompts)})
        context_builder = await self._get_context_builder()
        participant_filter = await self._get_participant_filter()
        legitimate_participants = participant_filter.build_legitimate_participants_set(context)

        async def _task(idx: int, step_name: str, prompt_template: str) -> Dict[str, Any]:
            try:
                async with self.concurrency_manager.semaphore:
                    cache_key = f"{cache_key_base}:{step_name}:{idx}"
                    logger.info(f"⚡ [{step_name}] 缓存 key: ...{cache_key[-10:]}")

                    allowed_markers = ALLOWED_PARALLEL_PERCEPTION_MARKERS.get(idx, set())
                    rendered_prompt = context_builder.inject_allowed_context(
                        prompt_template, context_desc_info, allowed_markers
                    )

                    prompt_records.setdefault(PARALLEL_PERCEPTION, []).append({
                        "step_name": step_name,
                        "prompt": rendered_prompt
                    })

                    data = await self.step_executor.execute_step(
                        prompt_template=rendered_prompt,
                        template_name=template_name,
                        step_name=step_name,
                        cache_key=cache_key,
                        prompt_type=PARALLEL_PERCEPTION
                    )

                    if data.get("__success") is True:
                        try:
                            await self.llm_cache.set(cache_key, data)
                        except Exception as cache_err:
                            logger.warning(
                                f"⚠️ 并行感知任务缓存写入失败 [{step_name}]: {type(cache_err).__name__}: {cache_err}",
                                extra={"step": step_name}
                            )
                    logger.debug(f"✅ 并行感知任务 [{step_name}] 执行完成")
                    return data

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[{step_name}] 并行感知任务兜底异常: {error_msg}")
                failure_resp = LLMResponse.from_system_error(
                    system_error=error_msg,
                    model=self.llm_model,
                    template_name=template_name,
                    step_name=step_name,
                    prompt_type=PARALLEL_PERCEPTION,
                    include_traceback=True
                )
                return failure_resp.to_dict()

        tasks = [
            _task(idx, step_name, prompt)
            for idx, (step_name, driven_by, prompt) in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)
        for idx, result in enumerate(results):
            try:
                await participant_filter.filter_perception_results(
                    context["user_input"], result, legitimate_participants, prompt_records, all_step_results
                )
                all_step_results.append(result)
                context_builder.update_context_from_result(
                    result, context, result.get("step_name")
                )
            except Exception as e:
                system_error = str(e)
                logger.error(
                    f"⚠️ 并行感知任务后处理失败 [idx={idx}, step={result.get('step_name', 'unknown')}]: {system_error}"
                )
                fallback_result = LLMResponse.from_system_error(
                    system_error=system_error,
                    model=self.llm_model,
                    template_name=template_name,
                    step_name=result.get("step_name", f"unknown_{idx}"),
                    prompt_type=PARALLEL_PERCEPTION,
                    include_traceback=True
                )
                all_step_results.append(fallback_result.to_dict())

        dynamic_desc = context_builder.build_perception_context_batch(context)
        if dynamic_desc:
            context_desc_info.append(dynamic_desc)

        legit_participants_ctx = context_builder.build_legitimate_participants_context(context)
        if legit_participants_ctx:
            context_desc_info.append(legit_participants_ctx)

        success_count = sum(1 for r in results if r.get("__success", False))
        logger.info(
            f"并行感知任务完成: {len(results)} 个任务, 成功 {success_count} 个",
            extra={"total": len(results), "success": success_count}
        )

    @async_timed
    async def _run_high_order_parallel_async(
        self,
        prompts: List[Tuple[str, str, str]],
        context: Dict[str, Any],
        template_name: str,
        cache_key_base: str,
        all_step_results: List[Dict],
        prompt_records: Dict,
        context_desc_info: List[str],
    ):
        """
        并发执行高阶推理三步链（策略锚定 / 矛盾暴露 / 操控机制解码）
        前提：context 已包含完整的并行感知结果，且 eligible=True
        """
        if not prompts:
            logger.info("⏭️ 无并行高阶任务")
            return

        if len(prompts) != 3:
            logger.warning(f"⚠️ 并行高阶任务数量异常，期望 3 个，实际 {len(prompts)} 个")

        logger.info("⚡ 执行并行高阶任务", extra={"count": len(prompts)})
        context_builder = await self._get_context_builder()

        async def _task(idx: int, step_name: str, driven_by: str, prompt_template: str) -> Dict[str, Any]:
            try:
                async with self.concurrency_manager.semaphore:
                    cache_key = f"{cache_key_base}:{step_name}:{idx}"
                    logger.info(f"⚡ [{step_name}] 缓存 key: ...{cache_key[-10:]}")

                    allowed_markers = ALLOWED_PARALLEL_HIGH_ORDER_MARKERS.get(idx, set())
                    rendered_prompt = context_builder.inject_allowed_context(
                        prompt_template, context_desc_info, allowed_markers
                    )

                    prompt_records.setdefault(PARALLEL_HIGH_ORDER, []).append({
                        "step_name": step_name,
                        "prompt": rendered_prompt
                    })

                    result = await self.step_executor.execute_step(
                        prompt_template=rendered_prompt,
                        template_name=template_name,
                        step_name=step_name,
                        cache_key=cache_key,
                        prompt_type=PARALLEL_HIGH_ORDER
                    )

                    if result.get("__success") is True:
                        try:
                            await self.llm_cache.set(cache_key, result)
                        except Exception as cache_err:
                            logger.warning(
                                f"⚠️ 并行高阶任务缓存写入失败 [{step_name}]: {type(cache_err).__name__}: {cache_err}",
                                extra={"step": step_name}
                            )

                    logger.debug(f"✅ 并行高阶任务 [{step_name}] 执行完成")
                    return result

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[{step_name}] 并行高阶任务异常: {error_msg}")
                failure_resp = LLMResponse.from_system_error(
                    system_error=error_msg,
                    model=self.llm_model,
                    template_name=template_name,
                    step_name=step_name,
                    prompt_type=PARALLEL_HIGH_ORDER,
                    include_traceback=True
                )
                return failure_resp.to_dict()

        tasks = [
            _task(idx, step_name, driven_by, prompt)
            for idx, (step_name, driven_by, prompt) in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)
        for idx, result in enumerate(results):
            step_name = result.get("step_name", f"unknown_high_order_{idx}")
            all_step_results.append(result)
            context_builder.update_context_from_result(result, context, step_name)
            context_builder.build_common_context(step_name, context, context_desc_info)

        success_count = sum(1 for r in results if r.get("__success", False))
        logger.info(
            f"并行高阶任务完成: {len(results)} 个任务, 成功 {success_count} 个",
            extra={"total": len(results), "success": success_count}
        )

    @async_timed
    async def _run_suggestion_serial_async(
            self,
            prompts: List[Tuple[str, str, str]],
            context: Dict[str, Any],
            template_name: str,
            cache_key_base: str,
            all_step_results: List[Dict],
            prompt_records: Dict,
            context_desc_info: List
    ):
        """串行执行任务，后续步骤可使用前面步骤注入的字段"""
        if not prompts:
            logger.info("⏭️ 无串行最小可行性建议任务")
            return

        logger.info("🔁 执行串行最小可行性建议任务", extra={"count": len(prompts)})
        context_builder = await self._get_context_builder()
        total_steps = len(prompts)

        for idx, (step_name, driven_by, prompt_template) in enumerate(prompts):
            cache_key = f"{cache_key_base}:{step_name}:{idx}"
            logger.info(f"⚡ [{step_name}] 缓存 key: ...{cache_key[-10:]}")
            rendered_prompt = prompt_template

            # === 关键：按 marker 动态筛选要注入的上下文 ===
            allowed = ALLOWED_SERIAL_SUGGESTION_MARKERS.get(idx, set())
            rendered_prompt = context_builder.inject_allowed_context(rendered_prompt, context_desc_info, allowed)

            prompt_records[SERIAL_SUGGESTION].append({"step_name": step_name, "prompt": rendered_prompt})
            result = await self.step_executor.execute_step(rendered_prompt, template_name, step_name,
                                                           cache_key, SERIAL_SUGGESTION)
            all_step_results.append(result)
            context_builder.update_context_from_result(result, context, step_name)
            if idx < total_steps - 1:
                context_builder.build_common_context(step_name, context, context_desc_info)

            if result.get("__success") is True:
                try:
                    await self.llm_cache.set(cache_key, result)
                except Exception as cache_err:
                    logger.warning(
                        f"⚠️ 串行最小可行性建议任务缓存写入失败 [{step_name}]: {type(cache_err).__name__}: {cache_err}",
                        extra={"step": step_name}
                    )
            logger.debug(f"✅ 串行最小可行性建议任务 [{step_name}] 执行完成")

    @staticmethod
    def _build_top_field_to_step_types() -> Dict[str, List[str]]:
        """
        从 REQUIRED_FIELDS_BY_CATEGORY 中提取所有顶级字段（如 'participants'），
        并记录它们所属的 step_type（如 LLM_SOURCE_EXTRACTION）。
        """
        mapping: Dict[str, List[str]] = {}
        for category, steps in REQUIRED_FIELDS_BY_CATEGORY.items():
            for step_name, field_tuples in steps.items():
                for field_path, *_ in field_tuples:
                    # 提取顶级字段名：取第一个 '.' 之前的部分
                    top_field = field_path.split('.')[0]
                    if top_field not in mapping:
                        mapping[top_field] = []
                    if step_name not in mapping[top_field]:
                        mapping[top_field].append(step_name)
        return mapping

    @staticmethod
    def _build_step_type_to_config() -> Dict[str, List[Tuple]]:
        config_map = {}
        for category, steps in REQUIRED_FIELDS_BY_CATEGORY.items():
            for step_name, tuples in steps.items():
                if step_name not in config_map:
                    config_map[step_name] = []
                config_map[step_name].extend(tuples)
        return config_map

    @async_timed
    async def _persist_extraction_artifacts(
            self,
            result: Dict[str, Any],
            aggregation: Dict[str, Any],
            template_name: str,
            user_input: str,
            prompt_records: Dict[str, List[Dict]],
            raw_response_records: Dict[str, List[Dict]],
            is_success: bool = True
    ) -> Optional[str]:
        """
        通用结果持久化函数，无论成功与否都保存诊断数据（dye vat），
        成功时额外保存结构化 raw 数据和生成报告。
        返回 report_url（仅成功时非空）。
        """
        filename = self.file_util.generate_filename(prefix=template_name, suffix=".json")
        report_url = ""

        try:
            # === 1. 总是保存诊断数据（dye vat）===
            dye_data = {
                "success": is_success,
                "partial_success": aggregation.get("__partial_success", False),
                "__valid_structure": aggregation.get("__valid_structure", False),
                "errors_summary": aggregation.get("__errors_summary", {}),
                "prompt_records": prompt_records,
                "raw_response_records": raw_response_records,
                "model": self.llm_model,
                "category": template_name,
                "user_input_preview": user_input[:200] if user_input else "",
                "timestamp": int(time.time()),
            }
            dye_file_path = self.DYE_VAT_DIR / filename
            if self.file_util.write_json(dye_data, dye_file_path):
                logger.info("💉 已保存验证诊断信息", extra={"path": str(dye_file_path), "success": is_success})

            # === 2. 仅成功时保存 raw + 生成报告 ===
            if is_success:
                raw_file_path = self.RAW_DATA_DIR / filename
                if self.file_util.write_json(result, raw_file_path):
                    logger.info("💾 已保存结构化数据", extra={"path": str(raw_file_path)})

                # 注入水印相关配置
                await self.result_assembler.inject_watermark_into_result(result)

                # 预处理相关步骤的数据
                await self.result_assembler.preprocess_for_html_rendering(result)

                outpath = self.report_generator.render_report_to_html(result)
                if outpath is None:
                    logger.error("❌ 报告生成失败，跳过 URL 构造")
                    report_url = ""
                else:
                    report_url = f"{self.REPORT_URL_PREFIX}{outpath.name}"
                    result["meta"]["report_url"] = report_url
                    logger.info("✅ 构造HTML报告成功", extra={"report_url": report_url})
        except Exception as e:
            logger.exception("持久化 extract 结果失败", extra={
                "category": template_name,
                "is_success": is_success,
                "error": str(e)
            })

        return report_url

    # @staticmethod
    # def _open_report_in_browser(outpath: Path) -> None:
    #     try:
    #         import webbrowser
    #         webbrowser.open(f"file://{outpath}")
    #         logger.info("🌐 已在浏览器中打开报告", extra={"outpath": str(outpath)})
    #     except Exception as e:
    #         logger.warning("❌ 无法自动打开浏览器", extra={"error": str(e)})
