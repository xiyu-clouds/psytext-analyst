import asyncio
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import List, Any, Tuple, Dict, Optional, Set
from src.state_of_mind.utils.registry import GlobalSingletonRegistry
from src.state_of_mind.cache.base import BaseCache
from src.state_of_mind.cache.redis import RedisLLMCache
from .prompter import Prompter
from ..cache.llm_cache import LLMCache
from ..config import config
from ..utils.async_decorators import async_timed
from ..utils.constants import ModelName, REQUIRED_FIELDS_BY_CATEGORY, LLM_PARTICIPANTS_EXTRACTION, \
    PREPROCESSING, PARALLEL, SERIAL, SEMANTIC_MODULES_L1, CATEGORY_SUGGESTION, LLM_EXPLICIT_MOTIVATION_EXTRACTION, \
    LLM_INFERENCE, PERCEPTION_LAYERS, ALLOWED_SERIAL_MARKERS, ALLOWED_PARALLEL_MARKERS, EXCLUDED_PRONOUNS, \
    CHINESE_PRONOUNS
from ..utils.file_util import FileUtil
from src.state_of_mind.utils.logger import LoggerManager as logger


class MetaCognitiveEngine:
    CHINESE_NAME = "MetaCognitiveEngine"
    _init_lock = asyncio.Lock()  # 用于保护并发初始化

    def __init__(self, backend_name: str, llm_model: str, recommended_params: dict):
        # 参数校验
        if not backend_name:
            raise ValueError("backend_name 不能为空")
        if not llm_model:
            raise ValueError("llm_model 不能为空")
        if not isinstance(recommended_params, dict):
            raise TypeError("recommended_params 必须是字典类型")

        # 校验 backend_name 是否支持
        if backend_name not in self.supported_backends():
            logger.warning(
                f"未知的 backend_name: {backend_name}，当前仅支持: {self.supported_backends()}"
            )
            # 可选择抛出异常或继续（取决于你的策略）
            raise ValueError(f"不支持的 backend_name: {backend_name}")

        # 校验 llm_model 是否在 MODEL_CONFIG 中存在
        if llm_model not in self.supported_models():
            logger.warning(
                f"未知的 llm_model: {llm_model}，当前仅支持: {self.supported_models()}"
            )
            # 同样，可选择抛出异常
            raise ValueError(f"不支持的 llm_model: {llm_model}")

        self.backend_name = backend_name
        self.llm_model = llm_model
        self.recommended_params = recommended_params
        self.prompter = Prompter()
        self.llm_cache = self._create_cache_backend(config)
        self.file_util = FileUtil()
        self._backend = None
        self._backend_configs = {"api_key": config.LLM_API_KEY, "timeout": 120} \
            if backend_name == ModelName.QWEN \
            else {"api_key": config.LLM_API_KEY, "base_url": config.LLM_API_URL, "timeout": 120}
        self._top_field_to_step_types = self._build_top_field_to_step_types()
        self._step_type_to_config = self._build_step_type_to_config()
        self.current_parallel_concurrency = config.get("CURRENT_PARALLEL_CONCURRENCY", 3)  # 默认3
        self._parallel_semaphore = asyncio.Semaphore(self.current_parallel_concurrency)
        logger.info(
            f"MetaCognitiveEngine 初始化成功，使用 backend: {backend_name}, model: {llm_model}"
        )

    @property
    async def backend(self):
        """惰性加载 backend 实例"""
        if self._backend is None:
            async with self._init_lock:
                if self._backend is None:  # double-check
                    logger.info("🔄 首次获取 backend，正在异步初始化...")
                    self._backend = await GlobalSingletonRegistry.get_backend_async(
                        self.backend_name,
                        self._backend_configs
                    )
                    if self._backend is None:
                        raise RuntimeError(f"无法获取 backend: {self.backend_name}")
        return self._backend

    @classmethod
    def supported_backends(cls) -> set:
        return {"qwen", "deepseek"}

    @classmethod
    def supported_models(cls) -> set:
        return {
            "qwen-max", "qwen3-max", "qwen-plus", "qwen-flash",
            "deepseek-chat"
        }

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

    def extract(self, template_name: str, user_input: str, suggestion_type: str, title: str = "文本多模态感知分析报告",
                **template_vars):
        """
        同步提取入口。仅适用于无 asyncio 事件循环的环境（如普通脚本）。
        在 Jupyter、FastAPI、异步测试等环境中，请使用 `await async_extract(...)`
        """
        return asyncio.run(self._async_extract(template_name, user_input, suggestion_type, title, **template_vars))

    async def async_extract(self, template_name: str, user_input: str, suggestion_type: str, title: str = "文本多模态感知分析报告",
                            **template_vars) -> Dict[str, Any]:
        """异步提取入口，适用于所有异步环境"""
        return await self._async_extract(template_name, user_input, suggestion_type, title, **template_vars)

    @async_timed
    async def _async_extract(self, template_name: str, user_input: str, suggestion_type: str,
                             title: str = "文本多模态感知分析报告", **template_vars) -> Dict[str, Any]:
        """异步核心流程"""
        context = template_vars.copy()
        context["user_input"] = user_input
        context["llm_model"] = self.llm_model
        self.user_input = user_input

        # 参数校验（同步）
        if not template_name or not isinstance(template_name, str):
            raise ValueError("template_name 必须是非空字符串")
        if user_input is not None and not isinstance(user_input, str):
            raise TypeError("user_input 必须是字符串或 None")

        cache_key = self.llm_cache.make_key(template_name, **context)
        logger.info(f"整体缓存 key: {cache_key}")
        cache_response = await self.llm_cache.get(cache_key)
        if cache_response.get("success"):
            cached_data = cache_response.get("data")
            if cached_data is not None:
                report_url = cached_data.get("meta", {}).get("report_url", "")
                res = {"report_url": report_url}
                logger.info("🔁 使用缓存结果", extra={"template": template_name, "report_url": report_url})
                return res

        prompt_result = self.prompter.build_raw(template_name, **context)
        preprocessing_prompts = prompt_result["preprocessing_prompts"]
        parallel_prompts = prompt_result["parallel_prompts"]
        serial_prompts = prompt_result["serial_prompts"]
        basic_data = prompt_result["basic_data"]

        # 收集所有步骤结果
        all_step_results = []
        # 收集所有步骤的最终prompt
        prompt_records = {PREPROCESSING: [], PARALLEL: [], SERIAL: []}
        # 收集所有步骤的原始响应
        raw_response_records = {PREPROCESSING: [], PARALLEL: [], SERIAL: []}
        # 收集可复用的动态生成的上下文描述信息
        context_desc_info = []

        # ✅ 预处理：必须等待完成
        await self._run_preprocessing_async(
            preprocessing_prompts, context, template_name, cache_key, all_step_results, prompt_records,
            context_desc_info
        )

        # ✅ 并行任务：并发执行
        await self._run_parallel_async(
            parallel_prompts, context, template_name, cache_key, all_step_results, prompt_records,
            context_desc_info
        )

        # ✅ 串行任务：必须等待完成
        await self._run_serial_async(
            serial_prompts, context, template_name, cache_key, all_step_results, prompt_records, context_desc_info
        )

        # ✅ 使用 basic_data 组装最终结果
        result = self._assemble_final_data(context, basic_data)
        aggregation = self._aggregate_step_results(all_step_results, raw_response_records)
        valid_result = self._validate_final_result(result)
        aggregation["__errors_summary"]["final_validation_errors"] = [
            {"step": "final_validation", "errors": valid_result["__final_validation_errors"]}
        ] if valid_result["__final_validation_errors"] else []
        result["meta"]["validity_level"] = valid_result["__validity_level"]

        report_url = ""

        # 只缓存完全成功的结果
        if valid_result.get("__success"):
            try:
                await self._inject_suggestion_into_result(result, user_input, suggestion_type, title)

                # 生成唯一文件名
                filename = self.file_util.generate_filename(prefix=template_name, suffix=".json")

                # === 写入 DATA_YUAN_RAW_DIR ===
                raw_file_path = config.DATA_YUAN_RAW_DIR / filename
                if self.file_util.write_json(result, raw_file_path):
                    logger.info("💾 已保存结构化数据", extra={"path": str(raw_file_path), "category": template_name})

                # === 写入 DATA_YUAN_DYE_VAT_DIR（验证诊断信息）===
                dye_data = {
                    "success": valid_result.get("__success", False),
                    "partial_success": aggregation.get("__partial_success", False),
                    "__valid_structure": aggregation.get("__valid_structure", False),
                    "errors_summary": aggregation.get("__errors_summary", {}),
                    "prompt_records": prompt_records,
                    "raw_response_records": raw_response_records,
                    "model": self.llm_model,
                    "category": template_name,
                    "timestamp": int(time.time()),
                }
                dye_file_path = config.DATA_YUAN_DYE_VAT_DIR / filename
                if self.file_util.write_json(dye_data, dye_file_path):
                    logger.info("💉 已保存验证诊断信息", extra={"path": str(dye_file_path), "category": template_name})

            except Exception as e:
                logger.exception("持久化 extract 结果失败", extra={"category": template_name, "error": str(e)})

            outpath = self._render_report_to_html(result)
            # ✅ 提取文件名（如 analysis_abc123.html）
            report_filename = outpath.name  # outpath 是 Path 对象
            # ✅ 构造可访问的 URL（前端能打开的）
            report_url = f"/reports/{report_filename}"
            logger.info("✅ 构造HTML报告成功", extra={"report_url": report_url})

            result["meta"]["report_url"] = report_url
            await self.llm_cache.set(cache_key, result)
            logger.info("✅ 缓存有效结果", extra={"cache_key": cache_key})
            # self._open_report_in_browser(outpath)
        else:
            logger.info("🟡 部分步骤成功但最终校验失败，不缓存",
                        extra={"cache_key": cache_key, "errors_summary": aggregation.get("__errors_summary", {})})

        return {"report_url": report_url}

    async def _run_preprocessing_async(
            self,
            prompts: List[Tuple[str, str, str]],
            context: Dict[str, Any],
            template_name: str,
            cache_key_base: str,
            all_step_results: List[Dict],
            prompt_records: Dict,
            context_desc_info: List
    ):
        """异步执行预处理（虽然是顺序，但为未来扩展）"""
        logger.info("🔧 执行预处理任务", extra={"count": len(prompts)})
        for idx, (step_name, driven_by, prompt_template) in enumerate(prompts):
            cache_key = f"{cache_key_base}:{step_name}:{idx}"

            rendered_prompt = self.build_user_input_only(prompt_template, context, context_desc_info)
            # 记录 prompt
            prompt_records[PREPROCESSING].append({"step_name": step_name, "prompt": rendered_prompt})

            result = await self._execute_single_step_async(rendered_prompt, template_name, step_name, cache_key,
                                                           PREPROCESSING)
            all_step_results.append(result)
            self._update_context_from_result(result, context, step_name)
            await self.llm_cache.set(cache_key, result)

    async def _run_parallel_async(
            self,
            prompts: List[Tuple[str, str, str]],
            context: Dict[str, Any],
            template_name: str,
            cache_key_base: str,
            all_step_results: List[Dict],
            prompt_records: Dict,
            context_desc_info: List
    ):
        """并发执行并行任务"""
        if not prompts:
            logger.info("⏭️ 无并行任务")
            return

        logger.info("⚡ 执行并行任务", extra={"count": len(prompts)})
        # 构建参与者有效信息，只需要一次
        self.build_parallel_context(
            step_name=LLM_PARTICIPANTS_EXTRACTION,
            context=context,
            context_desc_info=context_desc_info
        )

        # 显式定义返回类型，让类型检查器知道可能返回 Exception
        async def _task(idx: int, step_name: str, driven_by: str, prompt_template: str) -> Dict[str, Any]:
            async with self._parallel_semaphore:  # ← 限制并发
                cache_key = f"{cache_key_base}:{step_name}:{idx}"
                rendered_prompt = prompt_template

                # === 关键：按 marker 动态筛选要注入的上下文 ===
                allowed = ALLOWED_PARALLEL_MARKERS.get(idx, set())
                for ctx_str in context_desc_info:
                    if not ctx_str or not isinstance(ctx_str, str):
                        continue
                    # 检查该段是否以允许的 marker 开头
                    if any(ctx_str.lstrip().startswith(marker) for marker in allowed):
                        rendered_prompt += ctx_str

                prompt_records[PARALLEL].append({"step_name": step_name, "prompt": rendered_prompt})
                data = await self._execute_single_step_async(rendered_prompt, template_name, step_name, cache_key,
                                                             PARALLEL)
                await self.llm_cache.set(cache_key, data)
                return data

        # ✅ 所有 task 都返回 dict，不再可能返回 Exception
        tasks = [
            _task(idx, step_name, driven_by, prompt)
            for idx, (step_name, driven_by, prompt) in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        legitimate_participants = self._build_legitimate_participant_set(context)

        for idx, result in enumerate(results):
            # 调用封装的后处理函数
            await self._filter_perception_results_by_legitimate_participants(result, legitimate_participants)
            all_step_results.append(result)
            self._update_context_from_result(result, context, result.get("step_name"))

    async def _run_serial_async(
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
            logger.info("⏭️ 无串行任务")
            return

        logger.info("🔁 执行串行任务", extra={"count": len(prompts)})
        # === 关键：动态生成感知的上下文描述 ===
        dynamic_desc = self.build_serial_context_batch(context)
        context_desc_info.append(dynamic_desc)
        # 生成合法参与者数据
        legit_participants_ctx = self._build_participants_context_desc(context)
        if legit_participants_ctx:
            context_desc_info.append(legit_participants_ctx)

        total_steps = len(prompts)

        for idx, (step_name, driven_by, prompt_template) in enumerate(prompts):
            cache_key = f"{cache_key_base}:{step_name}:{idx}"
            rendered_prompt = prompt_template

            # === 关键：按 marker 动态筛选要注入的上下文 ===
            allowed = ALLOWED_SERIAL_MARKERS.get(idx, set())
            for ctx_str in context_desc_info:
                if not ctx_str or not isinstance(ctx_str, str):
                    continue
                # 检查该段是否以允许的 marker 开头
                if any(ctx_str.lstrip().startswith(marker) for marker in allowed):
                    rendered_prompt += ctx_str

            prompt_records[SERIAL].append({"step_name": step_name, "prompt": rendered_prompt})
            result = await self._execute_single_step_async(rendered_prompt, template_name, step_name,
                                                           cache_key, SERIAL)
            all_step_results.append(result)
            # 4. 更新 context（后续步骤可用）
            self._update_context_from_result(result, context, step_name)
            # 仅在非最后一次迭代时生成并注入并行上下文描述
            if idx < total_steps - 1:
                temp_context = {driven_by: context.get(driven_by)}
                self.build_parallel_context(step_name, temp_context, context_desc_info)

            await self.llm_cache.set(cache_key, result)

    async def _execute_single_step_async(
            self,
            prompt_template: str,
            template_name: str,
            step_name: str,
            cache_key: str,
            prompt_type: str
    ) -> Dict[str, Any]:
        """异步执行单个 LLM 调用，支持缓存"""
        # 查缓存（同步）
        cache_response = await self.llm_cache.get(cache_key)
        if cache_response.get("success"):
            cached_data = cache_response.get("data")
            if cached_data is not None:
                logger.info("🔁 使用缓存结果", extra={"template": template_name})
                return cached_data

        try:
            backend = await self.backend
            result = await backend.async_call(
                prompt=prompt_template,
                model=self.llm_model,
                template_name=template_name,
                step_name=step_name,
                params=self.recommended_params,
                prompt_type=prompt_type
            )
            return result
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"[{step_name}] LLM 调用异常",
                extra={"step": step_name}
            )
            return {
                "__success": False,
                "__valid_structure": False,
                "data": {},
                "__raw_response": None,
                "__validation_errors": [],
                "__api_error": None,
                "__system_error": error_msg,  # ← 统一使用 __system_error
                "model": self.llm_model,
                "template_name": template_name,
                "step_name": step_name,
                "prompt_type": prompt_type,
                "__traceback": traceback.format_exc()  # 可选，用于调试
            }

    @staticmethod
    def _update_context_from_result(
            result: Dict[str, Any],
            context: Dict[str, Any],
            step_name: str
    ):
        if not result.get("__success"):
            error_detail = result.get("__system_error") or result.get("__api_error") or "Unknown error"
            extra = {
                "step": step_name,
                "error": error_detail,  # ← 真实错误
                "system_error": result.get("__system_error"),
                "api_error": result.get("__api_error")
            }
            logger.warning(
                f"⚠️ 步骤失败，跳过更新: {extra}",
                module_name=MetaCognitiveEngine.CHINESE_NAME,
                extra=extra
            )
            return

        if not result.get("__valid_structure"):
            val_errors = result.get("__validation_errors", [])
            extra = {
                "step_name": result.get("step_name"),
                "template_name": result.get("template_name"),
                "validation_errors": val_errors,
                "raw_response": result.get("__raw_response")[:200] if result.get("__raw_response") else None
            }
            logger.warning(
                f"当前步骤 {step_name} 结构校验失败",
                module_name=MetaCognitiveEngine.CHINESE_NAME,
                extra=extra
            )
            return

        data = result.get("data")
        if data and isinstance(data, dict):
            clean_data = {k: v for k, v in data.items() if not k.startswith("__")}
            if clean_data:
                injected_keys = list(clean_data.keys())
                context.update(clean_data)
                logger.info("🟢 成功注入上下文字段", module_name=MetaCognitiveEngine.CHINESE_NAME,
                            extra={"step": step_name, "keys": injected_keys})
        elif data:
            raise ValueError(f"[{step_name}] data 必须是 dict，实际为 {type(data)}")
        else:
            logger.info("⚪ data 为空，跳过注入", module_name=MetaCognitiveEngine.CHINESE_NAME, extra={"step": step_name})

    def _assemble_final_data(
            self,
            context: Dict[str, Any],
            basic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        组装最终的有效数据结构。
        - 保留 basic_data 的完整骨架（尤其是 meta 结构）
        - 从 context 中提取非排除字段注入 result 顶层
        - 为 result 中的 participants 每个实体生成 entity_id
        - 计算 privacy_level 并注入 meta.privacy_scope
        """
        result = deepcopy(basic_data)

        excluded_fields = {"user_input", "llm_model"}

        # 第一步：注入非排除字段
        for key, value in context.items():
            if key.startswith("__") or key in excluded_fields:
                continue
            result[key] = value

        # 第二步：处理 participants（直接操作 result）
        participants = result.get("participants")
        if isinstance(participants, list) and participants:
            processed_participants = []
            for p in participants:
                if isinstance(p, dict) and "entity" in p and isinstance(p["entity"], str):
                    unique_suffix = uuid.uuid4().hex[:8]
                    p_new = deepcopy(p)
                    p_new["entity_id"] = f"{p['entity']}_{unique_suffix}"
                    processed_participants.append(p_new)
                else:
                    processed_participants.append(deepcopy(p))
            result["participants"] = processed_participants

        # 第三步：隐私度计算（仍基于 context，因为 count 需要排除逻辑）
        count = sum(
            1 for k in context.keys()
            if not k.startswith("__") and k not in excluded_fields
        )

        privacy_score = count * 0.05

        # 推理层
        inference = context.get("inference")
        if isinstance(inference, dict):
            has_inference = (
                    (isinstance(inference.get("events"), list) and len(inference["events"]) > 0) or
                    (isinstance(inference.get("summary"), str) and inference["summary"].strip()) or
                    (isinstance(inference.get("evidence"), list) and len(inference["evidence"]) > 0)
            )
            if has_inference:
                privacy_score += 0.05

        # 深度分析层（+0.05）
        explicit_motivation = context.get("explicit_motivation")
        if isinstance(explicit_motivation, dict):
            has_explicit_motivation = (
                    (isinstance(explicit_motivation.get("summary"), str) and explicit_motivation["summary"].strip()) or
                    (isinstance(explicit_motivation.get("core_driver"), list) and len(
                        explicit_motivation["core_driver"]) > 0) or
                    (isinstance(explicit_motivation.get("power_asymmetry"), dict) and explicit_motivation[
                        "power_asymmetry"]) or
                    (isinstance(explicit_motivation.get("narrative_distortion"), dict) and explicit_motivation[
                        "narrative_distortion"])
            )
            if has_explicit_motivation:
                privacy_score += 0.05

        # 合理建议层（+0.1）
        rational_advice = context.get("rational_advice")
        if isinstance(rational_advice, dict):
            has_rational_advice = (
                    (isinstance(rational_advice.get("summary"), str) and rational_advice["summary"].strip()) or
                    (isinstance(rational_advice.get("safety_first_intervention"), list) and len(
                        rational_advice["safety_first_intervention"]) > 0) or
                    (isinstance(rational_advice.get("incremental_strategy"), list) and len(
                        rational_advice["incremental_strategy"]) > 0)
            )
            if has_rational_advice:
                privacy_score += 0.1

        privacy_level = min(round(privacy_score, 2), 1.0)

        # 注入 meta
        meta = result.setdefault("meta", {})
        privacy_scope = meta.setdefault("privacy_scope", {})
        privacy_scope["privacy_level"] = float(privacy_level)

        # 关键：清理 result 中“全空”的顶层字段
        keys_to_remove = []
        for key, value in result.items():
            if not self._is_value_effective(value):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del result[key]
        return result

    def _is_value_effective(self, value) -> bool:
        """
        判断一个值是否“有效”（即不应被视为空）。
        - 字符串：非空且非纯空白 → 有效
        - 列表/元组：至少一个元素有效 → 有效
        - 字典：至少一个 value 有效 → 有效
        - None / 空字符串 / 空 list / 空 dict → 无效
        - bool / int / float → 一律视为有效（如 privacy_level=0.0 是有效的）
        """
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple)):
            return any(self._is_value_effective(item) for item in value)
        if isinstance(value, dict):
            return any(self._is_value_effective(v) for v in value.values())
        # bool, int, float, etc.
        return True

    @staticmethod
    def build_user_input_only(
            prompt_template: str,
            context: Dict[str, Any],
            context_desc_info: List
    ) -> str:
        user_input_text = context.get("user_input", "")
        build_context_desc = f"### USER_INPUT BEGIN（用户原始输入开始）\n{user_input_text}\n### USER_INPUT END（用户原始输入结束）\n"
        context_desc_info.append(build_context_desc)
        rendered_prompt = f"{prompt_template}{build_context_desc}"
        return rendered_prompt

    def build_parallel_context(
            self,
            step_name: str,
            context: Dict[str, Any],
            context_desc_info: List
    ):
        """
        构建最终渲染后的 prompt，支持动态描述生成。
        """
        field_config = self._step_type_to_config.get(step_name)
        wrapped_desc = ""
        if field_config:
            try:
                raw_desc = self.prompter.generate_description(
                    context=context,
                    field_config=field_config,
                    prefix=""
                )
                start_marker = ""
                end_marker = ""
                readable = ""
                if raw_desc:
                    if step_name == LLM_PARTICIPANTS_EXTRACTION:
                        start_marker = "### PARTICIPANTS_VALID_INFORMATION BEGIN"
                        end_marker = "### PARTICIPANTS_VALID_INFORMATION END"
                        readable = "参与者有效信息上下文"
                    elif step_name == LLM_INFERENCE:
                        start_marker = "### INFERENCE_CONTEXT BEGIN"
                        end_marker = "### INFERENCE_CONTEXT END"
                        readable = "合理推演有效信息上下文"
                    elif step_name == LLM_EXPLICIT_MOTIVATION_EXTRACTION:
                        start_marker = "### EXPLICIT_MOTIVATION_CONTEXT BEGIN"
                        end_marker = "### EXPLICIT_MOTIVATION_CONTEXT END"
                        readable = "显性动机有效信息上下文"

                    wrapped_desc = self._wrap_with_context_markers(
                        raw_desc, start_marker, end_marker, readable
                    )
                    context_desc_info.append(wrapped_desc)
            except Exception as e:
                logger.error(f"[{step_name}] 动态描述生成失败: {e}")

    def build_serial_context_batch(self, context: Dict[str, Any]) -> str:
        """
        遍历 context 中所有顶级字段（非 __ 开头，非系统字段），
        查找其对应的 step_type，用完整 field_tuples 生成描述。
        """
        excluded = {"user_input", "llm_model", "participants"}
        descriptions = []

        for key, value in context.items():
            if key.startswith("__") or key in excluded:
                continue

            # 查找该顶级字段关联的所有 step_type
            step_types = self._top_field_to_step_types.get(key)
            if not step_types:
                continue

            for st in step_types:
                field_tuples = self._step_type_to_config.get(st)
                if not field_tuples:
                    continue

                try:
                    desc = self.prompter.generate_description(
                        context=context,
                        field_config=field_tuples,
                        prefix=""
                    )
                    if desc.strip():
                        descriptions.append(desc.strip())
                except Exception as e:
                    logger.error(f"生成字段 {key} 的描述失败 (step_type={st}): {e}")

        full_content = "\n".join(descriptions)
        return self._wrap_with_context_markers(
            full_content,
            "### PERCEPTUAL_CONTEXT_BATCH BEGIN",
            "### PERCEPTUAL_CONTEXT_BATCH END",
            "批量感知层上下文"
        )

    @staticmethod
    def _validate_l0(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """校验原始输入有效性（L0）"""
        errors = []
        required_top = {"id", "type", "timestamp", "source", "meta"}
        for field in required_top:
            if field not in result:
                errors.append(f"L0缺失顶层字段: {field}")

        source = result.get("source", {})
        content = source.get("content")
        if not isinstance(content, str) or len(content.strip()) < 10:
            errors.append("L0: source.content 必须为非空字符串且长度≥10")

        return len(errors) == 0, errors

    @staticmethod
    def _validate_l1(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """校验语义结构有效性（L1）——每个模块必须同时满足：
        (A) 顶层 summary（非空str）+ evidence（非空list）
        (B) events 中至少一个 item 含 semantic_notation（非空str）+ evidence（非空list）
        只要任一模块同时满足 A 和 B，L1 即有效。
        """
        errors = []
        l1_valid = False
        present_but_empty = []
        missing_or_invalid = []

        for mod_name in SEMANTIC_MODULES_L1:
            mod = result.get(mod_name)
            if mod is None:
                missing_or_invalid.append(f"{mod_name} (缺失)")
                continue

            if isinstance(mod, dict):
                # --- (A) 顶层单元必须有效 ---
                top_summary = mod.get("summary")
                top_evidence = mod.get("evidence")
                top_valid = (
                        isinstance(top_summary, str) and top_summary.strip() and
                        isinstance(top_evidence, list) and len(top_evidence) > 0
                )

                # --- (B) events 中必须至少有一个完整事件项 ---
                events = mod.get("events")
                event_valid = False
                if isinstance(events, list) and len(events) > 0:
                    for item in events:
                        if isinstance(item, dict):
                            notation = item.get("semantic_notation")
                            evi = item.get("evidence")
                            if (
                                    isinstance(notation, str) and notation.strip() and
                                    isinstance(evi, list) and len(evi) > 0
                            ):
                                event_valid = True
                                break

                # --- 模块有效条件：A AND B ---
                if top_valid and event_valid:
                    l1_valid = True
                else:
                    reasons = []
                    if not top_valid:
                        reasons.append("顶层 summary/evidence 无效")
                    if not event_valid:
                        reasons.append("events 中缺少含 semantic_notation+evidence 的有效项")
                    present_but_empty.append(f"{mod_name} ({'; '.join(reasons)})")

            elif isinstance(mod, list):
                # 兼容旧结构（虽已统一为 dict，但保留）
                # 注意：list 结构无法同时满足 A+B（无顶层字段），故视为无效
                present_but_empty.append(f"{mod_name} (模块为列表，无法满足双重要求)")
            else:
                missing_or_invalid.append(f"{mod_name} (类型错误: {type(mod)})")

        if not l1_valid:
            if present_but_empty:
                errors.append("存在语义模块但未同时满足顶层与事件有效性: " + ", ".join(present_but_empty))
            if missing_or_invalid:
                errors.append("关键语义模块缺失或格式错误: " + ", ".join(missing_or_invalid))

        return l1_valid, errors

    @staticmethod
    def _validate_l2(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """校验认知干预有效性（L2）——必须同时满足：
        1. inference 模块：顶层 summary+evidence 有效 AND events 中至少一项含 semantic_notation+evidence
        2. explicit_motivation 模块：同上结构，同样双重要求
        3. rational_advice 模块：summary+evidence 有效 AND 至少一个建议字段有实质内容
        """
        errors = []
        l2_ok = True

        # -----------------------------
        # 1. Validate inference
        # -----------------------------
        inference = result.get("inference")
        if not isinstance(inference, dict):
            errors.append("L2: inference 缺失或非字典")
            l2_ok = False
        else:
            # (A) Top-level
            top_summary = inference.get("summary")
            top_evidence = inference.get("evidence")
            top_valid = (
                    isinstance(top_summary, str) and top_summary.strip() and
                    isinstance(top_evidence, list) and len(top_evidence) > 0
            )

            # (B) Events
            events = inference.get("events")
            event_valid = False
            if isinstance(events, list):
                for item in events:
                    if isinstance(item, dict):
                        sn = item.get("semantic_notation")
                        evi = item.get("evidence")
                        if (
                                isinstance(sn, str) and sn.strip() and
                                isinstance(evi, list) and len(evi) > 0
                        ):
                            event_valid = True
                            break

            if not (top_valid and event_valid):
                reasons = []
                if not top_valid:
                    reasons.append("顶层 summary/evidence 无效")
                if not event_valid:
                    reasons.append("events 中无 semantic_notation+evidence 有效项")
                errors.append(f"L2: inference 未同时满足双重要求 ({'; '.join(reasons)})")
                l2_ok = False

        # -----------------------------
        # 2. Validate explicit_motivation
        # -----------------------------
        explicit_motivation = result.get("explicit_motivation")
        if not isinstance(explicit_motivation, dict):
            errors.append("L2: explicit_motivation 缺失或非字典")
            l2_ok = False
        else:
            # (A) Top-level
            top_summary = explicit_motivation.get("summary")
            top_evidence = explicit_motivation.get("evidence")
            top_valid = (
                    isinstance(top_summary, str) and top_summary.strip() and
                    isinstance(top_evidence, list) and len(top_evidence) > 0
            )

            # (B) Events
            events = explicit_motivation.get("events")
            event_valid = False
            if isinstance(events, list):
                for item in events:
                    if isinstance(item, dict):
                        sn = item.get("semantic_notation")
                        evi = item.get("evidence")
                        if (
                                isinstance(sn, str) and sn.strip() and
                                isinstance(evi, list) and len(evi) > 0
                        ):
                            event_valid = True
                            break

            if not (top_valid and event_valid):
                reasons = []
                if not top_valid:
                    reasons.append("顶层 summary/evidence 无效")
                if not event_valid:
                    reasons.append("events 中无 semantic_notation+evidence 有效项")
                errors.append(f"L2: explicit_motivation 未同时满足双重要求 ({'; '.join(reasons)})")
                l2_ok = False

        # -----------------------------
        # 3. Validate rational_advice
        # -----------------------------
        rational_advice = result.get("rational_advice")
        if not isinstance(rational_advice, dict):
            errors.append("L2: rational_advice 缺失或非字典")
            l2_ok = False
        else:
            # rational_advice 无 events，只有顶层字段
            summary = rational_advice.get("summary")
            evidence = rational_advice.get("evidence")
            has_summary_evidence = (
                    isinstance(summary, str) and summary.strip() and
                    isinstance(evidence, list) and len(evidence) > 0
            )

            # Check if any substantive advice field is non-empty
            substantive_fields = {
                "safety_first_intervention",
                "systemic_leverage_point",
                "incremental_strategy",
                "stakeholder_tradeoffs",
                "long_term_exit_path",
                "cultural_adaptation_needed",
                "fallback_plan"
            }
            has_substantive_content = False
            for field in substantive_fields:
                val = rational_advice.get(field)
                if val not in (None, "", [], {}):
                    # For stakeholder_tradeoffs (dict), check if it has non-empty subfields
                    if isinstance(val, dict):
                        if any(v not in (None, "", [], {}) for v in val.values()):
                            has_substantive_content = True
                            break
                    else:
                        has_substantive_content = True
                        break

            if not (has_summary_evidence and has_substantive_content):
                reasons = []
                if not has_summary_evidence:
                    reasons.append("summary 或 evidence 无效")
                if not has_substantive_content:
                    reasons.append("无实质性建议内容")
                errors.append(f"L2: rational_advice 无效 ({'; '.join(reasons)})")
                l2_ok = False

        return l2_ok, errors

    def _validate_final_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        渐进式三级有效性校验（单次语义模块遍历），返回 validity_level 与分层错误。
        """
        # L0: 原始输入
        l0_valid, errors_l0 = self._validate_l0(result)
        if not l0_valid:
            return {
                "__success": False,
                "__validity_level": "invalid",
                "__final_validation_errors": {"L0": errors_l0, "L1": [], "L2": []}
            }

        # L1: 语义结构
        l1_valid, errors_l1 = self._validate_l1(result)

        # L2: 认知干预（仅当 L1 有效时尝试）
        l2_valid, errors_l2 = self._validate_l2(result) if l1_valid else (False, [])

        # 确定最终级别
        level = (
            "L2_actionable" if l2_valid else
            "L1_structured" if l1_valid else
            "L0_raw"
        )

        return {
            "__success": True,
            "__validity_level": level,
            "__final_validation_errors": {
                "L0": errors_l0,
                "L1": errors_l1,
                "L2": errors_l2
            }
        }

    @staticmethod
    def _aggregate_step_results(all_step_results: List[Dict[str, Any]], raw_response_records: Dict) -> Dict[str, Any]:
        """
        聚合所有步骤的结果，仅收集错误信息和执行状态。
        注意：__success 字段不在此处设置，留空（由 _validate_final_result 决定）。
        """
        system_errors = []
        api_errors = []
        validation_errors_all = []
        all_valid = True
        partial_success = False

        for step in all_step_results:
            if step.get("__success"):
                partial_success = True

            if not step.get("__valid_structure"):
                all_valid = False

            sys_err = step.get("__system_error")
            if sys_err:
                system_errors.append({
                    "step": step.get("step_name"),
                    "error": sys_err
                })

            api_err = step.get("__api_error")
            if api_err:
                api_errors.append({
                    "step": step.get("step_name"),
                    "error": api_err
                })

            val_errs = step.get("__validation_errors")
            if val_errs:
                validation_errors_all.append({
                    "step": step.get("step_name"),
                    "errors": val_errs
                })

            prompt_type = step.get("prompt_type")
            if prompt_type and prompt_type == PREPROCESSING:
                raw_response_records[PREPROCESSING].append({
                    "step_name": step.get("step_name"),
                    "raw_response": step.get("__raw_response")
                })
            elif prompt_type and prompt_type == PARALLEL:
                raw_response_records[PARALLEL].append({
                    "step_name": step.get("step_name"),
                    "raw_response": step.get("__raw_response")
                })
            else:
                raw_response_records[SERIAL].append({
                    "step_name": step.get("step_name"),
                    "raw_response": step.get("__raw_response")
                })

        return {
            "__valid_structure": all_valid,
            "__partial_success": partial_success,
            "__errors_summary": {
                "system_errors": system_errors,
                "api_errors": api_errors,
                "validation_errors": validation_errors_all,
                # 预留位置给最终校验错误，用不同 key 避免冲突
                "final_validation_errors": []  # 后续由 _validate_final_result 填入
            }
        }

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

    def _render_report_to_html(self, data: Dict[str, Any]) -> Optional[Path]:
        """
        将 result 数据注入 HTML 模板，生成报告。
        - 输出目录：config.STATIC_REPORTS_DIR
        - 文件名：通过 self.file_util.generate_filename 生成
        - 前缀："文本多模态感知分析报告"
        - 后缀：".html"
        - 模板读取：复用 self.file_util.read_file
        - 文件写入：复用 self.file_util.write_file
        - 上下文变量名：data
        """
        try:
            if not data or not isinstance(data, dict):
                return None

            # 1. 生成文件名
            filename = self.file_util.generate_filename(
                prefix="文本多模态感知分析报告",
                suffix=".html",
                include_timestamp=True
            )

            # 2. 确定输出路径
            output_path = config.REPORTS_DIR / filename

            # 3. 读取模板（复用工具）
            template_content = self.file_util.read_file(
                str(config.FILE_DEFAULT_TEMPLATE_PATH),
                encoding="utf-8",
                auto_decode=False
            )
            if not template_content:
                logger.error("❌ 模板文件为空或读取失败", extra={"template_path": str(config.FILE_DEFAULT_TEMPLATE_PATH)})
                return None

            # 4. 渲染模板
            from jinja2 import Template
            html_output = Template(template_content).render(data=data)

            # 5. ✅ 写入 HTML（复用 file_util.write_file）
            success = self.file_util.write_file(
                file_path=str(output_path),
                content=html_output,
                encoding="utf-8",
                as_json=False,
                file_type="html"  # ← 让日志显示“HTML”（若 write_file 支持该参数）
            )

            if not success:
                logger.error("❌ HTML 报告写入失败", extra={"path": str(output_path)})
                return None

            logger.info("📄 HTML 报告已生成", extra={"path": str(output_path)})
            return output_path

        except Exception as e:
            logger.exception("💥 HTML 报告生成失败", extra={"error": str(e)})
            return None

    async def _execute_suggestion(self, prompt: str) -> str:
        logger.info("🧠 开始生成 LLM 建议内容", module_name=self.CHINESE_NAME)
        try:
            backend = await self.backend
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

    async def _inject_suggestion_into_result(
            self,
            result: Dict[str, Any],
            user_input: str,
            suggestion_type: str,
            title: str = "文本多模态感知分析报告"
    ) -> None:
        """
        为已验证通过的 result 注入 LLM 生成的建议内容。
        - 标题注入到 result['meta']['title']
        - 建议内容注入到 result['analysis']['suggestion']（含元信息）
        - 保证即使失败也不破坏 result 结构
        """
        # 确保 meta 和 analysis 存在
        result.setdefault("meta", {})
        result.setdefault("analysis", {})

        # 先设置标题（总是成功）
        result["meta"]["title"] = title

        try:
            suggestion_prompt = self.prompter.build_suggestion(
                template_name=CATEGORY_SUGGESTION,
                user_input=user_input,
                suggestion_type=suggestion_type
            )
            suggestion_content = await self._execute_suggestion(suggestion_prompt)

            # 构造带元信息的 suggestion 对象
            suggestion_record = {
                "content": suggestion_content,
                "type": suggestion_type,
                "model": self.llm_model,
                "generated_at": int(time.time()),
                "success": not (suggestion_content.startswith(("生成失败", "[建议生成失败")))
            }

            result["analysis"]["suggestion"] = suggestion_record
            logger.info(
                "✅ LLM 建议已注入结果",
                extra={
                    "suggestion_type": suggestion_type,
                    "model": self.llm_model,
                    "success": suggestion_record["success"]
                }
            )
        except Exception as e:
            error_msg = f"[建议生成失败: {str(e)}]"
            logger.exception(
                "⚠️ 注入 LLM 建议失败",
                extra={
                    "suggestion_type": suggestion_type,
                    "error": str(e)
                }
            )
            # 即使失败也写入结构化占位，便于前端/后续处理
            result["analysis"]["suggestion"] = {
                "content": error_msg,
                "type": suggestion_type,
                "model": self.llm_model,
                "generated_at": int(time.time()),
                "success": False
            }

    @staticmethod
    def _open_report_in_browser(outpath: Path) -> None:
        try:
            import webbrowser
            webbrowser.open(f"file://{outpath}")
            logger.info("🌐 已在浏览器中打开报告", extra={"outpath": str(outpath)})
        except Exception as e:
            logger.warning("❌ 无法自动打开浏览器", extra={"error": str(e)})

    def _build_participants_context_desc(self, context: Dict[str, Any]) -> str:
        """基于合法参与者集合生成上下文描述字符串"""
        legit_items = sorted(self._build_legitimate_participant_set(context))  # 排序保证输出稳定（便于缓存/调试）
        if not legit_items:
            return ""

        prefix = "### LEGITIMATE_PARTICIPANTS BEGIN（合法的参与者实体或角色开始）\n"
        suffix = "\n### LEGITIMATE_PARTICIPANTS END（合法的参与者实体或角色结束）\n"
        return prefix + "\n".join(legit_items) + suffix

    @staticmethod
    def _build_legitimate_participant_set(context: Dict[str, Any]) -> Set[str]:
        """从 context['participants'] 构建合法标识集合（entity + name）"""
        participants = context.get("participants", [])
        if not isinstance(participants, list):
            return set()

        legit_set = set()
        for p in participants:
            if not isinstance(p, dict):
                continue
            entity = p.get("entity")
            name = p.get("name")
            if isinstance(entity, str) and entity.strip():
                legit_set.add(entity.strip())
            if isinstance(name, str) and name.strip():
                legit_set.add(name.strip())
        return legit_set

    async def _filter_perception_results_by_legitimate_participants(
            self,
            result: Dict[str, Any],
            legitimate_participants: Set[str]
    ) -> None:
        """
        过滤感知结果中的非法 experiencer。
        - 仅保留 experiencer 属于 legitimate_participants 的事件；
        - 支持两阶段解析：
            1. 简单代词映射（如 "他" → "张三"）
            2. LLM 批量兜底指代消解（最后手段）
        """
        logger.info(f"→ 进入感知结果过滤流程（合法参与者: {sorted(legitimate_participants)}）", extra={"module_name": self.CHINESE_NAME})

        if not isinstance(result, dict):
            return

        step_name = result.get("step_name")
        if step_name not in PERCEPTION_LAYERS:
            return

        data = result.get("data")
        if not isinstance(data, dict) or not data:
            return

        try:
            key, block = next(iter(data.items()))
        except StopIteration:
            return

        if not (isinstance(block, dict) and isinstance(block.get("events"), list)):
            return

        original_events = block["events"]
        if not original_events:
            return

        logger.info(
            f"→ 待处理事件 experiencer 列表: {[e.get('experiencer') for e in original_events if isinstance(e, dict)]}",
            extra={"module_name": self.CHINESE_NAME})

        # 第一步：扫描事件，标记合法项，并收集需 LLM 消解的代词
        valid_indices: Set[int] = set()
        pronoun_map: Dict[int, str] = {}  # idx -> pronoun

        for idx, evt in enumerate(original_events):
            if not isinstance(evt, dict):
                continue

            exp = evt.get("experiencer")
            if not isinstance(exp, str):
                continue

            # 情况1：已在合法名单中
            if exp in legitimate_participants:
                valid_indices.add(idx)
                continue

            # 情况2：尝试简单映射
            resolved = self._try_simple_resolution(exp, legitimate_participants)
            if resolved is not None:
                evt["experiencer"] = resolved  # 原地更新
                valid_indices.add(idx)
                continue

            # 情况3：需 LLM 兜底
            pronoun_map[idx] = exp

        # 第二步：批量调用 LLM 兜底（仅当有未解析项）
        llm_resolved: Dict[int, str] = {}
        if pronoun_map:
            try:
                llm_resolved = await self._perform_coreference_resolution(
                    index_to_pronoun=pronoun_map,
                    legitimate_participants=legitimate_participants
                )
            except Exception as e:
                logger.exception(
                    "LLM 兜底指代消解失败，跳过",
                    extra={"error": str(e), "module_name": self.CHINESE_NAME}
                )
                llm_resolved = {}

        # 应用 LLM 解析结果（原地更新）
        for idx, name in llm_resolved.items():
            if 0 <= idx < len(original_events) and isinstance(original_events[idx], dict):
                original_events[idx]["experiencer"] = name
                valid_indices.add(idx)

        # 第三步：按原始顺序保留有效事件
        filtered_events = [
            original_events[i] for i in range(len(original_events)) if i in valid_indices
        ]

        # 更新 block
        block["events"] = filtered_events

        # 清理空块
        if not filtered_events:
            block["evidence"] = [] if isinstance(block.get("evidence"), list) else []
            block["summary"] = "" if isinstance(block.get("summary"), str) else ""

        # 日志
        perception_type = step_name.replace("LLM_PERCEPTION_", "").replace("_EXTRACTION", "").lower()
        removed = len(original_events) - len(filtered_events)
        if removed > 0:
            kept_exps = [evt.get("experiencer") for evt in filtered_events if isinstance(evt, dict)]
            removed_exps = [
                original_events[i].get("experiencer")
                for i in range(len(original_events))
                if i not in valid_indices and isinstance(original_events[i], dict)
            ]
            logger.info(
                f"🧹 感知层 [{perception_type}] 过滤完成：保留 {kept_exps}，丢弃 {removed_exps}",
                extra={"module_name": self.CHINESE_NAME}
            )
        else:
            all_exps = [evt.get("experiencer") for evt in original_events if isinstance(evt, dict)]
            logger.info(
                f"✅ 感知层 [{perception_type}] 全部保留：{all_exps}",
                extra={"module_name": self.CHINESE_NAME}
            )

    def _try_simple_resolution(self, experiencer: str, legitimate_participants: Set[str]) -> Optional[str]:
        """
        尝试将代词或模糊指称解析为具体的合法参与者。

        策略：
          1. 若已是合法名 → 返回自身
          2. 若含 [uncertain] 标记 → 清理后判断
          3. 若为 EXCLUDED_PRONOUNS → 返回 None（不映射）
          4. 若为 CHINESE_PRONOUNS 且合法参与者唯一 → 映射到该唯一参与者
          5. 其他情况 → 无法解析，返回 None
        """
        logger.debug(f"→ 尝试简单指代解析: '{experiencer}'", extra={"module_name": self.CHINESE_NAME})

        if not isinstance(experiencer, str) or not legitimate_participants:
            return None

        # 已是合法参与者
        if experiencer in legitimate_participants:
            return experiencer

        # 清理可能的 uncertain 标记（兼容 LLM 输出）
        clean_exp = experiencer
        if "[uncertain]" in clean_exp:
            clean_exp = clean_exp.replace("[uncertain]", "").strip()
        if "(uncertain)" in clean_exp:
            clean_exp = clean_exp.replace("(uncertain)", "").strip()

        # 再一次判断，避免极端情况
        if clean_exp in legitimate_participants:
            logger.debug(f"← 清理后匹配合法参与者: '{clean_exp}'", extra={"module_name": self.CHINESE_NAME})
            return clean_exp

        # 明确排除的代词（如“别人”）→ 不映射
        if clean_exp in EXCLUDED_PRONOUNS:
            return None

        # 可尝试映射的代词
        if clean_exp in CHINESE_PRONOUNS:
            # 仅当合法参与者唯一时，才安全映射
            if len(legitimate_participants) == 1:
                resolved = next(iter(legitimate_participants))
                logger.debug(f"← 代词映射成功: '{experiencer}' → '{resolved}'", extra={"module_name": self.CHINESE_NAME})
                return resolved
            else:
                # 多人场景，无法确定 → 不映射
                return None

        # 非代词且非合法名 → 无法处理
        return None

    async def _perform_coreference_resolution(
            self,
            index_to_pronoun: Dict[int, str],
            legitimate_participants: Set[str]
    ) -> Dict[int, str]:
        """
        执行批量指代消解，直接调用 bottom_dissolving_pronouns。

        输入：{原始事件索引 -> 代词}
        输出：{原始事件索引 -> 确定的合法参与者名}（不确定的不返回）
        """
        logger.info(f"→ 启动 LLM 指代消解（待解析代词: {list(index_to_pronoun.values())}）",
                    extra={"module_name": self.CHINESE_NAME})

        if not index_to_pronoun or not legitimate_participants:
            return {}

        # 构造 prompt
        try:
            prompt = self.prompter._build_coref_prompt(
                user_input=self.user_input,
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
            backend = await self.backend
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

        # 底层已保证：resolved_from_llm 是合法 dict，失败时返回 {}
        # 我们只需做最终校验：key 是否在输入中，value 是否在合法名单里
        resolved_map: Dict[int, str] = {}
        for idx, name in resolved_from_llm.items():
            if isinstance(idx, int) and isinstance(name, str):
                if idx in index_to_pronoun and name in legitimate_participants:
                    resolved_map[idx] = name

        logger.info(f"← LLM 消解结果: {resolved_map}", extra={"module_name": self.CHINESE_NAME})
        return resolved_map

    @staticmethod
    def _wrap_with_context_markers(
            content: str,
            start_marker: str,
            end_marker: str,
            human_readable_name: str = ""
    ) -> str:
        """统一包装上下文片段，带可配置边界"""
        if not content.strip():
            return ""
        readable_start = f"（{human_readable_name}开始）" if human_readable_name else ""
        readable_end = f"（{human_readable_name}结束）" if human_readable_name else ""
        return (
            f"{start_marker}{readable_start}\n"
            f"{content.strip()}\n"
            f"{end_marker}{readable_end}\n"
        )
