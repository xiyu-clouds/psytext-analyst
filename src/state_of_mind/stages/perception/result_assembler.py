import re
from typing import Dict, Any, List, Tuple, Set, Optional
from copy import deepcopy
import uuid
import time
from src.state_of_mind.stages.perception.executor import StepExecutor
from src.state_of_mind.stages.perception.prompt_builder import PromptBuilder
from src.state_of_mind.utils.logger import LoggerManager as logger
from .constants import (
    CATEGORY_SUGGESTION, ALL_STEPS_FOR_FRONTEND, MENTION_TYPES_CONFIG, PARALLEL_PERCEPTION, PARALLEL_PREPROCESSING,
    PARALLEL_HIGH_ORDER, SERIAL_SUGGESTION, PARALLEL_PERCEPTION_KEYS, SERIAL_SUGGESTION_KEYS, PARALLEL_HIGH_ORDER_KEYS,
    PARALLEL_PREPROCESSING_KEYS, OTHER
)
from ...config import config


class ResultAssembler:
    CHINESE_NAME = "全息感知基底：组装数据+校验数据有效级别"

    def __init__(
            self,
            llm_model: str,
            prompt_builder: PromptBuilder,
            step_executor: StepExecutor,
    ):
        self.llm_model = llm_model
        self.prompt_builder = prompt_builder
        self.step_executor = step_executor

    def assemble_final_data(
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

        excluded_fields = {"user_input", "llm_model", "pre_screening", "eligibility"}

        # 第一步：注入非排除字段
        for key, value in context.items():
            if key.startswith("__") or key in excluded_fields:
                continue
            result[key] = value

        # 第二步：处理 participants（直接操作 result）
        self._process_participants_in_place(result)

        # 第三步：隐私度计算
        privacy_level = self._calculate_privacy_level(context)
        self._inject_privacy_level_into_meta(result, privacy_level)

        # 第四步：清理 result 中“全空”的顶层字段
        self._prune_ineffective_top_level_fields(result)

        return result

    @staticmethod
    def _process_participants_in_place(result: Dict[str, Any]) -> None:
        """为 participants 中每个实体生成唯一 entity_id"""
        participants = result.get("participants")
        if not isinstance(participants, list) or not participants:
            return

        processed = []
        for p in participants:
            if isinstance(p, dict) and "entity" in p and isinstance(p["entity"], str):
                unique_suffix = uuid.uuid4().hex[:8]
                p_new = deepcopy(p)
                p_new["entity_id"] = f"{p['entity']}_{unique_suffix}"
                processed.append(p_new)
            else:
                processed.append(deepcopy(p))
        result["participants"] = processed

    def _calculate_privacy_level(self, context: Dict[str, Any]) -> float:
        score = 0.0

        # ─── 1. 预处理：仅 "participants" 有效（+0.06）───────────────
        if "participants" in PARALLEL_PREPROCESSING_KEYS:
            data = context.get("participants")
            if data and self._is_valid_participants(data):
                score += 0.06

        # ─── 2. 感知层：12 步，每有效一步 +0.04 ─────────────────────
        for key in PARALLEL_PERCEPTION_KEYS:
            data = context.get(key)
            if data and self._is_valid_perception_module(data):
                score += 0.04

        # ─── 3. 高阶层（策略、矛盾、操控）：3 步，每步 +0.11 ─────────
        for key in PARALLEL_HIGH_ORDER_KEYS:
            data = context.get(key)
            if data and self._is_valid_high_order_module(data):
                score += 0.11

        # ─── 4. 建议层（单独在 SERIAL_SUGGESTION）：1 步，+0.11 ──────
        for key in SERIAL_SUGGESTION_KEYS:  # 通常只有一个
            data = context.get(key)
            if data and self._is_valid_suggestion_module(data):
                score += 0.11
                break  # 防止多个（但一般就一个）

        return min(round(score, 2), 1.0)

    @staticmethod
    def _is_valid_participants(data: Any) -> bool:
        if not isinstance(data, list) or not data:
            return False
        return any(
            isinstance(item, dict) and
            isinstance(item.get("entity"), str) and
            item["entity"].strip()
            for item in data
        )

    @staticmethod
    def _is_valid_perception_module(data: Any) -> bool:
        # 感知模块根必须是 dict（如 temporal, spatial, emotional 等）
        if not isinstance(data, dict):
            return False

        events = data.get("events")
        if not isinstance(events, list) or not events:
            return False

        # 至少一个 event 有有效的 semantic_notation + evidence
        return any(
            isinstance(ev, dict) and
            isinstance(ev.get("semantic_notation"), str) and ev["semantic_notation"].strip() and
            isinstance(ev.get("evidence"), list) and len(ev["evidence"]) > 0
            for ev in events
        )

    @staticmethod
    def _is_valid_high_order_module(data: Any) -> bool:
        """用于策略、矛盾、操控"""
        if not isinstance(data, dict):
            return False
        synthesis_ok = isinstance(data.get("synthesis"), str) and data["synthesis"].strip()
        evidence_ok = isinstance(data.get("evidence"), list) and len(data["evidence"]) > 0
        events_ok = isinstance(data.get("events"), list) and any(
            isinstance(ev, dict) and
            isinstance(ev.get("semantic_notation"), str) and ev["semantic_notation"].strip() and
            isinstance(ev.get("evidence"), list) and len(ev["evidence"]) > 0
            for ev in data["events"]
        )
        return synthesis_ok and evidence_ok and events_ok

    @staticmethod
    def _is_valid_suggestion_module(data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        synthesis_ok = isinstance(data.get("synthesis"), str) and data["synthesis"].strip()
        evidence_ok = isinstance(data.get("evidence"), list) and len(data["evidence"]) > 0
        if not (synthesis_ok and evidence_ok):
            return False

        events = data.get("events")
        if not isinstance(events, list):
            return False

        for ev in events:
            if not isinstance(ev, dict):
                continue
            counter = ev.get("counter_action")
            target = ev.get("targeted_mechanism")
            disrupt = ev.get("expected_disruption")
            sn = ev.get("semantic_notation")
            evi = ev.get("evidence")

            if (
                    isinstance(sn, str) and sn.strip() and
                    isinstance(evi, list) and len(evi) > 0 and
                    isinstance(counter, str) and counter.strip() and
                    isinstance(target, str) and target.strip() and
                    isinstance(disrupt, str) and disrupt.strip()
            ):
                return True
        return False

    @staticmethod
    def _inject_privacy_level_into_meta(result: Dict[str, Any], privacy_level: float) -> None:
        meta = result.setdefault("meta", {})
        privacy_scope = meta.setdefault("privacy_scope", {})
        privacy_scope["privacy_level"] = float(privacy_level)

    def _prune_ineffective_top_level_fields(self, result: Dict[str, Any]) -> None:
        keys_to_remove = [
            key for key, value in result.items()
            if not self._is_value_effective(value)
        ]
        for key in keys_to_remove:
            del result[key]

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

    # ======================
    # 校验逻辑（L0/L1/L2）
    # ======================

    @staticmethod
    def _validate_l0(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
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
        errors = []
        l1_valid = False
        present_but_empty = []
        missing_or_invalid = []

        for mod_name in PARALLEL_PERCEPTION_KEYS:
            mod = result.get(mod_name)
            if mod is None:
                missing_or_invalid.append(f"{mod_name} (缺失)")
                continue

            if isinstance(mod, dict):
                top_summary = mod.get("summary")
                top_evidence = mod.get("evidence")
                top_valid = (
                        isinstance(top_summary, str) and top_summary.strip() and
                        isinstance(top_evidence, list) and len(top_evidence) > 0
                )

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
        errors = []
        l2_ok = True

        def is_valid_module(data: Optional[Dict], module_name: str,
                            require_core_fields_in_events: bool = False) -> bool:
            """
            通用校验函数：
            - 要求 data 是 dict
            - synthesis 非空 str
            - evidence 非空 list
            - events 中至少一个 item 同时有 non-empty semantic_notation + non-empty evidence
            - 若 require_core_fields_in_events=True，则事件还需包含特定核心字段（用于 minimal_viable_advice）
            """
            if not isinstance(data, dict):
                errors.append(f"L2: {module_name} 缺失或非字典")
                return False

            # 顶层 synthesis + evidence
            has_synthesis = isinstance(data.get("synthesis"), str) and data["synthesis"].strip()
            has_global_evidence = isinstance(data.get("evidence"), list) and len(data["evidence"]) > 0
            if not (has_synthesis and has_global_evidence):
                reasons = []
                if not has_synthesis:
                    reasons.append("synthesis 无效")
                if not has_global_evidence:
                    reasons.append("全局 evidence 为空")
                errors.append(f"L2: {module_name} 顶层字段无效 ({'; '.join(reasons)})")
                return False

            # events 中至少一个有效事件
            events = data.get("events")
            if not isinstance(events, list):
                errors.append(f"L2: {module_name}.events 非列表")
                return False

            valid_event_found = False
            for item in events:
                if not isinstance(item, dict):
                    continue
                sn = item.get("semantic_notation")
                evi = item.get("evidence")
                has_sn = isinstance(sn, str) and sn.strip()
                has_evi = isinstance(evi, list) and len(evi) > 0

                if not (has_sn and has_evi):
                    continue

                # 针对 minimal_viable_advice 的额外字段校验
                if require_core_fields_in_events:
                    counter = item.get("counter_action")
                    target = item.get("targeted_mechanism")
                    disrupt = item.get("expected_disruption")
                    if not (
                            isinstance(counter, str) and counter.strip() and
                            isinstance(target, str) and target.strip() and
                            isinstance(disrupt, str) and disrupt.strip()
                    ):
                        continue  # 此事件不满足建议的核心三要素

                valid_event_found = True
                break

            if not valid_event_found:
                errors.append(f"L2: {module_name} 无有效事件（需 semantic_notation + evidence，建议类还需核心三字段）")
                return False

            return True

        # ──────────────── 校验前三项：策略 / 矛盾 / 操控 ────────────────
        strategy_ok = is_valid_module(result.get("strategy_anchor"), "strategy_anchor")
        contradiction_ok = is_valid_module(result.get("contradiction_map"), "contradiction_map")
        manipulation_ok = is_valid_module(result.get("manipulation_decode"), "manipulation_decode")

        at_least_one_of_first_three = strategy_ok or contradiction_ok or manipulation_ok
        if not at_least_one_of_first_three:
            # 清理前面可能累积的重复错误（可选），这里保留全部
            errors.append("L2: 策略锚定、矛盾暴露、操控机制解码三者均无效（至少需一项有效）")
            l2_ok = False

        # ──────────────── 校验第四项：最小可行性建议（必须有效）────────────────
        advice_ok = is_valid_module(
            result.get("minimal_viable_advice"),
            "minimal_viable_advice",
            require_core_fields_in_events=True  # 启用建议类特殊校验
        )
        if not advice_ok:
            l2_ok = False

        return l2_ok, errors

    def validate_final_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        l0_valid, errors_l0 = self._validate_l0(result)
        if not l0_valid:
            return {
                "__success": False,
                "__validity_level": "invalid",
                "__final_validation_errors": {"L0": errors_l0, "L1": [], "L2": []}
            }

        l1_valid, errors_l1 = self._validate_l1(result)
        l2_valid, errors_l2 = self._validate_l2(result) if l1_valid else (False, [])

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

    # ======================
    # 聚合 & 建议注入
    # ======================
    @staticmethod
    def aggregate_step_results(
            all_step_results: List[Dict[str, Any]],
            raw_response_records: Dict[str, List]
    ) -> Dict[str, Any]:
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

            if sys_err := step.get("__system_error"):
                system_errors.append({"step": step.get("step_name"), "error": sys_err})
            if api_err := step.get("__api_error"):
                api_errors.append({"step": step.get("step_name"), "error": api_err})
            if val_errs := step.get("__validation_errors"):
                validation_errors_all.append({"step": step.get("step_name"), "errors": val_errs})

            prompt_type = step.get("prompt_type")
            raw_record = {"step_name": step.get("step_name"), "raw_response": step.get("__raw_response")}
            if prompt_type == PARALLEL_PREPROCESSING:
                raw_response_records[PARALLEL_PREPROCESSING].append(raw_record)
            elif prompt_type == PARALLEL_PERCEPTION:
                raw_response_records[PARALLEL_PERCEPTION].append(raw_record)
            elif prompt_type == PARALLEL_HIGH_ORDER:
                raw_response_records[PARALLEL_HIGH_ORDER].append(raw_record)
            elif prompt_type == SERIAL_SUGGESTION:
                raw_response_records[SERIAL_SUGGESTION].append(raw_record)
            else:
                raw_response_records[OTHER].append(raw_record)

        return {
            "__valid_structure": all_valid,
            "__partial_success": partial_success,
            "__errors_summary": {
                "system_errors": system_errors,
                "api_errors": api_errors,
                "validation_errors": validation_errors_all,
                "final_validation_errors": []  # 由外部填入
            }
        }

    async def inject_suggestion_into_result(
            self,
            result: Dict[str, Any],
            user_input: str,
            suggestion_type: str,
            all_step_results: List[Dict],
            prompt_records: Dict,
            title: str = "全息感知基底分析报告",
    ) -> None:
        result.setdefault("meta", {})["title"] = title
        result.setdefault("analysis", {})

        try:
            suggestion_prompt = self.prompt_builder.build_suggestion(
                template_name=CATEGORY_SUGGESTION,
                user_input=user_input,
                suggestion_type=suggestion_type
            )
            step_name = "suggestion_generation"
            prompt_type = "suggestion"
            prompt_records.setdefault(OTHER, []).append({
                "step_name": step_name,
                "prompt": suggestion_prompt
            })
            suggestion_content = await self.step_executor.execute_suggestion(suggestion_prompt, step_name, prompt_type, all_step_results)
            suggestion_record = {
                "content": suggestion_content,
                "type": suggestion_type,
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
                extra={"suggestion_type": suggestion_type, "error": str(e)}
            )
            result["analysis"]["suggestion"] = {
                "content": error_msg,
                "type": suggestion_type,
                "generated_at": int(time.time()),
                "success": False
            }

    @staticmethod
    async def inject_watermark_into_result(
            result: Dict[str, Any],
            watermark_data: Dict[str, Any] = None
    ) -> Any:
        """
        将水印配置注入到 result 中，供前端模板渲染使用。
        """
        if not config.WATERMARK_ENABLED:
            return

        # 组装水印配置字典
        watermark_config = {
            "enabled": True,
            "text": config.WATERMARK_TEXT,
            "color": config.WATERMARK_COLOR,
            "opacity": config.WATERMARK_OPACITY,
            "fontSize": config.WATERMARK_FONT_SIZE,
            "angle": config.WATERMARK_ANGLE,
            "cols": config.WATERMARK_SPACING_COLS,
            "rows": config.WATERMARK_SPACING_ROWS,
            "padding": config.WATERMARK_PADDING,
        }
        result["watermark"] = watermark_config

    async def inject_global_semantic_signature(self, result: Dict[str, Any], user_input: str, all_step_results: List[Dict], prompt_records: Dict,):
        result.setdefault("meta", {})["global_semantic_signature"] = ""

        try:
            prompt = self.prompt_builder.build_global_signature_prompt(user_input=user_input)
            step_name = "global_semantic_signature"
            prompt_type = "semantic_signature"
            prompt_records.setdefault(OTHER, []).append({
                "step_name": step_name,
                "prompt": prompt
            })
            raw_output = await self.step_executor.execute_global_signature(prompt, step_name, prompt_type, all_step_results)

            cleaned = self._sanitize_global_signature(raw_output)
            if not cleaned:
                cleaned = "raw_complex_invalid_or_empty_signature"

            result["meta"]["global_semantic_signature"] = cleaned
            logger.info(
                "✅ 全局语义标识生成成功",
                extra={"model": self.llm_model, "signature_length": len(cleaned)}
            )
        except Exception as e:
            error_msg = f"全局语义标识生成失败: {str(e)}"
            logger.exception(error_msg, extra={"model": self.llm_model})
            result["meta"]["global_semantic_signature"] = "raw_complex_generation_failed"

    @staticmethod
    def _sanitize_global_signature(raw: str) -> str:
        if not raw or not isinstance(raw, str):
            return ""
        line = raw.strip().split("\n")[0].strip().lower()
        if not line.startswith("raw_") or len(line) > 256:
            return ""
        if not re.fullmatch(r"[a-z0-9_]+", line):
            return ""
        return line

    # ======================
    # 预处理注入html模板数据
    # ======================
    async def preprocess_for_html_rendering(self, result: Dict[str, Any]) -> None:
        """
        动态调度预处理：
        1. 从 ALL_STEPS_FOR_FRONTEND 获取所有非 preprocessing 步骤的 driven_by（即顶级字段名）
        2. 对 result 中存在的字段，按 naming convention 自动调用 _preprocess_{key}
        """
        candidate_keys = {
            step["driven_by"]
            for step in ALL_STEPS_FOR_FRONTEND
            if step.get("type") == PARALLEL_PERCEPTION
        }

        for key in candidate_keys:
            if not (key in result and result[key]):
                continue

            method_name = f"_preprocess_{key}"
            preprocessor = getattr(self, method_name, None)

            if preprocessor is None:
                logger.debug(f"ℹ️ 无预处理函数: {method_name}，跳过")
                continue

            try:
                preprocessor(result)
            except Exception as e:
                logger.warning(
                    f"⚠️ HTML 预处理失败: {key}",
                    extra={"error": str(e)}
                )

    def _preprocess_temporal(self, result: Dict[str, Any]) -> None:
        """
        预处理 temporal 数据，为 HTML 模板生成按类型分组的时间短语列表。
        输入：result（含 result["temporal"]）
        副作用：在每个 event 中注入 temporal_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "temporal", MENTION_TYPES_CONFIG["temporal"])

    def _preprocess_spatial(self, result: Dict[str, Any]) -> None:
        """
        预处理 spatial 数据，为 HTML 模板生成按类型分组的空间短语列表。
        输入：result（含 result["spatial"]）
        副作用：在每个 event 中注入 spatial_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "spatial", MENTION_TYPES_CONFIG["spatial"])

    def _preprocess_visual(self, result: Dict[str, Any]) -> None:
        """
        预处理 visual 数据，为 HTML 模板生成按类型分组的视觉短语列表。
        输入：result（含 result["visual"]）
        副作用：在每个 event 中注入 visual_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "visual", MENTION_TYPES_CONFIG["visual"])

    def _preprocess_auditory(self, result: Dict[str, Any]) -> None:
        """
        预处理 auditory 数据，为 HTML 模板生成按类型分组的听觉短语列表。
        输入：result（含 result["auditory"]）
        副作用：在每个 event 中注入 auditory_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "auditory", MENTION_TYPES_CONFIG["auditory"])

    def _preprocess_olfactory(self, result: Dict[str, Any]) -> None:
        """
        预处理 olfactory 数据，为 HTML 模板生成按类型分组的嗅觉短语列表。
        输入：result（含 result["olfactory"]）
        副作用：在每个 event 中注入 olfactory_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "olfactory", MENTION_TYPES_CONFIG["olfactory"])

    def _preprocess_tactile(self, result: Dict[str, Any]) -> None:
        """
        预处理 tactile 数据，为 HTML 模板生成按类型分组的触觉短语列表。
        输入：result（含 result["tactile"]）
        副作用：在每个 event 中注入 tactile_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "tactile", MENTION_TYPES_CONFIG["tactile"])

    def _preprocess_gustatory(self, result: Dict[str, Any]) -> None:
        """
        预处理 gustatory 数据，为 HTML 模板生成按类型分组的味觉短语列表。
        输入：result（含 result["gustatory"]）
        副作用：在每个 event 中注入 gustatory_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "gustatory", MENTION_TYPES_CONFIG["gustatory"])

    def _preprocess_interoceptive(self, result: Dict[str, Any]) -> None:
        """
        预处理 interoceptive 数据，为 HTML 模板生成按类型分组的内感受短语列表。
        输入：result（含 result["interoceptive"]）
        副作用：在每个 event 中注入 interoceptive_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "interoceptive", MENTION_TYPES_CONFIG["interoceptive"])

    def _preprocess_cognitive(self, result: Dict[str, Any]) -> None:
        """
        预处理 cognitive 数据，为 HTML 模板生成按类型分组的认知短语列表。
        输入：result（含 result["cognitive"]）
        副作用：在每个 event 中注入 cognitive_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "cognitive", MENTION_TYPES_CONFIG["cognitive"])

    def _preprocess_bodily(self, result: Dict[str, Any]) -> None:
        """
        预处理 bodily 数据，为 HTML 模板生成按类型分组的躯体化表现短语列表。
        输入：result（含 result["bodily"]）
        副作用：在每个 event 中注入 bodily_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "bodily", MENTION_TYPES_CONFIG["bodily"])

    def _preprocess_emotional(self, result: Dict[str, Any]) -> None:
        """
        预处理 emotional 数据，为 HTML 模板生成按类型分组的情感短语列表。
        输入：result（含 result["emotional"]）
        副作用：在每个 event 中注入 emotional_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "emotional", MENTION_TYPES_CONFIG["emotional"])

    def _preprocess_social_relation(self, result: Dict[str, Any]) -> None:
        """
        预处理 social_relation 数据，为 HTML 模板生成按类型分组的社会关系短语列表。
        输入：result（含 result["social_relation"]）
        副作用：在每个 event 中注入 social_relation_mentions_by_type 字段
        """
        self._preprocess_generic_mentions(result, "social_relation", MENTION_TYPES_CONFIG["social_relation"])

    @staticmethod
    def _preprocess_generic_mentions(
            result: Dict[str, Any],
            top_key: str,
            valid_types: Set[str]
    ) -> None:
        """
        基于 naming convention 的通用 mentions 预处理器。

        约定：
          - 输入 mentions 字段名：{top_key}_mentions
          - 输出分组字段名：{top_key}_mentions_by_type
        """
        root_obj = result.get(top_key)
        if not isinstance(root_obj, dict):
            return

        events = root_obj.get("events")
        if not isinstance(events, list):
            return

        mentions_key = f"{top_key}_mentions"
        output_key = f"{top_key}_mentions_by_type"

        for event in events:
            if not isinstance(event, dict):
                continue

            mentions = event.get(mentions_key)
            if not isinstance(mentions, list):
                continue

            grouped = {t: [] for t in valid_types}
            for item in mentions:
                if not isinstance(item, dict):
                    continue
                phrase = item.get("phrase")
                itype = item.get("type")
                if isinstance(phrase, str) and itype in valid_types:
                    grouped[itype].append(phrase)

            event[output_key] = grouped
