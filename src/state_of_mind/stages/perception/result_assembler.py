from typing import Dict, Any, List, Tuple
from copy import deepcopy
import uuid
import time
from src.state_of_mind.stages.perception.executor import StepExecutor
from src.state_of_mind.stages.perception.prompt_builder import PromptBuilder
from src.state_of_mind.utils.logger import LoggerManager as logger
from .constants import (
    SEMANTIC_MODULES_L1,
    PREPROCESSING, PARALLEL, SERIAL,
    CATEGORY_SUGGESTION
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
        """基于上下文内容丰富度计算隐私等级（0.0 ~ 1.0）"""
        excluded_fields = {"user_input", "llm_model", "pre_screening", "eligibility"}
        count = sum(
            1 for k in context.keys()
            if not k.startswith("__") and k not in excluded_fields
        )
        privacy_score = count * 0.05

        # 推理层 +0.05
        if self._has_valid_inference(context.get("inference")):
            privacy_score += 0.05

        # 显式动机层 +0.05
        if self._has_valid_explicit_motivation(context.get("explicit_motivation")):
            privacy_score += 0.05

        # 合理建议层 +0.1
        if self._has_valid_rational_advice(context.get("rational_advice")):
            privacy_score += 0.1

        return min(round(privacy_score, 2), 1.0)

    @staticmethod
    def _has_valid_inference(inference) -> bool:
        if not isinstance(inference, dict):
            return False
        has_summary_evidence = (
                isinstance(inference.get("summary"), str) and inference["summary"].strip() and
                isinstance(inference.get("evidence"), list) and len(inference["evidence"]) > 0
        )
        has_events = (
                isinstance(inference.get("events"), list) and
                any(
                    isinstance(item, dict) and
                    isinstance(item.get("semantic_notation"), str) and item["semantic_notation"].strip() and
                    isinstance(item.get("evidence"), list) and len(item["evidence"]) > 0
                    for item in inference["events"]
                )
        )
        return has_summary_evidence and has_events

    @staticmethod
    def _has_valid_explicit_motivation(explicit_motivation) -> bool:
        if not isinstance(explicit_motivation, dict):
            return False
        has_summary_evidence = (
                isinstance(explicit_motivation.get("summary"), str) and explicit_motivation["summary"].strip() and
                isinstance(explicit_motivation.get("evidence"), list) and len(explicit_motivation["evidence"]) > 0
        )
        has_events = (
                isinstance(explicit_motivation.get("events"), list) and
                any(
                    isinstance(item, dict) and
                    isinstance(item.get("semantic_notation"), str) and item["semantic_notation"].strip() and
                    isinstance(item.get("evidence"), list) and len(item["evidence"]) > 0
                    for item in explicit_motivation["events"]
                )
        )
        return has_summary_evidence and has_events

    def _has_valid_rational_advice(self, rational_advice) -> bool:
        if not isinstance(rational_advice, dict):
            return False
        has_summary_evidence = (
                isinstance(rational_advice.get("summary"), str) and rational_advice["summary"].strip() and
                isinstance(rational_advice.get("evidence"), list) and len(rational_advice["evidence"]) > 0
        )
        substantive_fields = {
            "safety_first_intervention",
            "systemic_leverage_point",
            "incremental_strategy",
            "stakeholder_tradeoffs",
            "long_term_exit_path",
            "cultural_adaptation_needed",
            "fallback_plan"
        }
        has_substantive = any(
            self._is_value_effective(rational_advice.get(field))
            for field in substantive_fields
        )
        return has_summary_evidence and has_substantive

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

        for mod_name in SEMANTIC_MODULES_L1:
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

        # 1. inference
        inference = result.get("inference")
        if not isinstance(inference, dict):
            errors.append("L2: inference 缺失或非字典")
            l2_ok = False
        else:
            top_valid = (
                    isinstance(inference.get("summary"), str) and inference["summary"].strip() and
                    isinstance(inference.get("evidence"), list) and len(inference["evidence"]) > 0
            )
            event_valid = False
            events = inference.get("events")
            if isinstance(events, list):
                for item in events:
                    if isinstance(item, dict):
                        sn = item.get("semantic_notation")
                        evi = item.get("evidence")
                        if isinstance(sn, str) and sn.strip() and isinstance(evi, list) and len(evi) > 0:
                            event_valid = True
                            break
            if not (top_valid and event_valid):
                reasons = ["顶层 summary/evidence 无效"] if not top_valid else []
                if not event_valid:
                    reasons.append("events 中无 semantic_notation+evidence 有效项")
                errors.append(f"L2: inference 未同时满足双重要求 ({'; '.join(reasons)})")
                l2_ok = False

        # 2. explicit_motivation
        explicit_motivation = result.get("explicit_motivation")
        if not isinstance(explicit_motivation, dict):
            errors.append("L2: explicit_motivation 缺失或非字典")
            l2_ok = False
        else:
            top_valid = (
                    isinstance(explicit_motivation.get("summary"), str) and explicit_motivation["summary"].strip() and
                    isinstance(explicit_motivation.get("evidence"), list) and len(explicit_motivation["evidence"]) > 0
            )
            event_valid = False
            events = explicit_motivation.get("events")
            if isinstance(events, list):
                for item in events:
                    if isinstance(item, dict):
                        sn = item.get("semantic_notation")
                        evi = item.get("evidence")
                        if isinstance(sn, str) and sn.strip() and isinstance(evi, list) and len(evi) > 0:
                            event_valid = True
                            break
            if not (top_valid and event_valid):
                reasons = ["顶层 summary/evidence 无效"] if not top_valid else []
                if not event_valid:
                    reasons.append("events 中无 semantic_notation+evidence 有效项")
                errors.append(f"L2: explicit_motivation 未同时满足双重要求 ({'; '.join(reasons)})")
                l2_ok = False

        # 3. rational_advice
        rational_advice = result.get("rational_advice")
        if not isinstance(rational_advice, dict):
            errors.append("L2: rational_advice 缺失或非字典")
            l2_ok = False
        else:
            has_summary_evidence = (
                    isinstance(rational_advice.get("summary"), str) and rational_advice["summary"].strip() and
                    isinstance(rational_advice.get("evidence"), list) and len(rational_advice["evidence"]) > 0
            )
            substantive_fields = {
                "safety_first_intervention",
                "systemic_leverage_point",
                "incremental_strategy",
                "stakeholder_tradeoffs",
                "long_term_exit_path",
                "cultural_adaptation_needed",
                "fallback_plan"
            }
            has_substantive_content = any(
                rational_advice.get(field) not in (None, "", [], {}) or
                (isinstance(rational_advice.get(field), dict) and
                 any(v not in (None, "", [], {}) for v in rational_advice[field].values()))
                for field in substantive_fields
            )
            if not (has_summary_evidence and has_substantive_content):
                reasons = []
                if not has_summary_evidence:
                    reasons.append("summary 或 evidence 无效")
                if not has_substantive_content:
                    reasons.append("无实质性建议内容")
                errors.append(f"L2: rational_advice 无效 ({'; '.join(reasons)})")
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
            if prompt_type == PREPROCESSING:
                raw_response_records[PREPROCESSING].append(raw_record)
            elif prompt_type == PARALLEL:
                raw_response_records[PARALLEL].append(raw_record)
            else:
                raw_response_records[SERIAL].append(raw_record)

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
            title: str = "全息感知基底分析报告"
    ) -> None:
        result.setdefault("meta", {})["title"] = title
        result.setdefault("analysis", {})

        try:
            suggestion_prompt = self.prompt_builder.build_suggestion(
                template_name=CATEGORY_SUGGESTION,
                user_input=user_input,
                suggestion_type=suggestion_type
            )
            suggestion_content = await self.step_executor.execute_suggestion(suggestion_prompt)

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
