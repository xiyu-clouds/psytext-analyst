from typing import Dict, List, Set, Tuple, Optional, Any
from src.state_of_mind.stages.perception.prompt_builder import PromptBuilder
from .constants import (
    LLM_PARTICIPANTS_EXTRACTION, LLM_INFERENCE,
    LLM_EXPLICIT_MOTIVATION_EXTRACTION
)
from src.state_of_mind.utils.logger import LoggerManager as logger
from .participant_filter import ParticipantFilter


class ContextBuilder:
    CHINESE_NAME = "全息感知基底：通用上下文构造器"

    def __init__(
            self,
            prompt_builder: PromptBuilder,
            participant_filter: ParticipantFilter,
            step_type_to_config: Dict[str, List[Tuple]],
            top_field_to_step_types: Dict[str, List[str]]
    ):
        self.prompt_builder = prompt_builder
        self.participant_filter = participant_filter
        self._step_type_to_config = step_type_to_config
        self._top_field_to_step_types = top_field_to_step_types

    @staticmethod
    def build_user_input_context(prompt_template: str, user_input: str, context_desc_info: List[str]) -> str:
        build_context_desc = f"\n### USER_INPUT BEGIN（用户原始输入开始）\n{user_input}\n### USER_INPUT END（用户原始输入结束）\n"
        context_desc_info.append(build_context_desc)
        rendered_prompt = f"{prompt_template}{build_context_desc}"
        return rendered_prompt

    def build_parallel_context(
            self,
            step_name: str,
            context: Dict[str, Any],
            context_desc_info: List[str]
    ) -> None:
        field_config = self._step_type_to_config.get(step_name)
        if not field_config:
            return
        try:
            raw_desc = self.prompt_builder.generate_description(context=context, field_config=field_config, prefix="")
            if not raw_desc:
                return
            start_marker = end_marker = readable = ""
            if step_name == LLM_PARTICIPANTS_EXTRACTION:
                start_marker, end_marker, readable = "### PARTICIPANTS_VALID_INFORMATION BEGIN", "### PARTICIPANTS_VALID_INFORMATION END", "参与者有效信息上下文"
            elif step_name == LLM_INFERENCE:
                start_marker, end_marker, readable = "### INFERENCE_CONTEXT BEGIN", "### INFERENCE_CONTEXT END", "合理推演有效信息上下文"
            elif step_name == LLM_EXPLICIT_MOTIVATION_EXTRACTION:
                start_marker, end_marker, readable = "### EXPLICIT_MOTIVATION_CONTEXT BEGIN", "### EXPLICIT_MOTIVATION_CONTEXT END", "显性动机有效信息上下文"
            wrapped = self.wrap_with_context_markers(raw_desc, start_marker, end_marker, readable)
            context_desc_info.append(wrapped)
        except Exception as e:
            logger.error(f"[{step_name}] 动态描述生成失败: {e}")

    """批量构造全部感知数据上下文"""
    def build_serial_context_batch(self, context: Dict[str, Any]) -> str:
        excluded = {"user_input", "llm_model", "participants"}
        descriptions = []
        for key, value in context.items():
            if key.startswith("__") or key in excluded:
                continue
            step_types = self._top_field_to_step_types.get(key)
            if not step_types:
                continue
            for st in step_types:
                field_tuples = self._step_type_to_config.get(st)
                if not field_tuples:
                    continue
                try:
                    desc = self.prompt_builder.generate_description(context=context, field_config=field_tuples,
                                                                    prefix="")
                    if desc.strip():
                        descriptions.append(desc.strip())
                except Exception as e:
                    logger.error(f"生成字段 {key} 的描述失败 (step_type={st}): {e}")
        full_content = "\n".join(descriptions)
        return self.wrap_with_context_markers(
            full_content,
            "### PERCEPTUAL_CONTEXT_BATCH BEGIN",
            "### PERCEPTUAL_CONTEXT_BATCH END",
            "批量感知层上下文"
        )

    """构造合法参与者上下文"""
    def build_legitimate_participants_context(self, context: Dict[str, Any]) -> Optional[str]:
        legit_set = self.participant_filter.build_legitimate_participants_set(context)
        if not legit_set:
            return None
        sorted_entities = sorted(legit_set)
        content = "\n".join(f"- {e}" for e in sorted_entities)
        return self.wrap_with_context_markers(
            content,
            "### LEGITIMATE_PARTICIPANTS BEGIN",
            "### LEGITIMATE_PARTICIPANTS END",
            "合法参与者列表"
        )

    """统一包装上下文片段，带可配置边界"""
    @staticmethod
    def wrap_with_context_markers(content: str, start: str, end: str, readable: str) -> str:
        return f"\n{start}（{readable}开始）\n{content}\n{end}（{readable}结束）\n"

    @staticmethod
    def inject_allowed_context(prompt: str, context_desc_info: List[str], allowed_markers: Set[str]) -> str:
        for ctx_str in context_desc_info:
            if not ctx_str or not isinstance(ctx_str, str):
                continue
            if any(ctx_str.lstrip().startswith(marker) for marker in allowed_markers):
                prompt += ctx_str
        return prompt

    @staticmethod
    def update_context_from_result(
            result: Dict[str, Any],
            context: Dict[str, Any],
            step_name: str
    ) -> None:
        """
        从单个 stage 的标准化结果中提取有效数据，安全地更新共享上下文。
        """
        if not result.get("__success"):
            error_detail = (
                result.get("__system_error") or
                result.get("__api_error") or
                "Unknown error"
            )

            logger.warning(
                f"⚠️ 步骤失败，跳过更新: {step_name}", module_name=ContextBuilder.CHINESE_NAME,
                extra={"step": step_name, "error": error_detail}
            )
            return

        if not result.get("__valid_structure"):
            val_errors = result.get("__validation_errors")
            logger.warning(
                f"当前步骤 {step_name} 结构校验失败", module_name=ContextBuilder.CHINESE_NAME,
                extra={"step": step_name, "error": val_errors}
            )
            return

        data = result.get("data")
        if data and isinstance(data, dict):
            clean_data = {k: v for k, v in data.items() if not k.startswith("__")}
            if clean_data:
                context.update(clean_data)
                logger.info(
                    "🟢 成功注入上下文字段", module_name=ContextBuilder.CHINESE_NAME,
                    extra={"step": step_name, "keys": list(clean_data.keys())}
                )
        else:
            logger.info(f"⚪ 跳过上下文注入：步骤 {step_name} 未返回有效数据", module_name=ContextBuilder.CHINESE_NAME,)
