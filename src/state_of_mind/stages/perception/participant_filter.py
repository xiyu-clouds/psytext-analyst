from typing import Dict, Any, Set, Optional, List
from .constants import EXCLUDED_PRONOUNS, CHINESE_PRONOUNS, PERCEPTION_LAYERS
from src.state_of_mind.utils.logger import LoggerManager as logger


# ----------------------------
# 🔧 独立工具函数：简单指代解析
# ----------------------------
def try_simple_resolution(experiencer: str, legitimate_participants: Set[str]) -> Optional[str]:
    """
    尝试将代词或模糊指称解析为具体的合法参与者。
    不依赖任何类状态，纯函数，便于测试。
    """
    if not isinstance(experiencer, str) or not legitimate_participants:
        return None

    # 已是合法参与者
    if experiencer in legitimate_participants:
        return experiencer

    # 清理 uncertain 标记
    clean_exp = experiencer
    for marker in ["[uncertain]", "(uncertain)"]:
        if marker in clean_exp:
            clean_exp = clean_exp.replace(marker, "").strip()

    if clean_exp in legitimate_participants:
        return clean_exp

    # 明确排除的代词
    if clean_exp in EXCLUDED_PRONOUNS:
        return None

    # 可尝试映射的中文代词（仅当唯一合法参与者时）
    if clean_exp in CHINESE_PRONOUNS and len(legitimate_participants) == 1:
        return next(iter(legitimate_participants))

    return None


class ParticipantFilter:
    CHINESE_NAME = "全息感知基底：感知数据基于合法参与者过滤处理"

    def __init__(self, prompt_builder, backend):
        self.prompt_builder = prompt_builder
        self.backend = backend

    @staticmethod
    def build_legitimate_participants_set(context: Dict[str, Any]) -> Set[str]:
        legit_set = set()
        participants = context.get("participants")
        if not isinstance(participants, list):
            return legit_set

        for p in participants:
            if isinstance(p, dict) and "entity" in p and isinstance(p["entity"], str):
                stripped = p["entity"].strip()
                if stripped:  # 忽略空字符串
                    legit_set.add(stripped)
        return legit_set

    async def filter_perception_results(
        self,
        user_input: str,
        result: Dict[str, Any],
        legitimate_participants: Set[str]
    ) -> None:
        """主入口：过滤感知结果中的非法 experiencer"""
        self._log_info(f"→ 进入感知结果过滤流程（合法参与者: {sorted(legitimate_participants)}）")

        if not self._is_valid_perception_result(result):
            return

        step_name = result["step_name"]
        block = self._extract_block_from_result(result)
        if block is None:
            return

        original_events = block.get("events")
        if not isinstance(original_events, list) or not original_events:
            return

        self._log_info(
            f"→ 待处理事件 experiencer 列表: {[e.get('experiencer') for e in original_events if isinstance(e, dict)]}"
        )

        # 扫描并分类
        valid_indices, pronoun_map = self._scan_and_classify_events(original_events, legitimate_participants)

        # LLM 兜底消解
        llm_resolved = await self._resolve_pronouns_with_llm(user_input, pronoun_map, legitimate_participants)

        # 应用 LLM 结果
        for idx, name in llm_resolved.items():
            if 0 <= idx < len(original_events) and isinstance(original_events[idx], dict):
                original_events[idx]["experiencer"] = name
                valid_indices.add(idx)

        # 保留有效事件
        filtered_events = [original_events[i] for i in range(len(original_events)) if i in valid_indices]
        block["events"] = filtered_events

        # 清理空块
        if not filtered_events:
            block["evidence"] = [] if isinstance(block.get("evidence"), list) else []
            block["summary"] = "" if isinstance(block.get("summary"), str) else ""

        # 日志总结
        self._log_filter_summary(step_name, original_events, filtered_events)

    def _log_info(self, msg: str):
        logger.info(msg, extra={"module_name": self.CHINESE_NAME})

    def _log_debug(self, msg: str):
        logger.debug(msg, extra={"module_name": self.CHINESE_NAME})

    @staticmethod
    def _is_valid_perception_result(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        step_name = result.get("step_name")
        return step_name in PERCEPTION_LAYERS

    @staticmethod
    def _extract_block_from_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = result.get("data")
        if not isinstance(data, dict) or not data:
            return None
        try:
            _, block = next(iter(data.items()))
        except StopIteration:
            return None
        if not (isinstance(block, dict) and isinstance(block.get("events"), list)):
            return None
        return block

    @staticmethod
    def _scan_and_classify_events(
        events: List[Dict],
        legitimate_participants: Set[str]
    ) -> tuple[Set[int], Dict[int, str]]:
        valid_indices: Set[int] = set()
        pronoun_map: Dict[int, str] = {}

        for idx, evt in enumerate(events):
            if not isinstance(evt, dict):
                continue
            exp = evt.get("experiencer")
            if not isinstance(exp, str):
                continue

            if exp in legitimate_participants:
                valid_indices.add(idx)
                continue

            resolved = try_simple_resolution(exp, legitimate_participants)
            if resolved is not None:
                evt["experiencer"] = resolved
                valid_indices.add(idx)
                continue

            pronoun_map[idx] = exp

        return valid_indices, pronoun_map

    async def _resolve_pronouns_with_llm(
        self,
        user_input: str,
        pronoun_map: Dict[int, str],
        legitimate_participants: Set[str]
    ) -> Dict[int, str]:
        if not pronoun_map:
            return {}

        try:
            return await self.backend.perform_coreference_resolution(
                user_input=user_input,
                index_to_pronoun=pronoun_map,
                legitimate_participants=legitimate_participants
            )
        except Exception as e:
            logger.exception(
                "LLM 兜底指代消解失败，跳过",
                extra={"error": str(e), "module_name": self.CHINESE_NAME}
            )
            return {}

    def _log_filter_summary(self, step_name: str, original: List, filtered: List):
        perception_type = (
            step_name
            .replace("LLM_PERCEPTION_", "")
            .replace("_EXTRACTION", "")
            .lower()
        )
        removed = len(original) - len(filtered)
        if removed > 0:
            kept_exps = [evt.get("experiencer") for evt in filtered if isinstance(evt, dict)]
            removed_exps = [
                original[i].get("experiencer")
                for i in range(len(original))
                if i not in {j for j, _ in enumerate(filtered)} and isinstance(original[i], dict)
            ]
            self._log_info(f"🧹 感知层 [{perception_type}] 过滤完成：保留 {kept_exps}，丢弃 {removed_exps}")
        else:
            all_exps = [evt.get("experiencer") for evt in original if isinstance(evt, dict)]
            self._log_info(f"✅ 感知层 [{perception_type}] 全部保留：{all_exps}")