from typing import Dict, Any, Set, Optional, List
from .constants import EXCLUDED_PRONOUNS
from src.state_of_mind.utils.logger import LoggerManager as logger


# ----------------------------
# 🔧 独立工具函数：简单指代解析
# ----------------------------
def try_simple_resolution(experiencer: str, legitimate_participants: Set[str]) -> Optional[str]:
    """
    返回：
      - 合法名字（str）→ 保留
      - "__EXCLUDED__" → 丢弃
      - None → 需 LLM 消解
    """
    if not isinstance(experiencer, str):
        return None

    stripped = experiencer.strip()
    if not stripped:
        return None  # 视为“无主体”

    if stripped in legitimate_participants:
        return stripped

    if stripped in EXCLUDED_PRONOUNS:
        return "__EXCLUDED__"

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
                if stripped:
                    legit_set.add(stripped)
        return legit_set

    async def filter_perception_results(
        self,
        user_input: str,
        result: Dict[str, Any],
        legitimate_participants: Set[str],
        prompt_records: Dict,
        all_step_results: List[Dict],

    ) -> None:
        step_name = result.get("step_name", "unknown")
        self._log_info(f"→ 进入 {step_name} 感知结果处理流程（合法参与者: {sorted(legitimate_participants)}）")

        block = self._extract_block_from_result(result)
        if block is None:
            return

        original_events = block.get("events")
        if not isinstance(original_events, list):
            return

        self._log_info(
            f"→  {step_name} 感知结果待处理事件 experiencer 列表: {[e.get('experiencer') for e in original_events if isinstance(e, dict)]}"
        )

        keep_indices, resolve_map, discard_indices = self._scan_and_classify_events(
            original_events, legitimate_participants
        )

        llm_resolved = await self._resolve_pronouns_with_llm(
            user_input, resolve_map, legitimate_participants, prompt_records, all_step_results
        )

        # 应用 LLM 成功解析的结果
        for idx, name in llm_resolved.items():
            if 0 <= idx < len(original_events) and isinstance(original_events[idx], dict):
                original_events[idx]["experiencer"] = name
                keep_indices.add(idx)

        # 构建最终保留列表（排除 discard + 未被 LLM 解析的 resolve 项）
        filtered_events = [
            original_events[i] for i in range(len(original_events))
            if i in keep_indices and i not in discard_indices
        ]

        block["events"] = filtered_events

        # 清理空块
        if not filtered_events:
            block["evidence"] = []
            block["summary"] = ""

        self._log_filter_summary(
            result["step_name"],
            original_events,
            filtered_events,
            discard_indices,
            resolve_map,
            llm_resolved
        )

    def _log_info(self, msg: str):
        logger.info(msg, extra={"module_name": self.CHINESE_NAME})

    def _log_debug(self, msg: str):
        logger.debug(msg, extra={"module_name": self.CHINESE_NAME})

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
    ) -> tuple[Set[int], Dict[int, str], Set[int]]:
        keep_indices: Set[int] = set()
        resolve_map: Dict[int, str] = {}
        discard_indices: Set[int] = set()

        for idx, evt in enumerate(events):
            if not isinstance(evt, dict):
                # 非 dict 事件？保守丢弃或保留？这里选择丢弃以保安全
                discard_indices.add(idx)
                continue

            exp = evt.get("experiencer")

            # 情况1: 无主体（None / 空字符串）→ 保留
            if exp is None or (isinstance(exp, str) and not exp.strip()):
                keep_indices.add(idx)
                continue

            # 情况2: 是字符串 → 正常解析
            if isinstance(exp, str):
                resolved = try_simple_resolution(exp, legitimate_participants)
                if resolved is None:
                    resolve_map[idx] = exp.strip()
                elif resolved == "__EXCLUDED__":
                    discard_indices.add(idx)
                else:
                    evt["experiencer"] = resolved
                    keep_indices.add(idx)
                continue

            # 情况3: 非字符串 → 尝试安全转换
            try:
                if isinstance(exp, (int, float, bool)):
                    exp_str = str(exp).strip()
                else:
                    # 复杂类型（dict/list/object）→ 视为无效，丢弃
                    discard_indices.add(idx)
                    continue
            except Exception:
                discard_indices.add(idx)
                continue

            # 对转换后的字符串走相同逻辑
            resolved = try_simple_resolution(exp_str, legitimate_participants)
            if resolved is None:
                resolve_map[idx] = exp_str
            elif resolved == "__EXCLUDED__":
                discard_indices.add(idx)
            else:
                evt["experiencer"] = resolved
                keep_indices.add(idx)

        return keep_indices, resolve_map, discard_indices

    async def _resolve_pronouns_with_llm(
        self,
        user_input: str,
        pronoun_map: Dict[int, str],
        legitimate_participants: Set[str],
        prompt_records: Dict,
        all_step_results: List[Dict],
    ) -> Dict[int, str]:
        if not pronoun_map:
            return {}

        try:
            raw_result = await self.backend.perform_coreference_resolution(
                user_input=user_input,
                index_to_pronoun=pronoun_map,
                legitimate_participants=legitimate_participants,
                prompt_records=prompt_records,
                all_step_results=all_step_results
            )

            resolved: Dict[int, str] = {}

            if not isinstance(raw_result, dict):
                logger.warning("LLM 指代消解返回非 dict，跳过", extra={"raw": raw_result})
                return resolved

            for k, v in raw_result.items():
                # 校验 key → int
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    logger.warning(f"LLM 返回非法索引 key: {k}，跳过")
                    continue

                # 校验 value → str 且在合法列表中
                if not isinstance(v, str):
                    logger.warning(f"LLM 返回非字符串值: {v}，跳过")
                    continue

                v_clean = v.strip()
                if not v_clean or v_clean not in legitimate_participants:
                    logger.warning(f"LLM 返回非法/不在列表中的参与者: '{v_clean}'，跳过")
                    continue

                # 校验索引是否在请求范围内
                if idx not in pronoun_map:
                    logger.warning(f"LLM 返回未请求的索引 {idx}，跳过")
                    continue

                resolved[idx] = v_clean

            return resolved

        except Exception as e:
            logger.exception(
                "LLM 兜底指代消解失败，跳过",
                extra={"error": str(e), "module_name": self.CHINESE_NAME}
            )
            return {}

    def _log_filter_summary(
        self,
        step_name: str,
        original: List,
        filtered: List,
        discard_indices: Set[int],
        resolve_map: Dict[int, str],
        llm_resolved: Dict[int, str]
    ):
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

            # 可选：记录 LLM 未解析项
            unresolved = set(resolve_map.keys()) - set(llm_resolved.keys())
            if unresolved:
                self._log_info(f"❓ LLM 未能解析的指代项（已丢弃）: {[resolve_map[i] for i in unresolved]}")
        else:
            all_exps = [evt.get("experiencer") for evt in original if isinstance(evt, dict)]
            self._log_info(f"✅ 感知层 [{perception_type}] 全部保留：{all_exps}")