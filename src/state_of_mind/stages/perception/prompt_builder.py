"""负责动态渲染"""
import json
from typing import Any, Dict, List, Tuple, Optional, Set
from src.state_of_mind.prompt_templates.prompt_templates import LLM_PROMPTS_SCHEMA
from src.state_of_mind.stages.perception.constants import get_effective_policy, \
    render_iron_law_from_policy, COREFERENCE_RESOLUTION_BATCH, CATEGORY_SUGGESTION, \
    ALL_STEPS_FOR_FRONTEND, PERCEPTION_LAYERS, CATEGORY_RAW, \
    GLOBAL_SEMANTIC_SIGNATURE, PARALLEL_PREPROCESSING_STEPS, PARALLEL_PREPROCESSING, PARALLEL_PERCEPTION, \
    PARALLEL_PERCEPTION_STEPS, SERIAL_SUGGESTION_STEPS, PARALLEL_HIGH_ORDER_STEPS, SERIAL_SUGGESTION, \
    PARALLEL_HIGH_ORDER, PARALLEL_PERCEPTION_KEYS, PARALLEL_HIGH_ORDER_KEYS, SERIAL_SUGGESTION_KEYS, \
    PARALLEL_PREPROCESSING_KEYS
# from src.state_of_mind.utils.ip_timezone import IPBasedTimezoneResolver
from src.state_of_mind.utils.logger import LoggerManager as logger
# from src.state_of_mind.utils.network import get_public_ip


class PromptBuilder:
    """
    Prompt 构造器
    """
    CHINESE_NAME = "Prompt构造器"

    def build_raw(self) -> Dict[str, Any]:
        return {
            "preprocessing_prompts": self._build_step_prompts(
                list(PARALLEL_PREPROCESSING_STEPS.values()), PARALLEL_PREPROCESSING
            ),
            "perception_prompts": self._build_step_prompts(
                list(PARALLEL_PERCEPTION_STEPS.values()), PARALLEL_PERCEPTION
            ),
            "high_order_prompts": self._build_step_prompts(
                list(PARALLEL_HIGH_ORDER_STEPS.values()), PARALLEL_HIGH_ORDER
            ),
            "suggestion_prompts": self._build_step_prompts(
                list(SERIAL_SUGGESTION_STEPS.values()), SERIAL_SUGGESTION
            ),
        }

    def build_suggestion(self, template_name: str, user_input: str, suggestion_type: str) -> str:
        logger.info("🔄 开始构建 build_suggestion Prompt", module_name=self.CHINESE_NAME)

        suggestion_schema = LLM_PROMPTS_SCHEMA.get(template_name)
        if not suggestion_schema:
            error_msg = f"模板未定义: {template_name}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        valid_types = LLM_PROMPTS_SCHEMA[CATEGORY_SUGGESTION].keys()
        if suggestion_type not in valid_types:
            error_msg = f"不支持的建议类型: '{suggestion_type}'。可用类型: {sorted(valid_types)}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        prompt_template = suggestion_schema.get(suggestion_type)
        if not prompt_template:
            error_msg = f"模板 '{template_name}' 中缺少建议类型 '{suggestion_type}' 的定义"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        try:
            final_prompt = prompt_template.format(user_input=user_input)
        except KeyError as e:
            error_msg = f"模板中包含未提供的字段: {e}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"模板渲染失败: {e}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        logger.info("✅ build_suggestion Prompt 构建成功", module_name=self.CHINESE_NAME)
        return final_prompt

    def build_global_signature_prompt(self, user_input):
        prompt_template = LLM_PROMPTS_SCHEMA.get(GLOBAL_SEMANTIC_SIGNATURE)
        if not prompt_template:
            error_msg = f"模板中缺少全局语义标识的 prompt 定义"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        try:
            final_prompt = prompt_template.format(user_input=user_input)
        except Exception as e:
            error_msg = f"模板渲染失败: {e}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        logger.info("✅ build_global_signature_prompt Prompt 构建成功", module_name=self.CHINESE_NAME)
        return final_prompt

    @staticmethod
    def build_coref_prompt(
            user_input: str,
            legitimate_participants: Set[str],
            index_to_pronoun: Dict[int, str]
    ) -> str:
        """
        构造指代消解 prompt。
        :param user_input:
        :param legitimate_participants:
        :param index_to_pronoun: {0: "他", 2: "她", ...} —— 原始事件中的索引到代词映射
        """
        participant_list_str = "\n".join(f"- {p}" for p in sorted(legitimate_participants))

        pronoun_lines = []
        for idx in sorted(index_to_pronoun.keys()):  # 按索引排序，便于阅读
            pronoun_lines.append(f"{idx} -> “{index_to_pronoun[idx]}”")
        pronoun_mapping_str = "\n".join(pronoun_lines)

        template = LLM_PROMPTS_SCHEMA[COREFERENCE_RESOLUTION_BATCH]
        return template.format(
            user_input=user_input,
            participant_list_str=participant_list_str,
            pronoun_mapping_str=pronoun_mapping_str
        )

    def pre_basic_data(self):
        raw_schema = LLM_PROMPTS_SCHEMA.get(CATEGORY_RAW)
        if not raw_schema:
            error_msg = f"模板未定义: {CATEGORY_RAW}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        pipeline = raw_schema.get("pipeline")
        if not isinstance(pipeline, list):
            error_msg = f"配置错误: {CATEGORY_RAW}.pipeline 必须是列表，当前值: {repr(pipeline)}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        if len(pipeline) == 0:
            logger.warning(f"⚠️ 警告: {CATEGORY_RAW}.pipeline 为空列表！将导致前端步骤为空！", module_name=self.CHINESE_NAME)
            raise ValueError("pipeline 不能为空")

        pipeline = raw_schema.get("pipeline")
        self._split_pipeline(pipeline)

    @staticmethod
    def _split_pipeline(pipeline: List[Dict]) -> None:
        """
        分离 pipeline 中的预处理、并行、串行任务
        """
        # 各类型步骤数据
        PARALLEL_PREPROCESSING_STEPS.clear()
        PARALLEL_PERCEPTION_STEPS.clear()
        PARALLEL_HIGH_ORDER_STEPS.clear()
        SERIAL_SUGGESTION_STEPS.clear()
        # 感知类型步骤名
        PERCEPTION_LAYERS.clear()
        # 各类型顶级键
        PARALLEL_PREPROCESSING_KEYS.clear()
        PARALLEL_PERCEPTION_KEYS.clear()
        PARALLEL_HIGH_ORDER_KEYS.clear()
        SERIAL_SUGGESTION_KEYS.clear()
        # 全部步骤相关数据
        ALL_STEPS_FOR_FRONTEND.clear()

        valid_types = {
            PARALLEL_PREPROCESSING,
            PARALLEL_PERCEPTION,
            PARALLEL_HIGH_ORDER,
            SERIAL_SUGGESTION
        }

        for idx, step in enumerate(pipeline):
            if not isinstance(step, dict) or "step_name" not in step:
                continue

            step_id = step["step_name"]
            step_type = step.get("type")
            label = step.get("label", step_id)
            driven_by = step.get("driven_by")

            if step_type not in valid_types:
                raise ValueError(
                    f"步骤 '{step_id}' 使用了非法类型: '{step_type}'。"
                    f"仅允许: {sorted(valid_types)}"
                )

            # 分组存储
            if step_type == PARALLEL_PREPROCESSING:
                PARALLEL_PREPROCESSING_STEPS[step_id] = step
                PARALLEL_PREPROCESSING_KEYS.add(driven_by)
            elif step_type == PARALLEL_PERCEPTION:
                PARALLEL_PERCEPTION_STEPS[step_id] = step
                PERCEPTION_LAYERS.add(step_id)
                PARALLEL_PERCEPTION_KEYS.add(driven_by)
            elif step_type == PARALLEL_HIGH_ORDER:
                PARALLEL_HIGH_ORDER_STEPS[step_id] = step
                PARALLEL_HIGH_ORDER_KEYS.add(driven_by)
            elif step_type == SERIAL_SUGGESTION:
                SERIAL_SUGGESTION_STEPS[step_id] = step
                SERIAL_SUGGESTION_KEYS.add(driven_by)

            # 【关键】注入全量前端配置
            ALL_STEPS_FOR_FRONTEND.append({
                "id": step_id,
                "label": label,
                "type": step_type,
                "driven_by": driven_by
            })
        logger.info(
            f"✅ 步骤分离完成 | "
            f"pre={len(PARALLEL_PREPROCESSING_STEPS)} | "
            f"percep={len(PARALLEL_PERCEPTION_STEPS)} | "
            f"high={len(PARALLEL_HIGH_ORDER_STEPS)} | "
            f"sugg={len(SERIAL_SUGGESTION_STEPS)}"
        )

    @staticmethod
    def _build_step_prompts(
            steps: List[Dict],
            step_type: str
    ) -> List[Tuple[str, str, str]]:
        """
        构建指定类型（并行/串行）的 prompt 列表，返回 (step_name, driven_by, full_prompt) 元组列表。
        每个 prompt 严格按以下顺序组织：
          1. role（角色）
          2. ### 核心原则（information_source + 通用策略）
          3. ### 步骤专属规则（来自 step_rules）
          4. 输出前缀（可选 来自output_prefix）
          5. 字段结构（来自fields JSON schema）
          6. 空结果兜底（来自empty_result_fallback）
          7. 输出后缀（可选 来自output_suffix）
        """
        prompts_with_fields = []
        missing_fields = []

        for idx, step in enumerate(steps):
            try:
                step_name = step["step_name"]
                role = step["role"]
                information_source = step["information_source"]
                fields = step["fields"]
                driven_by = step.get("driven_by")
                constraint_profile = step.get("constraint_profile")
                empty_fallback = step.get("empty_result_fallback", "")
                # 新：使用扁平化的 step_rules
                step_rules = step.get("step_rules", [])
                output_prefix = step.get("output_prefix", [])
                output_suffix = step.get("output_suffix", [])
            except KeyError as e:
                field = e.args[0]
                missing_fields.append(f"步骤{idx}.{field}")
                continue

            # === 渲染通用策略铁律（核心原则）===
            effective_policy = get_effective_policy(step_name)
            dynamic_iron_law = render_iron_law_from_policy(effective_policy).strip()

            # === 构建 prompt 各部分 ===
            parts = [role.strip()]

            # 核心原则
            core_principle_text = "### 核心原则\n" + information_source.strip() + dynamic_iron_law
            parts.append(core_principle_text)

            # 步骤专属规则
            if step_rules:
                rules_text = "\n".join(step_rules)
                parts.append(rules_text)

            # 输出前缀
            if output_prefix:
                parts.append("\n".join(output_prefix))

            # 字段结构（schema）
            fields_json_str = json.dumps(fields, ensure_ascii=False, indent=2)
            parts.append(fields_json_str)

            # 空结果兜底
            if empty_fallback.strip():
                parts.append(empty_fallback.strip())

            # 输出后缀
            if output_suffix:
                parts.append("\n".join(output_suffix))

            # 拼接完整 prompt
            full_prompt = "\n\n".join(parts).strip()
            prompts_with_fields.append((step_name, driven_by, full_prompt))

            # logger.info(
            #     f"📌 步骤 {step_name} 使用约束配置: {constraint_profile}",
            #     module_name=PromptBuilder.CHINESE_NAME
            # )

        # ❌ 字段缺失校验
        if missing_fields:
            error_msg = f"{step_type} 步骤中缺失字段: {', '.join(missing_fields)}"
            logger.error(error_msg, module_name=PromptBuilder.CHINESE_NAME)
            raise ValueError(error_msg)

        # ✅ 成功日志
        logger.info(
            f"🔧 已生成 {step_type} prompts 数量: {len(prompts_with_fields)}",
            module_name=PromptBuilder.CHINESE_NAME
        )
        return prompts_with_fields

    @staticmethod
    def generate_description(context: dict, field_config: List[Tuple[str, bool, Any, str]], prefix="") -> str:
        def _is_effectively_empty(value) -> bool:
            if value is None:
                return True
            if isinstance(value, str) and not value.strip():
                return True
            if isinstance(value, (list, dict)) and len(value) == 0:
                return True
            return False

        def _format_simple_value(value):
            if isinstance(value, list):
                non_empty = [str(v) for v in value if not _is_effectively_empty(v)]
                return ", ".join(non_empty)
            return str(value)

        # 预处理通配规则：提取所有 *. 路径
        wildcard_rules = {}
        normal_rules = {}
        top_fields = []

        for path, required, typ, desc in field_config:
            if ".*." in path:
                prefix_path = path.split(".*.", 1)[0]  # 如 "inference.events"
                field_name = path.split(".*.", 1)[1]  # 如 "inference_type"
                if prefix_path not in wildcard_rules:
                    wildcard_rules[prefix_path] = []
                wildcard_rules[prefix_path].append((field_name, desc))
            elif "." not in path:
                top_fields.append((path, desc))
            else:
                normal_rules[path] = desc

        output_lines = []

        # 如果没有顶层字段，fallback 到平铺渲染
        if not top_fields:
            for path, desc in normal_rules.items():
                val = context.get(path)
                if not _is_effectively_empty(val):
                    output_lines.append(f"## {desc.rstrip('：:').strip()}")
                    output_lines.append(f"  - {desc}{_format_simple_value(val)}")
            result = "\n".join(output_lines).strip()
            # logger.info(f"动态生成上下文（无顶层）:{result}", module_name=Prompter.CHINESE_NAME)
            return result

        # 处理每个顶层字段（支持多个）
        for top_path, top_desc in top_fields:
            top_value = context.get(top_path)
            if _is_effectively_empty(top_value):
                continue

            clean_top_desc = top_desc.rstrip("：:").strip()
            output_lines.append(f"## {clean_top_desc}")

            if isinstance(top_value, dict):
                # 渲染字典的每个子字段
                for key, val in top_value.items():
                    if _is_effectively_empty(val):
                        continue
                    full_sub_path = f"{top_path}.{key}"
                    # 检查是否是 list[dict] 且有通配规则
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        if full_sub_path in wildcard_rules:
                            # 获取该列表字段的完整描述（如 "events（推理事件列表）："）
                            list_desc = normal_rules.get(full_sub_path, f"{key}（列表）：")
                            for item in val:
                                item_lines = []
                                for field_name, field_desc in wildcard_rules[full_sub_path]:
                                    item_val = item.get(field_name)
                                    if not _is_effectively_empty(item_val):
                                        item_lines.append(f"    - {field_desc}{_format_simple_value(item_val)}")
                                if item_lines:
                                    output_lines.append(f"  - {list_desc}")
                                    output_lines.extend(item_lines)
                            continue  # 已处理，跳过默认逻辑

                    # 默认：简单格式化
                    desc = normal_rules.get(full_sub_path, f"{key}: ")
                    output_lines.append(f"  - {desc}{_format_simple_value(val)}")

            elif isinstance(top_value, list):
                # 顶层是列表（如 participants）
                if top_path in wildcard_rules:
                    for item in top_value:
                        if not isinstance(item, dict):
                            continue
                        item_lines = []
                        for field_name, field_desc in wildcard_rules[top_path]:
                            item_val = item.get(field_name)
                            if not _is_effectively_empty(item_val):
                                item_lines.append(f"    - {field_desc}{_format_simple_value(item_val)}")
                        if item_lines:
                            output_lines.append("  - 列表项：")
                            output_lines.extend(item_lines)
                else:
                    output_lines.append(f"  - {top_desc}{_format_simple_value(top_value)}")
            else:
                output_lines.append(f"  - {top_desc}{_format_simple_value(top_value)}")

        # 清理空行
        while output_lines and output_lines[-1] == "":
            output_lines.pop()

        result = "\n".join(output_lines).strip()
        # logger.info(f"动态生成上下文:{result}", module_name=Prompter.CHINESE_NAME)
        return result

    @staticmethod
    def extract_top_level_description(fields_spec: List[Tuple[str, bool, Any, str]]) -> Optional[str]:
        """
        从字段规范列表中提取顶层字段（路径中不含 '.' 的字段）的描述。
        若存在多个顶层字段（如 inference + context_clue），优先取第一个非通配、非列表项的。
        """
        for field_path, _, _, description in fields_spec:
            # 跳过带通配符的路径（如 participants.*.role）
            if ".*." in field_path or field_path.startswith("*."):
                continue
            parts = field_path.split(".")
            if len(parts) == 1:
                # 这是一个顶层字段，如 "participants", "inference", "context_clue"
                return description
        return None
