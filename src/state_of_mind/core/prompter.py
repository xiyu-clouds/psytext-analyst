import json
from typing import Any, Dict, List, Tuple, Optional, Set
import ulid
from datetime import datetime
from zoneinfo import ZoneInfo

from src.state_of_mind.config import config
from src.state_of_mind.utils.constants import PARALLEL, SERIAL, PREPROCESSING, CATEGORY_RAW, SuggestionType, \
    get_effective_policy, render_iron_law_from_policy, COREFERENCE_RESOLUTION_BATCH
# from src.state_of_mind.utils.ip_timezone import IPBasedTimezoneResolver
from src.state_of_mind.utils.logger import LoggerManager as logger
# from src.state_of_mind.utils.network import get_public_ip
from static.prompts.prompt import LLM_PROMPTS_SCHEMA


class Prompter:
    """
    Prompt 构造器
    """
    CHINESE_NAME = "Prompt构造器"

    def build_raw(self, template_name: str, **template_vars: Any) -> Dict[str, Any]:
        """
        构建 Prompt 并返回完整上下文数据
        """
        logger.info("🔄 开始构建 build_raw Prompt", module_name=self.CHINESE_NAME)

        # 1. 验证输入
        if "user_input" not in template_vars:
            error_msg = "缺失必需字段: user_input"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        user_input = template_vars["user_input"]
        llm_model = template_vars["llm_model"]

        # 2. 获取模板定义
        raw_schema = LLM_PROMPTS_SCHEMA.get(template_name)
        if not raw_schema:
            error_msg = f"模板未定义: {template_name}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        schema_version = raw_schema.get("version")
        pipeline = raw_schema.get("pipeline")

        # 3. 三路分离 pipeline
        preprocessing_steps, parallel_steps, serial_steps = Prompter._split_pipeline(pipeline)

        # 4. 构造三类 prompts
        preprocessing_prompts = Prompter._build_step_prompts(
            steps=preprocessing_steps,
            step_type=PREPROCESSING
        )

        parallel_prompts = Prompter._build_step_prompts(
            steps=parallel_steps,
            step_type=PARALLEL
        )

        serial_prompts = Prompter._build_step_prompts(
            steps=serial_steps,
            step_type=SERIAL
        )

        # 5. 生成基础元数据
        basic_data = Prompter.create_raw_basic_data(user_input, llm_model, schema_version)

        # 6. 记录完成日志
        logger.info(
            f"✅ Prompt 构建完成, preprocessing_count = {len(preprocessing_prompts)} | "
            f"parallel_count = {len(parallel_prompts)} | serial_count = {len(serial_prompts)} | "
            f"record_id = {basic_data['id']}",
            module_name=Prompter.CHINESE_NAME
        )

        # ✅ 返回完整结构，便于上层组装
        return {
            "template_name": template_name,
            "preprocessing_prompts": preprocessing_prompts,  # 新增
            "parallel_prompts": parallel_prompts,
            "serial_prompts": serial_prompts,
            "basic_data": basic_data
        }

    def build_suggestion(self, template_name: str, user_input: str, suggestion_type: str) -> str:
        logger.info("🔄 开始构建 build_suggestion Prompt", module_name=self.CHINESE_NAME)

        suggestion_schema = LLM_PROMPTS_SCHEMA.get(template_name)
        if not suggestion_schema:
            error_msg = f"模板未定义: {template_name}"
            logger.error(error_msg, module_name=self.CHINESE_NAME)
            raise ValueError(error_msg)

        # ✅ 使用 SuggestionType 定义的合法类型做校验
        valid_types = {
            SuggestionType.PSYCHOANALYSIS,
            SuggestionType.CONSISTENCY_SUGGESTION,
            SuggestionType.LITERARY_CRITIC,
            SuggestionType.IRONIC_DECONSTRUCTOR,
            SuggestionType.CRITICAL_THEORIST,
            SuggestionType.EXISTENTIAL_PHILOSOPHER,
            SuggestionType.CULTURAL_ANTHROPOLOGIST,
        }

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

    def _build_coref_prompt(
            self,
            user_input: str,
            legitimate_participants: Set[str],
            index_to_pronoun: Dict[int, str]
    ) -> str:
        """
        构造指代消解 prompt。

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

    @staticmethod
    def _split_pipeline(pipeline: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        分离 pipeline 中的预处理、并行、串行任务
        返回: (preprocessing_steps, parallel_steps, serial_steps)
        """
        if not isinstance(pipeline, list):
            error_msg = "pipeline 必须是列表类型"
            logger.error(error_msg)
            raise ValueError(error_msg)

        preprocessing = []
        parallel = []
        serial = []

        for idx, step in enumerate(pipeline):
            if not isinstance(step, dict):
                logger.warning(f"跳过非法 pipeline 步骤（非字典）: 索引={idx}")
                continue

            step_type = step.get("type", SERIAL)  # 默认为串行

            if step_type == PREPROCESSING:
                preprocessing.append(step)
            elif step_type == PARALLEL:
                parallel.append(step)
            else:
                serial.append(step)  # 包括 SERIAL 和 未知 type 都归为串行（安全兜底）

        logger.info(
            f"📊 pipeline 三路分离完成, preprocessing_count = {len(preprocessing)} | parallel_count = {len(parallel)} | "
            f"serial_count = {len(serial)} | total_steps = {len(pipeline)}", module_name=Prompter.CHINESE_NAME
        )
        return preprocessing, parallel, serial

    @staticmethod
    def _build_step_prompts(
            steps: List[Dict],
            step_type: str
    ) -> List[Tuple[str, str, str]]:
        """
        构建指定类型（并行/串行）的 prompt 列表，返回 (step_name, driven_by, full_prompt) 元组列表。

        每个 prompt 严格按以下顺序组织：
          1. role
          2. sole_mission
          3. ### 当前任务的核心铁律（必须绝对遵守）### （来自 input_requirements.data_and_anchor_constraints）
          4. ### 输出格式与结构强制要求 ### （来自 input_requirements.output_structure_constraints）
          5. ### 【必须遵守的铁律】 （来自 render_iron_law_from_policy）
          6. fields schema（JSON 转义后）
        """
        prompts_with_fields = []
        missing_fields = []

        for idx, step in enumerate(steps):
            try:
                step_name = step["step"]
                role = step["role"]
                sole_mission = step["sole_mission"]
                fields = step["fields"]
                driven_by = step.get("driven_by")
                constraint_profile = step.get("constraint_profile", "unknown")
                input_requirements = step.get("input_requirements", {})
            except KeyError as e:
                field = e.args[0]
                missing_fields.append(f"步骤{idx}.{field}")
                continue

            # 渲染通用策略铁律（如字面锚定、结构一致等）
            effective_policy = get_effective_policy(step_name)
            dynamic_iron_law = render_iron_law_from_policy(effective_policy)

            # 转义 fields schema，防止大模型误解析为指令
            fields_json = json.dumps(fields, ensure_ascii=False, indent=2)
            fields_escaped = fields_json.replace('{', '{{').replace('}', '}}')

            # === 构建三层铁律（按优先级从高到低）===
            iron_law_sections = []

            # 1️⃣ 任务专属数据与锚定约束（最高优先级）
            data_constraints = input_requirements.get("data_and_anchor_constraints")
            if data_constraints:
                iron_law_sections.append(
                    "### 当前任务的核心铁律（必须绝对遵守）###\n" +
                    "\n".join(data_constraints)
                )

            # 2️⃣ 输出结构与格式强制要求（直接影响生成行为）
            output_constraints = input_requirements.get("output_structure_constraints")
            if output_constraints:
                iron_law_sections.append(
                    "### 输出格式与结构强制要求 ###\n" +
                    "\n".join(output_constraints)
                )

            # 3️⃣ 通用策略铁律（方法论层，最低优先级）
            if dynamic_iron_law.strip():
                iron_law_sections.append(dynamic_iron_law.strip())

            combined_iron_law = "\n\n".join(iron_law_sections).strip()

            # === 拼接完整 prompt ===
            full_prompt_parts = [
                "### SYSTEM INSTRUCTIONS BEGIN ###\n",
                role.strip(),
                sole_mission.strip()
            ]
            if combined_iron_law:
                full_prompt_parts.append(combined_iron_law)
            full_prompt_parts.append(fields_escaped.strip())
            full_prompt_parts.append("### SYSTEM INSTRUCTIONS END ###")

            full_prompt = "\n".join(full_prompt_parts)

            prompts_with_fields.append((step_name, driven_by, full_prompt))

            # 📝 日志记录：便于后期审计与调试
            logger.info(
                f"📌 步骤 {step_name} 使用约束配置: {constraint_profile}",
                module_name=Prompter.CHINESE_NAME
            )

        # ❌ 字段缺失校验
        if missing_fields:
            error_msg = f"{step_type} 步骤中缺失字段: {', '.join(missing_fields)}"
            logger.error(error_msg, module_name=Prompter.CHINESE_NAME)
            raise ValueError(error_msg)

        # ✅ 成功日志
        logger.info(
            f"🔧 已生成 {step_type} prompts 数量: {len(prompts_with_fields)}",
            module_name=Prompter.CHINESE_NAME
        )
        return prompts_with_fields

    @staticmethod
    def create_raw_basic_data(user_input: str, llm_model: str, schema_version: str = "1.0.0") -> Dict[str, Any]:
        """
        构造原始事件的固定基础元数据
        可用于日志追踪、审计、溯源等
        """
        record_id = f"raw_{ulid.new().str}"

        # public_ip = get_public_ip()
        # tz_name = IPBasedTimezoneResolver.get_timezone_from_ip(public_ip) if public_ip else "UTC"

        # if not public_ip:
        #     logger.warning("⚠️ 无法获取公网IP，使用 UTC 时区", module_name=Prompter.CHINESE_NAME)

        tz = ZoneInfo("UTC")
        timestamp = datetime.now(tz).isoformat()

        formatter_time = ""
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][dt.weekday()]
            base_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 毫秒部分
            formatter_time = f"{base_time} {weekday}"
        except Exception as e:
            logger.warning(
                f"🕒 无法解析 timestamp 为 formatter_time: {e}",
                module_name=Prompter.CHINESE_NAME,
                extra={"timestamp": timestamp}
            )

        data = {
            "id": record_id,
            "type": CATEGORY_RAW,
            "schema_version": schema_version,
            "timestamp": timestamp,
            "formatter_time": formatter_time,
            "source": {
                "modality": "text/narrative",
                "content": user_input,
                "input_mode": "user_input",
                # "local_ip": public_ip,
                "timezone": "UTC"
            },
            "meta": {
                "library_version": config.VERSION,
                "created_by_ai": True,
                "llm_model": llm_model,
                "crystal_ids": [],
                "ontology_ids": [],
                "narrative_enriched": False,
                "privacy_scope": {
                    "allowed_modules": [],
                    "sync_to_cloud": False,
                    "notify_on_trigger": False,
                    "exportable": False
                }

            }
        }

        logger.info(f"📦 已生成基础元数据, id={record_id} | timezone=UTC", module_name=Prompter.CHINESE_NAME)
        return data

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
