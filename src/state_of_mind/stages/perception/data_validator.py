
from typing import Dict, Any, List, Union, Tuple
from .constants import REQUIRED_FIELDS_BY_CATEGORY, SEMANTIC_NULL_STRINGS
from src.state_of_mind.utils.logger import LoggerManager as logger
from ...types.perception import ValidationRule


class DataValidator:
    """
    轻量级数据校验器，封装通配路径解析、自动修复、空值清理等逻辑。
    """
    CHINESE_NAME = "数据结构校验"

    def __init__(self, auto_repair: bool = True):
        self.auto_repair = auto_repair

    # --- 工具方法 ---
    @staticmethod
    def _split_path(path: str) -> tuple:
        return tuple(path.split('.'))

    @staticmethod
    def deep_get(data: Any, path: str) -> Any:
        keys = DataValidator._split_path(path)
        for key in keys:
            try:
                if isinstance(data, dict):
                    if key in data:
                        data = data[key]
                    else:
                        return None
                elif isinstance(data, list):
                    if key == '*':
                        next_vals = []
                        rest = '.'.join(keys[keys.index('*') + 1:])
                        for item in data:
                            sub_val = DataValidator.deep_get(item, rest)
                            next_vals.append(sub_val)
                        return next_vals
                    else:
                        return None
                else:
                    return None
            except Exception as e:
                logger.error(f"deep_get error at key '{key}': {e}", module_name=DataValidator.CHINESE_NAME)
                return None
        return data

    @staticmethod
    def expand_wildcard_paths(data: Any, path: str) -> List[Tuple[str, Any]]:
        def _recurse(current_data, keys, current_path):
            if not keys:
                return [(current_path, current_data)]
            key, rest_keys = keys[0], keys[1:]
            results = []
            if key == '*':
                if isinstance(current_data, list):
                    for i, item in enumerate(current_data):
                        new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                        results.extend(_recurse(item, rest_keys, new_path))
            else:
                if isinstance(current_data, dict) and key in current_data:
                    new_path = f"{current_path}.{key}" if current_path else key
                    results.extend(_recurse(current_data[key], rest_keys, new_path))
            return results

        keys = DataValidator._split_path(path)
        return _recurse(data, keys, "")

    @staticmethod
    def deep_set(data: Any, path: str, value: Any) -> None:
        keys = DataValidator._split_path(path)
        current = data
        for i, key in enumerate(keys[:-1]):
            # 尝试将 key 转为 int（用于 list 索引）
            next_key = key
            is_int_key = False
            if isinstance(current, list):
                try:
                    idx = int(key)
                    if 0 <= idx < len(current):
                        next_key = idx
                        is_int_key = True
                    else:
                        # 越界：无法设置，报错或跳过？
                        logger.warning(f"deep_set: list index out of range: {key} in path {path}",
                                       module_name=DataValidator.CHINESE_NAME)
                        return
                except ValueError:
                    # key 不是数字，但 current 是 list → 无效操作
                    logger.warning(f"deep_set: cannot use non-integer key '{key}' on list in path {path}",
                                   module_name=DataValidator.CHINESE_NAME)
                    return

            if isinstance(current, dict):
                if next_key not in current:
                    # 下一层类型未知，但通常我们期望和 value 类型一致？这里保守创建 dict
                    current[next_key] = {}
                current = current[next_key]
            elif isinstance(current, list) and is_int_key:
                # 已确认 next_key 是有效 int 索引
                current = current[next_key]
            else:
                # 类型不匹配，无法继续深入
                logger.warning(f"deep_set: unsupported container type at key '{key}' in path {path}",
                               module_name=DataValidator.CHINESE_NAME)
                return

        # 设置最终值
        final_key = keys[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            try:
                idx = int(final_key)
                if 0 <= idx < len(current):
                    current[idx] = value
                else:
                    logger.warning(f"deep_set: final index out of range: {final_key} in path {path}",
                                   module_name=DataValidator.CHINESE_NAME)
            except ValueError:
                logger.warning(f"deep_set: cannot set non-integer key '{final_key}' on list in path {path}",
                               module_name=DataValidator.CHINESE_NAME)
        else:
            logger.warning(f"deep_set: target is not a container in path {path}",
                           module_name=DataValidator.CHINESE_NAME)

    @staticmethod
    def _is_semantic_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            s = value.strip()
            return s == "" or s in SEMANTIC_NULL_STRINGS
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        return False

    @staticmethod
    def _is_empty(value: Any) -> bool:
        return DataValidator._is_semantic_empty(value)

    @staticmethod
    def remove_nulls(data: Any) -> Any:
        if isinstance(data, (str, int, float, bool)):
            return None if DataValidator._is_semantic_empty(data) else data

        if isinstance(data, dict):
            cleaned = {
                k: DataValidator.remove_nulls(v)
                for k, v in data.items()
            }
            cleaned = {k: v for k, v in cleaned.items() if v is not None}
            return cleaned if cleaned else None

        if isinstance(data, list):
            cleaned = [DataValidator.remove_nulls(item) for item in data]
            cleaned = [x for x in cleaned if x is not None]
            return cleaned if cleaned else None

        return None if DataValidator._is_semantic_empty(data) else data

    @staticmethod
    def remove_meta_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if not k.startswith("__")}

    # --- 核心校验逻辑 ---
    def _validate_field(
            self,
            value: Any,
            field_path: str,
            required: bool,
            validator: Any
    ) -> List[str]:
        errors = []

        # 必填校验
        if required:
            if value is None:
                errors.append(f"[{field_path}] 字段为必填项: 值为 null 或缺失")
                return errors
            if isinstance(value, list) and len(value) == 0:
                errors.append(f"[{field_path}] 必填字段未匹配到任何有效元素: 列表为空")
                return errors

        # 非必填且为空 → 跳过
        if not required and self._is_empty(value):
            return errors

        try:
            valid = False
            if isinstance(validator, type):
                if validator == int:
                    valid = isinstance(value, int) and not isinstance(value, bool)
                elif validator == bool:
                    valid = isinstance(value, bool)
                else:
                    valid = isinstance(value, validator)
            elif callable(validator):
                valid = validator(value)
            else:
                errors.append(f"[{field_path}] 校验器无效: 类型 {type(validator).__name__}")
                return errors

            if not valid:
                actual_type = type(value).__name__
                expected_desc = getattr(validator, '__name__', str(validator))
                value_repr = repr(value)
                if len(value_repr) > 80:
                    value_repr = value_repr[:77] + "..."
                errors.append(
                    f"[{field_path}] 类型校验失败: 期望 {expected_desc}，实际为 {actual_type}（值: {value_repr}）"
                )

        except Exception as e:
            errors.append(f"[{field_path}] 校验异常: {e}")

        return errors

    def _maybe_repair_value(self, value: Any, field_path: str, validator: Any) -> Any:
        """
        根据 validator 类型，尝试安全修复 value。
        支持 IS_STR, IS_INT, IS_FLOAT, IS_BOOL, IS_LIST, IS_DICT。
        """
        if not self.auto_repair or value is None:
            return value

        # 辅助：判断是否为“可包装为 list 的单一值”
        def _is_scalar_like(val):
            return isinstance(val, (str, int, float, bool)) or val is None

        # 获取 validator 名称
        validator_name = getattr(validator, '__name__', str(validator))

        # --- 1. 期望是 LIST ---
        if validator_name == 'is_list':
            if isinstance(value, list):
                return value  # 已符合，无需修复

            # 情况 A: 单一标量值 → 包装为 [value]
            if _is_scalar_like(value):
                repaired = [value]
                logger.info(
                    f"自动修复字段 '{field_path}': 标量值 {repr(value)} 被包装为单元素列表",
                    module_name=DataValidator.CHINESE_NAME
                )
                return repaired

            # 情况 B: 是 dict → 通常不应包装！因为 dict 是对象，不是 list 元素
            # 但注意：[dict] 是合法 list！问题在于“单独一个 dict”是否该变 [dict]？
            # ✅ 我们允许！因为很多 LLM 会把“单个对象”直接输出为 dict，而业务期望是 list of objects
            if isinstance(value, dict):
                repaired = [value]
                logger.info(
                    f"自动修复字段 '{field_path}': 单个字典被包装为单元素列表（常见于 LLM 输出单对象）",
                    module_name=DataValidator.CHINESE_NAME
                )
                return repaired

            # 情况 C: 其他类型（如 set/tuple/function）→ 尝试转 list？
            # 保守起见：只处理 tuple（常见）
            if isinstance(value, tuple):
                repaired = list(value)
                logger.info(
                    f"自动修复字段 '{field_path}': tuple 被转换为 list",
                    module_name=DataValidator.CHINESE_NAME
                )
                return repaired

            # 情况 D: 字符串且含分隔符？→ 可选解析（需谨慎，避免误伤）
            # 暂不启用，除非你明确需要。可后续加开关。
            # if isinstance(value, str):
            #     stripped = value.strip()
            #     if ',' in stripped and not any(c in stripped for c in ['{', '[', '"']):
            #         repaired = [item.strip() for item in stripped.split(',') if item.strip()]
            #         logger.info(f"自动解析逗号分隔字符串为列表: {field_path}")
            #         return repaired

            # 默认：无法修复，保留原值（后续会报错）
            logger.warning(
                f"无法修复字段 '{field_path}' 为 list: 值类型为 {type(value).__name__}（{repr(value)[:60]}...）",
                module_name=DataValidator.CHINESE_NAME
            )
            return value

        # --- 2. 期望是 DICT ---
        elif validator_name == 'is_dict':
            if isinstance(value, dict):
                return value
            # dict 很难安全修复，除非是 JSON 字符串（暂不处理）
            logger.warning(
                f"无法修复字段 '{field_path}' 为 dict: 值类型为 {type(value).__name__}",
                module_name=DataValidator.CHINESE_NAME
            )
            return value

        # --- 3. 期望是 STR ---
        elif validator_name == 'is_str':
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float, bool)):
                repaired = str(value)
                logger.info(
                    f"自动修复字段 '{field_path}' 为字符串: {repr(repaired)}",
                    module_name=DataValidator.CHINESE_NAME
                )
                return repaired
            # dict/list 不转 str（会丢失结构）
            logger.warning(f"无法将复杂类型转为字符串: {field_path} = {type(value).__name__}", module_name=DataValidator.CHINESE_NAME)
            return value

        # --- 4. 期望是 INT ---
        elif validator_name == 'is_int':
            if isinstance(value, int) and not isinstance(value, bool):  # bool 是 int 子类！
                return value
            if isinstance(value, str):
                try:
                    if value.strip().isdigit():
                        repaired = int(value)
                        logger.info(f"自动修复字符串为整数: {field_path} = {repaired}", module_name=DataValidator.CHINESE_NAME)
                        return repaired
                except Exception:
                    pass
            logger.warning(f"无法修复为整数: {field_path} = {repr(value)}", module_name=DataValidator.CHINESE_NAME)
            return value

        # --- 5. 期望是 FLOAT ---
        elif validator_name == 'is_float':
            if isinstance(value, float):
                return value
            if isinstance(value, (int, str)) and not isinstance(value, bool):
                try:
                    repaired = float(value)
                    logger.info(f"自动修复为浮点数: {field_path} = {repaired}", module_name=DataValidator.CHINESE_NAME)
                    return repaired
                except Exception:
                    pass
            logger.warning(f"无法修复为浮点数: {field_path} = {repr(value)}", module_name=DataValidator.CHINESE_NAME)
            return value

        # --- 6. 期望是 BOOL ---
        elif validator_name == 'is_bool':
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v_lower = value.strip().lower()
                if v_lower in {'true', '1', 'yes', 'on'}:
                    return True
                elif v_lower in {'false', '0', 'no', 'off'}:
                    return False
            logger.warning(f"无法修复为布尔值: {field_path} = {repr(value)}", module_name=DataValidator.CHINESE_NAME)
            return value

        return value

    def _collect_errors(self, cleaned_data: Dict[str, Any], rules: List[ValidationRule]) -> List[str]:
        errors = []
        for rule in rules:
            try:
                field_path, required, validator, desc = rule
            except ValueError:
                err_msg = f"规则格式错误（应为4元组）: {rule}"
                logger.error(err_msg, module_name=DataValidator.CHINESE_NAME)
                errors.append(err_msg)
                continue

            if '*' in field_path:
                concrete_items = self.expand_wildcard_paths(cleaned_data, field_path)
                for concrete_path, value in concrete_items:
                    dot_path = concrete_path.replace('[', '.').replace(']', '')
                    repaired_value = self._maybe_repair_value(value, concrete_path, validator)
                    if repaired_value != value:
                        self.deep_set(cleaned_data, dot_path, repaired_value)
                        value = repaired_value
                    field_errors = self._validate_field(value, concrete_path, required, validator)
                    errors.extend(field_errors)
            else:
                value = self.deep_get(cleaned_data, field_path)
                repaired_value = self._maybe_repair_value(value, field_path, validator)
                if repaired_value != value:
                    self.deep_set(cleaned_data, field_path, repaired_value)
                    value = repaired_value
                field_errors = self._validate_field(value, field_path, required, validator)
                errors.extend(field_errors)

        return errors

    @staticmethod
    def _build_result(is_valid: bool, errors: List[str], cleaned_data: Any, template: str, step: str) -> Dict[str, Any]:
        result = {
            "is_valid": is_valid,
            "errors": errors,
            "cleaned_data": cleaned_data
        }
        if is_valid:
            logger.info(f"{template} - {step}: Validation passed", module_name=DataValidator.CHINESE_NAME)
        else:
            logger.warning(
                f"{template} - {step}: Validation failed",
                module_name=DataValidator.CHINESE_NAME,
                extra={"error_count": len(errors), "errors": errors}
            )
        return result

    # --- 对外接口 ---
    def validate(
            self,
            data: Union[Dict[str, Any], None],
            template_name: str,
            step_name: str
    ) -> Dict[str, Any]:
        if data is None:
            return {
                "is_valid": False,
                "errors": ["Input data is None"],
                "cleaned_data": None,
                "message": "Input data is None"
            }

        data = self.remove_meta_fields(data)
        cleaned_data = self.remove_nulls(data)

        if template_name not in REQUIRED_FIELDS_BY_CATEGORY:
            return {
                "is_valid": False,
                "errors": [f"不支持的 category: {template_name}"],
                "cleaned_data": cleaned_data
            }

        rules = REQUIRED_FIELDS_BY_CATEGORY[template_name].get(step_name)
        if not rules:
            return {
                "is_valid": False,
                "errors": [f"No rules for step_type '{step_name}'"],
                "cleaned_data": cleaned_data
            }

        errors = self._collect_errors(cleaned_data, rules)
        is_valid = len(errors) == 0
        return self._build_result(is_valid, errors, cleaned_data, template_name, step_name)
