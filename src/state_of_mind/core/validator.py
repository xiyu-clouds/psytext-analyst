import json
from typing import Dict, Any, List, Union, Tuple
from src.state_of_mind.utils.constants import REQUIRED_FIELDS_BY_CATEGORY, SUPPORTED_CATEGORIES, SEMANTIC_NULL_STRINGS
from src.state_of_mind.utils.logger import LoggerManager as logger

from functools import lru_cache

CHINESE_NAME = "数据结构校验"


@lru_cache(maxsize=128)
def _split_path(path: str) -> tuple:
    return tuple(path.split('.'))


def deep_get(data: Any, path: str) -> Any:
    """
    安全获取嵌套字段值。
    支持：
      - 字典嵌套: 'a.b.c'
      - 列表通配: 'a.*.b' 表示取 a 列表中每个元素的 b 字段
    """
    keys = _split_path(path)
    for key in keys:
        try:
            if isinstance(data, dict):
                if key in data:
                    data = data[key]
                else:
                    return None
            elif isinstance(data, list):
                if key == '*':
                    # 通配：提取列表中每个元素的下一个字段
                    next_vals = []
                    for item in data:
                        # 递归处理剩余路径（从下一个 key 开始）
                        sub_val = deep_get(item, '.'.join(keys[keys.index('*') + 1:]))
                        next_vals.append(sub_val)
                    return next_vals
                else:
                    return None  # 非 * 的 key 遇到 list 返回 None
            else:
                return None
        except Exception as e:
            logger.error(f"deep_get error at key '{key}': {e}", module_name=CHINESE_NAME)
            return None
    return data


def expand_wildcard_paths(data: Any, path: str) -> List[Tuple[str, Any]]:
    """
    将带通配符的路径（如 'a.*.b.*.c'）展开为所有实际存在的 (concrete_path, value) 对。
    例如：
        data = {"a": [{"b": [{"c": 1}, {"c": 2}]}, {"b": [{"c": 3}]}]}
        path = "a.*.b.*.c"
        → [("a[0].b[0].c", 1), ("a[0].b[1].c", 2), ("a[1].b[0].c", 3)]
    """

    def _recurse(current_data, keys, current_path):
        if not keys:
            return [(current_path, current_data)]

        key = keys[0]
        rest_keys = keys[1:]

        results = []

        if key == '*':
            if isinstance(current_data, list):
                for i, item in enumerate(current_data):
                    new_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                    results.extend(_recurse(item, rest_keys, new_path))
            # 如果不是 list，* 无意义，返回空
        else:
            if isinstance(current_data, dict) and key in current_data:
                new_path = f"{current_path}.{key}" if current_path else key
                results.extend(_recurse(current_data[key], rest_keys, new_path))
            # 否则路径不存在，返回空

        return results

    keys = _split_path(path)
    return _recurse(data, keys, "")


def validate_field(
        value: Any,
        field_path: str,
        required: bool,
        validator: Any
) -> List[str]:
    errors = []

    # 判断是否是通配路径（包含 *）
    is_wildcard_path = '*' in field_path

    # === 必填性校验 ===
    if required:
        if value is None:
            errors.append(f"字段 '{field_path}' 为必填项，但未提供或值为 null")
            return errors
        if is_wildcard_path and isinstance(value, list) and len(value) == 0:
            errors.append(f"通配路径 '{field_path}' 未匹配到任何有效字段，且该字段为必填")
            return errors

    # === 非必填字段：如果值为空（None, [], {}, ""），则跳过后续校验 ===
    if not required:
        if value is None:
            return errors
        if isinstance(value, list) and len(value) == 0:
            return errors
        if isinstance(value, dict) and len(value) == 0:
            return errors
        if isinstance(value, str) and value.strip() == "":
            return errors

    # 如果走到这里，说明 value 存在且非空，需要校验
    try:
        if isinstance(validator, type):
            # 类型校验器
            if is_wildcard_path and isinstance(value, list):
                for i, item in enumerate(value):
                    if item is None:
                        continue
                    if not isinstance(item, validator):
                        actual = type(item).__name__
                        expected = validator.__name__
                        errors.append(f"字段 '{field_path}[{i}]' 的实际类型为 {actual}，但期望类型为 {expected}")
            else:
                if not isinstance(value, validator):
                    actual = type(value).__name__
                    expected = validator.__name__
                    errors.append(f"字段 '{field_path}' 的实际类型为 {actual}，但期望类型为 {expected}")

        elif callable(validator):
            if is_wildcard_path and isinstance(value, list):
                for i, item in enumerate(value):
                    if item is None:
                        continue
                    if not validator(item):
                        errors.append(f"字段 '{field_path}[{i}]' 未通过自定义校验规则")
            else:
                if not validator(value):
                    errors.append(f"字段 '{field_path}' 未通过自定义校验规则")
        else:
            logger.warning(f"未知的校验器类型: {type(validator).__name__}", module_name=CHINESE_NAME)
            errors.append(f"字段 '{field_path}' 使用了无效的校验器")

    except Exception as e:
        errors.append(f"字段 '{field_path}' 校验时发生异常: {str(e)}")

    return errors


# --- 以下函数保持不变 ---
def deep_set(data: Dict[str, Any], path: str, value: Any) -> None:
    keys = _split_path(path)
    for key in keys[:-1]:
        if key not in data or not isinstance(data[key], dict):
            data[key] = {}
        data = data[key]
    data[keys[-1]] = value


def deep_del(data: Dict[str, Any], path: str) -> None:
    keys = _split_path(path)
    for key in keys[:-1]:
        if key not in data or not isinstance(data[key], dict):
            return
        data = data[key]
    if keys[-1] in data:
        del data[keys[-1]]


def remove_nulls(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            cleaned_v = remove_nulls(v)
            if cleaned_v is not None:  # 只保留非 None
                cleaned[k] = cleaned_v
        return cleaned if cleaned else None
    elif isinstance(data, list):
        cleaned_list = []
        for item in data:
            cleaned_item = remove_nulls(item)
            if cleaned_item is not None:
                cleaned_list.append(cleaned_item)
        return cleaned_list if cleaned_list else None
    elif isinstance(data, str):
        stripped = data.strip()
        if stripped == "" or stripped in SEMANTIC_NULL_STRINGS:
            return None
        return data
    elif isinstance(data, (int, float, bool)):
        return data
    else:
        return data


def safe_json_output(data: dict) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        safe_copy = {}
        for k, v in data.items():
            try:
                json.dumps({k: v}, ensure_ascii=False)
                safe_copy[k] = v
            except:
                safe_copy[k] = str(v)
        return json.dumps(safe_copy, ensure_ascii=False, indent=2)


def collect_validation_errors(
        cleaned_data: Dict[str, Any],
        rules: List[Tuple[str, bool, Any, str]]
) -> List[str]:
    errors = []
    for rule in rules:
        try:
            field_path, required, validator, desc = rule
        except ValueError:
            err_msg = f"规则格式错误（应为4元组）: {rule}"
            logger.error(err_msg, module_name=CHINESE_NAME)
            errors.append(err_msg)
            continue

        if '*' in field_path:
            logger.debug(f"检测到通配路径，正在展开: '{field_path}'", module_name=CHINESE_NAME)
            concrete_items = expand_wildcard_paths(cleaned_data, field_path)
            logger.debug(f"路径 '{field_path}' 展开后得到 {len(concrete_items)} 个具体字段", module_name=CHINESE_NAME)

            if not concrete_items:
                pass

            for concrete_path, value in concrete_items:
                logger.debug(f"校验具体字段: {concrete_path} = {repr(value)}", module_name=CHINESE_NAME)

                # === 通配路径：自动修复非列表值 → [value]（仅当 validator 是 IS_LIST）===
                if (hasattr(validator, '__name__') and validator.__name__ == 'is_list') and value is not None:
                    if not isinstance(value, list):
                        should_repair = True
                        if isinstance(value, str):
                            stripped = value.strip()
                            if stripped == "" or stripped in SEMANTIC_NULL_STRINGS:
                                should_repair = False
                        if should_repair:
                            repaired_value = [value]
                            # 转换 concrete_path: "a[0].b[1].c" → "a.0.b.1.c"
                            dot_path = concrete_path.replace('[', '.').replace(']', '')
                            deep_set(cleaned_data, dot_path, repaired_value)
                            logger.info(
                                f"自动修复通配字段 '{concrete_path}': 原值 {repr(value)} 不是列表，已包装为单元素列表",
                                module_name=CHINESE_NAME
                            )

                field_errors = validate_field(value, concrete_path, required, validator)
                errors.extend(field_errors)
        else:
            value = deep_get(cleaned_data, field_path)
            logger.debug(f"校验普通字段: {field_path} = {repr(value)}", module_name=CHINESE_NAME)

            # === 普通路径：自动修复非列表值 → [value]（仅当 validator 是 IS_LIST）===
            if (hasattr(validator, '__name__') and validator.__name__ == 'is_list') and value is not None:
                if not isinstance(value, list):
                    should_repair = True
                    if isinstance(value, str):
                        stripped = value.strip()
                        if stripped == "" or stripped in SEMANTIC_NULL_STRINGS:
                            should_repair = False
                    if should_repair:
                        repaired_value = [value]
                        deep_set(cleaned_data, field_path, repaired_value)
                        logger.info(
                            f"自动修复字段 '{field_path}': 原值 {repr(value)} 不是列表，已包装为单元素列表",
                            module_name=CHINESE_NAME
                        )

            field_errors = validate_field(value, field_path, required, validator)
            errors.extend(field_errors)

    return errors


def build_validation_result(
        is_valid: bool,
        errors: List[str],
        cleaned_data: Dict[str, Any],
        template_name: str,
        step_type: str
) -> Dict[str, Any]:
    result = {
        "is_valid": is_valid,
        "errors": errors,
        "cleaned_data": cleaned_data
    }
    if is_valid:
        logger.info(f"{template_name} - {step_type}: Validation passed", module_name=CHINESE_NAME)
    else:
        logger.warning(f"{template_name} - {step_type}: Validation failed", module_name=CHINESE_NAME,
                       extra={"error_count": len(errors), "errors": errors})
    return result


def remove_meta_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """移除所有以 __ 开头的元字段，防止污染业务数据"""
    if not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if not k.startswith("__")}


def validate_with_diagnosis(
        data: Union[Dict[str, Any], None],
        template_name: str,
        step_name: str
) -> Dict[str, Any]:
    if data is None:
        result = {"is_valid": False, "errors": [], "message": "Input data is None", "meta": {}, "cleaned_data": None}
        logger.warning("Validation failed: Input data is None", extra=result["meta"])
        return result

    data = remove_meta_fields(data)

    if template_name not in SUPPORTED_CATEGORIES:
        return {
            "is_valid": False,
            "message": f"不支持的 category: {template_name}",
            "errors": [],
            "meta": {},
            "cleaned_data": None
        }

    cleaned_data = remove_nulls(data)

    if template_name not in REQUIRED_FIELDS_BY_CATEGORY:
        return {
            "is_valid": False,
            "message": f"No validation rules found for template_name: {template_name}",
            "errors": [],
            "cleaned_data": cleaned_data
        }

    raw = REQUIRED_FIELDS_BY_CATEGORY.get(template_name)
    rules = raw.get(step_name)

    if not rules:
        return {
            "is_valid": False,
            "message": f"No rules for step_type '{step_name}'",
            "errors": [],
            "cleaned_data": cleaned_data
        }

    errors = collect_validation_errors(cleaned_data, rules)
    is_valid = len(errors) == 0
    return build_validation_result(is_valid, errors, cleaned_data, template_name, step_name)
