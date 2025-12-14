import os
import json


def try_decode_json_string(s: str):
    """
    尝试将一个可能是 JSON 字符串的字符串解码为 Python 对象。
    支持多层嵌套的 JSON 字符串（如 Redis 中常见的 double-encoded JSON）。
    """
    max_depth = 5  # 防止无限循环
    current = s
    for _ in range(max_depth):
        if isinstance(current, str):
            current = current.strip()
            # 如果看起来像 JSON（以 { 或 [ 开头），尝试解析
            if (current.startswith('{') and current.endswith('}')) or \
               (current.startswith('[') and current.endswith(']')):
                try:
                    current = json.loads(current)
                except (json.JSONDecodeError, TypeError):
                    break
            else:
                break
        else:
            break
    return current


def clean_redis_dump_json(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到文件 '{input_file}'")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_content = f.read().strip()

        # 尝试解析最外层是否为 JSON 对象或数组
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            # 如果不是，尝试当作单行 JSON 字符串处理（如你原始 dump 的格式）
            # 但你的输入实际是多个 {"key": "...", "value": "..."} 对象的 JSON 数组？
            # 根据你提供的数据，看起来是 **一个 JSON 数组**
            # 所以我们假设输入是一个 JSON 数组
            raise ValueError("输入文件必须是有效的 JSON 数组或对象")

        # 如果 data 是列表（常见于 Redis dump 导出）
        if isinstance(data, list):
            cleaned_data = []
            for item in data:
                if isinstance(item, dict) and "value" in item:
                    raw_value = item["value"]
                    # 解码 value 字段中的 JSON 字符串
                    decoded_value = try_decode_json_string(raw_value)
                    new_item = {**item, "value": decoded_value}
                    cleaned_data.append(new_item)
                else:
                    cleaned_data.append(item)
            result = cleaned_data
        elif isinstance(data, dict):
            # 单个对象情况
            if "value" in data:
                data["value"] = try_decode_json_string(data["value"])
            result = data
        else:
            result = data

        # 写出干净的 JSON
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✅ 成功：已清理嵌套 JSON 字符串，输出到 '{output_file}'")
        print("\n预览前 500 字符：")
        preview = json.dumps(result, ensure_ascii=False, indent=2)[:500]
        print(preview + ("..." if len(preview) == 500 else ""))

    except Exception as e:
        print(f"❌ 处理失败：{e}")
        import traceback
        traceback.print_exc()


def main():
    input_file = "output.json"
    output_file = "cleaned_output.json"
    clean_redis_dump_json(input_file, output_file)


if __name__ == "__main__":
    main()