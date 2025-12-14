import json
import re


def extract_json_safely(content: str) -> dict:
    if not content or not content.strip():
        return {"__error": "空响应内容"}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r'^```(?:json|text|markdown)?\s*', '', content, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace('\\\\', '\\').replace('\\\'', '\'').replace('\\"', '"')

    try:
        match = re.search(r'{.*}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass

    return {"__error": "无法提取有效 JSON", "__raw": content[:200]}


def remove_check(text: str) -> str:
    text = re.sub(r'^```(?:json|text|markdown)?\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?```$', '', text)
    return text