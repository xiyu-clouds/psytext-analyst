"""
🌊 全局常量池
"""
from pathlib import Path
from typing import Dict, Final, List, Set

# ======================================================================
# 🌐 根目录（唯一真实源）
# ======================================================================
ROOT_DIR: Final[Path] = Path(__file__).parent.parent.parent.parent.resolve()

# ======================================================================
# 📂 目录名称（字符串常量，用于组合）
# ======================================================================
# 根级目录
DIR_DATA = "data"
DIR_STATIC = "static"
DIR_XINJING = "src/state_of_mind"

# 子模块目录
DIR_LOGS = "logs"
DIR_LOGS_FALLBACK = "logs_fallback"
DIR_YUAN = "yuan"
DIR_RAW = "raw"
DIR_DYE_VAT = "dye_vat"
DIR_CONFIG = "config"
DIR_OTHER = "other"
DIR_PROMPTS = "prompt_templates"
DIR_TEMPLATES = "templates"

# ======================================================================
# 🛤️ 路径片段（Path 类型！不再是字符串）
# ======================================================================
# 使用 Path 对象统一管理路径，支持自然拼接 /
PATH_DATA = Path("/home/appuser/psytext_data")
PATH_STATIC = ROOT_DIR / DIR_STATIC
PATH_XINJING = ROOT_DIR / DIR_XINJING

# —————— 静态资源 ——————
PATH_STATIC_CONFIG = PATH_STATIC / DIR_CONFIG
PATH_STATIC_OTHER = PATH_STATIC / DIR_OTHER
PATH_STATIC_PROMPTS = PATH_STATIC / DIR_PROMPTS
PATH_STATIC_TEMPLATES = PATH_STATIC / DIR_TEMPLATES

# ======================================================================
# 📄 文件名（FILE_）——仍为字符串（文件名本身不含路径）
# ======================================================================
FILE_CONSTANTS = "constants.py"
FILE_ENUMS = "enums.py"
FILE_PROMPTS = "prompt_templates.py"
FILE_PYPROJECT = "pyproject.toml"
FILE_DEFAULT_TEMPLATE = "default_template.html"
FILE_CHAINA_IP_LIST = "china_ip_list.txt"
FILE_APP_JSON = "app.json"

# ======================================================================
# 📄 完整文件路径（基于前面路径 + 文件名拼接而成）
# ======================================================================
PATH_FILE_PROMPTS = PATH_STATIC_PROMPTS / FILE_PROMPTS
PATH_FILE_APP_JSON = PATH_STATIC_CONFIG / FILE_APP_JSON

PATH_FILE_PYPROJECT = ROOT_DIR / FILE_PYPROJECT

# 中国IP文件路径
PATH_FILE_CHAINA_IP_LIST = PATH_STATIC_OTHER / FILE_CHAINA_IP_LIST
# 默认模板
PATH_FILE_DEFAULT_TEMPLATE = PATH_STATIC_TEMPLATES / FILE_DEFAULT_TEMPLATE

# ======================================================================
# 📄 日志路径
# ======================================================================
LOG_KEEP_DAYS = 7
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 10
PATH_ROOT_LOGS = PATH_DATA / "logs"
PATH_ROOT_LOGS_FALLBACK = PATH_DATA / "logs_fallback"

# ======================================================================
# 🧩 枚举型常量（保持不变）
# ======================================================================
# 💾 存储后端
STORAGE_LOCAL = "local"
STORAGE_REDIS = "redis"


class LLMBackendConst:
    QWEN = "qwen"
    DEEPSEEK = "deepseek"

    @classmethod
    def all(cls) -> Set[str]:
        return {cls.QWEN, cls.DEEPSEEK}


class LLMModelConst:
    # Qwen 系列
    QWEN_MAX = "qwen-max"
    QWEN3_MAX = "qwen3-max"
    QWEN_PLUS = "qwen-plus"
    QWEN_FLASH = "qwen-flash"

    # DeepSeek 系列
    DEEPSEEK_CHAT = "deepseek-chat"

    @classmethod
    def all(cls) -> Set[str]:
        return {
            cls.QWEN_MAX,
            cls.QWEN3_MAX,
            cls.QWEN_PLUS,
            cls.QWEN_FLASH,
            cls.DEEPSEEK_CHAT,
        }

    @classmethod
    def by_backend(cls) -> Dict[str, List[str]]:
        return {
            LLMBackendConst.QWEN: [
                cls.QWEN_MAX,
                cls.QWEN3_MAX,
                cls.QWEN_PLUS,
                cls.QWEN_FLASH,
            ],
            LLMBackendConst.DEEPSEEK: [
                cls.DEEPSEEK_CHAT,
            ],
        }
