"""
🌊 心境配置中枢
"""
import importlib.metadata
import os
from pathlib import Path
from typing import Dict, Any
from src.state_of_mind import GlobalSingletonRegistry
from src.state_of_mind.utils.constants import (
    ROOT_DIR,
    STORAGE_LOCAL,
    LOG_KEEP_DAYS, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    PATH_FILE_PYPROJECT,
    PATH_FILE_CHAINA_IP_LIST, PATH_FILE_PROMPTS,
    PATH_FILE_DEFAULT_TEMPLATE, STORAGE_REDIS, PATH_FILE_APP_JSON, LLMBackendConst, LLMModelConst,
)
from src.state_of_mind.utils.file_util import FileUtil
from src.state_of_mind.utils.logger import FallbackLogger

try:
    import tomli as tomllib  # Python < 3.11
except ImportError:
    import tomllib  # Python 3.11+


class Config:
    CHINESE_NAME = "心海配置中枢"

    __slots__ = [
        'ROOT_DIR', 'OUTPUT_ROOT', 'VERSION', 'LLM_RECOMMENDED_PARAMS',
        'DATA_YUAN_RAW_DIR', 'DATA_YUAN_DYE_VAT_DIR', 'REPORTS_DIR',
        'LOGS_DIR', 'LOGS_FALLBACK_DIR', 'PATH_FILE_APP_JSON', 'REPORT_TITLE',
        'STATIC_PROMPTS_DIR', 'STATIC_REPORTS_DIR', 'SUGGESTION_TYPE',
        'FILE_PROMPTS_PATH', 'FILE_CHAINA_IP_LIST_PATH', 'FILE_DEFAULT_TEMPLATE_PATH',
        'STORAGE_BACKEND', 'STORAGE_LOCAL', 'STORAGE_REDIS', 'MEDIUM_PARALLEL_CONCURRENCY',
        'REDIS_HOST', 'REDIS_PORT', 'REDIS_DB', 'REDIS_PASSWORD', 'REDIS_TIMEOUT',
        'LLM_BACKEND', 'LLM_MODEL', 'LLM_API_URL', 'LLM_API_KEY', 'CURRENT_PARALLEL_CONCURRENCY',
        'LOG_KEEP_DAYS', 'LOG_MAX_BYTES', 'LOG_BACKUP_COUNT', 'LOG_ENABLE_INSPECT',
        'MAX_PARALLEL_CONCURRENCY', 'LLM_CACHE_MAX_SIZE', 'LLM_CACHE_TTL', 'LLM_API_TIMEOUT',
        'WATERMARK_ENABLED', 'WATERMARK_TEXT', 'WATERMARK_COLOR', 'WATERMARK_OPACITY',
        'WATERMARK_FONT_SIZE', 'WATERMARK_ANGLE', 'WATERMARK_SPACING_COLS', 'WATERMARK_SPACING_ROWS',
        'WATERMARK_PADDING',
        'logger', 'metadata', '_registry',
    ]

    def __init__(self, registry=None):
        FallbackLogger.info("🔄 正在初始化心海配置中枢...")
        self.ROOT_DIR = ROOT_DIR
        self.metadata = self._load_metadata()
        self.VERSION = self.metadata.get("version", "dev-local")
        self._load()

        """
        初始化配置中枢。
        :param registry: 全局单例注册中心，用于在配置重载时清理 LLM 缓存。
                         若未提供，则使用默认的 GlobalSingletonRegistry。
        """
        self._registry = registry or GlobalSingletonRegistry

    def _load(self):
        # === 从 app.json 读取配置 ===
        self.PATH_FILE_APP_JSON = PATH_FILE_APP_JSON
        raw_config = FileUtil().read_json_file(PATH_FILE_APP_JSON)

        # === 定义一个辅助函数：优先从环境变量取，其次从 raw_config，最后用默认值 ===
        def get_config(key: str, default, cast):
            if key in os.environ:
                val = os.environ[key]
                try:
                    if cast == int:
                        return int(val)
                    elif cast == float:
                        return float(val)
                    elif cast == bool:
                        return val.strip().lower() in ("true", "1", "yes", "on", "ok")
                    elif cast == dict:
                        import json
                        return json.loads(val)
                    elif cast == str:
                        return val
                    else:
                        return val  # 默认原样返回
                except Exception as e:
                    FallbackLogger.warning(
                        f"环境变量 {key}={val} 转换失败: {e}，使用默认值"
                    )
            return raw_config.get(key, default)

        self.OUTPUT_ROOT = Path("/home/appuser/psytext_data")

        # === 构建动态路径 ===
        self._setup_paths()

        # === 存储配置 ===
        self.STORAGE_LOCAL = STORAGE_LOCAL
        self.STORAGE_REDIS = STORAGE_REDIS
        self.STORAGE_BACKEND = get_config("XINJING_STORAGE_BACKEND", STORAGE_LOCAL, cast=str)
        self.LLM_CACHE_MAX_SIZE = get_config("XINJING_LLM_CACHE_MAX_SIZE", 4096, cast=int)
        self.LLM_CACHE_TTL = get_config("XINJING_LLM_CACHE_TTL", 3600, cast=int)
        self.REDIS_HOST = get_config("XINJING_REDIS_HOST", "redis", cast=str)
        self.REDIS_PORT = get_config("XINJING_REDIS_PORT", 6379, cast=int)
        self.REDIS_DB = get_config("XINJING_REDIS_DB", 0, cast=int)
        self.REDIS_PASSWORD = get_config("XINJING_REDIS_PASSWORD", None, cast=str)  # 注意：环境变量中 null 要传空字符串
        self.REDIS_TIMEOUT = get_config("XINJING_REDIS_TIMEOUT", 5, cast=int)
        self.REPORT_TITLE = get_config("XINJING_REPORT_TITLE", "全息感知基底分析报告", cast=str)

        # === LLM 配置（支持 env + 智能默认值 + 大小写归一）===
        raw_backend = get_config("XINJING_LLM_BACKEND", LLMBackendConst.DEEPSEEK, cast=str)
        self.LLM_BACKEND = str(raw_backend).strip().lower() if raw_backend else LLMBackendConst.DEEPSEEK

        raw_model = get_config("XINJING_LLM_MODEL", "", cast=str)
        self.LLM_MODEL = str(raw_model).strip().lower() if raw_model else LLMModelConst.DEEPSEEK_CHAT
        self.LLM_API_URL = get_config("XINJING_LLM_API_URL", None, cast=str)
        self.LLM_API_KEY = get_config("XINJING_LLM_API_KEY", None, cast=str)
        self.LLM_API_TIMEOUT = get_config("XINJING_LLM_API_TIMEOUT", 120, cast=int)
        self.SUGGESTION_TYPE = get_config("XINJING_SUGGESTION_TYPE", "ironic_deconstructor", cast=str)
        recommended_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_output_tokens": 2048,
            "result_format": "json_object"
        }
        self.LLM_RECOMMENDED_PARAMS = get_config("XINJING_LLM_RECOMMENDED_PARAMS", recommended_params, cast=dict)

        # === 智能默认值填充（保持你的原逻辑）===
        if self.LLM_BACKEND == LLMBackendConst.QWEN:
            if not self.LLM_MODEL:
                self.LLM_MODEL = LLMModelConst.QWEN3_MAX
            if not self.LLM_API_URL:
                self.LLM_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        elif self.LLM_BACKEND == LLMBackendConst.DEEPSEEK:
            if not self.LLM_MODEL:
                self.LLM_MODEL = LLMModelConst.DEEPSEEK_CHAT
            if not self.LLM_API_URL:
                self.LLM_API_URL = "https://api.deepseek.com"
        else:
            FallbackLogger.warning(
                f"未知 LLM 后端: {self.LLM_BACKEND}，请确保 LLM_API_URL 和 LLM_MODEL 已手动配置"
            )

        # === 动态修正 result_format（适配 Qwen3 系列）===
        if self.LLM_BACKEND == LLMBackendConst.QWEN and self.LLM_MODEL == LLMModelConst.QWEN3_MAX:
            current_fmt = self.LLM_RECOMMENDED_PARAMS.get("result_format")
            if current_fmt == "json_object":
                self.LLM_RECOMMENDED_PARAMS["result_format"] = "message"
                FallbackLogger.info(
                    "检测到 Qwen3 系列模型，已将 result_format 从 'json_object' 修正为 'message'"
                )

        # === 日志配置（这些是常量，不动）===
        self.LOG_KEEP_DAYS = get_config("XINJING_LOG_KEEP_DAYS", LOG_KEEP_DAYS, cast=int)
        self.LOG_MAX_BYTES = get_config("XINJING_LOG_MAX_BYTES", LOG_MAX_BYTES, cast=int)
        self.LOG_BACKUP_COUNT = get_config("XINJING_LOG_BACKUP_COUNT", LOG_BACKUP_COUNT, cast=int)

        # === 并发 ===
        self.MAX_PARALLEL_CONCURRENCY = get_config("XINJING_MAX_PARALLEL_CONCURRENCY", 10, cast=int)
        self.CURRENT_PARALLEL_CONCURRENCY = get_config("XINJING_CURRENT_PARALLEL_CONCURRENCY", 3, cast=int)
        self.MEDIUM_PARALLEL_CONCURRENCY = get_config("XINJING_MEDIUM_PARALLEL_CONCURRENCY", 5, cast=int)

        # === 水印相关 ===
        self.WATERMARK_ENABLED = get_config("XINJING_WATERMARK_ENABLED", True, cast=bool)
        self.WATERMARK_TEXT = get_config("XINJING_WATERMARK_TEXT", "内部审计严禁外传", cast=str)
        self.WATERMARK_COLOR = get_config("XINJING_WATERMARK_COLOR", "rgba(54, 52, 52, 0.9)", cast=str)
        self.WATERMARK_OPACITY = get_config("XINJING_WATERMARK_OPACITY", 0.12, cast=float)
        self.WATERMARK_FONT_SIZE = get_config("XINJING_WATERMARK_FONT_SIZE", 48, cast=int)
        self.WATERMARK_ANGLE = get_config("XINJING_WATERMARK_ANGLE", -30, cast=int)
        self.WATERMARK_SPACING_COLS = get_config("XINJING_WATERMARK_SPACING_COLS", 5, cast=int)
        self.WATERMARK_SPACING_ROWS = get_config("XINJING_WATERMARK_SPACING_ROWS", 8, cast=int)
        self.WATERMARK_PADDING = get_config("XINJING_WATERMARK_PADDING", 30, cast=int)

    @staticmethod
    def _load_metadata() -> Dict[str, Any]:
        # 关心的 URL 键（用于标准化输出）
        STANDARD_URL_KEYS = ("Homepage", "Repository", "Documentation")

        # 默认 fallback 值
        metadata = {
            "name": "psytext-analyst",
            "version": "dev-local",
            "description": "心镜文本分析系统",
            "authors": [],
            "license": "",
            "urls": {key: "" for key in STANDARD_URL_KEYS}
        }

        # 🔍 第一步：尝试通过 importlib.metadata 读取已安装包的元数据
        try:
            normalized_name = "psytext-analyst"
            pkg_metadata = importlib.metadata.metadata(normalized_name)

            metadata["name"] = pkg_metadata.get("Name", metadata["name"])
            metadata["version"] = pkg_metadata.get("Version", metadata["version"])
            metadata["description"] = pkg_metadata.get("Summary", metadata["description"])
            metadata["license"] = pkg_metadata.get("License", metadata["license"])

            # 处理作者：支持 Author + Author-email
            authors = []
            author = pkg_metadata.get("Author")
            author_email = pkg_metadata.get("Author-email")
            if author:
                authors.append({"name": author.strip(), "email": (author_email or "").strip()})
            metadata["authors"] = authors

            raw_urls = {}
            for item in pkg_metadata.get_all("Project-URL", []):
                label, sep, url = item.partition(",")
                if sep:
                    raw_urls[label.strip()] = url.strip()
                else:
                    FallbackLogger.warning(f"跳过无效 Project-URL 格式: {item}")

            if "Home-page" in pkg_metadata:
                raw_urls.setdefault("Homepage", pkg_metadata["Home-page"])

            metadata["urls"] = {
                key: raw_urls.get(key, "") for key in STANDARD_URL_KEYS
            }

        except importlib.metadata.PackageNotFoundError:
            pass
        except Exception as e:
            FallbackLogger.error(f"读取已安装包元数据失败: {e}")

        pyproject_path = PATH_FILE_PYPROJECT
        if not pyproject_path.exists():
            FallbackLogger.warning(f"pyproject.toml 不存在: {pyproject_path}")
            return metadata

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            proj = data.get("project", {})

            metadata["name"] = proj.get("name", metadata["name"])
            metadata["version"] = proj.get("version", metadata["version"])
            metadata["description"] = proj.get("description", metadata["description"])

            license_info = proj.get("license")
            if isinstance(license_info, dict):
                metadata["license"] = license_info.get("text", "")
            elif isinstance(license_info, str):
                metadata["license"] = license_info

            def _parse_persons(persons):
                result = []
                for p in persons or []:
                    if isinstance(p, dict):
                        name = p.get("name", "").strip()
                        email = p.get("email", "").strip()
                        if name:
                            result.append({"name": name, "email": email})
                return result

            authors = _parse_persons(proj.get("authors"))
            if not authors:
                authors = _parse_persons(proj.get("maintainers"))
            metadata["authors"] = authors

            # URLs：从 TOML 直接读取
            toml_urls = proj.get("urls", {})
            metadata["urls"] = {
                key: toml_urls.get(key, "") for key in STANDARD_URL_KEYS
            }

            return metadata

        except Exception as e:
            FallbackLogger.critical(f"读取 pyproject.toml 失败: {e}")
            return metadata

    def _setup_paths(self):
        """基于 OUTPUT_ROOT 动态构建所有输出路径"""
        # 输出目录
        self.DATA_YUAN_RAW_DIR = self.OUTPUT_ROOT / "raw"
        self.DATA_YUAN_DYE_VAT_DIR = self.OUTPUT_ROOT / "dye_vat"
        self.REPORTS_DIR = self.OUTPUT_ROOT / "reports"
        self.LOGS_DIR = self.OUTPUT_ROOT / "logs"
        self.LOGS_FALLBACK_DIR = self.OUTPUT_ROOT / "logs_fallback"

        for d in [self.DATA_YUAN_RAW_DIR, self.DATA_YUAN_DYE_VAT_DIR, self.REPORTS_DIR, self.LOGS_DIR,
                  self.LOGS_FALLBACK_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        self.FILE_PROMPTS_PATH = PATH_FILE_PROMPTS
        self.FILE_CHAINA_IP_LIST_PATH = PATH_FILE_CHAINA_IP_LIST
        self.FILE_DEFAULT_TEMPLATE_PATH = PATH_FILE_DEFAULT_TEMPLATE

    @staticmethod
    def _parse_bool(val: str) -> bool:
        return val.strip().lower() in ("true", "1", "yes", "on", "ok")

    def get(self, key: str, default=None):
        return getattr(self, key.upper(), default)

    async def reload(self):
        # 定义一个唯一哨兵对象
        _MISSING = object()
        # 只对比非私有配置字段
        config_keys = {k for k in self.__slots__ if not k.startswith("_")}

        old_config = {k: getattr(self, k, _MISSING) for k in config_keys}
        self._load()  # 重新加载
        new_config = {k: getattr(self, k, _MISSING) for k in config_keys}

        diff_keys = {k for k in config_keys if old_config[k] != new_config[k]}
        if diff_keys:
            FallbackLogger.info(f"配置变更项: {sorted(diff_keys)}")

        LLM_SENSITIVE_KEYS = {
            "LLM_BACKEND",
            "LLM_MODEL",
            "LLM_API_KEY",
            "LLM_API_URL",
            "LLM_API_TIMEOUT",
            "LLM_RECOMMENDED_PARAMS"
        }

        if diff_keys & LLM_SENSITIVE_KEYS:
            FallbackLogger.info(
                "检测到 LLM 敏感配置变更，触发缓存清理..."
            )
            try:
                await self._on_llm_config_changed()
            except Exception as e:
                FallbackLogger.error(
                    f"清理 LLM 缓存失败: {e}"
                )

    async def _on_llm_config_changed(self):
        """当 LLM 相关配置变更时触发的回调"""
        await self._registry.async_clear_llm_caches()


# 🌊 全局配置实例
config = Config()
