"""
🌊 心境配置中枢
"""
import importlib.metadata
import os
import threading
from pathlib import Path
from typing import Dict, Any
from src.state_of_mind.utils.constants import (
    ROOT_DIR,
    STORAGE_LOCAL,
    LOG_KEEP_DAYS, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    PATH_FILE_PYPROJECT,
    PATH_FILE_CHAINA_IP_LIST, PATH_FILE_PROMPTS,
    PATH_FILE_DEFAULT_TEMPLATE, STORAGE_REDIS, ModelName, PATH_FILE_APP_JSON,
)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.state_of_mind.utils.file_util import FileUtil
from src.state_of_mind.utils.logger import LoggerManager, _logger_dict

try:
    import tomli as tomllib  # Python < 3.11
except ImportError:
    import tomllib  # Python 3.11+


class Config:
    CHINESE_NAME = "心境配置中枢"

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
        'MAX_PARALLEL_CONCURRENCY', 'LLM_CACHE_MAX_SIZE', 'LLM_CACHE_TTL',
        '_observer', '_watcher_thread', '_stop_event', 'logger', 'metadata', '__setitem__',
    ]

    def __init__(self):
        self.ROOT_DIR = ROOT_DIR
        self._observer = None
        self._watcher_thread = None
        self._stop_event = threading.Event()
        self.metadata = self._load_metadata()
        self.VERSION = self.metadata.get("version", "dev-local")

        self.logger = LoggerManager  # 使用类接口
        self.logger.info("🌊 正在加载心海配置...")
        self._load()
        self.start_watcher()

        # 注入配置
        self.logger.inject_config(self)
        self.logger.initialize(configured=True)
        self.logger.info("🌊 心镜配置加载完成")

    def _load_metadata(self) -> Dict[str, Any]:
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

            # 解析 Project-URLs（格式: "Label, URL"）
            raw_urls = {}
            for item in pkg_metadata.get_all("Project-URL", []):
                label, sep, url = item.partition(",")
                if sep:  # 确保有逗号分隔
                    raw_urls[label.strip()] = url.strip()
                else:
                    # 兼容异常格式，跳过或记录警告（可选）
                    self.logger.debug(f"跳过无效 Project-URL 格式: {item}")

            # Home-page 是单独字段（旧式）
            if "Home-page" in pkg_metadata:
                raw_urls.setdefault("Homepage", pkg_metadata["Home-page"])

            # 标准化 urls：只保留关心的键，但允许扩展（可选）
            metadata["urls"] = {
                key: raw_urls.get(key, "") for key in STANDARD_URL_KEYS
            }

        except importlib.metadata.PackageNotFoundError:
            pass  # 包未安装，继续从 pyproject.toml 读
        except Exception as e:
            self.logger.warning(f"读取已安装包元数据失败: {e}")

        # 🔍 第二步：回退到读取本地 pyproject.toml
        pyproject_path = PATH_FILE_PYPROJECT
        if not pyproject_path.exists():
            self.logger.warning(f"⚠️ pyproject.toml 不存在: {pyproject_path}")
            return metadata

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            proj = data.get("project", {})

            # 覆盖基础字段
            metadata["name"] = proj.get("name", metadata["name"])
            metadata["version"] = proj.get("version", metadata["version"])
            metadata["description"] = proj.get("description", metadata["description"])

            # License（PEP 621 支持 str 或 {text: "..."}）
            license_info = proj.get("license")
            if isinstance(license_info, dict):
                metadata["license"] = license_info.get("text", "")
            elif isinstance(license_info, str):
                metadata["license"] = license_info

            # Authors（优先 authors，其次 maintainers）
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
            self.logger.exception(f"[CRITICAL] 读取 pyproject.toml 失败: {e}")
            return metadata

    def _load(self):
        # === Step 1: 从 app.json 读取配置 ===
        self.PATH_FILE_APP_JSON = PATH_FILE_APP_JSON
        raw_config = FileUtil().read_json_file(PATH_FILE_APP_JSON)

        # === Step 2: 定义一个辅助函数：优先从环境变量取，其次从 raw_config，最后用默认值 ===
        def get_config(key: str, default, cast):
            env_key = key
            if env_key in os.environ:
                val = os.environ[env_key]
                if cast == int:
                    return int(val)
                elif cast == bool:
                    return val.lower() in ("true", "1", "yes", "on")
                return val
            return raw_config.get(key, default)

        self.OUTPUT_ROOT = Path("/home/psytext_analyst/data")
        # self.OUTPUT_ROOT = Path(DEFAULT_OUTPUT_ROOT) # 本地测试使用

        # === Step 3: 构建动态路径 ===
        self._setup_paths()

        # === Step 4: 存储配置 ===
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

        # === Step 5: LLM 配置 ===
        self.LLM_BACKEND = str(raw_config.get("XINJING_LLM_BACKEND")).lower()
        self.LLM_MODEL = str(raw_config.get("XINJING_LLM_MODEL")).lower()
        self.LLM_API_URL = raw_config.get("XINJING_LLM_API_URL")
        self.LLM_API_KEY = raw_config.get("XINJING_LLM_API_KEY")
        self.SUGGESTION_TYPE = raw_config.get("XINJING_SUGGESTION_TYPE")
        self.LLM_RECOMMENDED_PARAMS = raw_config.get("XINJING_LLM_RECOMMENDED_PARAMS")

        self.REPORT_TITLE = get_config("XINJING_REPORT_TITLE", "文本多模态感知分析报告", cast=str)

        # 设置默认值（仅当统一字段未提供时）
        if self.LLM_BACKEND == ModelName.QWEN:
            if not self.LLM_MODEL:
                self.LLM_MODEL = ModelName.QWEN3_MAX
            if not self.LLM_API_URL:
                self.LLM_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        elif self.LLM_BACKEND == ModelName.DEEPSEEK:
            if not self.LLM_MODEL:
                self.LLM_MODEL = ModelName.DEEPSEEK_CHAT
            if not self.LLM_API_URL:
                self.LLM_API_URL = "https://api.deepseek.com"

        # === Step 6: 日志配置（这些是常量，不动）===
        self.LOG_KEEP_DAYS = int(raw_config.get("XINJING_LOG_KEEP_DAYS")) or int(LOG_KEEP_DAYS)
        self.LOG_MAX_BYTES = int(raw_config.get("XINJING_LOG_MAX_BYTES")) or int(LOG_MAX_BYTES)
        self.LOG_BACKUP_COUNT = int(raw_config.get("XINJING_LOG_BACKUP_COUNT")) or int(LOG_BACKUP_COUNT)

        # === Step 7: 并发 ===
        self.MAX_PARALLEL_CONCURRENCY = int(raw_config.get("XINJING_MAX_PARALLEL_CONCURRENCY", 10))
        self.CURRENT_PARALLEL_CONCURRENCY = int(raw_config.get("XINJING_CURRENT_PARALLEL_CONCURRENCY", 3))
        self.MEDIUM_PARALLEL_CONCURRENCY = int(raw_config.get("XINJING_MEDIUM_PARALLEL_CONCURRENCY", 5))

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

    def _parse_bool(self, val: str) -> bool:
        return val.strip().lower() in ("true", "1", "yes", "on")

    def get(self, key: str, default=None):
        return getattr(self, key.upper(), default)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__ if not k.startswith("_") and hasattr(self, k)}

    def reload(self):
        old_config = self.as_dict()
        self._load()
        new_config = self.as_dict()
        self.logger.info("🔄 心镜配置已热重载")

        # 计算变更项
        diff_keys = {k for k in old_config if old_config[k] != new_config[k]}
        if diff_keys:
            self.logger.info("🔧 配置变更项: %s", sorted(diff_keys))

        # === 1. 处理日志系统重建 ===
        LOG_KEYS = {"LOG_KEEP_DAYS", "LOG_MAX_BYTES", "LOG_BACKUP_COUNT"}
        if diff_keys & LOG_KEYS:
            self.logger.clear_cache()
            for logger_obj in _logger_dict.values():
                if hasattr(logger_obj, "_handlers_added"):
                    logger_obj._needs_reconfigure = True

        # === 2. 处理 LLM 引擎缓存清理（精准判断）===
        LLM_SENSITIVE_KEYS = {
            "LLM_BACKEND",
            "LLM_MODEL",
            "LLM_API_KEY",
            "LLM_API_URL",
            "LLM_RECOMMENDED_PARAMS"
        }

        if diff_keys & LLM_SENSITIVE_KEYS:
            self.logger.info("🔑 检测到 LLM 敏感配置变更，触发缓存清理...")
            try:
                from src.state_of_mind.utils.registry import GlobalSingletonRegistry
                GlobalSingletonRegistry.clear_llm_caches()
            except Exception as e:
                self.logger.exception(f"❌ 清理 LLM 缓存失败: {e}")

    # ==================== 文件监听 ====================

    def _start_observer(self):
        observer = Observer()
        event_handler = self._create_event_handler()
        observer.schedule(event_handler, str(self.ROOT_DIR), recursive=False)
        observer.start()
        self._observer = observer

        try:
            while not self._stop_event.wait(1):
                pass
        except Exception as e:
            self.logger.exception(f"🛑 监听器异常: {e}")
        finally:
            observer.stop()
            observer.join()
            self.logger.info("🛑 监听器已停止")

    def _create_event_handler(self):
        config = self

        class EnvFileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if p.name in PATH_FILE_APP_JSON.resolve():
                    config.logger.info(f"📁 检测到配置文件变更: {p.name}，触发热重载...")
                    config.reload()

        return EnvFileHandler()

    def start_watcher(self):
        if self._watcher_thread and self._watcher_thread.is_alive():
            self.logger.warning("👀 监听器已在运行中...")
            return
        self._stop_event.clear()
        self._watcher_thread = threading.Thread(target=self._start_observer, daemon=True)
        self._watcher_thread.start()
        self.logger.info("👀 已启动配置文件监听器（自动热重载）")

    def stop_watcher(self):
        if self._stop_event.is_set():
            self.logger.info("🛑 监听器未运行，无需停止")
            return
        self._stop_event.set()
        if self._observer:
            self._observer.stop()
        self.logger.info("🛑 正在停止监听器...")
        if self._watcher_thread:
            self._watcher_thread.join(timeout=3)
            if self._watcher_thread.is_alive():
                self.logger.warning("⚠️ 监听线程未能及时退出")
            else:
                self.logger.info("✅ 监听器已安全停止")


# 🌊 全局配置实例
config = Config()
