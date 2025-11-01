"""
日志管理器模块（终极优化版 v3 —— 终极形态）
- ✅ 完全解耦 fallback 日志系统
- ✅ 安全 inspect 栈帧处理（context=0，显式释放）
- ✅ 原子化异步日志清理（仅执行一次）
- ✅ 支持配置变更后自动重建 handler
- ✅ 清晰统一的类方法日志接口
- ✅ 支持 trace 级别 + 动态日志级别控制
- ✅ 多级 fallback：主目录 → 备用目录 → 临时目录 → 控制台
- ✅ 严格资源管理，避免泄漏
"""
import atexit
import inspect
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from pathlib import Path
import colorlog

from src.state_of_mind.utils.constants import (
    LOG_KEEP_DAYS as DEFAULT_LOG_KEEP_DAYS,
    PATH_ROOT_LOGS,
    LOG_MAX_BYTES as DEFAULT_LOG_MAX_BYTES,
    LOG_BACKUP_COUNT as DEFAULT_LOG_BACKUP_COUNT,
    PATH_ROOT_LOGS_FALLBACK,
)


# ======================================================================
# 1. 独立 FallbackLogger 模块（完全解耦，无外部依赖）
# ======================================================================


class FallbackLogger:
    """完全独立的 fallback 日志系统，避免任何递归或依赖问题"""
    _logger: Optional[logging.Logger] = None
    _lock = threading.Lock()

    # 日志级别映射
    _level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    @classmethod
    def _ensure_logger(cls):
        if cls._logger is not None:
            return cls._logger

        with cls._lock:
            if cls._logger is not None:
                return cls._logger

            logger = logging.getLogger("fallback.core")
            logger.setLevel(logging.INFO)
            logger.propagate = False

            # === 控制台输出（必选）===
            console = logging.StreamHandler(sys.stdout)
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s | FALLBACK | %(levelname)-8s | %(message)s%(reset)s",
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            )
            console.setFormatter(console_formatter)
            console.setLevel(logging.DEBUG)
            logger.addHandler(console)

            # === 文件输出（可选）===
            try:
                now = datetime.now().strftime("%Y-%m-%d")
                log_dir = PATH_ROOT_LOGS_FALLBACK / now
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "fallback.log"

                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=DEFAULT_LOG_MAX_BYTES,
                    backupCount=DEFAULT_LOG_BACKUP_COUNT,
                    encoding='utf-8'
                )
                file_formatter = logging.Formatter(
                    '%(asctime)s | FALLBACK | %(levelname)-8s | %(name)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(logging.INFO)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"[FALLBACK] Failed to create file handler: {e}", file=sys.stderr)

            cls._logger = logger
            return logger

    @classmethod
    def log(cls, level: str, msg: str, *args, **kwargs):
        logger = cls._ensure_logger()
        lvl = cls._level_map.get(level.upper(), logging.INFO)
        logger.log(lvl, msg, *args, **kwargs)

    @classmethod
    def debug(cls, msg, *args, **kwargs):
        cls.log('DEBUG', msg, *args, **kwargs)

    @classmethod
    def info(cls, msg, *args, **kwargs):
        cls.log('INFO', msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg, *args, **kwargs):
        cls.log('WARNING', msg, *args, **kwargs)

    @classmethod
    def error(cls, msg, *args, **kwargs):
        cls.log('ERROR', msg, *args, **kwargs)

    @classmethod
    def critical(cls, msg, *args, **kwargs):
        cls.log('CRITICAL', msg, *args, **kwargs)


# ======================================================================
# 2. 全局状态与配置加载（支持注入 + 环境变量 fallback）
# ======================================================================

# 自定义 TRACE 级别
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def trace(self: logging.Logger, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


logging.Logger.trace = trace  # type: ignore

# --- 全局状态 ---
_logger_dict: Dict[str, logging.Logger] = {}
_cleanup_done = False
_cleanup_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LogCleanup-")

_use_primary_logging = False
_primary_init_lock = threading.Lock()

_config_cache: Optional[Dict[str, int]] = None
_config_lock = threading.Lock()

_config_instance = None


def _get_config() -> Dict[str, Any]:
    """获取日志配置，优先使用注入的 config，其次环境变量，最后默认值"""
    global _config_cache
    with _config_lock:
        if _config_cache is not None:
            return _config_cache

        config = {}

        # 优先：注入的 config 实例
        if _config_instance is not None:
            try:
                config = {
                    "keep_days": int(getattr(_config_instance, "LOG_KEEP_DAYS", DEFAULT_LOG_KEEP_DAYS)),
                    "max_bytes": int(getattr(_config_instance, "LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES)),
                    "backup_count": int(getattr(_config_instance, "LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT)),
                    "logs_dir": Path(getattr(_config_instance, "LOGS_DIR", PATH_ROOT_LOGS)),
                    "logs_fallback_dir": Path(getattr(_config_instance, "LOGS_FALLBACK_DIR", PATH_ROOT_LOGS_FALLBACK)),
                }
                FallbackLogger.info("✅ 使用注入的 config 配置日志参数")
            except Exception as e:
                FallbackLogger.warning(f"⚠️ 从 config 实例读取日志配置失败: {e}")

        if not config:
            config = {
                "keep_days": DEFAULT_LOG_KEEP_DAYS,
                "max_bytes": DEFAULT_LOG_MAX_BYTES,
                "backup_count": DEFAULT_LOG_BACKUP_COUNT,
                "logs_dir": Path(PATH_ROOT_LOGS),
                "logs_fallback_dir": Path(PATH_ROOT_LOGS_FALLBACK),
            }
            FallbackLogger.info("✅ 使用默认日志配置")

        _config_cache = config
        return config


def _create_console_handler() -> logging.Handler:
    """创建统一的彩色控制台 handler"""
    handler = logging.StreamHandler(sys.stdout)
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(custom_module)s | %(custom_location)s | %(message)s%(reset)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
            'TRACE': 'blue',
        }
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    return handler


# ======================================================================
# 3. 主日志管理器（核心）
# ======================================================================

class LoggerManager:
    """终极版日志管理器"""

    CHINESE_NAME = "日志管理"

    _level_map = {
        'trace': TRACE,
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }

    @classmethod
    def initialize(cls, configured: bool = True):
        """初始化日志系统模式"""
        global _use_primary_logging
        with _primary_init_lock:
            _use_primary_logging = bool(configured)
            mode = "正式模式" if configured else "预初始化模式（fallback）"
            FallbackLogger.info(f"✅ 日志系统已切换至 {mode}")

    @classmethod
    def inject_config(cls, config_obj):
        """注入配置对象，触发缓存刷新"""
        global _config_instance, _config_cache
        if _config_instance is None:
            _config_instance = config_obj
            _config_cache = None
            FallbackLogger.info("🔧 已注入 config 实例，日志配置将刷新")

    @classmethod
    def clear_cache(cls):
        """清除配置缓存，强制重建所有 handler"""
        global _config_cache
        _config_cache = None
        # 标记所有 logger 需要重建
        for logger in _logger_dict.values():
            if isinstance(logger, logging.Logger):
                logger._needs_reconfigure = True  # type: ignore
        FallbackLogger.info("♻️ 日志配置缓存已清除，后续日志将重建 handler")

    @classmethod
    def set_global_level(cls, level: str):
        """动态设置全局日志级别"""
        lvl = cls._level_map.get(level.lower())
        if lvl is None:
            FallbackLogger.warning(f"⚠️ 无效的日志级别: {level}")
            return
        for logger in _logger_dict.values():
            logger.setLevel(lvl)
        FallbackLogger.info(f"🔧 全局日志级别已设置为: {level.upper()}")

    @classmethod
    def get_logger(cls, name: str = '心镜文本分析系统') -> logging.Logger:
        """获取或创建 logger"""
        name = name or "default"
        if name in _logger_dict:
            return _logger_dict[name]

        with threading.Lock():
            if name in _logger_dict:
                return _logger_dict[name]

            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False

            # 添加控制台（始终存在）
            logger.addHandler(_create_console_handler())

            # 标记状态
            logger._handlers_added = False  # type: ignore
            logger._needs_reconfigure = False  # type: ignore

            _logger_dict[name] = logger
            return logger

    @classmethod
    def _ensure_handlers(cls, logger: logging.Logger):
        """确保 logger 拥有正确的文件 handler（惰性初始化 + 重建）"""
        if not _use_primary_logging:
            return

        needs_reconfigure = getattr(logger, "_needs_reconfigure", False)
        if getattr(logger, "_handlers_added", False) and not needs_reconfigure:
            return

        with threading.Lock():
            if getattr(logger, "_handlers_added", False) and not needs_reconfigure:
                return

            config = _get_config()
            current_date = datetime.now().strftime("%Y-%m-%d")
            candidates = [
                config["logs_dir"] / current_date
            ]

            log_directory = None
            for candidate in candidates:
                try:
                    candidate.mkdir(parents=True, exist_ok=True)
                    log_directory = candidate
                    break
                except Exception as e:
                    FallbackLogger.debug(f"📁 路径创建失败: {candidate} -> {e}")
                    continue

            if not log_directory:
                FallbackLogger.error("❌ 所有日志路径均创建失败，仅使用控制台输出")
                logger._handlers_added = True
                logger._needs_reconfigure = False
                return

            # 移除旧的文件 handler
            for h in logger.handlers[:]:
                if isinstance(h, RotatingFileHandler):
                    logger.removeHandler(h)
                    h.close()

            # Info handler（INFO 及以上）
            info_file = log_directory / "info.log"
            try:
                handler = RotatingFileHandler(info_file, maxBytes=config["max_bytes"],
                                              backupCount=config["backup_count"], encoding='utf-8')
                formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | %(name)s | %(custom_module)s | %(custom_location)s | %(message)s | %(pathname)s:%(lineno)d',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)
            except Exception as e:
                FallbackLogger.warning(f"⚠️ 创建 info handler 失败: {e}")

            # Error handler（ERROR 及以上）
            error_file = log_directory / "error.log"
            try:
                handler = RotatingFileHandler(error_file, maxBytes=config["max_bytes"],
                                              backupCount=config["backup_count"], encoding='utf-8')
                formatter = logging.Formatter(
                    '%(asctime)s | ERROR | %(name)s | %(custom_module)s | %(custom_location)s | %(message)s | %(pathname)s:%(lineno)d',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                handler.setLevel(logging.ERROR)
                logger.addHandler(handler)
            except Exception as e:
                FallbackLogger.warning(f"⚠️ 创建 error handler 失败: {e}")

            logger._handlers_added = True
            logger._needs_reconfigure = False

    @classmethod
    def _async_cleanup(cls):
        """异步清理过期日志目录（原子化，仅执行一次）"""
        global _cleanup_done
        if not _cleanup_lock.acquire(blocking=False):
            return

        try:
            if _cleanup_done:
                return
            _cleanup_done = True
        finally:
            _cleanup_lock.release()

        config = _get_config()
        cutoff = datetime.now() - timedelta(days=config["keep_days"])

        for root_path in [config["logs_dir"], config["logs_fallback_dir"]]:
            if not root_path.exists():
                FallbackLogger.info(f"🔍 日志根目录不存在，跳过清理: {root_path}")
                continue

            deleted_count = 0
            for item in root_path.iterdir():
                if not item.is_dir():
                    continue
                try:
                    dir_date = datetime.strptime(item.name, "%Y-%m-%d")
                    if dir_date < cutoff:
                        # 安全删除所有子文件
                        for file in item.rglob("*"):
                            if file.is_file():
                                try:
                                    file.unlink()
                                except Exception as e:
                                    FallbackLogger.warning(f"⚠️ 删除文件失败: {file} -> {e}")
                        # 删除空目录
                        try:
                            item.rmdir()
                            FallbackLogger.info(f"🗑️ 已删除过期日志目录: {item}")
                            deleted_count += 1
                        except Exception as e:
                            FallbackLogger.error(f"❌ 删除目录失败: {item} -> {e}")
                except ValueError:
                    continue  # 无法解析的目录名，跳过

            if deleted_count == 0:
                FallbackLogger.info(f"✅ 无过期日志需清理: {root_path}")

    @classmethod
    def _log(
            cls,
            level: str,
            msg: str,
            *args,
            module_name: Optional[str] = None,
            location: Optional[str] = None,
            **kwargs
    ):
        """统一日志入口：固定字段进 formatter，动态 extra 拼到 message 末尾"""
        # 启动异步清理（仅一次）
        if not _cleanup_done:
            _executor.submit(cls._async_cleanup)

        logger = cls.get_logger()
        cls._ensure_handlers(logger)

        # === 安全获取调用上下文 ===
        frame = None
        try:
            # 限制上下文为0，避免深度遍历
            stack = inspect.stack(context=0)
            if len(stack) < 3:
                raise ValueError("Stack too shallow")

            frame = stack[2].frame
            f_locals = frame.f_locals
            func_name = frame.f_code.co_name or "<module>"
            class_name = None
            chinese_name = module_name or "未知模块"

            # 尝试获取类实例或类
            if 'self' in f_locals:
                instance = f_locals['self']
                cls_type = instance.__class__
                class_name = cls_type.__name__
                chinese_name = getattr(cls_type, 'CHINESE_NAME', chinese_name)
            elif 'cls' in f_locals and isinstance(f_locals['cls'], type):
                cls_type = f_locals['cls']
                class_name = cls_type.__name__
                chinese_name = getattr(cls_type, 'CHINESE_NAME', chinese_name)

            final_location = location or (f"{class_name}.{func_name}" if class_name else func_name)

            # === 1. 分离固定字段 和 动态 extra ===
            user_extra = kwargs.get('extra', {}).copy()  # 用户传的 extra
            fixed_extra = {
                'custom_module': chinese_name,
                'custom_location': final_location,
                'password': "",
                'token': ""
            }

            # 动态部分 = 用户 extra 减去固定字段（避免重复）
            dynamic_extra = {
                k: v for k, v in user_extra.items()
                if k not in fixed_extra
            }

            # === 2. 构建最终 message（拼接 dynamic_extra）===
            final_msg = msg
            if dynamic_extra:
                try:
                    from json import dumps
                    # 紧凑格式，避免换行/空格破坏日志结构
                    extra_str = dumps(dynamic_extra, ensure_ascii=False, separators=(',', ':'), default=str)
                    final_msg = f"{msg} | extra:{extra_str}"
                except Exception:
                    final_msg = f"{msg} | extra:<serialize failed>"

            # === 3. 构建最终 extra（只包含固定字段，供 formatter 使用）===
            final_extra = {**fixed_extra, **{k: v for k, v in user_extra.items() if k in fixed_extra}}
            # 注意：这里也可以直接用 fixed_extra，除非你允许用户覆盖 custom_module 等

            # === 4. 清理 kwargs，只保留 logging 支持的参数 ===
            supported_keys = {'exc_info', 'stack_info', 'stacklevel', 'extra'}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_keys}
            filtered_kwargs['extra'] = final_extra  # 注入固定字段

            # === 5. 记录日志 ===
            log_func = getattr(logger, level, logger.info)
            log_func(final_msg, *args, **filtered_kwargs)

        except Exception as e:
            FallbackLogger.error(f"📌 日志记录失败: {e} | msg='{msg}'")
        finally:
            del frame  # 显式释放
            del stack  # 显式释放

    # === 公共日志接口 ===
    @classmethod
    def trace(cls, msg: str, *args, module_name: str = None, location: str = None, **kwargs):
        cls._log('trace', msg, *args, module_name=module_name, location=location, **kwargs)

    @classmethod
    def debug(cls, msg: str, *args, module_name: str = None, location: str = None, **kwargs):
        cls._log('debug', msg, *args, module_name=module_name, location=location, **kwargs)

    @classmethod
    def info(cls, msg: str, *args, module_name: str = None, location: str = None, **kwargs):
        cls._log('info', msg, *args, module_name=module_name, location=location, **kwargs)

    @classmethod
    def warning(cls, msg: str, *args, module_name: str = None, location: str = None, **kwargs):
        cls._log('warning', msg, *args, module_name=module_name, location=location, **kwargs)

    @classmethod
    def error(cls, msg: str, *args, module_name: str = None, location: str = None, **kwargs):
        cls._log('error', msg, *args, module_name=module_name, location=location, **kwargs)

    @classmethod
    def critical(cls, msg: str, *args, module_name: str = None, location: str = None, **kwargs):
        cls._log('critical', msg, *args, module_name=module_name, location=location, **kwargs)

    @classmethod
    def exception(cls, msg: str, *args, module_name: str = None, location: str = None, **kwargs):
        kwargs['exc_info'] = True
        cls.error(msg, *args, module_name=module_name, location=location, **kwargs)


# ======================================================================
# 4. 资源清理与退出钩子
# ======================================================================
@atexit.register
def _cleanup_resources():
    """程序退出时清理资源"""
    FallbackLogger.info("🛑 正在关闭日志系统...")
    _executor.shutdown(wait=True)
    FallbackLogger.info("✅ 日志系统已关闭")
