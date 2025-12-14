from __future__ import annotations
from typing import Type, Dict, ClassVar
import hashlib
import json
import asyncio
from src.state_of_mind.llm.base import LLMBackend
from src.state_of_mind.utils.logger import LoggerManager as logger


class GlobalSingletonRegistry:
    """
    全局注册中心
    - 注册 LLM 后端类（如 qwen、deepseek）
    - 按连接参数缓存 LLMBackend 实例（线程安全 + 异步初始化）
    - 支持运行时清除缓存以实现配置热重载
    """
    CHINESE_NAME = "全局注册中心"

    _backends: Dict[str, Type[LLMBackend]] = {}
    _backend_instances: Dict[str, LLMBackend] = {}  # backend 实例缓存
    # 使用 asyncio.Lock，但注意：不能在类定义时直接实例化（需延迟）
    _lock: ClassVar[asyncio.Lock] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[LLMBackend]):
        """注册 LLM 后端类"""
        if not issubclass(backend_class, LLMBackend):
            raise TypeError(f"Backend must inherit from LLMBackend, got {backend_class}")
        cls._backends[name] = backend_class
        logger.info("✅ 注册 LLM 后端: %s", name)

    @classmethod
    def _make_backend_key(cls, name: str, llm_config: dict) -> str:
        """
        基于 backend 名称和连接级配置生成唯一 key
        """
        key_data = {
            "backend": name,
            "api_key_hash": hashlib.md5(llm_config["api_key"].encode()).hexdigest()[:8],
            "timeout": llm_config["timeout"]
        }
        # 可选字段：只有当 backend 实际使用时才加入
        backend_class = cls._backends[name]
        if getattr(backend_class, '_uses_api_url', True):  # 默认 True
            key_data["api_url"] = llm_config.get("api_url", "")
        config_str = json.dumps(key_data, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    @classmethod
    async def get_backend_async(cls, name: str) -> LLMBackend:
        if name not in cls._backends:
            raise ValueError(f"未知的 LLM 后端: {name}")

        # === 统一配置合并逻辑 ===
        llm_config = cls._resolve_backend_configs()
        key = cls._make_backend_key(name, llm_config)

        lock = cls._get_lock()
        async with lock:
            if key not in cls._backend_instances:
                logger.info(f"🆕 创建 {name} LLMBackend 实例（配置变更）")
                try:
                    instance = cls._backends[name]()
                    await instance.init(llm_config)
                    cls._backend_instances[key] = instance
                except Exception as e:
                    logger.error(f"❌ 初始化 {name} backend 失败: {e}")
                    raise
            return cls._backend_instances[key]

    @classmethod
    def _resolve_backend_configs(cls) -> dict:
        from src.state_of_mind.config import config
        llm_config = {
            "api_key": config.LLM_API_KEY,
            "timeout": config.get("LLM_API_TIMEOUT", 120),
            "api_url": config.LLM_API_URL
        }
        return llm_config

    @classmethod
    async def async_clear_llm_caches(cls):
        async with cls._get_lock():
            for instance in cls._backend_instances.values():
                if hasattr(instance, 'close') and callable(instance.close):
                    try:
                        instance.close()
                    except Exception as e:
                        logger.warning(f"⚠️ 关闭 backend 实例时出错: {e}")
            cls._backend_instances.clear()
            logger.info("🧹 已清除所有 LLM backend 缓存实例")
