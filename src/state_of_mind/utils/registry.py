import threading
from typing import Type, Dict
import hashlib
import json
from src.state_of_mind.utils.logger import LoggerManager as logger


class GlobalSingletonRegistry:
    """
    全局单例注册中心
    - 注册 backend 类
    - 缓存 backend 实例（按连接参数唯一）
    - 缓存 MetaCognitiveEngine 实例（按完整配置唯一）
    """
    CHINESE_NAME = "GlobalSingletonRegistry"

    _backends: Dict[str, Type['LLMBackend']] = {}
    _backend_instances: Dict[str, 'LLMBackend'] = {}  # backend 实例缓存
    _engine_instances = {}  # MetaCognitiveEngine 实例缓存
    _lock = threading.Lock()

    @classmethod
    def register_backend(cls, name: str, backend_class: Type['LLMBackend']):
        """注册 LLM 后端类"""
        from src.state_of_mind.core.llm.base import LLMBackend
        if not issubclass(backend_class, LLMBackend):
            raise TypeError(f"Backend must inherit from LLMBackend, got {backend_class}")
        cls._backends[name] = backend_class
        logger.info("✅ 注册 LLM 后端: %s", name)

    @classmethod
    def _make_backend_key(cls, name: str, configs: dict) -> str:
        """
        基于 backend 名称和连接级配置生成唯一 key
        注意：只包含影响 client 初始化的参数
        """
        relevant_config = {
            "name": name,
            "api_key": configs.get("api_key"),  # 认证
            "timeout": configs.get("timeout"),  # 超时
        }
        config_str = json.dumps(relevant_config, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    @classmethod
    async def get_backend_async(cls, name: str, configs: dict = None) -> 'LLMBackend':
        """
        异步获取 LLMBackend 实例（带缓存）
        config: 仅包含连接级参数（如 api_key, timeout）
        """
        if name not in cls._backends:
            raise ValueError(f"未知的 LLM 后端: {name}")

        backend_class = cls._backends[name]
        key = cls._make_backend_key(name, configs or {})

        with cls._lock:
            if key not in cls._backend_instances:
                logger.info("🆕 创建 LLMBackend 实例", extra={"backend": name, "key": key[:8]})
                instance = backend_class()
                await instance.init(configs)  # 将连接配置传入
                cls._backend_instances[key] = instance
            return cls._backend_instances[key]

    @classmethod
    def get_extractor_instance(cls, backend_name: str, llm_model: str, recommended_params: dict):
        """
        获取 MetaCognitiveEngine 实例（单例缓存）
        cache_key 包含所有影响行为的参数
        """
        # ✅ 安全地生成可哈希键
        params_key = json.dumps(
            recommended_params or {},
            sort_keys=True,
            default=str,
            ensure_ascii=False
        )
        cache_key = (backend_name, llm_model, params_key)
        logger.info(f"Cache key: {cache_key}")

        if cache_key not in cls._engine_instances:
            with cls._lock:
                if cache_key not in cls._engine_instances:
                    logger.info(f"🆕 新建 MetaCognitiveEngine 实例: {backend_name}/{llm_model}")
                    from src.state_of_mind.core.engine import MetaCognitiveEngine
                    cls._engine_instances[cache_key] = MetaCognitiveEngine(
                        backend_name=backend_name,
                        llm_model=llm_model,
                        recommended_params=recommended_params
                    )
        return cls._engine_instances[cache_key]

    @classmethod
    def clear_llm_caches(cls):
        """
        清除所有 LLM 相关缓存实例（backend + engine）
        线程安全，适用于配置热重载场景
        """
        with cls._lock:
            # 先关闭 backend 实例（如有 close 方法）
            for instance in cls._backend_instances.values():
                if hasattr(instance, 'close') and callable(instance.close):
                    try:
                        instance.close()
                    except Exception as e:
                        logger.warning(f"⚠️ 关闭 backend 实例时出错: {e}")

            # 清空缓存
            cls._backend_instances.clear()
            cls._engine_instances.clear()
            logger.info("🧹 已清除所有 LLM backend 和 engine 缓存实例")