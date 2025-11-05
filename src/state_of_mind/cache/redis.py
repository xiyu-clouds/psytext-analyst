# import json
from typing import List, Optional, Dict, Any
from aiocache import Cache
# from aiocache.serializers import JsonSerializer

from src.state_of_mind.cache.serializer import UTF8JsonSerializer
from src.state_of_mind.utils.logger import LoggerManager as logger
from src.state_of_mind.cache.base import BaseCache


class RedisLLMCache(BaseCache):
    CHINESE_NAME = "Redis LLM 缓存后端"

    def __init__(
            self,
            config,
            default_ttl: Optional[int] = None,
    ):
        try:
            import redis  # noqa
        except ImportError:
            raise RuntimeError(
                "❌ Redis 缓存需要安装 'redis' 包。请在 requirements.txt 中添加 'redis' 并重建镜像。"
            )

        self.config = config
        self.default_ttl = default_ttl or int(config.LLM_CACHE_TTL)
        if not isinstance(self.default_ttl, int) or self.default_ttl < 0:
            raise ValueError("default_ttl 必须是非负整数")

        self._cache = Cache(
            Cache.REDIS,
            endpoint=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            password=config.REDIS_PASSWORD or None,
            timeout=config.REDIS_TIMEOUT,
            # serializer=JsonSerializer(),
            serializer=UTF8JsonSerializer(),
            namespace="psytext_analyst",
        )

        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(
            f"🔌 使用 Redis 缓存后端，连接: redis://{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}, "
            f"namespace={self._cache.namespace}, serializer={self._cache.serializer.__class__.__name__}"
        )

    # ========== 实现 BaseCache 的异步抽象方法 ==========
    async def _aget_raw(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            value = await self._cache.get(key)
            if value is not None:
                self._cache_hits += 1
                return value
            else:
                self._cache_misses += 1
                return None
        except Exception as e:
            logger.error(f"Redis aget 失败 (key={key}): {e}")
            self._cache_misses += 1
            return None

    async def _aset_raw(self, key: str, value: Dict[str, Any]) -> None:
        try:
            await self._cache.set(key, value, ttl=self.default_ttl)
        except Exception as e:
            logger.error(f"Redis aset 失败 (key={key}): {e}")

    async def _adelete_raw(self, key: str) -> None:
        try:
            await self._cache.delete(key)
        except Exception as e:
            logger.warning(f"Redis delete 失败 (key={key}): {e}")

    async def _aclear_raw(self) -> None:
        try:
            await self._cache.clear()
        except Exception as e:
            logger.error(f"Redis clear 失败: {e}")

    async def _akeys_raw(self) -> List[str]:
        try:
            redis_client = self._cache.client
            namespace = self._cache.namespace or ""
            pattern = f"{namespace}:*" if namespace else "*"
            keys = []
            cursor = b'0'
            while cursor:
                cursor, batch = await redis_client.scan(cursor, match=pattern, count=100)
                keys.extend([k.decode('utf-8') for k in batch])
            # 去掉 namespace 前缀
            prefix_len = len(namespace) + 1 if namespace else 0
            return [k[prefix_len:] for k in keys]
        except Exception as e:
            logger.warning(f"获取 Redis keys 失败: {e}")
            return []

    def stats(self) -> str:
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return "📊 Redis LLM 缓存: 无本地调用统计"
        hit_rate = self._cache_hits / total
        return f"📊 Redis 缓存命中率（本地统计）: {hit_rate:.2%} | 命中={self._cache_hits} | 未命中={self._cache_misses}"
