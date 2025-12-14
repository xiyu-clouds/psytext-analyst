import asyncio
from aiocache import Cache
from aiocache.serializers import JsonSerializer

from src.state_of_mind.config import config


async def test_redis_connection():
    cache = Cache(
        Cache.REDIS,
        endpoint=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        password=config.REDIS_PASSWORD or None,
        timeout=config.REDIS_TIMEOUT,
        serializer=JsonSerializer(),
        namespace="llm_cache_test",
    )

    test_key = "connection_test"
    test_value = {"status": "ok"}

    try:
        # ✅ 必须 await
        await cache.set(test_key, test_value, ttl=60)
        value = await cache.get(test_key)

        if value == test_value:
            print("✅ Redis 连接成功，读写正常")
            return True
        else:
            print("❌ Redis 读写不一致（可能值未正确序列化/反序列化）")
            return False

    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")
        return False
    finally:
        pass
        # try:
        #     await cache.delete(test_key)
        # except:
        #     pass


if __name__ == "__main__":
    # ✅ 正确执行异步函数
    asyncio.run(test_redis_connection())
