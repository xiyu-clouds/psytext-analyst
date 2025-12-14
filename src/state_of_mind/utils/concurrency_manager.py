import asyncio
from typing import List, Callable, Any, Awaitable


class ConcurrencyManager:
    """通用并发控制器，用于限制同时执行的异步任务数量"""
    CHINESE_NAME = "通用并发控制器"

    def __init__(self, max_concurrent: int = 3):
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run_tasks(self, tasks: List[Callable[[], Awaitable[Any]]]) -> List[Any]:
        """
        并发执行一组无参异步任务，受内部信号量限制。
        注意：每个 task 必须是 **无参可调用对象**（如 lambda 或 partial），
              且返回一个 awaitable（通常是协程）。
        """
        if not tasks:
            return []

        async def _limited_task(task: Callable[[], Awaitable[Any]]) -> Any:
            async with self.semaphore:
                return await task()

        results = await asyncio.gather(*[_limited_task(t) for t in tasks], return_exceptions=False)
        return list(results)
