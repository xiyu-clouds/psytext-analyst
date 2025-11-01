from functools import wraps
from typing import Callable, Any

from src.state_of_mind.utils.context_manager import Timeout, Timer


class PerformanceGuard:
    """性能与超时组合卫士（上下文管理器）"""

    def __init__(self, func_name: str, timeout: float = 60.0, module: str = "服务监控"):
        self.func_name = func_name
        self.timeout = timeout
        self.module = module
        self.timer = None
        self.timeout_ctx = None

    def __enter__(self):
        self.timer = Timer(self.func_name, self.module)
        self.timeout_ctx = Timeout(self.timeout, self.func_name, self.module)
        self.timer.__enter__()
        self.timeout_ctx.__enter__()
        return self

    def __exit__(self, *args):
        self.timeout_ctx.__exit__(*args)
        self.timer.__exit__(*args)


# =============================================================================
# ✅ 装饰器版本（最常用！）
# =============================================================================

def performance_guard(timeout: float = 60.0, module: str = "服务监控"):
    """
    装饰器：为函数添加性能统计 + 超时保护

    Usage:
        @performance_guard(timeout=10, module="LLM调用")
        def call_llm(prompt):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with PerformanceGuard(func.__name__, timeout, module):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# 方式 1：装饰器（推荐，最简洁）
# @performance_guard(timeout=10, module="LLM调用")
# def call_llm(prompt: str) -> str:
#     time.sleep(8)
#     if "错误" in prompt:
#         raise ValueError("输入不合法")
#     return "这是AI的回答"
#
# 方式 2：上下文管理器（灵活控制范围）
# def complex_process():
#     with Timer("数据清洗", "数据处理"), Timeout(30, "数据清洗", "数据处理"):
#         # 只对关键步骤监控
#         clean_data()
#         transform_data()
#
# 方式 3：组合卫士（手动控制）
# def risky_task():
#     with PerformanceGuard("风险操作", timeout=5, module="任务系统"):
#         time.sleep(6)  # 会超时