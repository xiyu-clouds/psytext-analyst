import time
import eventlet
from typing import Optional
from .decorator_utils import log_function_event


class Timer:
    """性能计时上下文管理器"""

    def __init__(self, name: str, module: str = "性能统计"):
        self.name = name
        self.module = module
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        log_function_event(
            action="start",
            func_name=self.name,
            module_name=self.module,
            start_timestamp=self.start_time
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time

        if exc_type is None:
            log_function_event(
                action="success",
                func_name=self.name,
                module_name=self.module,
                duration=self.duration
            )
        else:
            log_function_event(
                action="failure",
                func_name=self.name,
                module_name=self.module,
                duration=self.duration,
                exception=f"{exc_type.__name__}: {str(exc_val)}"
            )
        return False  # 不吞异常


class Timeout:
    """超时控制上下文管理器"""

    def __init__(self, timeout: float, name: str, module: str = "超时控制"):
        self.timeout = timeout
        self.name = name
        self.module = module
        self.timer = None

    def __enter__(self):
        log_function_event(
            action="start",
            func_name=self.name,
            module_name=self.module,
            timeout=self.timeout
        )
        self.timer = eventlet.Timeout(self.timeout, False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 检查是否超时
        if self.timer is not None and not self.timer.pending:
            log_function_event(
                action="timeout",
                func_name=self.name,
                module_name=self.module,
                timeout=self.timeout
            )
            # 保留原异常或抛出 TimeoutError
            if exc_type is None:
                raise TimeoutError(f"函数 {self.name} 执行超时，超过 {self.timeout} 秒")
        else:
            if exc_type is None:
                log_function_event(
                    action="success",
                    func_name=self.name,
                    module_name=self.module
                )
            else:
                log_function_event(
                    action="exception",
                    func_name=self.name,
                    module_name=self.module,
                    exception=f"{exc_type.__name__}: {str(exc_val)}"
                )
        return False
