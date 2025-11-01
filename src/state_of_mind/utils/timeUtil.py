from functools import wraps

from src.state_of_mind.utils.context_manager import Timer, Timeout


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__, module="性能统计"):
            return func(*args, **kwargs)

    return wrapper


def with_timeout(timeout=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timeout(timeout, func.__name__, module="超时控制"):
                return func(*args, **kwargs)

        return wrapper

    return decorator
