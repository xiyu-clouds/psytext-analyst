from src.state_of_mind.utils.logger import LoggerManager as logger


def log_function_event(
        action: str,
        func_name: str,
        module_name: str,
        **kwargs
):
    """
    统一日志记录接口，用于函数执行监控。

    Args:
        action: 动作类型，如 'start', 'success', 'timeout', 'exception', 'failure'
        func_name: 函数名
        module_name: 模块名（用于日志分类）
        **kwargs: 其他上下文字段
    """
    log_data = {
        "func_name": func_name,
        "module_name": module_name,
        **kwargs
    }

    message = f"函数 {func_name} {action}"

    if action == "start":
        logger.info(message, **log_data)
    elif action in ["success", "completed"]:
        logger.info(f"{message}，耗时 {kwargs.get('duration', 0):.4f} 秒", **log_data)
    elif action == "timeout":
        logger.error(message, **log_data)
    elif action == "exception":
        logger.exception(f"{message}: {kwargs.get('exception')}", **log_data)
    elif action == "failure":
        logger.exception(f"{message}，耗时 {kwargs.get('duration', 0):.4f} 秒，异常: {kwargs.get('exception')}", **log_data)
    else:
        logger.info(message, **log_data)
