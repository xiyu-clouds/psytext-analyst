from src.state_of_mind.utils.logger import LoggerManager as logger

# 全局描述函数注册表
CONTEXT_DESCRIPTORS = []


def register_descriptor(*keys):
    """
    装饰器：注册描述函数，并声明它依赖的顶层键
    自动记录注册日志，便于调试和确认上下文感知能力
    """
    def decorator(func):
        descriptor_info = {
            "func": func,
            "keys": keys,
            "name": func.__name__
        }
        CONTEXT_DESCRIPTORS.append(descriptor_info)

        # ✅ 正确方式：使用 extra 传入自定义字段
        logger.info(
            "ContextDescriptorRegistered",
            extra={
                "descriptor_name": func.__name__,
                "dependent_keys": list(keys),
                "total_registered": len(CONTEXT_DESCRIPTORS)
            },
            module_name="上下文描述注册器",
            location="register_descriptor"
        )

        return func

    return decorator
