from src.state_of_mind.config import config
from src.state_of_mind.utils.logger import LoggerManager as logger

if __name__ == "__main__":
    logger.info("📦 Metadata:")
    logger.info(f"  名称: {config.metadata['name']}")
    logger.info(f"  版本: {config.metadata['version']}")
    logger.info(f"  描述: {config.metadata['description']}")
    logger.info(f"  作者: {config.metadata['authors']}")
    logger.info(f"  许可: {config.metadata['license']}")
    logger.info("  URLs:")
    for k, v in config.metadata["urls"].items():
        logger.info(f"    {k}: {v}")


