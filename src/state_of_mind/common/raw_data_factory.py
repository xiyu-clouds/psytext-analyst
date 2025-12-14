from datetime import datetime
from typing import Dict, Any
from zoneinfo import ZoneInfo
from src.state_of_mind.config import config
from src.state_of_mind.stages.perception.constants import CATEGORY_RAW
from src.state_of_mind.utils.logger import LoggerManager as logger
import ulid
CHINESE_NAME = "第一阶段：全息感知基底基础数据构造"


def create_raw_basic_data(user_input: str, llm_model: str, schema_version: str = "1.0.0") -> Dict[str, Any]:
    """
    构造原始事件的固定基础元数据
    可用于日志追踪、审计、溯源等
    """
    record_id = f"raw_{ulid.new().str}"

    # public_ip = get_public_ip()
    # tz_name = IPBasedTimezoneResolver.get_timezone_from_ip(public_ip) if public_ip else "UTC"

    # if not public_ip:
    #     logger.warning("⚠️ 无法获取公网IP，使用 UTC 时区", module_name=Prompter.CHINESE_NAME)

    tz = ZoneInfo("UTC")
    timestamp = datetime.now(tz).isoformat()

    formatter_time = ""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][dt.weekday()]
        base_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 毫秒部分
        formatter_time = f"{base_time} {weekday}"
    except Exception as e:
        logger.warning(
            f"🕒 无法解析 timestamp 为 formatter_time: {e}",
            module_name=CHINESE_NAME,
            extra={"timestamp": timestamp}
        )

    data = {
        "id": record_id,
        "type": CATEGORY_RAW,
        "schema_version": schema_version,
        "timestamp": timestamp,
        "formatter_time": formatter_time,
        "source": {
            "modality": "text/narrative",
            "content": user_input,
            "input_mode": "user_input",
            # "local_ip": public_ip,
            "timezone": "UTC"
        },
        "meta": {
            "library_version": config.VERSION,
            "created_by_ai": True,
            "llm_model": llm_model,
            "crystal_ids": [],
            "ontology_ids": [],
            "narrative_enriched": False,
            "privacy_scope": {
                "allowed_modules": [],
                "sync_to_cloud": False,
                "notify_on_trigger": False,
                "exportable": False
            }
        }
    }

    logger.info(f"📦 已生成基础元数据, id={record_id} | timezone=UTC", module_name=CHINESE_NAME)
    return data
