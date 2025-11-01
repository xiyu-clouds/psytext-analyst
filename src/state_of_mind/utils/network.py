import requests
from src.state_of_mind.utils.logger import LoggerManager as logger


def get_public_ip() -> str:
    """
    获取当前设备的公网出口 IP
    """
    # 可靠的免费服务（国内访问快）
    urls = [
        "https://ipinfo.io/ip",
        "https://httpbin.org/ip",
        "https://ifconfig.me/ip"
    ]

    for url in urls:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                ip = response.text.strip()
                if ip is None:
                    ip = response.json().get("origin")
                logger.info(f"🌐 获取公网IP成功: {ip} (via {url})", module_name="NetworkUtil", location="get_public_ip")
                return ip
        except Exception as e:
            logger.warning(f"⚠️ 获取IP失败: {url} | {e}", module_name="NetworkUtil", location="get_public_ip")

    logger.error("❌ 所有IP查询服务均失败", module_name="NetworkUtil", location="get_public_ip")
    return ""
