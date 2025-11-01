from netaddr import IPNetwork, IPAddress
import os

from src.state_of_mind.config import config
from src.state_of_mind.utils.logger import LoggerManager as logger


class IPBasedTimezoneResolver:
    """
    基于 IP 地址判断是否为中国用户，并返回对应时区（Asia/Shanghai / UTC）
    使用 17mon 中国 IP 列表，本地加载，零外部依赖。
    """
    CHINESE_NAME = "IPBasedTimezoneResolver"
    _CN_CIDRS = None
    _IP_LIST_FILE = "china_ip_list.txt"
    _TIMEZONE_CN = "Asia/Shanghai"
    _TIMEZONE_DEFAULT = "UTC"

    @classmethod
    def load_china_ips(cls, ip_list_path: str = config.FILE_CHAINA_IP_LIST_PATH):
        """
        加载中国 IP 段列表
        :param ip_list_path: 自定义路径，若为 None 则使用默认文件
        """
        if ip_list_path is None:
            ip_list_path = cls._IP_LIST_FILE

        if not os.path.exists(ip_list_path):
            error_msg = f"❌ 中国IP段文件未找到: {ip_list_path}，请先下载 https://github.com/17mon/china_ip_list"
            logger.error(error_msg, module_name=cls.__name__)
            raise FileNotFoundError(error_msg)

        cls._CN_CIDRS = []
        try:
            with open(ip_list_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        cls._CN_CIDRS.append(IPNetwork(line))
                    except Exception as e:
                        logger.warning(f"⚠️ 第 {line_num} 行格式无效，跳过: {line} | 错误: {e}", module_name=cls.__name__)
            logger.info(f"✅ 成功加载 {cls._CN_CIDRS.__len__()} 个中国IP段", module_name=cls.__name__)
        except Exception as e:
            logger.error(f"❌ 加载IP列表失败: {e}", module_name=cls.__name__)
            raise

    @classmethod
    def is_chinese_ip(cls, ip: str) -> bool:
        """
        判断 IP 是否属于中国
        :param ip: IPv4 地址字符串
        :return: 是否为中国 IP
        """
        if cls._CN_CIDRS is None:
            cls.load_china_ips()

        try:
            ip_addr = IPAddress(ip)
            result = any(ip_addr in cidr for cidr in cls._CN_CIDRS)
            logger.info(f"🔍 IP检查: {ip} -> {'中国' if result else '非中国'}", module_name=cls.__name__)
            return result
        except Exception as e:
            logger.error(f"❌ IP格式错误: {ip} | 错误: {e}", module_name=cls.__name__)
            return False

    @classmethod
    def get_timezone_from_ip(cls, ip: str) -> str:
        """
        根据 IP 返回对应时区
        :param ip: 客户端 IP 地址
        :return: 时区字符串，如 "Asia/Shanghai" 或 "UTC"
        """
        if not ip:
            logger.warning("⚠️ IP为空，使用默认时区 UTC", module_name=cls.__name__)
            return cls._TIMEZONE_DEFAULT

        is_cn = cls.is_chinese_ip(ip)
        timezone = cls._TIMEZONE_CN if is_cn else cls._TIMEZONE_DEFAULT
        logger.info(f"🌐 IP → 时区: {ip} -> {timezone}", module_name=cls.__name__)
        return timezone

    @classmethod
    def reload(cls, ip_list_path: str = None):
        """
        重新加载 IP 列表（热更新用）
        """
        cls._CN_CIDRS = None
        cls.load_china_ips(ip_list_path)
        logger.info("🔄 IP列表已重新加载", module_name=cls.__name__)
