from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template
from src.state_of_mind.config import config
from src.state_of_mind.utils.file_util import FileUtil
from src.state_of_mind.utils.logger import LoggerManager as logger


class ReportGenerator:
    CHINESE_NAME = "全息感知基底：生成报告"

    def __init__(self, file_util: FileUtil):
        self.file_util = file_util

    def render_report_to_html(self, data: Dict[str, Any]) -> Optional[Path]:
        """
        将 result 数据注入 HTML 模板，生成报告。
        - 输出目录：config.REPORTS_DIR
        - 文件名：通过 self.file_util.generate_filename 生成
        - 前缀："全息感知基底分析报告"
        - 后缀：".html"
        - 模板读取：复用 self.file_util.read_file
        - 文件写入：复用 self.file_util.write_file
        - 上下文变量名：data
        """
        try:
            if not data or not isinstance(data, dict):
                return None

            filename = self.file_util.generate_filename(
                prefix="全息感知基底分析报告",
                suffix=".html",
                include_timestamp=True
            )

            output_path = config.REPORTS_DIR / filename
            template_content = self.file_util.read_file(
                str(config.FILE_DEFAULT_TEMPLATE_PATH),
                encoding="utf-8",
                auto_decode=False
            )
            if not template_content:
                logger.error(
                    "❌ 模板文件为空或读取失败",
                    extra={
                        "template_path": str(config.FILE_DEFAULT_TEMPLATE_PATH),
                        "module_name": self.CHINESE_NAME
                    }
                )
                return None

            html_output = Template(template_content).render(data=data)

            success = self.file_util.write_file(
                file_path=str(output_path),
                content=html_output,
                encoding="utf-8",
                as_json=False,
                file_type="html"
            )
            if not success:
                logger.error(
                    "❌ HTML 报告写入失败",
                    extra={"path": str(output_path), "module_name": self.CHINESE_NAME}
                )
                return None

            logger.info(
                "📄 HTML 报告已生成",
                extra={"path": str(output_path), "module_name": self.CHINESE_NAME}
            )
            return output_path
        except Exception as e:
            logger.exception(
                "💥 HTML 报告生成失败",
                extra={"error": str(e), "module_name": self.CHINESE_NAME}
            )
            return None
