# 使用 Python 3.10（匹配你的环境）
FROM python:3.10-slim

# 设置时区（可选）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 创建普通用户
RUN useradd --create-home --shell /bin/bash appuser

USER root
RUN mkdir -p /home/appuser/psytext_data/{raw,dye_vat,reports,logs,logs_fallback}
RUN chown -R appuser:appuser /home/appuser/psytext_data

USER appuser
WORKDIR /home/appuser/psytext_analyst

# 复制依赖文件
COPY --chown=appuser:appuser requirements.txt .

# 安装依赖
# 清华源
#RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# 或腾讯云：
#RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple/ -r requirements.txt
# 阿里云
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
# 官方源
#RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY --chown=appuser:appuser . .

# 编译所有 .py 文件（包括 static/prompts/prompt.py）
RUN python -m compileall -q .

EXPOSE 8000

# 启动命令（用 python -m 最可靠）
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]