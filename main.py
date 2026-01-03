import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from src.state_of_mind.core.orchestration import MetaCognitiveOrchestrator
from src.state_of_mind.config import config
from src.state_of_mind.stages.perception.constants import DEFAULT_API_URLS, ALL_STEPS_FOR_FRONTEND
from src.state_of_mind.utils.constants import PATH_FILE_APP_JSON, LLMModelConst
from src.state_of_mind.utils.file_util import FileUtil
from src.state_of_mind.utils.logger import LoggerManager as logger
logger.inject_config(config)
CHINESE_NAME = "FastAPI启动中心"
logger.info("🚀 应用启动中...", module_name=CHINESE_NAME)
app = FastAPI(title="心海")
# 挂载静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"📁 静态文件已挂载: {static_dir}", module_name=CHINESE_NAME)
else:
    logger.warning(f"⚠️ 静态目录不存在: {static_dir}", module_name=CHINESE_NAME)

orchestrator = MetaCognitiveOrchestrator()

# 允许前端跨域（开发时用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS 中间件已加载", module_name=CHINESE_NAME)


class AnalysisRequest(BaseModel):
    text: str
    title: str = "文本多模态感知分析报告"


# === 配置读取接口 ===
@app.get("/api/config")
async def get_config():
    """返回当前 app.json 配置（含敏感字段，前端需谨慎展示）"""
    logger.info("📥 收到 /api/config GET 请求", module_name=CHINESE_NAME)
    try:
        data = FileUtil().read_json_file(PATH_FILE_APP_JSON)
        logger.info(f"📄 读取配置成功: {data}", module_name=CHINESE_NAME)
        return data
    except Exception as e:
        logger.error(f"❌ 读取配置失败: {e}", module_name=CHINESE_NAME)
        raise HTTPException(status_code=500, detail=f"读取配置失败: {str(e)}")


# === 配置保存接口 ===
@app.post("/api/config")
async def save_config(request: Request):
    """接收 JSON 配置，校验字段类型与值合法性，标准化路径，保存并重载"""
    logger.info("📥 收到 /api/config POST 请求", module_name=CHINESE_NAME)
    try:
        new_config = await request.json()
        if not isinstance(new_config, dict):
            logger.warning("⚠️ 配置不是 JSON 对象", module_name=CHINESE_NAME)
            raise HTTPException(status_code=400, detail="配置必须是 JSON 对象")

        errors = []

        # --- 字段校验与标准化 ---
        # 1. XINJING_STORAGE_BACKEND: str, 限定值
        backend = new_config.get("XINJING_STORAGE_BACKEND")
        if backend is not None:
            if not isinstance(backend, str) or backend not in {"local", "redis"}:
                errors.append("XINJING_STORAGE_BACKEND 必须是 'local' 或 'redis'")

        # 2. XINJING_LLM_CACHE_MAX_SIZE: int > 0
        cache_size = new_config.get("XINJING_LLM_CACHE_MAX_SIZE")
        if cache_size is not None:
            if not isinstance(cache_size, int) or cache_size <= 0:
                errors.append("XINJING_LLM_CACHE_MAX_SIZE 必须是正整数")

        # 3. XINJING_LLM_CACHE_TTL: int >= 0
        ttl = new_config.get("XINJING_LLM_CACHE_TTL")
        if ttl is not None:
            if not isinstance(ttl, int) or ttl < 0:
                errors.append("XINJING_LLM_CACHE_TTL 必须是非负整数")

        # 4. XINJING_REDIS_HOST: str
        redis_host = new_config.get("XINJING_REDIS_HOST")
        if redis_host is not None and not isinstance(redis_host, str):
            errors.append("XINJING_REDIS_HOST 必须是字符串")

        # 5. XINJING_REDIS_PORT: int in [1, 65535]
        redis_port = new_config.get("XINJING_REDIS_PORT")
        if redis_port is not None:
            if not isinstance(redis_port, int) or not (1 <= redis_port <= 65535):
                errors.append("XINJING_REDIS_PORT 必须是 1~65535 之间的整数")

        # 6. XINJING_REDIS_DB: int >= 0
        redis_db = new_config.get("XINJING_REDIS_DB")
        if redis_db is not None:
            if not isinstance(redis_db, int) or redis_db < 0 or redis_db > 15:
                errors.append("XINJING_REDIS_DB 必须是 0 到 15 之间的整数（Redis 默认最多支持 16 个数据库）")

        # 7. XINJING_REDIS_PASSWORD: str or null
        redis_pwd = new_config.get("XINJING_REDIS_PASSWORD")
        if redis_pwd is not None and redis_pwd is not None and not isinstance(redis_pwd, str):
            errors.append("XINJING_REDIS_PASSWORD 必须是字符串或 null")

        # 8. XINJING_REDIS_TIMEOUT: int > 0
        redis_timeout = new_config.get("XINJING_REDIS_TIMEOUT")
        if redis_timeout is not None:
            if not isinstance(redis_timeout, int) or redis_timeout <= 0:
                errors.append("XINJING_REDIS_TIMEOUT 必须是正整数")

        # 9. XINJING_MAX_PARALLEL_CONCURRENCY: int in [1, 10]
        concurrency = new_config.get("XINJING_CURRENT_PARALLEL_CONCURRENCY")
        max_parallel = new_config.get("XINJING_MAX_PARALLEL_CONCURRENCY")
        medium_parallel = new_config.get("XINJING_MEDIUM_PARALLEL_CONCURRENCY")
        if concurrency is not None:
            if not isinstance(concurrency, int):
                errors.append("XINJING_CURRENT_PARALLEL_CONCURRENCY 必须是整数")
            elif concurrency < 1:
                errors.append("XINJING_CURRENT_PARALLEL_CONCURRENCY 必须 ≥ 1")
            elif concurrency > max_parallel:
                errors.append(
                    f"XINJING_MAX_PARALLEL_CONCURRENCY 超出当前允许的最大值（{max_parallel}）。"
                    "大模型 API 普通密钥通常仅支持 3~5 并发，过高设置会导致请求被限流或拒绝。"
                )
            elif concurrency > medium_parallel:
                if medium_parallel is not None and concurrency > medium_parallel:
                    errors.append(
                        f"XINJING_CURRENT_PARALLEL_CONCURRENCY ({concurrency}) 超过推荐值 ({medium_parallel})。"
                        "普通大模型 API 密钥在并发 >5 时容易因服务端限流导致部分请求失败，"
                        "建议保持 3~5 以获得稳定响应。"
                    )

        # 10. LOG_KEEP_DAYS: int > 0
        log_days = new_config.get("LOG_KEEP_DAYS")
        if log_days is not None:
            if not isinstance(log_days, int) or log_days <= 0:
                errors.append("LOG_KEEP_DAYS 必须是正整数")

        # 11. LOG_MAX_BYTES: int > 0
        log_max = new_config.get("LOG_MAX_BYTES")
        if log_max is not None:
            if not isinstance(log_max, int) or log_max <= 0:
                errors.append("LOG_MAX_BYTES 必须是正整数")

        # 12. LOG_BACKUP_COUNT: int > 0
        log_backup = new_config.get("LOG_BACKUP_COUNT")
        if log_backup is not None:
            if not isinstance(log_backup, int) or log_backup <= 0:
                errors.append("LOG_BACKUP_COUNT 必须是正整数")

        # 13. XINJING_OUTPUT_ROOT: str, 路径标准化
        # output_root = new_config.get("XINJING_OUTPUT_ROOT")
        # if output_root is not None:
        #     if not isinstance(output_root, str) or not output_root.strip():
        #         errors.append("XINJING_OUTPUT_ROOT 必须是非空字符串")
        #     else:
        #         output_root = output_root.strip().replace("\\", "/")
        #         output_root = re.sub(r"/+$", "", output_root)
        #         # 关键：只禁止 ..，不禁止冒号（D: 是合法的）
        #         if ".." in output_root:
        #             errors.append("XINJING_OUTPUT_ROOT 路径不能包含 '..'（防止路径穿越）")
        #         else:
        #             new_config["XINJING_OUTPUT_ROOT"] = output_root

        # 14. XINJING_LLM_BACKEND: str, 限定值
        llm_backend = new_config.get("XINJING_LLM_BACKEND")
        if llm_backend is not None:
            if not isinstance(llm_backend, str) or llm_backend not in {"deepseek", "qwen"}:
                errors.append("XINJING_LLM_BACKEND 必须是 'deepseek'或'qwen'")

        # 15. XINJING_LLM_MODEL: str
        llm_model = new_config.get("XINJING_LLM_MODEL")
        if llm_model is not None:
            if not isinstance(llm_model, str):
                errors.append("XINJING_LLM_MODEL 必须是字符串")
            elif llm_model not in LLMModelConst.all():
                errors.append(
                    f"XINJING_LLM_MODEL '{llm_model}' 不受支持。"
                    f"当前支持的模型：{', '.join(sorted(LLMModelConst.all()))}"
                )

        # 16. XINJING_LLM_API_URL: str, 简单 URL 校验
        api_url = new_config.get("XINJING_LLM_API_URL")
        if api_url is not None:
            if not isinstance(api_url, str) or not (api_url.startswith("http://") or api_url.startswith("https://")):
                errors.append("XINJING_LLM_API_URL 必须是有效的 HTTP/HTTPS URL")

        if llm_backend and api_url:
            expected_url_prefix = DEFAULT_API_URLS.get(llm_backend, "")
            if expected_url_prefix and not api_url.startswith(expected_url_prefix):
                errors.append(
                    f"检测到 XINJING_LLM_BACKEND='{llm_backend}'，但 XINJING_LLM_API_URL='{api_url}' "
                    f"与官方默认地址 '{expected_url_prefix}' 不匹配。"
                    "若调用失败，请前往官方平台确认最新 API 地址："
                    "DeepSeek: https://platform.deepseek.com/docs/api-reference | "
                    "Qwen: https://help.aliyun.com/zh/dashscope/developer-reference/"
                )
        if llm_backend and api_url is None:
            new_config["XINJING_LLM_API_URL"] = DEFAULT_API_URLS[llm_backend]

        # 17. XINJING_LLM_API_KEY: str or null
        api_key = new_config.get("XINJING_LLM_API_KEY")
        if api_key is not None and not isinstance(api_key, str):
            errors.append("XINJING_LLM_API_KEY 必须是字符串或 null")

        # --- 如果有校验错误，直接返回 ---
        if errors:
            error_msg = "配置校验失败:\n" + "\n".join(errors)
            logger.warning(f"⚠️ 配置校验失败: {error_msg}", module_name=CHINESE_NAME)
            raise HTTPException(status_code=400, detail="配置校验失败:\n" + "\n".join(errors))

        # --- 保存文件 ---
        FileUtil().write_json(new_config, PATH_FILE_APP_JSON)
        logger.info("💾 配置已写入文件", module_name=CHINESE_NAME)

        # --- 重载配置 ---
        await config.reload()

        return {"status": "success", "message": "配置已保存并重载"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("💥 保存配置时发生未预期错误")  # 自动记录 traceback
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@app.get("/reports/{filename}", response_class=HTMLResponse)
async def serve_report(filename: str):
    """提供 HTML 报告服务"""
    logger.info(f"📄 请求报告: {filename}", module_name=CHINESE_NAME)
    if not filename.endswith(".html"):
        logger.warning(f"⚠️ 非法文件类型: {filename}", module_name=CHINESE_NAME)
        raise HTTPException(status_code=400, detail="仅支持 .html 文件")

    safe_filename = Path(filename).name  # 防路径穿越
    report_path = config.REPORTS_DIR / safe_filename
    logger.debug(f"🔍 报告完整路径: {report_path}", module_name=CHINESE_NAME)

    if not report_path.exists():
        logger.error(f"❌ 报告不存在: {report_path}", module_name=CHINESE_NAME)
        raise HTTPException(status_code=404, detail="报告不存在")

    logger.info(f"✅ 成功返回报告: {filename}", module_name=CHINESE_NAME)
    return HTMLResponse(report_path.read_text(encoding="utf-8"))


@app.get("/api/steps")
async def get_steps():
    logger.info("开始预加载全息感知基底阶段的各环节基础 Prompt 数据...")
    try:
        from src.state_of_mind.stages.perception.prompt_builder import PromptBuilder
        prompt_builder = PromptBuilder()
        prompt_builder.pre_basic_data()
        logger.info(f"全息感知基底 Prompt 预加载完成")
        return ALL_STEPS_FOR_FRONTEND
    except Exception as e:
        logger.exception("预加载 /api/steps 失败：构建 Prompt 基础数据时发生异常")
        raise


@app.post("/api/analyze")
async def analyze_text(request: AnalysisRequest):
    logger.info(f"🧠 收到分析请求，标题: {request.title[:30]}...", module_name=CHINESE_NAME)
    logger.info(f"📝 原始文本长度: {len(request.text)} 字符", module_name=CHINESE_NAME)
    try:
        result = await orchestrator.run(stage_name="perception", user_input=request.text)
        logger.info("✅ 文本分析完成", module_name=CHINESE_NAME)
        return result
    except Exception as e:
        logger.exception("💥 文本分析过程中发生错误", module_name=CHINESE_NAME)  # 记录完整堆栈
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

logger.info("🎉 FastAPI 应用初始化完成！", module_name=CHINESE_NAME)
logger.info("📜 本工具基于 MIT 许可证发布，商业/个人使用前请查阅 LICENSE 与 EULA 文件。", module_name=CHINESE_NAME)