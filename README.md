# 🌊 PsyText Analyst — 开发与部署手册

> **项目名称**：心镜文本分析系统（PsyText Analyst）
> **功能特性**：支持本地开发、Docker 容器化部署，兼容内存 / Redis 缓存后端。

------

## 🧑‍💻 一、本地开发环境（Ubuntu）

### 1. 系统账号

- **用户名**：`xinhai`

- **密码**：`xinhai`

- #### 🐧 WSL 安装轻量级 Ubuntu 系统

  1.管理员身份打开cmd窗口，执行： wsl --install -d Ubuntu-22.04

  ⏱ 首次安装会自动下载并注册，完成后会提示你设置 **Linux 用户名和密码**（与 Windows 账户无关）

  2.安装完成后，打开 **开始菜单**，搜索 “Ubuntu 22.04” 并启动，或在 PowerShell 中运行：wsl -d Ubuntu-22.04

  3.当你首次启动时，终端会显示类似以下内容（可能略有延迟，耐心等几秒）：

  ```
  Installing, this may take a few minutes...
  Please create a default UNIX user account. The username does not need to match your Windows username.
  Enter new UNIX username:
  ```

  4.输入你的 Linux 用户名（建议小写，无空格）

  ⌨️ 你输入 `xinhai` 然后按回车（输入时**不会显示字符**，这是正常的安全设计）

  5.设置密码（输入两次）

  ```
  New password:
  Retype new password:
  ```

  如果两次一致，你会看到：

  ```
  Installation successful!
  To run a command as administrator (user "root"), use "sudo <command>".
  See "man sudo_root" for details.
  
  Welcome to Ubuntu 22.04.4 LTS (GNU/Linux 5.15.153.1-microsoft-standard-WSL2 x86_64)
  
   * Documentation:  https://help.ubuntu.com
   * Management:     https://landscape.canonical.com
   * Support:        https://ubuntu.com/advantage
  
  This message is shown once a day. To disable it please create the
  /home/xinhai/.hushlogin file.
  
  xinhai@DESKTOP-XXXXXX:~$
  ```

### 2. 安装并启动 Redis（可选）

> 仅当使用 Redis 缓存（`XINJING_STORAGE_BACKEND=redis`）时需要。

```bash
# 更新包列表
sudo apt update

# 安装 Redis
sudo apt install redis-server -y

# 启动 Redis 服务
sudo service redis-server start

# 测试连接（应返回 PONG）
redis-cli ping
```

💡 **提示**：若使用本地内存缓存（`XINJING_STORAGE_BACKEND=local`），可跳过 Redis 安装。

------

## 🚀 二、本地运行服务（开发模式）

确保当前工作目录为项目根目录（包含 `pyproject.toml` 和 `main.py`）：

```bash
# 启动 FastAPI 服务（带热重载）
uvicorn main:app --reload --port 8000
```

访问前端页面：
👉 http://localhost:8000/static/index.html

------

## 📦 三、安装项目包（开发/测试）

### 1. 安装打包工具

```bash
pip install build
```

### 2. 构建并安装（二选一）

#### ✅ 推荐：Editable 模式（开发时实时生效）

```bash
pip install -e .
```

#### 或：构建并安装 wheel 包

```bash
python -m build
pip install dist/psytext_analyst-*.whl
```

### 3. 验证安装

- **Linux/macOS**

  ```bash
  pip list | grep -i psytext
  pip show psytext-analyst
  ```

- **Windows (PowerShell)**

  ```powershell
  pip list | findstr -i psytext
  ```

### 4. 卸载（如需）

```bash
pip uninstall psytext-analyst
```

------

## 📂 四、查看项目结构

- **Windows**

  ```cmd
  tree /F
  ```

- **Linux/macOS**

  ```bash
  tree -L 2
  ```

------

## 🐳 五、Docker 部署指南

### 1. 准备基础镜像（支持离线环境）

```bash
# 拉取基础镜像
docker pull python:3.10-slim

# 导出为 tar（便于离线传输）
docker save python:3.10-slim > python_3.10_slim.tar
```

在目标机器加载：

```bash
# 加载基础镜像
docker load -i ./python_3.10_slim.tar

# 验证
docker images
```

### 2. 构建应用镜像

```bash
# 使用本地已有基础镜像构建（不联网拉取）
docker build --pull=false -t psytext_analyst:latest .
```

✅ **前提**：项目根目录存在 `Dockerfile`。

------

## 🧩 六、多模式缓存部署（通过 `docker-compose.yml`）

项目支持三种缓存模式，通过修改 `docker-compose.yml` 中的配置即可切换。

### 📄 `docker-compose.yml` 核心配置说明

```yaml
# ==================================================
# PsyText Analyst + Redis 多模式部署配置
# 支持三种缓存模式：
#   1. Redis（Docker 内部） ← 默认推荐
#   2. 本地内存（local）
#   3. 连接 Windows 本地 Redis
# ==================================================

services:
  psytext:
  	build: .
    image: psytext_analyst:latest
    ports:
      - "8000:8000"
    volumes:
      - D:/psytext_data/raw:/home/psytext_analyst/data/raw
      - D:/psytext_data/dye_vat:/home/psytext_analyst/data/dye_vat
      - D:/psytext_data/reports:/home/psytext_analyst/data/reports
      - D:/psytext_data/logs:/home/psytext_analyst/data/logs
      - D:/psytext_data/logs_fallback:/home/psytext_analyst/data/logs_fallback
    restart: unless-stopped
    environment:
      # ┌──────────────────────────────────────┐
      # │ 模式 1：使用 Docker 内部 Redis（默认）│
      # └──────────────────────────────────────┘
      XINJING_STORAGE_BACKEND: redis          # ← 改为 "local" 切换到内存模式
      XINJING_REDIS_HOST: redis               # ← Docker 服务名（仅当 backend=redis 时生效）
      XINJING_REDIS_PORT: 6379
      XINJING_REDIS_DB: 0
      XINJING_REDIS_PASSWORD: ""
      XINJING_REDIS_TIMEOUT: 5

      # 缓存通用参数
      XINJING_LLM_CACHE_MAX_SIZE: 4096
      XINJING_LLM_CACHE_TTL: 3600

    depends_on:
      - redis  # ← 仅当使用内部 Redis 时保留

  # ========== 【Redis 服务（Docker 内部）】==========
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

------

### 🔧 三种部署场景切换指南

#### ✅ 场景 1：使用 Docker 内部 Redis（推荐开发/部署）

- 保持 `docker-compose.yml` 默认配置不变。

- 确保：

  ```yaml
  XINJING_STORAGE_BACKEND: redis
  XINJING_REDIS_HOST: redis
  ```

- 保留 `depends_on` 和 `redis` 服务块。

✅ **优点**：完全隔离、一键启动、数据持久化、不依赖宿主机环境。

------

#### ✅ 场景 2：使用本地内存缓存（`local`）

- 修改 

  ```
  environment
  XINJING_STORAGE_BACKEND: local  # ← 关键！
  # XINJING_REDIS_HOST 可删或保留
  ```

  注释或删除以下部分：

  ```yaml
  depends_on:
    - redis
  ```

- 注释或删除整个 `redis` 服务块（从 `redis:` 开始到 `volumes:` 之前）。

✅ **优点**：启动更快、无外部依赖、适合轻量测试。

------

#### ✅ 场景 3：连接 Windows 本地 Redis

- 确保 Windows 上 Redis 正在运行，并监听 `127.0.0.1` 或 `0.0.0.0`。

- 修改 

  ```
  environment
  XINJING_STORAGE_BACKEND: redis
  XINJING_REDIS_HOST: host.docker.internal  # ← 关键！指向 Windows 主机
  XINJING_REDIS_PORT: 6379
  ```

  注释或删除：

  ```yaml
  depends_on:
    - redis
  ```

- 注释或删除整个 `redis` 服务块。

✅ **优点**：复用已有 Redis 实例（如 WSL、桌面版 Redis）。

------

### 🔁 完整重建流程（推荐用于环境清理）

```bash
# 1. 停止并清理
docker-compose down

# 2. 删除旧镜像（避免标签复用缓存）
docker rmi psytext_analyst:latest

# 3. 用 Compose 重新构建（带 --no-cache 确保干净）
docker-compose build --no-cache psytext

# 4. 启动
docker-compose up -d  后台运行，不实时输出日志
docker-compose up  前台运行，有实时日志

# 5. 查看日志
docker-compose logs -f psytext  持续输出此psytext服务所对应容器的日志，psytext：指定服务名（对应 docker-compose.yml 中的 services.psytext）
docker-compose logs --tail=100 psytext  查看最近 100 行日志（不跟踪）
docker-compose logs -f  查看所有服务的日志（带服务名前缀）
docker-compose logs -f --timestamps psytext  实时日志 + 时间戳（调试用）
docker-compose logs psytext  如果容器崩溃了，也可以用 logs 看错误原因

# 6. 打包镜像
docker save -o psytext_analyst_latest.tar psytext_analyst:latest （推荐）
docker save psytext_analyst:latest > psytext_analyst_latest.tar （兜底）
```

> 若上述流程因网络或环境限制失败，可退而使用：
>
> ```bash
> docker build --pull=false -t psytext_analyst:latest .
> ```

------

### 🔍 验证容器内环境变量

```bash
# 进入容器
docker exec -it psytext_analyst-psytext-1 bash

# 检查关键环境变量
echo $XINJING_REDIS_HOST
# 场景1应输出：redis
# 场景3应输出：host.docker.internal

# Python 验证
python -c "import os; print(os.getenv('XINJING_REDIS_HOST'))"
```

✅ 如果输出符合预期，说明配置已正确加载！

------

## 🔍 七、容器管理与调试

| 操作                     | 命令                                              |
| ------------------------ | ------------------------------------------------- |
| 查看运行中容器           | `docker ps`                                       |
| 停止容器                 | `docker stop psytext`                             |
| 强制终止                 | `docker kill psytext`                             |
| 删除容器                 | `docker rm psytext`                               |
| 进入容器调试             | `docker exec -it psytext bash`                    |
| 调试文件结构             | `docker run -it --rm psytext_analyst:latest bash` |
| 进入redis容器            | docker exec -it psytext_analyst-redis-1 redis-cli |
| 一键删除所有已停止的容器 | docker container prune                            |
| 一键清理所有悬空镜像     | docker image prune                                |

示例：

```bash
docker run -it --rm psytext_analyst:latest bash
ls -la /home/psytext_analyst/src/
exit
```

------

## 📝 八、配置说明

- 所有运行时配置通过 `app.json` 提供默认值。
- **环境变量优先级高于 `app.json`**（代码已实现覆盖逻辑）。

------

## ✅ 最佳实践建议

| 阶段          | 推荐方案                                |
| ------------- | --------------------------------------- |
| **开发**      | `pip install -e .` + `uvicorn --reload` |
| **测试/演示** | `local` 缓存模式 + `docker-compose`     |
| **生产**      | `redis` 模式（Docker 内部或外部集群）   |

------

> 📌 **文档版本**：v1.0
> **最后更新**：2025年10月29日
