你是一个资深的 AI 全栈工程师。我已经在一个拥有 RTX 5880 显卡（48GB 显存）的 Linux 服务器上，通过 Docker Compose 启动了一个名为 `nerfstudio_3dgs` 的容器。该容器内置了完整的 Nerfstudio 环境，并在后台持续运行。

我的宿主机与容器的目录挂载关系如下：
- 宿主机 `./data/uploads` 挂载到容器 `/workspace/inputs`
- 宿主机 `./data/outputs` 挂载到容器 `/workspace/outputs`
- 宿主机 `./data/exports` 挂载到容器 `/workspace/exports`

现在，请帮我编写完整的前后端代码，实现一个完整的“上传照片 -> 自动生成 3DGS 模型 -> 前端渲染”的系统。

### 技术栈与环境要求：
1. 后端：Python 3.10 + FastAPI + Uvicorn。
2. **包管理：必须使用 `Pixi` 来管理 Python 环境和依赖。请勿使用 pip 或 venv。**
3. 前端：原生 HTML + Vanilla JavaScript + WebGL 3DGS 渲染库（推荐使用开源的 `mkkellogg/GaussianSplats3D` 或 `antimatter15/splat`，如果太复杂，请提供指导我引入相关的 CDN，或者先只提供模型下载链接）。
4. 进程管理：后端需要能异步执行 Shell 命令与 Docker 容器交互。

### 一、 后端需求 (FastAPI)
请编写一个 `main.py`，包含以下接口和逻辑：
1. **POST `/api/upload`**：
   - 接收前端上传的多张图片文件。
   - 生成一个唯一的 UUID 作为 `task_id`。
   - 将接收到的图片保存到宿主机的 `./data/uploads/{task_id}/images/` 目录下。
   - 响应成功并返回 `task_id`，然后在后台异步启动 3D 重建流程，不能阻塞主线程。

2. **异步 3D 重建流程 (核心逻辑)**：
   请使用 Python 的 `subprocess` 模块，按顺序向正在运行的 `nerfstudio_3dgs` 容器发送以下 `docker exec` 命令：
   - 步骤 1 (计算相机位姿)：`docker exec nerfstudio_3dgs ns-process-data images --data /workspace/inputs/{task_id}/images --output-dir /workspace/inputs/{task_id}_processed`
   - 步骤 2 (训练 3DGS 模型)：`docker exec nerfstudio_3dgs ns-train splatfacto --data /workspace/inputs/{task_id}_processed --max-num-iterations 7000 --output-dir /workspace/outputs/{task_id}` (注意：必须指定 max-num-iterations 7000，否则训练不会自动停止)。
   - 步骤 3 (导出通用模型)：利用 shell 命令找到刚才训练生成的 `config.yml` 文件路径，然后执行导出：`docker exec nerfstudio_3dgs ns-export gaussian-splat --load-config <找到的config.yml路径> --output-dir /workspace/exports/{task_id}`

3. **GET `/api/status/{task_id}`**：
   - 查询当前任务的状态（如：正在处理图片、正在训练、训练完成、失败）。请在后端使用简单的内存字典或 SQLite 记录状态即可。

4. **静态文件服务**：
   - 将宿主机的 `./data/exports/` 目录挂载为静态文件目录，以便前端可以通过 URL（如 `/exports/{task_id}/splat.ply`）直接获取生成的 3D 模型文件。

### 二、 前端需求 (HTML/JS)
请提供一个精简且美观的 `index.html`，包含以下功能：
1. 一个多文件上传的 input 框，支持选择数十张照片。
2. 一个“开始生成 3D 模型”的按钮。
3. 点击按钮后，通过 AJAX/Fetch 将照片提交到 `/api/upload`。
4. 提交成功后，启动一个轮询机制（每 5 秒调用一次 `/api/status/{task_id}`），并在页面上显示当前的进度状态（最好有简单的进度条或文字提示）。
5. 当状态变为“完成”时，提供生成的 `.ply` 文件的下载链接，或者（如果是高级能力）在网页中直接渲染该 `.ply` 3D 模型。

### 三、 交付要求
请提供：
1. 初始化项目所需的 `pixi init` 和 `pixi add` 命令，或者直接提供 `pixi.toml` 的内容。
2. 完整的 `main.py` 后端代码。
3. 完整的 `index.html` 前端代码。
4. 使用 `pixi run` 启动后端服务的完整命令。代码需要考虑基础的异常捕获（如 docker 命令执行失败时的状态更新）。