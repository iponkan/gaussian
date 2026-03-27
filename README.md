你是一个资深的 AI 全栈工程师。我已经在一个拥有 RTX 5880 显卡（48GB 显存）的 Linux 服务器上，通过 Docker Compose 启动了一个名为 `nerfstudio_3dgs` 的容器。该容器内置了完整的 Nerfstudio 环境，并在后台持续运行。

我的宿主机与容器的目录挂载关系如下：
- 宿主机 `./data/uploads` 挂载到容器 `/workspace/inputs`
- 宿主机 `./data/outputs` 挂载到容器 `/workspace/outputs`
- 宿主机 `./data/exports` 挂载到容器 `/workspace/exports`

# TripoSR 瞬时 3D 大模型生成系统

本项目是一个基于 **TripoSR** 的本地化单图到三维模型生成工作流。通过上传单张或少量图片，利用基于 LRM (Large Reconstruction Model) 架构的生成式 3D AI，秒级输出高保真的 3D 实体模型 (`.obj` / `.glb`)，彻底解决传统摄影测量算法 (例如 COLMAP) 对于反光、纯色抛光表面 (如汽车、金属) 几何重建失败的痛点。

本项目采用前后端分离设计，完全基于本地计算 (如 RTX 5880) 完成 AI 生成。

---

## 🏗️ 系统架构

*   **前端 (HTML/JS/CSS)**：
    *   单页应用 (SPA)，使用纯原生 JavaScript 和 Tailwind 风格的 CSS 实现现代暗黑界面。
    *   **3D 模型渲染**：采用 Google 的 `<model-viewer>` 组件直接在浏览器中渲染输出的 3D 模型 (.obj 或 .glb格式)。
    *   拖拽式上传、WebSocket 风格的状态轮询以展示 AI 推理进度。

*   **后端 (FastAPI + Python)**：
    *   使用 `FastAPI` 构建轻量异步 API 服务器。
    *   **AI 抠图管道**：使用 `rembg[gpu]` 自动剥离图片背景，净化生成质量。
    *   **TripoSR 引擎集成**：利用 PyTorch 前馈大模型对输入图片进行秒级张量推理，直接合成完整 Mesh 结构并在本地输出 3D 模型。

*   **环境依赖 (Pixi)**：
    *   整个 Python 运行环境及所有前沿 AI 库基于轻量化的 `pixi.toml` 管理，不污染宿主机系统。

## ⚙️ 环境自动配置 (Pixi)

本项目依赖以下核心环境包，可通过 `pixi run start` 自动激活与加载：
- `fastapi`, `uvicorn`, `python-multipart`
- `torch`, `torchvision`, `torchaudio`
- `transformers`, `accelerate`, `xformers`
- `trimesh`, `rembg[gpu]`, `omegaconf`, `einops`

## 🚀 启动与运行

1. 安装依赖包：
```bash
pixi install
```

2. 点击运行后端：
```bash
pixi run start
```
如果需要调试，可尝试：`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`。

前端默认在 `http://127.0.0.1:8000` 启动，上传 1 张带主体的图片后即可见证 3 秒钟大模型奇迹。
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