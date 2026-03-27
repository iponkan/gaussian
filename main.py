"""
3DGS 重建系统后端
上传照片 -> 自动生成 3DGS 模型 -> 前端渲染
"""

import asyncio
import glob
import os
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ──────────────────────── 配置 ────────────────────────

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
EXPORT_DIR = BASE_DIR / "data" / "exports"

CONTAINER_NAME = "nerfstudio_3dgs"
MAX_ITERATIONS = 7000

# 确保目录存在
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────── 状态管理 ────────────────────────

class TaskStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing_images"
    TRAINING = "training"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status: TaskStatus = TaskStatus.UPLOADING
        self.progress: int = 0        # 0-100
        self.message: str = "等待上传完成..."
        self.created_at: str = datetime.now().isoformat()
        self.error: str | None = None
        self.model_url: str | None = None
        self.file_count: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "error": self.error,
            "model_url": self.model_url,
            "file_count": self.file_count,
        }


# 内存字典存储任务状态
tasks: Dict[str, TaskInfo] = {}


# ──────────────────────── FastAPI 应用 ────────────────────────

app = FastAPI(title="3DGS 重建系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ──────────────────────── 工具函数 ────────────────────────

async def run_command(cmd: str, task: TaskInfo, timeout: int = 7200) -> str:
    """异步执行 Shell 命令，并实时打印日志"""
    print(f"[{task.task_id}] 开始命令: {cmd}")
    try:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        all_stdout = []

        async def read_stream(stream, is_stderr=False):
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    prefix = " (STDERR)" if is_stderr else ""
                    print(f"[{task.task_id}]{prefix} {text}")
                    all_stdout.append(text)
                    # 可以在这里根据日志内容更精细地更新进度
                    if "Step" in text and "/7000" in text:
                        try:
                            # 提取类似 [Step 1200/7000] 的进度
                            parts = text.split("Step")[1].split("/")[0].strip()
                            step = int(parts)
                            # 训练进度占总进度的 35-80% 之间
                            task.progress = int(35 + (step / 7000) * 45)
                        except:
                            pass

        # 并行读取 stdout 和 stderr
        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, is_stderr=True)
        )

        await asyncio.wait_for(process.wait(), timeout=timeout)

        if process.returncode != 0:
            raise RuntimeError(f"命令执行失败 (exit code {process.returncode})")

        return "\n".join(all_stdout)
    except asyncio.TimeoutError:
        print(f"[{task.task_id}] 命令超时!")
        raise RuntimeError(f"命令执行超时 ({timeout}s)")


async def find_config_path(task_id: str) -> str:
    """在 outputs 目录中查找训练生成的 config.yml 路径（容器内路径）"""
    # 在宿主机上查找
    host_pattern = str(OUTPUT_DIR / task_id / "**" / "config.yml")
    config_files = glob.glob(host_pattern, recursive=True)

    if not config_files:
        raise RuntimeError(f"未找到 config.yml 文件，搜索路径: {host_pattern}")

    # 取最新的一个（按修改时间排序）
    config_files.sort(key=os.path.getmtime, reverse=True)
    host_config_path = config_files[0]

    # 将宿主机路径转换为容器内路径
    rel_path = os.path.relpath(host_config_path, str(OUTPUT_DIR))
    container_config_path = f"/workspace/outputs/{rel_path}"
    print(f"找到 config.yml: {host_config_path} -> 容器路径: {container_config_path}")
    return container_config_path


# ──────────────────────── 3D 重建流程 ────────────────────────

async def reconstruction_pipeline(task: TaskInfo):
    """异步执行完整的 3D 重建流程"""
    task_id = task.task_id

    try:
        # ──── 动态参数配置 ────
        img_count = task.file_count
        if img_count <= 30:
            # 极少照片时：为了彻底把 150MB 降到 30MB 左右
            # 我们需要让高斯球更难分裂、更容易被剔除，并且提早结束训练。
            matching_method = "exhaustive"
            densify_thresh = 0.0003     # 升高增殖门槛，防止飞絮产生
            cull_thresh = 0.2           # 大幅升高透明度剔除，无情切掉背景杂乱模糊的高斯球
            max_iters = 3500            # 图片少时，过多的训练回合会导致过度拟合空洞，早早喊停
        elif img_count <= 100:
            matching_method = "vocab_tree"
            densify_thresh = 0.0002
            cull_thresh = 0.1
            max_iters = 7000
        else:
            matching_method = "vocab_tree"
            densify_thresh = 0.0002
            cull_thresh = 0.1
            max_iters = 7000

        # ──── 步骤 1: 计算相机位姿 (COLMAP) ────
        task.status = TaskStatus.PROCESSING
        task.progress = 10
        task.message = f"正在计算相机位姿 (匹配模式: {matching_method})..."

        cmd_process = (
            f"docker exec {CONTAINER_NAME} "
            f"ns-process-data images "
            f"--data /workspace/inputs/{task_id}/images "
            f"--output-dir /workspace/inputs/{task_id}_processed "
            f"--matching-method {matching_method}"
        )
        await run_command(cmd_process, task)
        task.progress = 30
        task.message = "相机位姿计算完成"

        # ──── 步骤 2: 训练 3DGS 模型 ────
        task.status = TaskStatus.TRAINING
        task.progress = 35
        task.message = f"正在训练 3DGS 模型 (最大回合数: {max_iters})..."

        cmd_train = (
            f"docker exec {CONTAINER_NAME} "
            f"ns-train splatfacto "
            f"--data /workspace/inputs/{task_id}_processed "
            f"--max-num-iterations {max_iters} "
            f"--output-dir /workspace/outputs/{task_id} "
            f"--viewer.quit-on-train-completion True "
            f"--pipeline.model.densify-grad-thresh {densify_thresh} "
            f"--pipeline.model.cull-alpha-thresh {cull_thresh}"
        )
        await run_command(cmd_train, task, timeout=7200)
        task.progress = 80
        task.message = "模型训练完成"

        # ──── 步骤 3: 导出通用模型 ────
        task.status = TaskStatus.EXPORTING
        task.progress = 85
        task.message = "正在导出 3D 模型..."

        # 查找 config.yml
        config_path = await find_config_path(task_id)

        cmd_export = (
            f"docker exec {CONTAINER_NAME} "
            f"ns-export gaussian-splat "
            f"--load-config {config_path} "
            f"--output-dir /workspace/exports/{task_id}"
        )
        await run_command(cmd_export, task)

        # ──── 完成 ────
        # 查找导出的模型文件
        export_path = EXPORT_DIR / task_id
        ply_files = list(export_path.glob("*.ply"))
        if ply_files:
            model_filename = ply_files[0].name
            task.model_url = f"/exports/{task_id}/{model_filename}"
        else:
            # 尝试查找 splat 文件
            splat_files = list(export_path.glob("*.splat"))
            if splat_files:
                model_filename = splat_files[0].name
                task.model_url = f"/exports/{task_id}/{model_filename}"
            else:
                task.model_url = f"/exports/{task_id}/"

        task.status = TaskStatus.COMPLETED
        task.progress = 100
        task.message = "3D 模型生成完成！"
        print(f"[{task_id}] ✅ 完成！模型地址: {task.model_url}")

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.message = f"任务失败: {str(e)[:200]}"
        print(f"[{task_id}] ❌ 失败: {e}")


# ──────────────────────── API 接口 ────────────────────────

@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """上传图片并启动 3D 重建"""
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张图片")

    # 过滤非图片文件
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    valid_files = [f for f in files if Path(f.filename or "").suffix.lower() in allowed_ext]

    if not valid_files:
        raise HTTPException(status_code=400, detail="未检测到有效的图片文件（支持 jpg/png/bmp/tiff/webp）")

    if len(valid_files) < 3:
        raise HTTPException(status_code=400, detail="3D 重建至少需要 3 张图片，请上传更多照片")

    # 生成任务 ID
    task_id = str(uuid.uuid4())

    # 创建任务
    task = TaskInfo(task_id)
    task.file_count = len(valid_files)
    tasks[task_id] = task

    # 保存图片到宿主机
    save_dir = UPLOAD_DIR / task_id / "images"
    save_dir.mkdir(parents=True, exist_ok=True)

    for f in valid_files:
        file_path = save_dir / f.filename
        content = await f.read()
        with open(file_path, "wb") as fp:
            fp.write(content)

    task.message = f"已上传 {len(valid_files)} 张图片，正在启动 3D 重建..."
    task.progress = 5

    # 异步启动重建流程（不阻塞主线程）
    asyncio.create_task(reconstruction_pipeline(task))

    return JSONResponse(content={
        "success": True,
        "task_id": task_id,
        "file_count": len(valid_files),
        "message": f"上传成功！{len(valid_files)} 张图片已接收，3D 重建任务已启动。",
    })


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """查询任务状态"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return JSONResponse(content=task.to_dict())


@app.get("/api/tasks")
async def list_tasks():
    """列出所有任务"""
    return JSONResponse(content={
        "tasks": [t.to_dict() for t in sorted(
            tasks.values(),
            key=lambda x: x.created_at,
            reverse=True
        )]
    })


# ──────────────────────── 前端页面路由 ────────────────────────

from fastapi.responses import FileResponse


@app.get("/")
async def serve_frontend():
    """服务前端页面"""
    return FileResponse(str(BASE_DIR / "static" / "index.html"))
