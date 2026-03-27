import sys
import asyncio
import uuid
import shutil
import time
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum

# ==========================================
# 1. API Models & Global State
# ==========================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    MODELING = "modeling"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    message: str
    file_count: int
    result_url: str = ""

# In-memory task dict (use a DB in production)
tasks: Dict[str, Task] = {}

# Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TripoSR 3D Generation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/api/exports", StaticFiles(directory=str(OUTPUT_DIR)), name="exports")

# ==========================================
# 2. Async Command Runner
# ==========================================

async def run_command(cmd_str: str, task: Task, timeout: int = 900):
    """
    Run a shell command asynchronously and stream its stdout/stderr to console.
    """
    print(f"\n[Command Start: {task.task_id}] {cmd_str}\n" + "="*50)
    process = await asyncio.create_subprocess_shell(
        cmd_str,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(BASE_DIR)
    )
    
    async def read_stream(stream, is_stderr=False):
        while True:
            line = await stream.readline()
            if not line:
                break
            line_str = line.decode('utf-8', errors='replace').rstrip()
            print(f"[{'ERR' if is_stderr else 'OUT'}] {line_str}")

    t_stdout = asyncio.create_task(read_stream(process.stdout))
    t_stderr = asyncio.create_task(read_stream(process.stderr, True))
    
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        process.terminate()
        task.status = TaskStatus.FAILED
        task.message = f"指令超时 ({timeout}s)"
        print(f"\n[Command Timeout] {cmd_str}")
        raise RuntimeError(f"Command timed out after {timeout}s : {cmd_str}")

    await t_stdout
    await t_stderr

    if process.returncode != 0:
        task.status = TaskStatus.FAILED
        task.message = "处理发生错误"
        print(f"\n[Command Failed] process returned {process.returncode}")
        raise RuntimeError(f"Command failed with return code {process.returncode}")
    print(f"\n[Command Success: {task.task_id}]\n" + "="*50)

# ==========================================
# 3. Task Pipeline (TripoSR)
# ==========================================

async def reconstruction_pipeline(task: Task):
    """
    AI 瞬时生成 3D (TripoSR)
    """
    task_id = task.task_id
    try:
        images_dir = UPLOAD_DIR / task_id / "images"
        out_dir = OUTPUT_DIR / task_id
        
        # 寻找第一张有效图片进行生成
        img_files = list(images_dir.glob("*.*"))
        if not img_files:
            raise RuntimeError("没有找到图片文件")
        
        target_image = img_files[0] # TripoSR 基于单图工作，我们取第一张图
        
        # 步骤 1: 启动推理
        task.status = TaskStatus.MODELING
        task.progress = 20
        task.message = "大模型开始推理 (TripoSR 推断中)..."

        # 调用本地克隆的 TripoSR 预测脚本 (去背景模式为默认开启)
        cmd_run = (
            f"HF_ENDPOINT=https://hf-mirror.com python TripoSR_App/run.py {target_image} "
            f"--output-dir {out_dir} "
            f"--model-save-format glb "
            f"--device cuda:0"
        )
        await run_command(cmd_run, task, timeout=300)

        # 完成
        task.status = TaskStatus.COMPLETED
        task.progress = 100
        task.message = "模型生成成功！"
        # TripoSR 生成的模型位于 {out_dir}/0/mesh.glb
        task.result_url = f"/api/exports/{task_id}/0/mesh.glb"
        
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.message = f"失败: {str(e)}"
        print(f"[Error] Pipeine failed for task {task_id}: {e}")

# ==========================================
# 4. Endpoints
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<h1>前端文件 static/index.html 不存在</h1>"

@app.post("/api/upload")
async def upload_images(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    task_id = str(uuid.uuid4())
    img_dir = UPLOAD_DIR / task_id / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    valid_files = [f for f in files if f.filename and not f.filename.startswith(".")]
    if not valid_files:
        raise HTTPException(status_code=400, detail="没有合法的图片文件")

    # 限制上传文件数，TripoSR 单次单图效果即可，这里为了友好允许用户框选多张，但后端只拿第一张图生成
    file_count = len(valid_files)
    
    for f in valid_files:
        content = await f.read()
        file_path = img_dir / f.filename
        with open(file_path, "wb") as fp:
            fp.write(content)

    new_task = Task(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0,
        message="文件上传完成，排队中",
        file_count=file_count
    )
    tasks[task_id] = new_task

    background_tasks.add_task(reconstruction_pipeline, new_task)
    return JSONResponse(content={"task_id": task_id, "message": "上传成功"})

@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task.model_dump()
