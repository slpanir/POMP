from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import uvicorn
import torch
from comet import download_model, load_from_checkpoint
from pydantic import BaseModel, Field
from typing import List
import asyncio

app = FastAPI()

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "1"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息
# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

class TranslationItem(BaseModel):
    src: str = Field(..., example="source text")
    mt: str = Field(..., example="machine translated text")
    ref: str = Field(..., example="reference text")

@app.post("/xcomet_score")
async def xcomet_score(comet_hyp: List[TranslationItem]):
    global xcomet_model
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        try:
            # 准备数据以供模型使用
            formatted_data = [{"src": item.src, "mt": item.mt, "ref": item.ref} for item in comet_hyp]

            # 调用模型进行评分
            model_hyp = xcomet_model.predict(formatted_data, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]
            torch_gc()
            return {"scores": model_hyp}
        except Exception as e:
            print(e)
            torch_gc()
            attempts += 1  # 增加尝试次数
            if attempts >= max_attempts:
                raise HTTPException(status_code=500, detail=str(e))
            await asyncio.sleep(1)  # 简单的延迟，避免立即重试

if __name__ == "__main__":
    # 加载模型
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Available CUDA Devices:")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU instead.")
    xcomet_model = load_from_checkpoint('./models/checkpoints/model.ckpt', reload_hparams=True).to(device).eval()
    uvicorn.run(app, host="0.0.0.0", port=8081)
