from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import uvicorn
import torch
from comet import download_model, load_from_checkpoint
from pydantic import BaseModel, Field
from typing import List

app = FastAPI()

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
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
    hyp: str = Field(..., example="hypothesis text")

@app.post("/xcomet_score")
async def xcomet_score(comet_hyp: List[TranslationItem]):
    global xcomet_model
    try:
        # 准备数据以供模型使用
        formatted_data = [{"src": item.src, "mt": item.mt, "hyp": item.hyp} for item in comet_hyp]

        # 调用模型进行评分
        model_hyp = xcomet_model.predict(formatted_data, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]
        torch_gc()
        return {"scores": model_hyp}
    except Exception as e:
        print(e)
        torch_gc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xcomet_model = load_from_checkpoint('./models/checkpoints/model.ckpt', reload_hparams=True).to(device).eval()
    uvicorn.run(app, host="0.0.0.0", port=8080)
