import torch
from comet import download_model, load_from_checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xcomet_model = load_from_checkpoint('./models/checkpoints/model.ckpt', reload_hparams=True).eval()
print(xcomet_model)