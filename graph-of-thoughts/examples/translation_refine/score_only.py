from comet import download_model, load_from_checkpoint
import json
import torch

data_path = "/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/"
srcref_pair = "data/x2x/et2en/"
hyp_pair = "results/et2en/test/"
src_path = data_path + srcref_pair + "src"
hyp_path = data_path + hyp_pair + "chatgpt-super_direct_refine_got_2023-12-29_22-33-55/" + "test_got_refine"
ref_path = data_path + srcref_pair + "ref"
result_path = data_path + hyp_pair + "chatgpt-super_direct_refine_got_2023-12-29_22-33-55/" + "xcomet.json"

with open(src_path, 'r', encoding='utf8') as f:
    src = f.readlines()
with open(hyp_path, 'r', encoding='utf8') as f:
    hyp = f.readlines()
with open(ref_path, 'r', encoding='utf8') as f:
    ref = f.readlines()

src = [x.strip() for x in src]
hyp = [x.strip() for x in hyp]
ref = [x.strip() for x in ref]


comet_hyp = []
comet_input = []
for i in range(len(ref)):
    comet_hyp.append({'src': src[i], 'mt': hyp[i], 'ref': ref[i]})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = download_model("Unbabel/XCOMET-XL", saving_directory='/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/models/')
comet_model = load_from_checkpoint('/mnt/e/unmt/acl22-sixtp/graph-of-thoughts/examples/translation_refine/models/checkpoints/model.ckpt').to(device).eval()
model_hyp = comet_model.predict(comet_hyp, batch_size=8, gpus=1, num_workers=0).to_tuple()[1]

eval_results = {'xcomet_hyp': model_hyp}
with open(result_path, 'w', encoding='utf8') as f:
    json.dump(eval_results, f)

