import openai
import time
import torch
from fairseq.models.transformer import TransformerModel
import os
# openai.api_key = "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"
# openai.api_base = "https://slpansir.com/v1"
# # openai.organization = "org-947ErSZH9Ow6CK4igZlOXYrj"
# with open("prompt.txt", 'r', encoding='utf-8') as f:
#     prompt = "".join(f.readlines())
#
# # 计算时间
# start = time.time()
# response = openai.ChatCompletion.create(
#                 model='gpt-3.5-turbo',
#                 messages=[{"role": "user", "content": "如何添加L2正则化项约束，举个例子"}],
#             )
# end = time.time()
# print(response.choices[0].message['content'])
# print("prompt_tokens:" + str(response["usage"]["prompt_tokens"]))
# print("completion_tokens:" + str(response["usage"]["completion_tokens"]))
# print("time cost: ", end-start)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nmt_model_path = '/mnt/e/unmt/acl22-sixtp/models/x2x'
xlmr_model_path = '/mnt/e/unmt/acl22-sixtp/models/xlmrL_base'
nmt_model = TransformerModel.from_pretrained(
    nmt_model_path,
    checkpoint_file='x2x.pt',
    bpe='sentencepiece',
    bpe_codes=os.path.join(nmt_model_path, 'sentencepiece.bpe.model'),
    fixed_dictionary=os.path.join(nmt_model_path, 'dict.txt'),
    xlmr_modeldir=xlmr_model_path,
    eval_bleu=False,
    enable_lang_proj=True,
    fp16=True,
    source_lang='zh',
    target_lang='en',
    enable_lang_ids=True,

).eval().to(device)

translated = nmt_model.translate('hello world')
