import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

model_name_or_path=r"D:\LiJinQi\cache\models--internlm--internlm2-chat-7b\snapshots\f7dc28191037a297c086b5b70c6a226e2134e46d"
tokenizer=AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map)