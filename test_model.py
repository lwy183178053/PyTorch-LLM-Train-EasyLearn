import random
import numpy as np
import torch
from transformers import AutoTokenizer
from model import MyModel
import time

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained('tokenizer')
# 模型
model = MyModel(vocab_size=tokenizer.vocab_size, embed_dim=768, num_layers=8, num_heads=8).to(device)

model.load_state_dict(torch.load("model_save/768emb_8layer_8head/sft_model.pth", map_location=device, weights_only=True))
model.eval()  # 切换到推理模式

# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(random.randint(0, 2048))
#setup_seed(512)  # 如需固定每次输出则换成【固定】的随机种子
messages = [{"role": "user", "content": "请简要解释一下什么是区块链技术？"}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt",truncation=True).to(device)  # 将输入数据移动到设备上

# 记录开始结束时间
start_time = time.time()
response_ids = model.generate(inputs.input_ids, max_length=512, temperature=0.6,top_k=50)[0].tolist()#[0][len(inputs.input_ids[0]):].tolist()
end_time = time.time()

# 计算生成的token数量
generated_token_count = len(response_ids) - len(inputs.input_ids[0])
# 计算所用时间
elapsed_time = end_time - start_time
# 计算生成速度
generation_speed = generated_token_count / elapsed_time

# 解码
response = tokenizer.decode(response_ids)#, skip_special_tokens=True)
print(response)
messages.append({"role": "assistant", "content": response})
print(f"模型生成速度: {generation_speed:.2f} token/s")
