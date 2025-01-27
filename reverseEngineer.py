# pip install accelerate
import torch
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForCausalLM
checkpoint = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for fp16 use `torch_dtype=torch.float16` instead
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
print(model)
summary(model)

inputs = tokenizer.encode("Gravity is", return_tensors="pt").to("mps")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

'''
(venv) shriti@Shritis-MacBook-Pro GithubCode % python3 reverseEngineer.py 
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
======================================================================
Layer (type:depth-idx)                        Param #
======================================================================
LlamaForCausalLM                              --
├─LlamaModel: 1-1                             --
│    └─Embedding: 2-1                         28,311,552
│    └─ModuleList: 2-2                        --
│    │    └─LlamaDecoderLayer: 3-1            3,540,096
│    │    └─LlamaDecoderLayer: 3-2            3,540,096
│    │    └─LlamaDecoderLayer: 3-3            3,540,096
│    │    └─LlamaDecoderLayer: 3-4            3,540,096
│    │    └─LlamaDecoderLayer: 3-5            3,540,096
│    │    └─LlamaDecoderLayer: 3-6            3,540,096
│    │    └─LlamaDecoderLayer: 3-7            3,540,096
│    │    └─LlamaDecoderLayer: 3-8            3,540,096
│    │    └─LlamaDecoderLayer: 3-9            3,540,096
│    │    └─LlamaDecoderLayer: 3-10           3,540,096
│    │    └─LlamaDecoderLayer: 3-11           3,540,096
│    │    └─LlamaDecoderLayer: 3-12           3,540,096
│    │    └─LlamaDecoderLayer: 3-13           3,540,096
│    │    └─LlamaDecoderLayer: 3-14           3,540,096
│    │    └─LlamaDecoderLayer: 3-15           3,540,096
│    │    └─LlamaDecoderLayer: 3-16           3,540,096
│    │    └─LlamaDecoderLayer: 3-17           3,540,096
│    │    └─LlamaDecoderLayer: 3-18           3,540,096
│    │    └─LlamaDecoderLayer: 3-19           3,540,096
│    │    └─LlamaDecoderLayer: 3-20           3,540,096
│    │    └─LlamaDecoderLayer: 3-21           3,540,096
│    │    └─LlamaDecoderLayer: 3-22           3,540,096
│    │    └─LlamaDecoderLayer: 3-23           3,540,096
│    │    └─LlamaDecoderLayer: 3-24           3,540,096
│    │    └─LlamaDecoderLayer: 3-25           3,540,096
│    │    └─LlamaDecoderLayer: 3-26           3,540,096
│    │    └─LlamaDecoderLayer: 3-27           3,540,096
│    │    └─LlamaDecoderLayer: 3-28           3,540,096
│    │    └─LlamaDecoderLayer: 3-29           3,540,096
│    │    └─LlamaDecoderLayer: 3-30           3,540,096
│    └─LlamaRMSNorm: 2-3                      576
│    └─LlamaRotaryEmbedding: 2-4              --
├─Linear: 1-2                                 28,311,552
======================================================================
Total params: 162,826,560
Trainable params: 162,826,560
Non-trainable params: 0
======================================================================
'''
