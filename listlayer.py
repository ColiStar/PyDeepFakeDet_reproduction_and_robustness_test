import yaml
import torch
from PyDeepFakeDet.models.m2tr import M2TR

# 1. 先把配置读进来，构造一个 M2TR 实例
full_cfg = yaml.safe_load(open("configs/m2tr.yaml", "r"))
model_cfg = full_cfg["MODEL"]
net = M2TR(model_cfg)

# 2. 把 net 里所有子模块的“名字 → 模块”都打印一下
def print_modules(module, prefix=""):
    """
    递归打印 module 以及它的所有子 module，类似：
     > prefix= "" 时
     backbone （是 net.model 的别名或父级）
       └─ _conv_stem：...
       └─ _blocks[0]： ...
       └─ _conv_head： ...
       ...
     head1
     head2
    """
    for name, child in module.named_children():
        full_name = prefix + (name if prefix=="" else "." + name)
        print(full_name, "→", child.__class__.__name__)
        print_modules(child, full_name)

print("=== M2TR 整个网络结构 ===")
print_modules(net)