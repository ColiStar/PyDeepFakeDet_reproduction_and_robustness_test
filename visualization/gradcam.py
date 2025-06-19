# visualization/gradcam.py

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision               # ← 新增此行：为了解决类型注解里使用 torchvision.transforms 时找不到名称
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import yaml
import sys
import os
import argparse

# 为了能够通过 "from PyDeepFakeDet import models" 找到我们的模型定义，需要把项目根目录加到 PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyDeepFakeDet import models
from PyDeepFakeDet.utils.checkpoint import load_checkpoint


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        size=(224, 224),
        num_cls=1000,
        mean=None,
        std=None,
    ) -> None:
        """
        model: 已经构造并加载权重的 nn.Module 网络实例
        target_layer: 形如 "model._conv_head" 的字符串，表示在 model 对象下寻找子模块来挂钩
        size: 输入网络时先 resize 到的尺寸（如 (224,224)）
        num_cls: 网络输出类别数（比如 DeepFake 分 2 类，就传 2）
        mean/std: 输入归一化用的均值和方差
        """
        self.model = model
        self.model.eval()

        # —— 下面这段代码实现“对 target_layer 进行点分层拆解并递归 getattr”，
        #     比如 target_layer="model._conv_head" 就会先 getattr(self.model, "model")，
        #     再 getattr(上一层的结果, "_conv_head")，最后拿到真正的 Conv2dStaticSamePadding 层。
        modules = target_layer.split(".")
        sub_mod = self.model
        for m in modules:
            sub_mod = getattr(sub_mod, m)
        # sub_mod 现在指向 net.model._conv_head 这样要挂钩的卷积层

        # 在该卷积层上注册 forward/backward hook
        sub_mod.register_forward_hook(self.__forward_hook)
        sub_mod.register_backward_hook(self.__backward_hook)

        self.size = size
        self.origin_size = None
        self.num_cls = num_cls

        # 如果用户传了自定义的 mean/std，就使用它们，否则使用 ImageNet 默认值
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if mean and std:
            self.mean, self.std = mean, std

        self.grads = []
        self.fmaps = []

    def __img_transform(
        self, img_arr: np.ndarray, transform: torchvision.transforms
    ) -> torch.Tensor:
        img = img_arr.copy()
        img = Image.fromarray(np.uint8(img))
        img = transform(img).unsqueeze(0)  # (1, C, H, W)
        return img

    def __img_preprocess(self, img_in: np.ndarray) -> torch.Tensor:
        # 记录原始尺寸，后面会把 CAM heatmap 还原到这个尺寸
        self.origin_size = (img_in.shape[1], img_in.shape[0])  # (width, height)
        img = img_in.copy()
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )
        img_tensor = self.__img_transform(img, transform)
        return img_tensor

    def __forward_hook(self, module, input, output):
        # 在 forward 时把 当前层的 feature map 保留下来
        self.fmaps.append(output)

    def __backward_hook(self, module, grad_in, grad_out):
        # 在 backward 时把 当前层的梯度 保存下来
        # grad_out[0] 是 forward hook 对应层的梯度
        self.grads.append(grad_out[0].detach())

    def __compute_loss(self, logit, index=None):
        # 如果没传 label，就用网络预测的最大值作为要反向的目标
        if index is None:
            index = np.argmax(logit.cpu().data.numpy())
        else:
            index = np.array(index)

        # 构造 one-hot 向量
        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros((1, self.num_cls)).scatter_(1, index, 1)
        one_hot.requires_grad = True
        loss = torch.sum(one_hot * logit)
        return loss

    def __compute_cam(self, feature_map, grads):
        # feature_map: numpy, shape = (C, H', W')
        # grads: numpy, 同样是对应层梯度 (C, H', W')
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # (H', W')
        alpha = np.mean(grads, axis=(1, 2))  # (C,)
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, self.size)  # 先把 CAM 放大到与输入一致
        cam = (cam - np.min(cam)) / np.max(cam)  # 归一化到 [0,1]
        return cam

    def __show_cam_on_image(
        self, img: np.ndarray, mask: np.ndarray, if_show=True, if_write=False, path=""
    ):
        # img: 原始图，已经归一化到 [0,1]
        # mask: CAM heatmap，[0,1]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # (H, W, 3), BGR
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        if if_write and path:
            cv2.imwrite(path, cam)
        if if_show:
            plt.imshow(cam[:, :, ::-1])  # 转回 RGB
            plt.axis("off")
            plt.show()

    def forward(self, img_arr: np.ndarray, label=None, show=True, write=False, path=""):
        """
        img_arr: numpy 数组格式的 BGR 原始图（还没有做归一化），shape = (H, W, 3)
        label: 如果想针对特定类别（0/1）做 GradCAM，可以把类别编号传进来；默认 None 则让网络自己选最大预测
        show: 是否实时弹窗显示
        write: 是否写到硬盘
        path: 如果 write=True 就把可视化图保存到这个路径
        """
        img_input = self.__img_preprocess(img_arr.copy())  # 变成归一化后 (1,3,H',W')

        output_dict = self.model({"img": img_input})
        logits = output_dict["logits"]                                 # (1, num_cls)
        idx = np.argmax(logits.cpu().data.numpy())                    # 网络预测的类别

        self.model.zero_grad()
        loss = self.__compute_loss(logits, label)                      # one-hot loss
        loss.backward()                                                # 反向传播，计算梯度

        grads_val = self.grads[0].cpu().data.numpy().squeeze()         # (C, H', W')
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()              # (C, H', W')
        cam = self.__compute_cam(fmap, grads_val)                      # (H', W') 归一化好

        cam_show = cv2.resize(cam, self.origin_size)                   # 放回原图尺寸
        img_show = img_arr.astype(np.float32) / 255.0                  # 原图也归一化到 [0,1]
        self.__show_cam_on_image(img_show, cam_show, if_show=show, if_write=write, path=path)

        # 清空 buffer
        self.fmaps.clear()
        self.grads.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("gradcam script", add_help=False)
    parser.add_argument("--model", type=str, required=True, help="模型名字，比如 M2TR")
    parser.add_argument("--pth", type=str, required=True, help="权重文件路径")
    parser.add_argument("--layer", type=str, required=True, help="要挂钩的层，比如 model._conv_head")
    parser.add_argument("--img", type=str, required=True, help="要可视化的单张图片文件路径")
    parser.add_argument("--save_path", type=str, required=True, help="输出 GradCAM 可视化图的保存路径")
    args = parser.parse_args()

    # ── 第一步：从配置文件里读取 model_cfg —— 
    full_cfg = yaml.safe_load(open("configs/m2tr.yaml", "r"))
    model_cfg = full_cfg["MODEL"]
    # 注意：M2TR 构造函数里会用到 model_cfg["IMG_SIZE"]、["BACKBONE"]、["NUM_CLASSES"] 等字段

    # ── 第二步：构造网络实例并载入权重 —— 
    net = getattr(models, args.model)(model_cfg)                 # 等同于 models.M2TR(model_cfg)
    checkpoint = torch.load(args.pth, map_location="cpu")
    load_checkpoint(args.pth, net, False)

    # ── 第三步：读入要可视化的单张图片 —— 
    img = cv2.imread(args.img, 1)
    assert img is not None, f"无法读取输入图片: {args.img}"

    # ── 第四步：创建 GradCAM 对象，指定 target_layer —— 
    grad_cam = GradCAM(
        net,
        target_layer=args.layer,
        size=(320, 320),                       # M2TR 默认输入尺度是 320，所以这里设置 (320,320)
        num_cls=model_cfg["NUM_CLASSES"],      # 例如 2
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # ── 第五步：执行前向 + 反向，输出热力图并保存 —— 
    grad_cam.forward(img, show=False, write=True, path=args.save_path)
    print("GradCAM 图已保存到:", args.save_path)