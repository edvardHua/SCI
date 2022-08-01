# -*- coding: utf-8 -*-
# @Time : 2022/8/1 2:47 PM
# @Author : zihua.zeng
# @File : convert.py

import torch
from torchprofile import profile_macs
from model import Finetunemodel


def to_onnx():
    model = Finetunemodel("./weights/medium.pt")
    model.eval()

    dummy_input = torch.randn((1, 3, 1280, 720))
    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "low_light_enhancement.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['image'],  # the model's input names
                      output_names=['enhanced_image'])


def calc_macs():
    model = Finetunemodel("./weights/medium.pt")
    model.eval()
    dummy_input = torch.randn((1, 3, 1280, 720))
    out = profile_macs(model, dummy_input)
    print(out / 1e9)


if __name__ == '__main__':
    calc_macs()