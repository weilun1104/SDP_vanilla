import numpy as np
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
import sys
import os
pythonpath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,pythonpath)
from Simulation.Data.lyngspice_master.lyngspice_master.examples.simulation_ota3_tsmc40 import OTA_three_simulation_gmid

class Circuit_40_OTA_three(SyntheticTestFunction):
    def __init__(
        self,
        bounds: list,
        noise_std: float = 0,
        negate: bool = False,  # 设置为 False，以便最小化目标函数
    ) -> None:
        self.dim = len(bounds)  # 参数维度
        self._bounds = torch.Tensor(bounds)  # 输入参数的边界

        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        # 将输入转换为电路仿真工具需要的格式
        x_tensor = X.to(torch.float32).to('cuda:0')
        scaling_factors = torch.tensor([1e-13,1e-13,1e-1,1e+0,
                                        1e+0,1e+0,1e-6,1e-6,
                                        1e-7,1e-7,1e-6,1e-7,
                                        1e-7,1e-7,1e-6,1e+0,
                                        1e+0,1e+0,1e+0,1e+0,
                                        1e+0,1e+0,1e+0,1e+0]).to('cuda:0')
        x_tensor = x_tensor * scaling_factors  # 按比例缩小回原来的大小
        # 调用电路仿真函数
        results = OTA_three_simulation_gmid(x_tensor)
        gain, dc_current, phase, GBW = results[0]  # 提取仿真结果
        print(f"Gain: {gain}, DC Current: {dc_current}, Phase: {phase}, GBW: {GBW}")
        return Tensor([gain.item(), dc_current.item(), phase.item(), GBW.item()])
