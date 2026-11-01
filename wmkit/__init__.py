"""wmkit —— 纯 numpy + Pillow 的水印分析 / 去除工具包。

主要模块：
- analysis : 背景估计、叠加层颜色分析、像素掩膜、候选区域检测
- inpaint  : Telea（FMM）图像修复
- imageio  : 读写与可视化
- morph    : 形态学 / 连通域工具
"""

__version__ = "1.0.0"

from . import analysis, imageio, inpaint, morph  # noqa: F401
