# 版权声明，表明代码归属 Bytedance Ltd. 及其关联公司所有
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# 许可证声明，采用 Apache License 2.0，允许合规使用和分发
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则按“原样”分发，不提供任何明示或暗示的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 os 模块，用于文件路径和系统相关操作
import os

# 导入 base 子模块，便于后续使用其内容
from . import base
# 导入 base 子模块中的所有公开对象
from .base import *

# 获取当前文件的绝对路径，并定位到其所在文件夹
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# 版本号获取说明：此处已弃用 single_controller.__version__，但仍保留读取机制
# 通过打开上级目录下 version/version 文件，读取版本号字符串
with open(os.path.join(os.path.join(version_folder, os.pardir), "version/version")) as f:
    __version__ = f.read().strip()

# 设置 __all__，用于控制模块对外暴露的接口，引用 base 子模块的 __all__
__all__ = base.__all__
