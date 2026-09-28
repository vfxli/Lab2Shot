"""算法目录：几个节点共用的基础算法（选人、选取条目、框转遮罩、图像合成），以数据形式描述，由一个通用执行器执行。

- `ops.toml`：每条算法一项；新增的算法属于词汇表中已有的运算种类时，只需修改此文件。
- `vocab.py`：词汇表（支持的运算种类）及精度、取整规则。
- `run.py`：执行器（numpy）。

节点通过 `NodeDef.ops` 声明所用算法（建类时核对算法存在，nodes/applies.py check_declarations），在 cook 中以
`run(算法 id, 参数)` 调用；算法本体统一定义于本目录，不在节点中定义。算法只在服务器上执行：浏览器不计算任何节点。
"""

from .run import CATALOG, describe, run
from .vocab import BadOp

__all__ = ["CATALOG", "BadOp", "describe", "run"]
