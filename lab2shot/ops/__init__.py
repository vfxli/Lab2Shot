"""算法目录：可在服务器与浏览器两端执行的基础算法。算法以数据形式描述一次，两端各有一个通用执行器。

- `ops.toml`：每条算法一项；新增算法只需修改此文件。
- `vocab.py`：词汇表（支持的运算种类）及精度、取整规则。
- `run.py`：服务器端执行器（numpy）；浏览器端执行器为 `webui/src/ops/run.ts`。

节点仅通过 `NodeDef.ops` 声明所用算法，算法本体统一定义于本目录，不在节点中定义。
"""

from .run import CATALOG, describe, run
from .vocab import DISPLAY_TOLERANCE, BadOp

__all__ = ["CATALOG", "DISPLAY_TOLERANCE", "BadOp", "describe", "run"]
