"""由算法目录（lab2shot/ops/ops.toml）生成供浏览器使用的 TypeScript 文件 webui/src/ops/ops.gen.ts。

`ops.toml` 是算法目录的唯一来源。浏览器不具备 TOML 解析能力，且页面启动时不应依赖向服务器请求目录，
因此由同一文件生成静态 TS。`tests/test_ops_catalog.py::test_browser_catalog_is_generated` 校验生成结果与 ops.toml 一致。

用法：uv run python tools/gen_ops_ts.py        （--check 仅校验，不写入）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab2shot.ops.run import CATALOG  # noqa: E402

OUT = ROOT / "webui" / "src" / "ops" / "ops.gen.ts"


def text() -> str:
    lines = [
        "// 由 tools/gen_ops_ts.py 从 lab2shot/ops/ops.toml 生成，**不要手改**。",
        "// 算法目录只有一个源（需求 6：算法只写一次）；改算法改那个 toml，然后跑：",
        "//   uv run python tools/gen_ops_ts.py",
        "export const OPS = {",
    ]
    for op_id, desc in CATALOG.items():
        lines.append(f"  {json.dumps(op_id, ensure_ascii=False)}: {json.dumps(desc, ensure_ascii=False)},")
    lines += ["} as const;", "", "export type OpId = keyof typeof OPS;", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    want = text()
    if "--check" in sys.argv:
        got = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if got != want:
            print(f"{OUT} 和 ops.toml 对不上：跑 uv run python tools/gen_ops_ts.py")
            raise SystemExit(1)
        print("ops.gen.ts 是最新的")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(want, encoding="utf-8")
        print(f"写好 {OUT}")
