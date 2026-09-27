"""从 ASGI 的回答上数字节，记在提出请求的那个账号上（数字和读法在 lab2shot/traffic.py，那一层不认识 HTTP）。"""

from __future__ import annotations

import asyncio

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..traffic import SCOPE_USER, count, flush


class Meter:
    """最外层的 ASGI 中间件：数一次请求真正答出去了多少字节，记在提出请求的那个账号上。

    必须放在最外面（app.py 里最后 add）：压缩已经做完，数到的才是线上真正走的字节；里面的
    server/access.py Guard 已把「这次是谁」写进同一个 scope（`SCOPE_USER`），这里直接读。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        sent = 0

        async def metered(message: Message) -> None:
            nonlocal sent
            if message["type"] == "http.response.body":
                sent += len(message.get("body", b"") or b"")
            await send(message)

        try:
            await self.app(scope, receive, metered)
        finally:
            # 记数只碰内存；count 返回真表示该落盘了，落盘是 sqlite 写，放到线程上，不卡事件循环
            if count(int(scope.get(SCOPE_USER) or 0), sent):
                await asyncio.to_thread(flush)
