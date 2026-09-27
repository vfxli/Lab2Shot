"""单个通道一帧在传输时的字节格式（`"L2C1"`）。浏览器端的实现是 `webui/src/transfer/plane.ts`，读取同一张表。

视图只接收代理：`lab2shot/view/proxy.py` 先按管理员设定的档位等比缩放，再交由本模块编码为字节；不存在无损 / 有损
之类的第二档可供切换。「单个通道如何编码为字节」属于视图层，不属于 HTTP（`server/wire.py` 只负责发送哪个文件、
浏览器能否长期保留）；view 不 import server，核心各层只向下 import。

格式的取舍：

* 浏览器可以直接送入 GPU 纹理。四种格式分别对应 WebGL2 的 R8 / R16 / R16F / R32F，浏览器将 gunzip 之后的
  ArrayBuffer 从第 16 个字节起映射为 Uint8Array / Uint16Array / Float32Array 即可调用 texImage2D，无需任何转换
  （头部 16 字节是 4 的倍数，恰好对齐）。
* 位数由数据本身决定（`_channel_format` 逐值检查能否容纳），不按类型名推测：8 位画面按 8 位发送（每个值一个字节），
  其他一律使用尾数低位置零的半精度（见下一条）；半精度无法容纳的保留 float32：绝对值超过 HALF_TRIMMED_MAX（65280）的值
  （以厘米计超过 650 米的深度、世界坐标位置图）转为半精度即为无穷，到浏览器后全部为 inf。
* 外层压缩使用 gzip（`Content-Encoding`，浏览器以原生代码解压，页面无需编写任何 JS）。横向差分、
  将半精度的位按 16 位 PNG 写出，在实拍素材上反而更大（1.61 / 1.56 MB 对 1.29 MB），在平滑素材上只节省一成多，
  且浏览器需多执行一遍；字节平面（`server/view_data.py` 的 pack 为三维数据使用的方法）对高熵数据可节省两成，对低熵数据反而
  增大一倍多，且浏览器需先将四个平面拼回，不再能直接送入纹理。以上方案均不采用。
* 尾数低位置零：半精度尾数保留 7 位，显示上的相对误差 ≤0.4%，即 8 位屏幕上的一级；视图不要求逐位一致。
  代理本身即为用于查看的副本，这一步与等比缩放同属降低数据量的处理。
"""

from __future__ import annotations

CHANNEL_MEDIA = "application/vnd.lab2shot.channel"
CHANNEL_MAGIC = b"L2C1"  # 标识数据的类型与版本：版本变更时旧页面可立即识别，不会按旧规则读取新字节
CHANNEL_HEAD = 16  # 头部字节数（像素数据从此处开始；16 是 4 的倍数，Float32Array 也可零拷贝映射）
CHANNEL_U8, CHANNEL_U16, CHANNEL_F16, CHANNEL_F32 = 0, 1, 2, 3
# 值的解释方式：两个整数档为「k 分之几」（8 位画面本身即 k/255），两个浮点档即值本身
CHANNEL_SCALE = {CHANNEL_U8: 255.0, CHANNEL_U16: 65535.0, CHANNEL_F16: 1.0, CHANNEL_F32: 1.0}
CHANNEL_DTYPE = {CHANNEL_U8: "<u1", CHANNEL_U16: "<u2", CHANNEL_F16: "<f2", CHANNEL_F32: "<f4"}
CHANNEL_TRIMMED = 1  # flags 第 0 位：尾数低位已置零（8 位档无可置零的位，因此为 0）
HALF_MANTISSA = 7  # 保留的尾数位数（半精度共 10 位）：2^-8 = 0.39%，即 8 位屏幕上的一级
# 尾数低位置零后仍可保留在半精度中的最大值：0x7BF8。半精度最大值为 65504（0x7BFF），但 _trimmed_half 先加半级再置零，
# 0x7BFC..0x7BFF 加上半级后会进位为 0x7C00 = 无穷，因此界限设在 0x7BF8 而非 65504。
HALF_TRIMMED_MAX = 65280.0


def _from_words(k, top: float):
    """将 k 分之几还原为浮点：乘以倒数而非相除。读图层（OpenImageIO）将 8 位、16 位像素转为浮点时使用的正是
    该公式（256 个 8 位值中，正确舍入的 k/255 有 126 个与之相差一位，乘以倒数的为 0 个），因此按它判断
    能否容纳并按它还原，8 位画面才能真正原样走 8 位档。浏览器端使用显卡的 R8 / R16 归一化纹理，
    硬件给出的是正确舍入的 k/255，与此处最多相差一个末位（6e-8），在显示和计算上均无影响。"""
    import numpy as np

    return k.astype(np.float32) * np.float32(1.0 / top)


def _fits(v, top: int, dtype: str):
    """每个值都恰好为 k/top（k 为整数）时返回这组 k，否则为 None：8 位画面读为浮点后仍是 k/255，
    原样按 8 位发送不损失任何位，且只需 1 个字节。view_data.py 的 colours() 对显示颜色做的是同样的处理。"""
    import numpy as np

    if not v.size:
        return None
    k = np.rint(v.astype(np.float64) * top)
    if k.min() < 0 or k.max() > top or not np.array_equal(_from_words(k.astype(dtype), top), v):
        return None
    return k.astype(dtype)


def _channel_format(v):
    """该通道发送时使用的档位（格式号，该档的数组）：能容纳时使用较小的档，否则使用较大的档。"""
    import numpy as np

    for top, code in ((255, CHANNEL_U8), (65535, CHANNEL_U16)):
        k = _fits(v, top, CHANNEL_DTYPE[code])
        if k is not None:
            return code, k
    half = v.astype("<f2")
    if np.array_equal(half.astype(np.float32), v, equal_nan=True):  # 本身即为半精度的 EXR：原样使用
        return CHANNEL_F16, half
    return CHANNEL_F32, np.ascontiguousarray(v, "<f4")


def _trimmed_half(v):
    """半精度，尾数低位四舍五入后置零（NaN 和无穷保持不变）。

    「四舍五入」即先加上被置零部分的一半再置零：直接截断会使所有值一律偏小，累积后形成系统性偏差。"""
    import numpy as np

    half = np.ascontiguousarray(v, np.float16)
    drop = 10 - HALF_MANTISSA
    bits = half.view(np.uint16).copy()
    fin = np.isfinite(half)
    bits[fin] = ((bits[fin].astype(np.uint32) + (1 << (drop - 1))) & ((0xFFFF << drop) & 0xFFFF)).astype(np.uint16)
    return bits.view("<f2")


def channel_blob(values) -> bytes:
    """单个通道的一帧（已缩放到代理档位，由 `view/proxy.py` 完成），编码为上述字节格式。
    `values`：[H, W] 的 float32（该帧该通道的值本身，未经任何显示处理）。

    头部（小端）：
        0  4  "L2C1"            通道数据，第 1 版
        4  1  格式              0 = u8(k/255)  1 = u16(k/65535)  2 = 半精度  3 = float32
        5  1  标志              第 0 位 = 尾数低位已置零
        6  2  保留（0）
        8  4  宽（像素）
       12  4  高（像素）
       16  …  宽 × 高 个值，行优先，从左上角开始

    8 位档不做尾数低位置零：它本身即为每个值一个字节，置零既不节省空间也没有意义。
    """
    import numpy as np

    v = np.ascontiguousarray(values, np.float32)
    if v.ndim != 2:
        raise ValueError(f"a channel is one plane of pixels, not {v.shape}")
    height, width = v.shape
    flat = v.reshape(-1)
    code, data = _channel_format(flat)
    trimmed = False
    if code != CHANNEL_U8:
        # 代理一律使用尾数低位置零的半精度，但半精度无法容纳的除外：有限值的绝对值超过 HALF_TRIMMED_MAX 时保留 float32 原值
        # （见开头「位数由数据本身决定」一条）。8 位档以外的整数档（k/65535）值在 0..1 之间，照常处理。
        # 原生半精度的数据同样需要检查：65280–65504 区间在尾数低位置零并四舍五入后会进位为 inf，与 float32 分支一样保留原值。
        finite = flat[np.isfinite(flat)]
        if not finite.size or float(np.abs(finite).max()) <= HALF_TRIMMED_MAX:
            code, data, trimmed = CHANNEL_F16, _trimmed_half(flat), True
    head = bytearray(CHANNEL_HEAD)
    head[0:4] = CHANNEL_MAGIC
    head[4] = code
    head[5] = CHANNEL_TRIMMED if trimmed else 0
    head[8:12] = int(width).to_bytes(4, "little")
    head[12:16] = int(height).to_bytes(4, "little")
    return bytes(head) + data.tobytes()


def read_channel_blob(blob: bytes):
    """channel_blob 的逆过程：(值 [H, W] float32, 编码方式)。浏览器做的是同样的处理
    （它无需转为 float32，直接将该数据映射为纹理）；本函数供自检和服务器自身使用。"""

    import numpy as np

    if len(blob) < CHANNEL_HEAD or blob[0:4] != CHANNEL_MAGIC:
        raise ValueError("not a Lab2Shot channel blob")
    code = blob[4]
    said = {"format": code, "trimmed": bool(blob[5] & CHANNEL_TRIMMED),
            "width": int.from_bytes(blob[8:12], "little"), "height": int.from_bytes(blob[12:16], "little")}
    data = np.frombuffer(blob, CHANNEL_DTYPE[code], offset=CHANNEL_HEAD)
    top = CHANNEL_SCALE[code]
    values = _from_words(data, top) if top != 1.0 else data.astype(np.float32)
    return values.reshape(said["height"], said["width"]), said
