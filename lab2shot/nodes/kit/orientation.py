"""方向场：灰度图中每个像素处纹理的走向，以及该判断的可信度。

使用一组 Gabor 滤波器（同一尺度，角度均匀覆盖 0–180°）与画面卷积，每个像素取响应最强的角度；
响应在各角度上越集中，该角度越可信。这是纹理方向估计的标准方法。

每个角度使用正交对（同一个核的偶相位和奇相位各一个，以 `hypot` 作为能量）。这是 Gabor 能量的标准定义，
且是必要的：只用偶核时，响应会随条纹相位周期性过零，过零处所有角度的响应都接近 0，
取最大值便失去意义；在发丝真值上，正交对的角度误差明显小于只用偶核。

输出的角度是纹理本身的走向（发丝或布纹的延伸方向），而非滤波器的参数角：
`cv2.getGaborKernel` 一类实现中的 θ 是条纹的法线方向，与走向相差 90°，直接输出会使下游整体旋转 90°。
本模块在 `orientation_field` 内部补回该差值，调用方得到的始终是走向。

方向不区分正负（一根头发朝左上和朝右下是同一方向），因此角度只在 0–180° 范围内，比较时一律使用
`angle_difference`，不得直接相减。

仅使用 numpy：卷积通过 FFT 完成（图像的 FFT 只做一次，每个角度只增加一次核的 FFT 和一次逆变换），
因此角度数加倍只使时间加倍，内存始终为 O(H·W)，不会展开 [H, W, 角度数] 的完整数组
（在 4K 分辨率下将达数十 GB）。
"""

from __future__ import annotations

import numpy as np

# 方向的两端表示同一方向：所有角度都在 [0, π) 内
HALF_TURN = np.pi


def gabor_kernel(size: int, sigma: float, theta: float, wavelength: float, gamma: float,
                 phase: float = 0.0) -> np.ndarray:
    """实数 Gabor 核 [size, size]，条纹法线朝向 `theta`，相位为 `phase`。

    `phase=0` 时与 OpenCV 的 `getGaborKernel(ksize, sigma, theta, lambd, gamma, psi=0)` 公式相同：
    沿法线方向的高斯宽度为 `sigma`，沿条纹方向为 `sigma / gamma`，
    条纹周期为 `wavelength` 像素。`phase=π/2` 为其奇相位对应核，两者构成正交对。
    """
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    cos, sin = np.cos(theta), np.sin(theta)
    across = x * cos + y * sin  # 垂直于条纹的方向
    along = -x * sin + y * cos  # 沿条纹的方向
    sigma_along = sigma / gamma
    envelope = np.exp(-0.5 * (across ** 2 / sigma ** 2 + along ** 2 / sigma_along ** 2))
    return envelope * np.cos(2 * np.pi * across / wavelength + phase)


class _Bank:
    """一组角度的 Gabor 滤波：图像的 FFT 只计算一次，逐个角度取得响应（内存 O(H·W)）。"""

    def __init__(self, image: np.ndarray, size: int, sigma: float, wavelength: float, gamma: float):
        self.shape = image.shape
        self.size, self.sigma, self.wavelength, self.gamma = size, sigma, wavelength, gamma
        self.pad = size // 2
        # 边缘先镜像填充一圈再做循环卷积：否则 FFT 会使画面左右两侧互相渗透
        padded = np.pad(np.asarray(image, np.float64), self.pad, mode="reflect")
        self.full = padded.shape
        self.spectrum = np.fft.rfft2(padded)

    def _filter(self, theta: float, phase: float) -> np.ndarray:
        kernel = np.zeros(self.full)
        kernel[:self.size, :self.size] = gabor_kernel(self.size, self.sigma, theta, self.wavelength, self.gamma, phase)
        # 将核的中心移到原点（循环卷积的相位零点），结果才不会整体平移；移动量为核的半径，而非整张图的一半
        kernel = np.roll(kernel, (-self.pad, -self.pad), axis=(0, 1))
        out = np.fft.irfft2(self.spectrum * np.fft.rfft2(kernel), self.full)
        # 再裁掉填充的一圈
        return out[self.pad:self.pad + self.shape[0], self.pad:self.pad + self.shape[1]]

    def response(self, theta: float) -> np.ndarray:
        """该角度上每个像素的 Gabor 能量：偶相位与奇相位平方和的平方根，与条纹所处相位无关。"""
        return np.hypot(self._filter(theta, 0.0), self._filter(theta, np.pi / 2))


def angle_difference(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """两个方向之间的夹角（弧度，0 到 π/2）。方向不区分正负，因此相差 179° 与相差 1° 等价。"""
    return HALF_TURN / 2 - np.abs(np.abs(np.asarray(a) - np.asarray(b)) - HALF_TURN / 2)


def orientation_field(gray: np.ndarray, *, size: int = 31, sigma: float = 2.0, wavelength: float = 3.0,
                      gamma: float = 0.5, angles: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """灰度图 [H, W] → (方向 [H, W]，可信度 [H, W])。

    方向是纹理的走向，单位为弧度，范围为 0 到 π（不含 π）。可信度为 0–1：响应越集中于单一角度越接近 1
    （按整张图中最集中的像素归一化，因此是该图内的相对可信度，而非绝对物理量）。
    `size` 为滤波核边长（像素），`wavelength` 为待检测的条纹周期（发丝粗细量级），
    `sigma` 为高斯包络宽度，`gamma` 为包络的长宽比，`angles` 为角度的档数。
    """
    gray = np.asarray(gray, np.float64)
    if gray.ndim != 2:
        raise ValueError(f"orientation_field wants one channel, got {gray.shape}")
    thetas = np.linspace(0.0, HALF_TURN, angles, endpoint=False)
    bank = _Bank(gray, size, sigma, wavelength, gamma)

    strongest = np.full(gray.shape, -np.inf)
    which = np.zeros(gray.shape, np.int32)
    total = np.zeros(gray.shape)
    for i, theta in enumerate(thetas):  # 第一遍：求最强角度及响应之和
        r = bank.response(theta)
        np.copyto(which, i, where=r > strongest)
        np.maximum(strongest, r, out=strongest)
        total += r
    normal = thetas[which]  # 滤波器的参数角（条纹的法线方向）

    spread = np.zeros(gray.shape)
    for i, theta in enumerate(thetas):  # 第二遍：按与最强角度的距离加权响应，计算其分散程度
        d = angle_difference(normal, thetas[i])
        spread += d * d * bank.response(theta)
    spread /= total + 1e-7

    confidence = np.ones(gray.shape, np.float32)
    spread_out = spread > 0
    if spread_out.any():
        sharp = 1.0 / spread[spread_out] ** 2
        confidence[spread_out] = (sharp / sharp.max()).astype(np.float32)
    # 参数角 → 纹理走向（见模块说明）
    direction = np.mod(normal + HALF_TURN / 2, HALF_TURN).astype(np.float32)
    return direction, confidence
