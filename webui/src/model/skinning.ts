/** 线性混合蒙皮（LBS）的 CPU 算法：每个点任意多个关节影响。显卡上的蒙皮（three 的 SkinnedMesh）每点只收四个，多于四个
 * 影响的网格平时用服务器逐帧求好的点（view/elements3d.tsx EvaluatedBody）；只摆一个姿势（双骨架编辑、骨架姿势的蒙皮
 * 预览，view/stageLayers.tsx SkinnedPose）时服务器没有那个姿势可求，就在这里按全部影响算一次。
 *
 *   点' = Σ_k w_k · (关节世界_j · 绑定_j⁻¹) · 点（绑定姿势的点在世界里，与 UsdSkel 相同）
 *
 * 纯模块，矩阵布局写明在参数上；不引 three，Node 可直接跑（model/rigPair.test.ts）。 */

/** 每个关节的蒙皮矩阵（行主序 3x4，[J*12]）：`world` 行主序 3x4 [J*12]（这一姿势的关节世界），`bind` 行主序 4x4 [J*16]
 * （绑定姿势的关节世界）。绑定矩阵不可逆（缩放为 0）的关节按单位逆处理（同 SkinnedBody）。 */
export function skinMatrices(world: ArrayLike<number>, bind: ArrayLike<number>, joints: number): Float32Array {
  const out = new Float32Array(joints * 12);
  for (let j = 0; j < joints; j++) {
    const b = Array.from({ length: 12 }, (_, k) => bind[j * 16 + k]);
    const inv = invertAffine(b) ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 4; c++) {
        let v = 0;
        for (let k = 0; k < 3; k++) v += world[j * 12 + r * 4 + k] * inv[k * 4 + c];
        if (c === 3) v += world[j * 12 + r * 4 + 3];
        out[j * 12 + r * 4 + c] = v;
      }
    }
  }
  return out;
}

/** 行主序 3x4 仿射的逆（同样布局）；不可逆为 null。 */
export function invertAffine(m: readonly number[]): number[] | null {
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
  const r = [
    (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
    (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
    (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ];
  const t = [m[3], m[7], m[11]];
  return [
    r[0], r[1], r[2], -(r[0] * t[0] + r[1] * t[1] + r[2] * t[2]),
    r[3], r[4], r[5], -(r[3] * t[0] + r[4] * t[1] + r[5] * t[2]),
    r[6], r[7], r[8], -(r[6] * t[0] + r[7] * t[1] + r[8] * t[2]),
  ];
}

/** 蒙皮后的点：`points` [V*3]，`indices` / `weights` 每点 `k` 个（[V*k]），`skin` 为 skinMatrices 的结果。权重按原样
 * 用（服务端给的已归一）；全为 0 的点留在原处。 */
export function skinPoints(points: ArrayLike<number>, indices: ArrayLike<number>, weights: ArrayLike<number>, k: number, skin: ArrayLike<number>): Float32Array {
  const n = Math.floor(points.length / 3);
  const out = new Float32Array(n * 3);
  for (let v = 0; v < n; v++) {
    const x = points[v * 3], y = points[v * 3 + 1], z = points[v * 3 + 2];
    let ox = 0, oy = 0, oz = 0, total = 0;
    for (let q = 0; q < k; q++) {
      const w = weights[v * k + q];
      if (!w) continue;
      const m = indices[v * k + q] * 12;
      ox += w * (skin[m] * x + skin[m + 1] * y + skin[m + 2] * z + skin[m + 3]);
      oy += w * (skin[m + 4] * x + skin[m + 5] * y + skin[m + 6] * z + skin[m + 7]);
      oz += w * (skin[m + 8] * x + skin[m + 9] * y + skin[m + 10] * z + skin[m + 11]);
      total += w;
    }
    if (total) out.set([ox, oy, oz], v * 3);
    else out.set([x, y, z], v * 3);
  }
  return out;
}
