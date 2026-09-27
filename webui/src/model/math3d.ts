/** The 3D viewer's maths that needs no three.js: bounds and framing.
 *
 * Pure: no imports, so node's own test runner reads it as it is. */

export interface Bounds {
  min: [number, number, number];
  max: [number, number, number];
}

/** Bounds of one frame's points, from at most `sample` of them (the spread order makes the first ones a fair
 * sample). null when there are none. */
export function spanBounds(data: Float32Array, start: number, count: number, stride: number, sample = 200_000): Bounds | null {
  const n = Math.min(count, sample);
  if (n <= 0) return null;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const p = (start + i) * stride;
    for (let c = 0; c < 3; c++) {
      const v = data[p + c];
      if (v < min[c]) min[c] = v;
      if (v > max[c]) max[c] = v;
    }
  }
  return { min, max };
}

/** How far a perspective camera stands from a sphere's centre to see it whole with a margin (fov in degrees,
 * vertical; aspect = width / height). */
export function fitDistance(radius: number, fovDeg: number, aspect: number, margin = 1.15): number {
  const half = (fovDeg * Math.PI) / 360;
  const halfX = Math.atan(Math.tan(half) * aspect);
  return (Math.max(radius, 1e-6) * margin) / Math.sin(Math.min(half, halfX));
}

/** The orthographic zoom (pixels per scene unit) that shows a sphere whole with a margin in a view of w x h pixels. */
export const fitZoom = (radius: number, w: number, h: number, margin = 1.15) => Math.min(w, h) / (2 * Math.max(radius, 1e-6) * margin);

/** Counts written short: 1234 -> "1234", 45_600 -> "4.6万", 2_300_000 -> "230万". */
export function shortCount(n: number): string {
  if (n < 10_000) return String(n);
  if (n < 100_000_000) return `${Number((n / 10_000).toFixed(n < 100_000 ? 1 : 0))}万`;
  return `${Number((n / 100_000_000).toFixed(1))}亿`;
}
