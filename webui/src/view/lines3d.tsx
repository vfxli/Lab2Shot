import { useEffect, useMemo } from "react";
import { ORDER } from "./drawOrder";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";

/** Lines on the 3D stage with a width in screen pixels (native WebGL lines are always one pixel wide). Segments are
 * given as point pairs [x0 y0 z0 x1 y1 z1, ...]. Geometry is created once per segment count; new positions (e.g. the
 * next frame of a skeleton) are written into the same buffer. When `colors` is given, it holds one colour per segment
 * end (same layout as `segments`) and the single `color` is ignored; this is how 三维曲线 keep their own colours. */

interface Props {
  segments: Float32Array;
  color: string;
  colors?: Float32Array | null; // one colour per segment end (0..1 RGB); null uses `color`
  width: number; // screen pixels
  opacity?: number;
  dashed?: boolean;
  dash?: number; // dash length in scene units; gaps have the same length
  overlay?: boolean; // draw on top of everything (e.g. bones over the body)
  renderOrder?: number;
}

export function FatLines({ segments, color, colors = null, width, opacity = 1, dashed = false, dash = 10, overlay = false, renderOrder }: Props) {
  const invalidate = useThree((s) => s.invalidate);
  const count = segments.length / 6;
  const painted = !!colors;
  const line = useMemo(() => {
    const g = new LineSegmentsGeometry();
    g.setPositions(new Float32Array(count * 6));
    if (painted) g.setColors(new Float32Array(count * 6));
    const l = new LineSegments2(g, new LineMaterial({ worldUnits: false, vertexColors: painted }));
    l.frustumCulled = false;
    return l;
  }, [count, painted]);
  useEffect(
    () => () => {
      line.geometry.dispose();
      line.material.dispose();
    },
    [line],
  );
  useEffect(() => {
    const start = line.geometry.attributes.instanceStart as THREE.InterleavedBufferAttribute;
    (start.data.array as Float32Array).set(segments);
    start.data.needsUpdate = true;
    if (dashed) line.computeLineDistances();
    invalidate();
  }, [line, segments, dashed, invalidate]);
  useEffect(() => {
    if (!colors) return;
    const start = line.geometry.attributes.instanceColorStart as THREE.InterleavedBufferAttribute;
    (start.data.array as Float32Array).set(colors);
    start.data.needsUpdate = true;
    invalidate();
  }, [line, colors, invalidate]);
  useEffect(() => {
    const m = line.material;
    // The shader multiplies vertex colours by the material colour. With per-segment colours the material colour must
    // stay at its default white, otherwise the data colours would be tinted; this implements "`color` is ignored".
    if (!painted) m.color.set(color);
    m.linewidth = width;
    m.opacity = opacity;
    m.transparent = opacity < 1 || overlay;
    m.dashed = dashed;
    m.dashSize = dash;
    m.gapSize = dash;
    m.depthTest = !overlay;
    m.depthWrite = !overlay;
    m.needsUpdate = true;
    line.renderOrder = renderOrder ?? (overlay ? ORDER.lines : 0);
    invalidate();
  }, [line, painted, color, width, opacity, dashed, dash, overlay, renderOrder, invalidate]);
  return <primitive object={line} />;
}

/** Converts polylines (ordered points) into segments: a single polyline, or several consecutive ones when `counts`
 * gives the point count of each (a set of 三维曲线). Accepts any three-component per-point values, i.e. positions or
 * colours for `FatLines`. */
export function polylineSegments(points: ArrayLike<number>, counts?: ArrayLike<number> | null): Float32Array {
  const lines = counts ?? [points.length / 3];
  let segments = 0;
  for (let k = 0; k < lines.length; k++) segments += Math.max(0, lines[k] - 1);
  const out = new Float32Array(segments * 6);
  let at = 0; // index of the first point of the current polyline
  let s = 0; // index of the segment being written
  for (let k = 0; k < lines.length; k++) {
    for (let i = 0; i + 1 < lines[k]; i++, s++)
      for (let c = 0; c < 3; c++) {
        out[s * 6 + c] = points[(at + i) * 3 + c];
        out[s * 6 + 3 + c] = points[(at + i + 1) * 3 + c];
      }
    at += lines[k];
  }
  return out;
}

/** The twelve edges of a box. */
export function boxSegments(box: THREE.Box3): Float32Array {
  const { min: a, max: b } = box;
  const c = [
    [a.x, a.y, a.z], [b.x, a.y, a.z], [b.x, b.y, a.z], [a.x, b.y, a.z],
    [a.x, a.y, b.z], [b.x, a.y, b.z], [b.x, b.y, b.z], [a.x, b.y, b.z],
  ];
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  return new Float32Array(edges.flatMap(([i, j]) => [...c[i], ...c[j]]));
}
