import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { matrixAt } from "./matrix3d";
import { msg } from "../messages/message";
import { heldSample, sampleAt} from "../model/viewFormat";
import { useViewerNote } from "../state/viewer";
import { boxSegments, FatLines, polylineSegments } from "./lines3d";
import { project, usePickable, type PickRay } from "./stageState";
import type { CurveData, CurveSample } from "./sceneTypes";
import type { ViewOptions } from "../model/viewOptions";

/** 三维曲线 (hair, guide curves, motion trails) on the 3D stage. Each curve is drawn as a polyline through its points
 * in order, using the shared line width (view option 线宽, in screen pixels), in either per-vertex colours or a single
 * colour (曲线着色 / 曲线单色).
 *
 * The display budget is the total number of points. A curve set within the budget is drawn in full, every curve and
 * every point. A set over the budget is drawn as its bounding box only, and the viewer reports the reason together
 * with the curve and point counts. Curves are never thinned (所见即所得); data, cook and delivery are unaffected.
 * Budgets may only decrease. */

/** Maximum number of points of one curve set drawn by the 3D view. */
const CURVE_POINTS_MAX = 500_000;

interface Props {
  src: CurveData;
  frame: number;
  o: ViewOptions;
  pickKey: string;
  version: number;
}

/** Per-segment-end colours of a sample in the layout expected by FatLines, or null when a single colour is used. */
function segmentColours(sample: CurveSample, o: ViewOptions): Float32Array | null {
  if (o.curveColor === "constant") return null;
  const c = sample.colors;
  const top = c instanceof Uint8Array ? 255 : c instanceof Uint16Array ? 65535 : 0;
  const rgb = top ? Float32Array.from(c, (v) => v / top) : (c as Float32Array);
  return polylineSegments(rgb, sample.vertexCounts);
}

export function Curves({ src, frame, o, pickKey }: Props) {
  const say = useViewerNote((s) => s.say);
  const i = sampleAt(src.frames, frame);
  // If this frame has not arrived, draw the nearest available sample instead of nothing (same rule as point clouds and point caches: viewFormat.ts heldSample).
  const sample = src.still ?? heldSample(src.samples, i)?.sample ?? null;
  const over = !!sample && sample.count > CURVE_POINTS_MAX;
  const matrix = useMemo(() => matrixAt(src.world, i), [src.world, i]);

  const segments = useMemo(() => (sample && !over ? polylineSegments(sample.points, sample.vertexCounts) : null), [sample, over]);
  const colours = useMemo(() => (sample && segments ? segmentColours(sample, o) : null), [sample, segments, o.curveColor]); // eslint-disable-line react-hooks/exhaustive-deps
  // Over budget: draw the bounding box so the set keeps its place in the scene and can still be framed and picked.
  const box = useMemo(() => (sample && over ? new THREE.Box3().setFromArray(sample.points as unknown as number[]) : null), [sample, over]);

  const pickable = useMemo(() => {
    const local = box ?? (segments?.length ? new THREE.Box3().setFromArray(segments as unknown as number[]) : null);
    return {
      label: src.name,
      bounds: () => (local ? local.clone().applyMatrix4(matrix) : null),
      hit: (p: PickRay) => {
        if (!segments) return null;
        let best: number | null = null;
        const v = new THREE.Vector3();
        for (let k = 0; k < segments.length; k += 3) {
          const s = project(v.set(segments[k], segments[k + 1], segments[k + 2]).applyMatrix4(matrix), p);
          if (s && Math.hypot(s.x - p.px.x, s.y - p.px.y) < 8 && (best === null || s.depth < best)) best = s.depth;
        }
        return best;
      },
    };
  }, [segments, box, matrix, src.name]);
  usePickable(pickKey, pickable);

  // Reported once per curve set rather than per frame; the viewer note keeps only the latest message per code.
  useEffect(() => {
    if (!over || !sample) return;
    say(msg("N-VIEW-CURVESTOOMANY"), msg("I-VIEW-CURVESTOOMANYWHY", { name: src.name, strands: sample.strands, points: sample.count, limit: CURVE_POINTS_MAX }));
  }, [over, sample, say, src.name]);

  if (!sample) return null;
  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      {box ? (
        <FatLines segments={boxSegments(box)} color={o.curveTint} width={Math.max(1, o.lineWidth)} opacity={0.9} dashed />
      ) : (
        segments &&
        segments.length > 0 && <FatLines segments={segments} colors={colours} color={o.curveTint} width={o.lineWidth} />
      )}
    </group>
  );
}
