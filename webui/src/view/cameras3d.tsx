/** Scene cameras in the 3D view: per-frame camera state, body and frustum, and camera path. */

import { useMemo } from "react";
import * as THREE from "three";
import { FatLines, polylineSegments } from "./lines3d";
import type { ViewOptions } from "../model/viewOptions";
import { matrixAt } from "./matrix3d";
import { type CameraData } from "./sceneData";
import { project, usePickable, type PickRay } from "./stageState";
import { sampleAt } from "../model/viewFormat";

/** Camera state at a frame: camera-to-world matrix and the half extents of the resolution gate at depth 1. */
export function cameraAt(cam: CameraData, frame: number): { matrix: THREE.Matrix4; focalMm: number; tanX: number; tanY: number } {
  const i = sampleAt(cam.ref.frames, frame);
  const at = (a: Float32Array) => a[Math.min(i, a.length - 1)];
  const focalMm = at(cam.focalMm);
  const tanX = at(cam.hAperture) / 2 / focalMm;
  return { matrix: matrixAt(cam.world, i), focalMm, tanX, tanY: tanX / (cam.ref.width / cam.ref.height) };
}

/** Shot camera frustum at its pose on the given frame, looking down -Z (USD / GL convention). */
export function ShotCamera({ cam, frame, o, pickKey }: { cam: CameraData; frame: number; o: ViewOptions; pickKey: string }) {
  const { matrix, tanX, tanY } = cameraAt(cam, frame);
  const segments = useMemo(() => {
    const depth = 70;
    const hw = depth * tanX;
    const hh = depth * tanY;
    const c = [[-hw, -hh, -depth], [hw, -hh, -depth], [hw, hh, -depth], [-hw, hh, -depth]];
    const pts: number[] = [];
    c.forEach((p, i) => pts.push(0, 0, 0, ...p, ...p, ...c[(i + 1) % 4]));
    pts.push(-hw * 0.3, hh * 1.08, -depth, 0, hh * 1.35, -depth, 0, hh * 1.35, -depth, hw * 0.3, hh * 1.08, -depth);
    return new Float32Array(pts);
  }, [tanX, tanY]);
  const key = matrix.elements.join();
  const pickable = useMemo(
    () => ({
      label: cam.ref.path.split("/").pop() || "相机",
      bounds: () => new THREE.Box3().setFromArray(segments).applyMatrix4(matrix),
      hit: (p: PickRay) => {
        const s = project(new THREE.Vector3().setFromMatrixPosition(matrix), p);
        return s && Math.hypot(s.x - p.px.x, s.y - p.px.y) < 14 ? s.depth : null;
      },
    }),
    [cam.ref.path, segments, key], // eslint-disable-line react-hooks/exhaustive-deps
  );
  usePickable(pickKey, pickable);
  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      <FatLines segments={segments} color={o.cameraColor} width={o.lineWidth} />
      <mesh position={[0, 0, 6]}>
        <boxGeometry args={[10, 7, 12]} />
        <meshStandardMaterial color="#8a8a90" roughness={0.35} metalness={0.2} />
      </mesh>
    </group>
  );
}

/** Path of the shot camera over the whole shot, drawn as a solid line. */
export function CameraPath({ cam, o }: { cam: CameraData; o: ViewOptions }) {
  const segments = useMemo(() => {
    const pts: number[] = [];
    const n = cam.world.length / 16;
    for (let i = 0; i < n; i++) {
      const m = matrixAt(cam.world, i);
      pts.push(m.elements[12], m.elements[13], m.elements[14]);
    }
    return polylineSegments(pts);
  }, [cam]);
  if (!segments.length) return null;
  return <FatLines segments={segments} color={o.cameraColor} width={o.lineWidth} opacity={0.55} />;
}

