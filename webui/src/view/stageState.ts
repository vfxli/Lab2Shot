import { createContext, useContext, useEffect } from "react";
import * as THREE from "three";

/** What the parts of one 3D stage share outside React's state, read every frame: what can be picked and framed. */

export interface PickRay {
  raycaster: THREE.Raycaster;
  camera: THREE.Camera;
  px: { x: number; y: number }; // the click in the canvas, CSS pixels
  size: { width: number; height: number };
}

export interface Pickable {
  label: string;
  bounds: () => THREE.Box3 | null; // in the world, on the current frame
  hit: (p: PickRay) => number | null; // how far along the view the click hits it (null: missed)
}

export class StageState {
  pickables = new Map<string, Pickable>();
}

export const StageContext = createContext<StageState>(new StageState());
export const useStage = () => useContext(StageContext);

/** Registers something that can be picked and framed while it is shown. */
export function usePickable(key: string, p: Pickable | null): void {
  const stage = useStage();
  useEffect(() => {
    if (!p) return;
    stage.pickables.set(key, p);
    return () => {
      if (stage.pickables.get(key) === p) stage.pickables.delete(key);
    };
  }, [stage, key, p]);
}

/** Screen position (CSS pixels) of a world point, and its distance along the view; null behind the camera. */
export function project(v: THREE.Vector3, p: PickRay): { x: number; y: number; depth: number } | null {
  const q = v.clone().applyMatrix4(p.camera.matrixWorldInverse);
  const depth = -q.z;
  q.applyMatrix4(p.camera.projectionMatrix);
  if (!(p.camera as THREE.OrthographicCamera).isOrthographicCamera && depth <= 0) return null;
  return { x: ((q.x + 1) / 2) * p.size.width, y: ((1 - q.y) / 2) * p.size.height, depth };
}
