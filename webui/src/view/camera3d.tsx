import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import { fitDistance, fitZoom } from "../model/math3d";
import type { ViewOptions } from "../model/viewOptions";
import { useStage, type Pickable } from "./stageState";
import { useViewCamera, type ViewName } from "../state/viewer";
import { useShallow } from "zustand/react/shallow";

/** The 3D stage's camera: perspective or orthographic, the free view or looking straight down an axis (顶 / 前 / 侧, as
 * Houdini's orthographic views, where the left button pans), orbit mouse navigation (tumble left, dolly middle, pan
 * right; the only navigation scheme), framing everything (H) or the selection (F), and the clipping planes; or, when
 * looking through a scene camera, that camera. */

const FOV = 38;
const THREE_QUARTER = new THREE.Vector3(330, 170, 300).normalize();
const DIRECTION: Record<Exclude<ViewName, "persp">, THREE.Vector3> = {
  top: new THREE.Vector3(0, 1, 1e-4).normalize(),
  front: new THREE.Vector3(0, 0, 1),
  side: new THREE.Vector3(1, 0, 0),
};

/** The viewer's 3D camera, kept across displayed nodes and across redraws of the stage (as in Houdini's viewport, the
 * data shown changes but the view does not). Null until the first 3D display, which frames its content; after that only
 * 框显全部 (H) and 框显选中 (F) reframe. */
interface KeptCamera {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  up: THREE.Vector3;
  fov: number;
  pivot: THREE.Vector3;
  ortho: { position: THREE.Vector3; quaternion: THREE.Quaternion; zoom: number };
  free: { position: THREE.Vector3; pivot: THREE.Vector3 } | null; // the free view to return to after looking through a camera
  looking: boolean; // exited while looking through a scene camera (in the 3D stage)
}
let kept: KeptCamera | null = null;
let framedOnce = false; // the session's first 3D display has been framed

/** The bounds of all visible objects on this frame (or of the selected one only). */
export function boundsOf(pickables: Iterable<Pickable>): THREE.Box3 | null {
  const box = new THREE.Box3();
  for (const p of pickables) {
    const b = p.bounds();
    if (b && !b.isEmpty() && Number.isFinite(b.min.x) && Number.isFinite(b.max.x)) box.union(b);
  }
  return box.isEmpty() ? null : box;
}

/** A scene camera seen through, on this frame: its camera-to-world, the vertical field of view of its resolution
 * gate (degrees), the gate's width / height. */
export interface Lens {
  pose: THREE.Matrix4;
  fovV: number;
  aspect: number;
}

/** Where the camera's gate sits on the canvas, CSS pixels. */
export interface Gate {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Props {
  o: ViewOptions;
  frameKey: string; // the displayed content: reframed when it changes (another node)
  selected: string | null;
  lens: Lens | null; // looking through a scene camera: the view is that camera's view, with its gate at `gate`
  gate: Gate | null;
  onLeave: () => void; // tumbling while looking through a camera: the view continues as 透视 from the same position
}

export function ViewCamera({ o, frameKey, selected, lens, gate, onLeave }: Props) {
  const stage = useStage();
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  const setThree = useThree((s) => s.set);
  const { view, ortho, frameAsk, viewAsk } = useViewCamera(useShallow((s) => ({ view: s.view, ortho: s.ortho, frameAsk: s.frameAsk, viewAsk: s.viewAsk })));
  const persp = useMemo(() => {
    const c = new THREE.PerspectiveCamera(FOV, 1, 1, 1e6);
    if (kept) {
      c.position.copy(kept.position);
      c.quaternion.copy(kept.quaternion);
      c.up.copy(kept.up);
      c.fov = kept.fov;
    } else c.position.set(330, 260, 50);
    return Object.assign(c, { manual: true }); // its aspect is set here (a scene camera's gate is not the canvas)
  }, []);
  const orthoCam = useMemo(() => {
    const c = Object.assign(new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 1e6), { manual: true });
    if (kept) {
      c.position.copy(kept.ortho.position);
      c.quaternion.copy(kept.ortho.quaternion);
      c.zoom = kept.ortho.zoom;
    }
    return c;
  }, []);
  const active = ortho && !lens ? orthoCam : persp;
  const controls = useRef<OrbitControlsImpl>(null);
  const pivot = useRef(kept ? kept.pivot.clone() : new THREE.Vector3(0, 90, -250));
  const freePose = useRef<{ position: THREE.Vector3; pivot: THREE.Vector3 } | null>(kept?.free ? { position: kept.free.position.clone(), pivot: kept.free.pivot.clone() } : null);

  // the frustums follow the canvas and the clipping planes; when looking through a camera, its pose and lens on this
  // frame, with its gate where the stage draws it (the canvas shows the gate and its surroundings: an off-centre window of its view)
  const wasLooking = useRef(kept?.looking ?? false);
  const justLeft = useRef(false); // exited a camera in this very change: its pose is not the free view to keep
  /** Adopts the current camera as the viewer's. */
  const keep = () => {
    kept = {
      position: persp.position.clone(),
      quaternion: persp.quaternion.clone(),
      up: persp.up.clone(),
      fov: persp.fov,
      pivot: pivot.current.clone(),
      ortho: { position: orthoCam.position.clone(), quaternion: orthoCam.quaternion.clone(), zoom: orthoCam.zoom },
      free: freePose.current ? { position: freePose.current.position.clone(), pivot: freePose.current.pivot.clone() } : null,
      looking: wasLooking.current,
    };
  };
  const keepRef = useRef(keep);
  keepRef.current = keep;
  useEffect(() => () => keepRef.current(), []);
  useEffect(() => {
    // for the browser walks (tests/ui/ui_smoke.py): the viewer camera's world matrix and lens, to verify that switching
    // nodes leaves it unchanged
    (window as unknown as { lab2shotCamera?: () => number[] }).lab2shotCamera = () => [...active.matrixWorld.toArray(), active === orthoCam ? orthoCam.zoom : persp.fov];
  }, [active, orthoCam, persp]);
  useLayoutEffect(() => {
    if (lens && gate) {
      if (!wasLooking.current) freePose.current = { position: persp.position.clone(), pivot: pivot.current.clone() }; // to return to later
      const q = new THREE.Quaternion();
      lens.pose.decompose(persp.position, q, new THREE.Vector3());
      persp.quaternion.copy(q);
      persp.up.set(0, 1, 0).applyQuaternion(q); // its own up vector: a rolled camera stays rolled
      persp.fov = lens.fovV;
      persp.aspect = lens.aspect;
      persp.setViewOffset(gate.w, gate.h, -gate.x, -gate.y, size.width, size.height);
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
      const reach = Math.max(persp.position.distanceTo(pivot.current), 50);
      pivot.current.copy(persp.position).addScaledVector(forward, reach);
      controls.current?.target.copy(pivot.current);
      wasLooking.current = true;
    } else {
      if (wasLooking.current) {
        // exited the camera: the same view without its gate (the whole canvas at the gate's scale)
        const h = gate?.h ?? size.height;
        persp.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(persp.fov) / 2) * (size.height / Math.max(h, 1))));
        persp.up.set(0, 1, 0);
        wasLooking.current = false;
        justLeft.current = true;
      }
      persp.clearViewOffset();
      persp.aspect = size.width / Math.max(size.height, 1);
    }
    Object.assign(orthoCam, { left: -size.width / 2, right: size.width / 2, top: size.height / 2, bottom: -size.height / 2 });
    for (const c of [persp, orthoCam]) {
      c.near = o.near;
      c.far = o.far;
      c.updateProjectionMatrix();
    }
    persp.updateMatrixWorld();
    invalidate();
  }, [persp, orthoCam, lens, gate, size.width, size.height, o.near, o.far, invalidate]);

  // switching projection preserves the view: the same direction and the same size at the pivot
  const previous = useRef(active);
  useLayoutEffect(() => {
    const from = previous.current;
    previous.current = active;
    const tanHalf = Math.tan((FOV * Math.PI) / 360);
    if (from !== active) {
      const dir = from.position.clone().sub(pivot.current).normalize();
      if (active === orthoCam) {
        const dist = from.position.distanceTo(pivot.current);
        orthoCam.zoom = size.height / (2 * dist * tanHalf);
        orthoCam.position.copy(from.position);
      } else {
        const dist = size.height / (2 * orthoCam.zoom * tanHalf);
        persp.position.copy(pivot.current).addScaledVector(dir, dist);
      }
      active.up.set(0, 1, 0);
      active.lookAt(pivot.current);
      active.updateProjectionMatrix();
    }
    setThree({ camera: active });
    invalidate();
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  const place = (center: THREE.Vector3, radius: number, dir: THREE.Vector3) => {
    pivot.current.copy(center);
    const c = controls.current;
    if (active === persp) {
      persp.position.copy(center).addScaledVector(dir, fitDistance(radius, FOV, size.width / Math.max(size.height, 1)));
    } else {
      orthoCam.position.copy(center).addScaledVector(dir, Math.max(radius * 4, 100));
      orthoCam.zoom = fitZoom(radius, size.width, size.height);
    }
    active.up.set(0, 1, 0);
    active.lookAt(center);
    active.updateProjectionMatrix();
    if (c) {
      c.target.copy(center);
      c.update();
    }
    keep();
    invalidate();
  };

  const frame = (what: "all" | "selected") => {
    const pick = what === "selected" && selected ? stage.pickables.get(selected) : undefined;
    const box = boundsOf(pick ? [pick] : stage.pickables.values());
    if (!box) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const dir = active.position.clone().sub(pivot.current);
    place(sphere.center, Math.max(sphere.radius, 5), dir.lengthSq() > 1e-9 ? dir.normalize() : THREE_QUARTER.clone());
  };

  // framed only on the session's first 3D display, from three quarters above (switching nodes or redrawing the stage
  // never moves the viewer's camera); otherwise requested by H / F or the buttons. Objects register as they load (a
  // large scene may take a while, each result as it arrives): framing happens once registration has paused briefly,
  // unless the user has moved the view
  const touched = useRef(false);
  const lensRef = useRef(lens);
  lensRef.current = lens;
  useEffect(() => {
    touched.current = false;
    if (framedOnce || kept) return;
    let n = 0;
    let seen = -1;
    let still = 0;
    const id = window.setInterval(() => {
      const size = stage.pickables.size;
      still = size === seen ? still + 1 : 0;
      seen = size;
      // 透过相机查看时视角属于该场景相机，不得自动框显（否则切换显隐会使视角跳回透视）；
      // 用户已操作视图或等待超时后，也不再框显
      if (touched.current || lensRef.current || ++n > 1500) return window.clearInterval(id);
      if (!size || still !== 3) return;
      const box = boundsOf(stage.pickables.values());
      if (box) {
        const s = box.getBoundingSphere(new THREE.Sphere());
        place(s.center, Math.max(s.radius, 5), view === "persp" ? THREE_QUARTER.clone() : DIRECTION[view].clone());
        framedOnce = true;
        // 框显一次后即停止：若轮询持续运行，之后切换某类数据的显隐会使登记数量变化并再次稳定，
        // 从而再次框显，透过相机查看的视角会被重置为透视
        window.clearInterval(id);
      }
    }, 100);
    return () => window.clearInterval(id);
  }, [frameKey]); // eslint-disable-line react-hooks/exhaustive-deps
  // requests made while this stage is shown (a request made before it was drawn is not replayed: that would move the camera)
  const seenFrame = useRef(frameAsk.n);
  useEffect(() => {
    if (frameAsk.n === seenFrame.current) return;
    seenFrame.current = frameAsk.n;
    frame(frameAsk.what);
  }, [frameAsk]); // eslint-disable-line react-hooks/exhaustive-deps

  // a requested view: looking down an axis at the current content, or back to the free view as it was left
  const lastView = useRef<ViewName>(view);
  const seenView = useRef(viewAsk);
  useEffect(() => {
    if (viewAsk === seenView.current) return;
    seenView.current = viewAsk;
    if (lastView.current === "persp" && view !== "persp" && !justLeft.current) freePose.current = { position: persp.position.clone(), pivot: pivot.current.clone() };
    lastView.current = view;
    const box = boundsOf(stage.pickables.values());
    const sphere = box ? box.getBoundingSphere(new THREE.Sphere()) : new THREE.Sphere(pivot.current.clone(), 200);
    if (view === "persp") {
      persp.fov = FOV; // restore the free view's own lens (after a camera's)
      persp.updateProjectionMatrix();
      const back = freePose.current;
      if (back) {
        pivot.current.copy(back.pivot);
        persp.position.copy(back.position);
        persp.up.set(0, 1, 0);
        persp.lookAt(back.pivot);
        controls.current?.target.copy(back.pivot);
        controls.current?.update();
        invalidate();
      } else place(sphere.center, Math.max(sphere.radius, 5), THREE_QUARTER.clone());
    } else {
      place(sphere.center, Math.max(sphere.radius, 5), DIRECTION[view].clone());
    }
  }, [viewAsk]); // eslint-disable-line react-hooks/exhaustive-deps

  // a new controls object for each camera: it starts at the pivot
  useEffect(() => {
    const c = controls.current;
    if (!c) return;
    c.target.copy(pivot.current);
    c.update();
  }, [active]);

  useEffect(() => {
    justLeft.current = false; // after this change's effects have run
  });

  const locked = view !== "persp" && !lens; // straight down an axis: the left button pans, as in Houdini
  const buttons = {
    LEFT: locked ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE,
    // while looking through a camera, the middle button and the wheel move the canvas (view2dState.ts), not the camera
    MIDDLE: lens ? (-1 as THREE.MOUSE) : THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      camera={active}
      enableDamping
      dampingFactor={0.12}
      enableRotate={!locked}
      enableZoom={!lens}
      mouseButtons={buttons}
      onStart={() => {
        touched.current = true; // the view now belongs to the user: nothing arriving later reframes it
        if (lens) onLeave(); // tumbling out of a camera: 透视 from its position (as in Houdini; the camera is data and is never moved)
      }}
      onChange={() => {
        if (controls.current) pivot.current.copy(controls.current.target);
        keep();
      }}
    />
  );
}

/** A click (not a drag) selects the nearest object under the pointer; a click on empty space clears the selection. */
export function Picker({ onPick }: { onPick: (key: string | null) => void }) {
  const stage = useStage();
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  useEffect(() => {
    const el = gl.domElement;
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => {
      down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
    };
    const onUp = (e: PointerEvent) => {
      if (!down || e.button !== 0 || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
      down = null;
      const r = el.getBoundingClientRect();
      const px = { x: e.clientX - r.left, y: e.clientY - r.top };
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2((px.x / r.width) * 2 - 1, -(px.y / r.height) * 2 + 1), camera);
      let best: { key: string; at: number } | null = null;
      for (const [key, p] of stage.pickables) {
        const at = p.hit({ raycaster, camera, px, size: { width: r.width, height: r.height } });
        if (at !== null && (!best || at < best.at)) best = { key, at };
      }
      onPick(best?.key ?? null);
    };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointerup", onUp);
    return () => {
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointerup", onUp);
    };
  }, [gl, camera, size, stage, onPick]);
  return null;
}
