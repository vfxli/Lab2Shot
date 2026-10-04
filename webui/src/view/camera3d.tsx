import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import { fitDistance, fitZoom } from "../model/math3d";
import type { ViewOptions } from "../model/viewOptions";
import { endSlot, onSlotEnd, picked, useStage, type Pickable } from "./stageState";
import { slotCamera, useViewCamera, VIEWER_SLOT, type ViewName } from "../state/viewer";
import { followDrag, followPress } from "../platform/drag";
import { useShallow } from "zustand/react/shallow";

/** The 3D stage's camera: perspective or orthographic, the free view or looking straight down an axis (顶 / 前 / 侧, as
 * Houdini's orthographic views, where the left button pans), Houdini's mouse navigation (tumble left, pan middle, dolly
 * right: a drag to the right comes closer, to the left moves away; the only navigation scheme), framing everything (H)
 * or the selection (F), and the clipping planes; or, when looking through a scene camera, that camera. */

const FOV = 38;
/** Right-drag dolly: the distance to the pivot (or the orthographic zoom) changes by e^(-pixels × this), so equal drags
 * change it by equal ratios wherever the camera is; 200 px to the right brings it to about a third of the distance. */
const DOLLY_PER_PX = 0.0055;
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
// per camera slot: the viewer's one ("viewer"), and any other 3D stage on the page with a camera of its own (a stage in
// a dialog), so that one never moves the other's view
const keptBy = new Map<string, KeptCamera>();
const framedOnce = new Set<string>(); // the slots whose first 3D display has been framed

// a stage slot that ends (a dialog stage gone, view/stageState.ts endSlot): its kept camera, its one framing and its
// choices go, so the next stage in that slot frames its own content instead of looking where the last one looked
onSlotEnd((slot) => {
  keptBy.delete(slot);
  framedOnce.delete(slot);
  useViewCamera.getState().forget(slot);
});

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
  /** The lens centre off the picture's centre as a part of the picture's width / height, +x right +y up (cameras3d.tsx
   * cameraAt): the picture sits off the lens axis by the opposite amount, so the frustum's window and the backdrop move
   * together by it (frustum, ImagePlane); the geometry projects as the solver saw it. */
  shift: { x: number; y: number };
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
  frameKey: string; // the displayed content: a change restarts the wait for the session's one automatic framing, never reframes after it
  selected: string | null;
  lens: Lens | null; // looking through a scene camera: the view is that camera's view, with its gate at `gate`
  gate: Gate | null;
  onLeave: () => void; // tumbling while looking through a camera: the view continues as 透视 from the same position
  slot?: string; // whose camera it is (default the viewer's): kept and framed per slot
  ephemeral?: boolean; // the slot ends with this stage (a dialog's): on unmount everything kept for it goes (endSlot), nothing is kept
}

export function ViewCamera({ o, frameKey, selected, lens, gate, onLeave, slot = VIEWER_SLOT, ephemeral = false }: Props) {
  const kept = keptBy.get(slot) ?? null;
  const stage = useStage();
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  const setThree = useThree((s) => s.set);
  const { view, ortho, frameAsk, viewAsk } = useViewCamera(useShallow((s) => slotCamera(s, slot)));
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
    keptBy.set(slot, {
      position: persp.position.clone(),
      quaternion: persp.quaternion.clone(),
      up: persp.up.clone(),
      fov: persp.fov,
      pivot: pivot.current.clone(),
      ortho: { position: orthoCam.position.clone(), quaternion: orthoCam.quaternion.clone(), zoom: orthoCam.zoom },
      free: freePose.current ? { position: freePose.current.position.clone(), pivot: freePose.current.pivot.clone() } : null,
      looking: wasLooking.current,
    });
  };
  const keepRef = useRef(keep);
  keepRef.current = keep;
  // the camera is kept for the next stage in this slot, or (a slot that ends with the stage) everything of the slot goes;
  // this runs last of all the stage's parts (it unmounts with the canvas's own root), so nothing writes the slot after
  useEffect(() => () => (ephemeral ? endSlot(slot) : keepRef.current()), []); // eslint-disable-line react-hooks/exhaustive-deps
  useLayoutEffect(() => {
    if (lens && gate) {
      if (!wasLooking.current) freePose.current = { position: persp.position.clone(), pivot: pivot.current.clone() }; // to return to later
      const q = new THREE.Quaternion();
      lens.pose.decompose(persp.position, q, new THREE.Vector3());
      persp.quaternion.copy(q);
      persp.up.set(0, 1, 0).applyQuaternion(q); // its own up vector: a rolled camera stays rolled
      persp.fov = lens.fovV;
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
    }
    frustum(size.width, size.height);
    persp.updateMatrixWorld();
    invalidate();
  }, [persp, orthoCam, lens, gate, size.width, size.height, o.near, o.far, invalidate]);

  // The frustum (aspect, the gate window when looking through a camera, the orthographic sides, near and far) depends
  // only on the canvas size and the lens, and the effect above and the per-frame check below share this one place.
  // While the parameter panel's splitter is dragged, R3F resizes the canvas's drawing buffer at once, but the effect
  // above runs only after this component re-renders: a frame drawn in between would put the old aspect's projection on
  // the new canvas, stretched sideways. So before every frame the projection is checked against the canvas size of the
  // moment (nothing happens when it is unchanged), and the aspect always matches the canvas
  const projected = useRef({ w: -1, h: -1 });
  function frustum(w: number, h: number) {
    if (lens && gate) {
      persp.aspect = lens.aspect;
      // the symmetric frustum is centred on the lens axis; the picture (the gate) sits off it by -shift, so the
      // canvas's window into the full picture starts that much further along (CSS pixels, y down)
      persp.setViewOffset(gate.w, gate.h, -gate.x - lens.shift.x * gate.w, -gate.y + lens.shift.y * gate.h, w, h);
    } else {
      persp.clearViewOffset();
      persp.aspect = w / Math.max(h, 1);
    }
    Object.assign(orthoCam, { left: -w / 2, right: w / 2, top: h / 2, bottom: -h / 2 });
    for (const c of [persp, orthoCam]) {
      c.near = o.near;
      c.far = o.far;
      c.updateProjectionMatrix();
    }
    projected.current = { w, h };
  }
  useFrame((state) => {
    const p = projected.current;
    if (p.w !== state.size.width || p.h !== state.size.height) frustum(state.size.width, state.size.height);
  }, -1);

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

  // the content fits into the part of the canvas no panel covers (StageState.insets), centred there: the view is fitted to
  // that rectangle, then camera and pivot slide sideways so the centre lands in its middle
  const place = (center: THREE.Vector3, radius: number, dir: THREE.Vector3) => {
    const c = controls.current;
    const { left, right, top, bottom } = stage.insets;
    const w = Math.max(size.width - left - right, 40), h = Math.max(size.height - top - bottom, 40);
    const H = Math.max(size.height, 1);
    let perPixel: number; // scene units per screen pixel at the centre's depth
    if (active === persp) {
      const fov = (2 * Math.atan(Math.tan((FOV * Math.PI) / 360) * (h / H)) * 180) / Math.PI;
      const dist = fitDistance(radius, fov, w / h);
      persp.position.copy(center).addScaledVector(dir, dist);
      perPixel = (2 * dist * Math.tan((FOV * Math.PI) / 360)) / H;
    } else {
      orthoCam.position.copy(center).addScaledVector(dir, Math.max(radius * 4, 100));
      orthoCam.zoom = fitZoom(radius, w, h);
      perPixel = 1 / orthoCam.zoom;
    }
    active.up.set(0, 1, 0);
    active.lookAt(center);
    const shift = new THREE.Vector3();
    if (left || right || top || bottom) {
      // the free rectangle's middle sits (left − right) / 2 px right of the canvas's and (top − bottom) / 2 px below it:
      // the camera moves the other way sideways (the content then shows right of the middle), and up by the downward offset
      const rightAxis = new THREE.Vector3(1, 0, 0).applyQuaternion(active.quaternion);
      const upAxis = new THREE.Vector3(0, 1, 0).applyQuaternion(active.quaternion);
      shift.addScaledVector(rightAxis, -((left - right) / 2) * perPixel).addScaledVector(upAxis, ((top - bottom) / 2) * perPixel);
    }
    active.position.add(shift);
    pivot.current.copy(center).add(shift);
    active.updateMatrixWorld();
    active.updateProjectionMatrix();
    if (c) {
      c.target.copy(pivot.current);
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
    if (framedOnce.has(slot) || keptBy.has(slot)) return;
    let n = 0;
    let seen = -1;
    let still = 0;
    const id = window.setInterval(() => {
      const size = stage.pickables.size;
      still = size === seen ? still + 1 : 0;
      seen = size;
      // looking through a camera, the view belongs to that scene camera and is never framed automatically (otherwise
      // toggling visibility would throw the view back to 透视); nor once the user has moved the view or the wait timed out
      if (touched.current || lensRef.current || ++n > 1500) return window.clearInterval(id);
      if (!size || still !== 3) return;
      const box = boundsOf(stage.pickables.values());
      if (box) {
        const s = box.getBoundingSphere(new THREE.Sphere());
        place(s.center, Math.max(s.radius, 5), view === "persp" ? THREE_QUARTER.clone() : DIRECTION[view].clone());
        framedOnce.add(slot);
        // stops after framing once: a poll still running would see the registrations change and settle again when a
        // kind of data is shown or hidden, frame again, and reset a view through a camera to 透视
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
    // while looking through a camera, the middle and right buttons and the wheel move the canvas (state/view2d.ts),
    // not the camera
    MIDDLE: lens ? (-1 as THREE.MOUSE) : THREE.MOUSE.PAN,
    RIGHT: -1 as THREE.MOUSE, // the dolly below: OrbitControls' own only knows an up / down drag
  };

  // Right-drag dolly, horizontal as in Houdini: the camera moves along its line to the pivot (perspective) or the
  // orthographic zoom changes. Not while looking through a camera: there the right button zooms the gate.
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    const el = gl.domElement;
    const onDown = (e: PointerEvent) => {
      if (e.button !== 2 || lensRef.current || !controls.current) return;
      e.preventDefault();
      const c = controls.current;
      let last = e.clientX;
      followDrag(
        (ev) => {
          const dx = ev.clientX - last;
          last = ev.clientX;
          if (!dx) return;
          touched.current = true;
          const f = Math.exp(-dx * DOLLY_PER_PX);
          const cam = c.object;
          if (cam instanceof THREE.PerspectiveCamera) {
            const off = cam.position.clone().sub(c.target);
            const d = THREE.MathUtils.clamp(off.length() * f, c.minDistance, c.maxDistance);
            cam.position.copy(c.target).addScaledVector(off.normalize(), d);
          } else if (cam instanceof THREE.OrthographicCamera) {
            cam.zoom = THREE.MathUtils.clamp(cam.zoom / f, c.minZoom, c.maxZoom);
            cam.updateProjectionMatrix();
          }
          c.update(); // dispatches change: the pivot and the kept camera follow, and the canvas redraws
        },
        () => {},
      );
    };
    el.addEventListener("pointerdown", onDown);
    return () => el.removeEventListener("pointerdown", onDown);
  }, [gl]);
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

/** A click (not a drag) selects the nearest object under the pointer; a click on empty space clears the selection
 * (`onPick(null)`: the stage clears the selections it holds, its own and the pickables' own). A press a handle's gizmo took (StageState.claimedAt, view/dragGizmo.tsx) is never a pick, however short. */
// `kept`: the click went to something that keeps its own selection (Pickable.choose), which is told next
export function Picker({ onPick }: { onPick: (key: string | null, kept: boolean) => void }) {
  const stage = useStage();
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  useEffect(() => {
    const el = gl.domElement;
    let stop: (() => void) | null = null;
    // a click or a drag (a tumble) is platform/drag.ts followPress's call, the page's one rule
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      const at = e.timeStamp;
      stop?.();
      stop = followPress(e, () => undefined, (how, up) => {
        stop = null;
        // the gizmo's own listener may run before or after this one: compare times, not order
        if (how === "click" && up && stage.claimedAt < at) pick(up);
      });
    };
    const pick = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      const px = { x: e.clientX - r.left, y: e.clientY - r.top };
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2((px.x / r.width) * 2 - 1, -(px.y / r.height) * 2 + 1), camera);
      const ray = { raycaster, camera, px, size: { width: r.width, height: r.height } };
      const best = picked(stage.pickables, ray);
      // the stage's selection first (cleared when the pick keeps its own), then the pick's own
      onPick(best && !best.p.choose ? best.key : null, !!best?.p.choose);
      best?.p.choose?.(best.p.part?.(ray) ?? null, { alt: e.altKey, shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey, x: e.clientX, y: e.clientY });
    };
    el.addEventListener("pointerdown", onDown);
    return () => {
      el.removeEventListener("pointerdown", onDown);
      stop?.();
    };
  }, [gl, camera, size, stage, onPick]);
  return null;
}
