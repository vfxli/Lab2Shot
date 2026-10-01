import { useEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { type Bounds } from "../model/math3d";
import type { ViewOptions } from "../model/viewOptions";
import { matrixAt } from "./matrix3d";
import { type CloudData, type CloudSample, type GridSample } from "./sceneData";
import {columnMajor, heldSample } from "../model/viewFormat";
import { sampleAt } from "../model/timelineMath";
import { project, usePickable, type PickRay } from "./stageState";
import { COLOR_MODES, POINTS_FRAG, POINTS_VERT } from "./pointShaders";

/** A point cloud on the 3D stage, drawn like Houdini's point display: 圆片 (round, soft-edged, facing the camera),
 * sized in screen pixels regardless of distance (显示选项 大小), and coloured by the points' own colour or a single
 * colour.
 *
 * Every point is drawn exactly as the data holds it (the server's 点云上限 is the only place points are thinned, and
 * the view's notice reports it). A frame's points are uploaded once per displayed frame (the last few are kept on the
 * GPU). A cloud derived from a depth map (GRID) is uploaded as its depths and colours in two textures, and the vertex
 * shader reconstructs each point with the arithmetic of viewFormat.ts gridPoints (the server verified that every frame
 * yields the cloud's own points). */

/** A sample's bounds: computed by the worker (together with its points for a depth cloud) as soon as something needs
 * them (scene.wantBounds); null until they arrive, and the subsequent redraw uses them. */
const boundsOf = (c: CloudData, s: CloudSample | null): Bounds | null => {
  if (!s) return null;
  if (!s.bounds) c.scene.wantBounds(s);
  return s.bounds;
};

function uniforms() {
  return {
    uColor: { value: 0 },
    uTint: { value: new THREE.Color() },
    uSize: { value: 3 },
    uPixelRatio: { value: 1 },
    uViewH: { value: 1 },
    uDepth: { value: null as THREE.Texture | null },
    uRgb: { value: null as THREE.Texture | null },
    uHasRgb: { value: 0 },
    uGridTint: { value: new THREE.Vector3(0.7, 0.7, 0.7) },
    uRamp: { value: new THREE.Vector3(0, 1, 0) },
    uRampNear: { value: new THREE.Vector3() },
    uRampSlope: { value: new THREE.Vector3() },
    uGrid: { value: new THREE.Vector4(1, 1, 1, 1) },
    uFocal: { value: 1 },
    uPrincipal: { value: new THREE.Vector2() },
    uCam: { value: new THREE.Matrix4() },
  };
}

const colours = (c: CloudSample["colors"]) => new THREE.BufferAttribute(c, 3, !(c instanceof Float32Array));

// a depth cloud drawn without positions: one attribute (never read) per grid size, uploaded once
const placeholders = new Map<number, THREE.BufferAttribute>();
function placeholder(n: number): THREE.BufferAttribute {
  let a = placeholders.get(n);
  if (!a) {
    placeholders.set(n, (a = new THREE.BufferAttribute(new Float32Array(n), 1)));
    if (placeholders.size > 4) placeholders.delete(placeholders.keys().next().value!);
  }
  return a;
}

/** A depth cloud's frame as two textures: its depths (32-bit float) and its colours with alpha 1 where a pixel is a
 * point (bytes when the colours are bytes, else float: 16-bit words as k/65535). */
function gridTextures(g: GridSample): THREE.DataTexture[] {
  const { gw, gh } = g; // this frame's own size: over 「点云上限」 the server has dropped points and the grid is smaller
  const n = gw * gh;
  const depth = new THREE.DataTexture(g.depth, gw, gh, THREE.RedFormat, THREE.FloatType);
  const bytes = !g.colors || g.colors instanceof Uint8Array;
  const rgba = bytes ? new Uint8Array(n * 4) : new Float32Array(n * 4);
  const top = bytes ? 255 : 1;
  const scale = g.colors instanceof Uint16Array ? 1 / 65535 : 1;
  for (let k = 0, point = 0; k < n; k++) {
    if (g.depth[k] !== g.depth[k]) continue;
    if (g.colors) for (let ch = 0; ch < 3; ch++) rgba[k * 4 + ch] = g.colors[point * 3 + ch] * scale; // one colour per kept pixel, in order
    rgba[k * 4 + 3] = top;
    point++;
  }
  const rgb = new THREE.DataTexture(rgba, gw, gh, THREE.RGBAFormat, bytes ? THREE.UnsignedByteType : THREE.FloatType);
  for (const t of [depth, rgb]) {
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
  }
  return [depth, rgb];
}

function release(g: THREE.BufferGeometry): void {
  g.dispose();
  for (const t of (g.userData.textures as THREE.Texture[] | undefined) ?? []) t.dispose();
}

/** A sample's geometry: its points and colours (bytes read as k/255) as attributes. */
function geometryOf(s: CloudSample): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  if (s.grid) {
    g.setAttribute("position", placeholder(s.grid.gw * s.grid.gh));
    g.userData.textures = gridTextures(s.grid);
    return g;
  }
  const points = s.points!; // an explicit sample always has its points (a grid sample is the branch above)
  g.setAttribute("position", new THREE.BufferAttribute(points, 3));
  g.setAttribute("rgb", colours(s.colors));
  return g;
}

interface Props {
  src: CloudData;
  frame: number;
  o: ViewOptions;
  pickKey: string;
  version: number; // the view's version: incremented when new samples arrive
}

export function Cloud({ src, frame, o, pickKey, version }: Props) {
  const { gl, size, invalidate } = useThree();
  const i = sampleAt(src.frames, frame);
  // A frame not yet arrived draws the nearest available sample, never nothing (otherwise scrubbing the timeline flickers).
  // The frame actually shown is stated by the one view notice area; this only keeps the picture from going empty
  const sample = src.still ?? heldSample(src.samples, i)?.sample ?? null;
  const count = sample ? sample.count : 0;
  const grid = !!sample?.grid;

  const obj = useMemo(() => {
    const m = new THREE.ShaderMaterial({ uniforms: uniforms(), vertexShader: POINTS_VERT, fragmentShader: POINTS_FRAG, defines: grid ? { GRID: "" } : {} });
    const object = new THREE.Points(new THREE.BufferGeometry(), m);
    object.frustumCulled = false;
    return { object, material: m };
  }, [grid]);
  // the samples' geometries, the last few kept (their buffers stay on the graphics card while kept)
  const kept = useMemo(() => new Map<CloudSample, THREE.BufferGeometry>(), [obj]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(
    () => () => {
      for (const g of kept.values()) release(g);
      kept.clear();
      obj.material.dispose();
    },
    [obj, kept],
  );
  const geometry = useMemo(() => {
    if (!sample) return null;
    let g = kept.get(sample);
    if (!g) {
      kept.set(sample, (g = geometryOf(sample)));
      while (kept.size > 6) {
        const [oldest, og] = kept.entries().next().value!;
        if (oldest === sample) break;
        release(og);
        kept.delete(oldest);
      }
    }
    return g;
  }, [sample, kept]);
  useEffect(() => invalidate(), [geometry, version, invalidate]);

  useFrame(() => {
    const g = geometry;
    if (!g) {
      obj.object.visible = false;
      return;
    }
    obj.object.visible = true;
    if (obj.object.geometry !== g) obj.object.geometry = g;
  });

  // the look: size and colour
  const world = useMemo(() => matrixAt(src.world, i), [src, i]);
  useFrame(() => {
    obj.object.matrixAutoUpdate = false;
    obj.object.matrix.copy(world);
    const u = obj.material.uniforms;
    u.uSize.value = o.pointPx;
    u.uPixelRatio.value = gl.getPixelRatio();
    u.uViewH.value = size.height;
    u.uColor.value = COLOR_MODES[o.pointColor];
    u.uTint.value.set(o.pointTint);
    const gs: GridSample | null = sample?.grid ?? null;
    if (gs && geometry) {
      const [depthTex, rgbTex] = geometry.userData.textures as THREE.Texture[];
      u.uDepth.value = depthTex;
      u.uRgb.value = rgbTex;
      u.uHasRgb.value = gs.colors ? 1 : 0;
      if (gs.tint) u.uGridTint.value.set(gs.tint[0], gs.tint[1], gs.tint[2]);
      const ramp = gs.ref.ramp;
      const colour = gs.ref.ramp_colour;  // the ramp comes from the server (view_data.py RAMP_NEAR / RAMP_SLOPE)
      u.uRamp.value.set(ramp ? ramp[0] : 0, ramp ? ramp[1] : 1, ramp && colour ? 1 : 0);
      if (colour) {
        u.uRampNear.value.fromArray(colour[0]);
        u.uRampSlope.value.fromArray(colour[1]);
      }
      u.uGrid.value.set(gs.gw, gs.step, gs.ref.width, gs.ref.height);
      u.uFocal.value = gs.focal;
      u.uPrincipal.value.set(gs.principal ? gs.principal[0] : gs.ref.width * 0.5, gs.principal ? gs.principal[1] : gs.ref.height * 0.5);
      u.uCam.value.fromArray(columnMajor(gs.cam, 0, 4));
    }
    const m = obj.material;
    const coverage = o.antialias === "msaa"; // the discs' soft edge: via coverage under MSAA, otherwise via blending
    if (m.alphaToCoverage !== coverage || m.transparent !== !coverage) {
      m.alphaToCoverage = coverage;
      m.transparent = !coverage;
      m.needsUpdate = true;
    }
  }, -1);

  // picked by a click near one of its points
  const pickable = useMemo(
    () => ({
      label: src.name,
      bounds: () => {
        const b = boundsOf(src, sample);
        return b ? new THREE.Box3(new THREE.Vector3(...b.min), new THREE.Vector3(...b.max)).applyMatrix4(world) : null;
      },
      hit: (p: PickRay) => {
        if (!sample || !sample.points) return null; // a depth cloud's points not yet reconstructed: requested together with its bounds
        const n = Math.min(count, 200_000);
        const step = Math.max(1, Math.floor(count / n));
        const v = new THREE.Vector3();
        let best: number | null = null;
        const radius = Math.max(6, o.pointPx * 0.75);
        for (let k = 0; k < count; k += step) {
          v.set(sample.points[k * 3], sample.points[k * 3 + 1], sample.points[k * 3 + 2]).applyMatrix4(world);
          const s = project(v, p);
          if (s && Math.hypot(s.x - p.px.x, s.y - p.px.y) < radius && (best === null || s.depth < best)) best = s.depth;
        }
        return best;
      },
    }),
    [src, sample, count, world, o.pointPx],
  );
  usePickable(pickKey, pickable);

  return <primitive object={obj.object} />;
}
