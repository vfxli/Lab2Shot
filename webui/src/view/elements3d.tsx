import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { FatLines } from "./lines3d";
import { personTint, type ViewOptions } from "../model/viewOptions";
import { matrixAt } from "./matrix3d";
import { type CharacterData, type CharacterMeshData, type ModelData } from "./sceneData";
import { project, usePickable, type PickRay } from "./stageState";
import { boneSegments, columnMajor, heldSample, invertAffine, sampleAt, type Typed} from "../model/viewFormat";

export { CameraPath, ShotCamera, cameraAt } from "./cameras3d";

/** Scene objects on the 3D stage: models (static, transform-animated, or point caches), skinned characters (skinned
 * on the GPU from bind pose, skin weights, blend shapes and joints, matching the server's evaluation), skeletons,
 * cameras with frustum and path, and dome lights. Lines use the configured width; every object can be picked and
 * framed. Meshes are always drawn double-sided and normals are not displayed. */

// UV checker texture, created once.
let checker: THREE.Texture | null = null;
function checkerTexture(): THREE.Texture {
  if (checker) return checker;
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d")!;
  const n = 8;
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      g.fillStyle = (x + y) % 2 ? "#3a3a40" : "#d8d8dc";
      g.fillRect((x * c.width) / n, (y * c.height) / n, c.width / n, c.height / n);
    }
  g.fillStyle = "#0a84ff";
  g.fillRect(0, c.height - 32, 32, 32); // UV origin (0, 0) marked in blue; red is reserved for production-risk indicators.
  checker = new THREE.CanvasTexture(c);
  checker.colorSpace = THREE.SRGBColorSpace;
  checker.wrapS = checker.wrapT = THREE.RepeatWrapping;
  checker.anisotropy = 4;
  return checker;
}

const SURFACE = "#e9e6e1";

/** Surface materials according to the display options: shaded (smooth, flat or UV checker) and/or wireframe.
 * `person` selects the per-person overlay colour for 按人物; undefined means the object does not belong to a person
 * and uses the common tint. */
function surface(o: ViewOptions, through: boolean, checkered: boolean, person?: number) {
  const side = THREE.DoubleSide;
  const shaded = o.shading !== "wire";
  const wire = o.shading === "wire" || o.shading === "wireShaded";
  const flat = o.shading === "flat";
  // Toggling a texture or flat shading changes the shader, so a new (keyed) material is created; updating the
  // existing material's settings would keep the old shader.
  const material = checkered ? (
    <meshStandardMaterial key={`checker.${flat}`} map={checkerTexture()} roughness={0.7} metalness={0} side={side} flatShading={flat} />
  ) : (
    // When viewing through a camera over the plate, apply the overlay (叠加 opacity and colour from the display
    // options) directly here, without a cook.
    <meshStandardMaterial key={`plain.${flat}`} color={through ? (person !== undefined && o.overlayByPerson ? personTint(person) : o.overlayTint) : SURFACE} roughness={0.62} metalness={0.04} side={side} flatShading={flat} transparent={through && o.overlayOpacity < 1} opacity={through ? o.overlayOpacity : 1} polygonOffset={wire} polygonOffsetFactor={1} polygonOffsetUnits={1} />
  );
  const wireMaterial = <meshBasicMaterial color={shaded ? "#2a2a30" : "#c8c8cc"} wireframe transparent opacity={shaded ? 0.55 : 0.9} depthWrite={false} side={side} />;
  return { shaded, wire, material, wireMaterial };
}

/** Builds non-indexed geometry with per-corner UVs (UVs are stored per face corner) from indexed geometry. */
function cornerGeometry(indexed: THREE.BufferGeometry, uv: Float32Array): THREE.BufferGeometry {
  const g = indexed.toNonIndexed();
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  return g;
}

function indexOf(faces: Typed): THREE.BufferAttribute {
  return new THREE.BufferAttribute(faces instanceof Uint32Array ? faces : new Uint32Array(faces), 1);
}

/** A 模型 at the given frame: its static shape or the frame's point-cache sample, placed by its transform. */
export function ModelMesh({ m, frame, o, pickKey, through, version, person }: { m: ModelData; frame: number; o: ViewOptions; pickKey: string; through: boolean; version: number; person?: number }) {
  const invalidate = useThree((s) => s.invalidate);
  const i = sampleAt(m.ref.frames, frame);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(m.ref.vertices * 3), 3));
    g.setIndex(indexOf(m.faces));
    return g;
  }, [m]);
  // If this point-cache frame has not arrived, draw the nearest available sample instead of nothing (viewFormat.ts heldSample).
  const shape = m.points ?? heldSample(m.samples, i)?.sample ?? null;
  // Positions are written during render so that the normals computed below use this frame.
  const shown = useMemo(() => {
    if (!shape) return null;
    const attr = geometry.attributes.position as THREE.BufferAttribute;
    (attr.array as Float32Array).set(shape);
    attr.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();
    return {};
  }, [geometry, shape]);
  const checkered = o.uvChecker && !!m.uv;
  const corners = useMemo(() => (checkered && shown ? cornerGeometry(geometry, m.uv!) : null), [checkered, geometry, m.uv, shown]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useEffect(() => () => corners?.dispose(), [corners]);
  useEffect(() => invalidate(), [shown, version, invalidate]);
  const matrix = useMemo(() => matrixAt(m.world, i), [m, i]);
  const ref = useRef<THREE.Group>(null);
  const pickable = useMemo(
    () => ({
      label: m.ref.name,
      bounds: () => (geometry.boundingBox && ref.current ? geometry.boundingBox.clone().applyMatrix4(ref.current.matrixWorld) : null),
      hit: (p: PickRay) => {
        const obj = ref.current?.children[0];
        const hits = obj ? p.raycaster.intersectObject(obj, false) : [];
        return hits.length ? hits[0].distance : null;
      },
    }),
    [geometry, m.ref.name],
  );
  usePickable(pickKey, pickable);
  const s = surface(o, through, checkered, person);
  return (
    <group ref={ref} matrix={matrix} matrixAutoUpdate={false}>
      <mesh geometry={corners ?? geometry} visible={s.shaded && !!shown}>
        {s.material}
      </mesh>
      {s.wire && (
        <mesh geometry={geometry} renderOrder={1} visible={!!shown}>
          {s.wireMaterial}
        </mesh>
      )}
    </group>
  );
}

/** One mesh of a 蒙皮角色, skinned on the GPU from its bind pose, four joint indices and weights per point, and blend
 * shapes. Per frame only the joint transforms and blend-shape weights change. */
function SkinnedBody({ c, mesh, frame, o, pickKey, through, person }: { c: CharacterData; mesh: CharacterMeshData; frame: number; o: ViewOptions; pickKey: string; through: boolean; person: number | undefined }) {
  const invalidate = useThree((s) => s.invalidate);
  const i = sampleAt(c.ref.frames, frame);
  const joints = c.ref.joints.length;
  const shapes = mesh.ref.shapes.length;
  const { geometry, skeleton } = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(mesh.points, 3));
    g.setAttribute("skinIndex", new THREE.BufferAttribute(mesh.jointIndices instanceof Uint8Array || mesh.jointIndices instanceof Uint16Array ? mesh.jointIndices : new Uint16Array(mesh.jointIndices!), 4));
    g.setAttribute("skinWeight", new THREE.BufferAttribute(mesh.jointWeights!, 4));
    g.setIndex(indexOf(mesh.faces));
    if (shapes) {
      const v = mesh.ref.vertices * 3;
      g.morphAttributes.position = mesh.ref.shapes.map((_, b) => new THREE.BufferAttribute(mesh.shapeOffsets!.subarray(b * v, (b + 1) * v), 3));
      g.morphTargetsRelative = true; // Offsets are added to the bind points before skinning, as in UsdSkel.
    }
    g.computeVertexNormals();
    // Bones are not added to the scene; their world transforms are set directly from the data each frame.
    const bones = Array.from({ length: joints }, () => Object.assign(new THREE.Bone(), { matrixAutoUpdate: false, matrixWorldAutoUpdate: false }));
    const inverses = Array.from({ length: joints }, (_, j) => new THREE.Matrix4().fromArray(invertAffine(columnMajor(c.bind, j, 4))));
    return { geometry: g, skeleton: new THREE.Skeleton(bones, inverses) };
  }, [c, mesh, joints, shapes]);
  const checkered = o.uvChecker && !!mesh.uv;
  const corners = useMemo(() => (checkered ? cornerGeometry(geometry, mesh.uv!) : null), [checkered, geometry, mesh.uv]);
  const ref = useRef<THREE.SkinnedMesh>(null);
  useEffect(
    () => () => {
      geometry.dispose();
      skeleton.dispose();
    },
    [geometry, skeleton],
  );
  useEffect(() => () => corners?.dispose(), [corners]);
  // Apply this frame's joint transforms and blend-shape weights.
  useLayoutEffect(() => {
    skeleton.bones.forEach((bone, j) => bone.matrixWorld.fromArray(columnMajor(c.anim, i * joints + j, 3)));
    const target = ref.current;
    if (target) {
      if (!target.skeleton || target.skeleton !== skeleton) {
        target.bindMode = THREE.DetachedBindMode; // Bones carry world transforms; the mesh's own transform is not applied.
        target.bind(skeleton, new THREE.Matrix4());
      }
      if (shapes && target.morphTargetInfluences?.length !== shapes) target.updateMorphTargets();
      if (shapes && target.morphTargetInfluences) for (let b = 0; b < shapes; b++) target.morphTargetInfluences[b] = mesh.shapeWeights![i * shapes + b];
    }
    skeleton.update(); // Update skinning matrices now; bounds and picking read them before the next render.
    invalidate();
  }, [skeleton, c, mesh, i, joints, shapes, invalidate, corners]);
  const pickable = useMemo(
    () => ({
      label: c.ref.name,
      bounds: () => {
        const t = ref.current;
        if (!t) return null;
        t.computeBoundingBox(); // skinned bounds at the current frame
        return t.boundingBox ? t.boundingBox.clone().applyMatrix4(t.matrixWorld) : null;
      },
      hit: (p: PickRay) => {
        const t = ref.current;
        if (!t) return null;
        t.computeBoundingSphere();
        const hits = p.raycaster.intersectObject(t, false);
        return hits.length ? hits[0].distance : null;
      },
    }),
    [c.ref.name],
  );
  usePickable(pickKey, pickable);
  const s = surface(o, through, checkered, person);
  return (
    <>
      <skinnedMesh ref={ref} geometry={corners ?? geometry} visible={s.shaded} frustumCulled={false}>
        {s.material}
      </skinnedMesh>
      {s.wire && <SkinnedWire geometry={geometry} skeleton={skeleton} material={s.wireMaterial} i={i} shapes={shapes} weights={mesh.shapeWeights} />}
    </>
  );
}

/** Wireframe of a skinned mesh, using the same skinning. */
function SkinnedWire({ geometry, skeleton, material, i, shapes, weights }: { geometry: THREE.BufferGeometry; skeleton: THREE.Skeleton; material: React.ReactNode; i: number; shapes: number; weights: Float32Array | null }) {
  const ref = useRef<THREE.SkinnedMesh>(null);
  useLayoutEffect(() => {
    const t = ref.current;
    if (!t) return;
    if (t.skeleton !== skeleton) {
      t.bindMode = THREE.DetachedBindMode;
      t.bind(skeleton, new THREE.Matrix4());
    }
    if (shapes && t.morphTargetInfluences?.length !== shapes) t.updateMorphTargets();
    if (shapes && t.morphTargetInfluences) for (let b = 0; b < shapes; b++) t.morphTargetInfluences[b] = weights![i * shapes + b];
  }, [skeleton, i, shapes, weights]);
  return (
    <skinnedMesh ref={ref} geometry={geometry} renderOrder={1} frustumCulled={false}>
      {material}
    </skinnedMesh>
  );
}

/** A 蒙皮角色 mesh that cannot be skinned on the GPU (more than four joints per point); uses the server's per-frame evaluation. */
function EvaluatedBody({ c, mesh, frame, o, pickKey, through, person }: { c: CharacterData; mesh: CharacterMeshData; frame: number; o: ViewOptions; pickKey: string; through: boolean; person: number | undefined }) {
  const model = useMemo<ModelData>(
    () => ({ ref: { name: c.ref.name, path: c.ref.path, frames: c.ref.frames, faces: mesh.ref.faces, vertices: mesh.ref.vertices, per_frame: true, uv: mesh.ref.uv }, faces: mesh.faces, world: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), points: null, samples: mesh.samples, uv: mesh.uv }),
    [c, mesh],
  );
  return <ModelMesh m={model} frame={frame} o={o} pickKey={pickKey} through={through} version={mesh.samples.size} person={person} />;
}

/** A 蒙皮角色 (skinned meshes) or a 骨架动画 (no mesh), drawing meshes and bones according to their visibility. */
export function Character({ c, frame, o, pickKey, through, meshes, bones, person }: { c: CharacterData; frame: number; o: ViewOptions; pickKey: string; through: boolean; meshes: boolean; bones: boolean; person: number | undefined }) {
  return (
    <>
      {meshes &&
        c.meshes.map((m, k) =>
          m.jointIndices ? (
            <SkinnedBody key={k} c={c} mesh={m} frame={frame} o={o} pickKey={`${pickKey}/mesh/${k}`} through={through} person={person} />
          ) : (
            <EvaluatedBody key={k} c={c} mesh={m} frame={frame} o={o} pickKey={`${pickKey}/mesh/${k}`} through={through} person={person} />
          ),
        )}
      {bones && <Bones c={c} frame={frame} o={o} pickKey={`${pickKey}/skeleton`} />}
    </>
  );
}

/** Skeleton bones at the given frame, drawn from each joint to its parent. */
function Bones({ c, frame, o, pickKey }: { c: CharacterData; frame: number; o: ViewOptions; pickKey: string }) {
  const i = sampleAt(c.ref.frames, frame);
  const segments = useMemo(() => boneSegments(c.anim, c.ref.parents, i), [c, i]);
  const pickable = useMemo(
    () => ({
      label: `${c.ref.name} 骨架`,
      bounds: () => (segments.length ? new THREE.Box3().setFromArray(segments) : null),
      hit: (p: PickRay) => {
        let best: number | null = null;
        const v = new THREE.Vector3();
        for (let k = 0; k < segments.length; k += 3) {
          const s = project(v.set(segments[k], segments[k + 1], segments[k + 2]), p);
          if (s && Math.hypot(s.x - p.px.x, s.y - p.px.y) < 8 && (best === null || s.depth < best)) best = s.depth;
        }
        return best;
      },
    }),
    [segments, c.ref.name],
  );
  usePickable(pickKey, pickable);
  if (!segments.length) return null;
  return <FatLines segments={segments} color={o.boneColor} width={o.lineWidth * 1.3} overlay opacity={0.95} />;
}

/** Camera state at a frame: camera-to-world matrix and the half extents of the resolution gate at depth 1. */
