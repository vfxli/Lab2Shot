import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { ORDER } from "./drawOrder";
import { useFrame, useThree } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { personTint, type ViewOptions } from "../model/viewOptions";
import { matrixAt } from "./matrix3d";
import { type CharacterData, type CharacterMeshData, type ModelData } from "./sceneData";
import { project, segmentsHit, surfaceHit, usePickable, type PickRay } from "./stageState";
import {boneSegments, columnMajor, heldSample, skinReach, type Typed } from "../model/viewFormat";
import { sampleAt } from "../model/timelineMath";
import { IDENTITY, inverse } from "../model/math3d";
import { FatLines } from "./lines3d";
import { t } from "../i18n/t";
import { useLang } from "../i18n/lang";

export { CameraPath, ShotCamera, cameraAt } from "./cameras3d";

/** 三维舞台上的场景物体：模型（静止、变换动画或点缓存）、蒙皮角色（在 GPU 上由绑定姿势、蒙皮权重、混合变形与关节
 * 蒙皮，与服务器的求值一致）、骨架、带视锥与路径的相机（view/cameras3d.tsx，在此转出）。线用设定的线宽；每个物体都可
 * 拾取、可框显。网格按显示选项绘制正面或双面，不显示法线。 */

// UV 棋盘格纹理，只创建一次。
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
  g.fillRect(0, c.height - 32, 32, 32); // UV 原点 (0, 0) 用蓝色标出；红色留给表示制作风险的标记。
  checker = new THREE.CanvasTexture(c);
  checker.colorSpace = THREE.SRGBColorSpace;
  checker.wrapS = checker.wrapT = THREE.RepeatWrapping;
  checker.anisotropy = 4;
  return checker;
}

const SURFACE = "#e9e6e1";

/** 按显示选项生成表面材质：明暗（平滑、平面或 UV 棋盘格）和 / 或线框。
 * `person` 用于「按人物」时选取该人的叠加色；undefined 表示该物体不属于某个人，使用公共颜色。 */
function surface(o: ViewOptions, through: boolean, checkered: boolean, person?: number) {
  const side = o.backFaces ? THREE.DoubleSide : THREE.FrontSide; // 关掉背面：只画朝向相机的面
  const shaded = o.shading !== "wire";
  const wire = o.shading === "wire" || o.shading === "wireShaded";
  const flat = o.shading === "flat";
  const transparent = through && o.overlayOpacity < 1;
  // 纹理、平面明暗、面朝向和透明模式都会改变着色器，因此用 key 区分材质。
  // 不透明着色器将 alpha 固定为 1；进入透明模式时需换材质，随后只改 opacity 即可。
  const material = checkered ? (
    <meshStandardMaterial key={`checker.${flat}.${side}`} map={checkerTexture()} roughness={0.7} metalness={0} side={side} flatShading={flat} />
  ) : (
    // 透过相机叠在背板上看时，叠加效果（显示选项中叠加的不透明度与颜色）直接在这里套用，无需计算。
    <meshStandardMaterial key={`plain.${flat}.${side}.${transparent}`} color={through ? (person !== undefined && o.overlayByPerson ? personTint(person) : o.overlayTint) : SURFACE} roughness={0.62} metalness={0.04} side={side} flatShading={flat} transparent={transparent} opacity={through ? o.overlayOpacity : 1} polygonOffset={wire} polygonOffsetFactor={1} polygonOffsetUnits={1} />
  );
  const wireMaterial = <meshBasicMaterial color={shaded ? "#2a2a30" : "#c8c8cc"} wireframe transparent opacity={shaded ? 0.55 : 0.9} depthWrite={false} side={side} />;
  return { shaded, wire, material, wireMaterial };
}

/** 由带索引的几何体生成不带索引、按角存 UV 的几何体（UV 按面的每个角存储）。 */
function cornerGeometry(indexed: THREE.BufferGeometry, uv: Float32Array): THREE.BufferGeometry {
  const g = indexed.toNonIndexed();
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  return g;
}

function indexOf(faces: Typed): THREE.BufferAttribute {
  return new THREE.BufferAttribute(faces instanceof Uint32Array ? faces : new Uint32Array(faces), 1);
}

/** 某一帧的模型：它的静止形状或该帧的点缓存样本，按它的变换摆放。 */
export function ModelMesh({ m, frame, o, pickKey, through, version, person }: { m: ModelData; frame: number; o: ViewOptions; pickKey: string; through: boolean; version: number; person?: number }) {
  const invalidate = useThree((s) => s.invalidate);
  const i = sampleAt(m.ref.frames, frame);
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(m.ref.vertices * 3), 3));
    g.setIndex(indexOf(m.faces));
    return g;
  }, [m]);
  // 点缓存的这一帧尚未到达时，画最近可用的一份而不是什么都不画（viewFormat.ts heldSample）。
  const shape = m.points ?? heldSample(m.samples, i)?.sample ?? null;
  // 位置在渲染期间写入，下面计算的法线才用的是这一帧。
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
      hit: (p: PickRay) => surfaceHit(p, ref.current?.children[0]),
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
        <mesh geometry={geometry} renderOrder={ORDER.body} visible={!!shown}>
          {s.wireMaterial}
        </mesh>
      )}
    </group>
  );
}

/** 蒙皮角色的一个网格，在 GPU 上由绑定姿势、每点四个关节下标与权重、混合变形蒙皮。每帧只变关节变换与混合变形权重。 */
function SkinnedBody({ c, mesh, frame, o, pickKey, through, person }: { c: CharacterData; mesh: CharacterMeshData; frame: number; o: ViewOptions; pickKey: string; through: boolean; person: number | undefined }) {
  const invalidate = useThree((s) => s.invalidate);
  const i = sampleAt(c.ref.frames, frame);
  const joints = c.ref.joints.length;
  const shapes = mesh.ref.shapes.length;
  const { geometry, skeleton, margin } = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(mesh.points, 3));
    g.setAttribute("skinIndex", new THREE.BufferAttribute(mesh.jointIndices instanceof Uint8Array || mesh.jointIndices instanceof Uint16Array ? mesh.jointIndices : new Uint16Array(mesh.jointIndices!), 4));
    g.setAttribute("skinWeight", new THREE.BufferAttribute(mesh.jointWeights!, 4));
    g.setIndex(indexOf(mesh.faces));
    if (shapes) {
      const v = mesh.ref.vertices * 3;
      g.morphAttributes.position = mesh.ref.shapes.map((_, b) => new THREE.BufferAttribute(mesh.shapeOffsets!.subarray(b * v, (b + 1) * v), 3));
      g.morphTargetsRelative = true; // 偏移量在蒙皮之前加到绑定点上，与 UsdSkel 相同。
    }
    g.computeVertexNormals();
    // 骨骼不加入场景；每帧直接按数据设定它们的世界变换。
    const bones = Array.from({ length: joints }, () => Object.assign(new THREE.Bone(), { matrixAutoUpdate: false, matrixWorldAutoUpdate: false }));
    // 退化的绑定矩阵（缩放 0）不可逆：逆绑定按单位阵，那个关节的顶点按它这一帧的世界变换整个搬过去（不再相对绑定姿势），
    // 形状可能不对，但不会让整副网格变成 NaN
    const inverseOf = (m: number[]) => inverse(m) ?? IDENTITY;
    const inverses = Array.from({ length: joints }, (_, j) => new THREE.Matrix4().fromArray(inverseOf(columnMajor(c.bind, j, 4))));
    // 包围框的边距：绑定姿势下每个点离带它的主关节（四个影响里权重最大的那个，model/viewFormat.ts skinReach：包里的顺序不保证）多远，取最大。每帧的包围框 = 这一帧关节位置的包围框
    // 各向放宽这么多：点随主关节转、挪，离它的距离不变，所以角色怎么转身都包得住（按轴量的突出量转 45° 就包不住）。不在
    // CPU 上蒙皮整副网格（十万个点一次要二十多毫秒）；混合变形推得更远的部分、几个关节平分的点被拉开的部分不算在内
    const jointAt = Array.from({ length: joints }, (_, j) => { const m = columnMajor(c.bind, j, 4); return [m[12], m[13], m[14]] as const; });
    const margin = skinReach(mesh.points, g.getAttribute("skinIndex").array as ArrayLike<number>, mesh.jointWeights!, jointAt);
    return { geometry: g, skeleton: new THREE.Skeleton(bones, inverses), margin };
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
  // 套用这一帧的关节变换与混合变形权重。
  useLayoutEffect(() => {
    skeleton.bones.forEach((bone, j) => bone.matrixWorld.fromArray(columnMajor(c.anim, i * joints + j, 3)));
    const target = ref.current;
    if (target) {
      if (!target.skeleton || target.skeleton !== skeleton) {
        target.bindMode = THREE.DetachedBindMode; // 骨骼带的是世界变换；不套用网格自身的变换。
        target.bind(skeleton, new THREE.Matrix4());
      }
      if (shapes && target.morphTargetInfluences?.length !== shapes) target.updateMorphTargets();
      if (shapes && target.morphTargetInfluences) for (let b = 0; b < shapes; b++) target.morphTargetInfluences[b] = mesh.shapeWeights![i * shapes + b];
    }
    skeleton.update(); // 当场更新蒙皮矩阵：包围盒与拾取在下一次渲染之前就要读取它们。
    invalidate();
  }, [skeleton, c, mesh, i, joints, shapes, invalidate, corners]);
  const pickable = useMemo(
    () => ({
      label: c.ref.name,
      bounds: () => {
        if (!ref.current || !skeleton.bones.length) return null;
        const box = new THREE.Box3();
        for (const bone of skeleton.bones) box.expandByPoint(new THREE.Vector3().setFromMatrixPosition(bone.matrixWorld));
        return box.expandByScalar(margin); // 骨骼带的是世界变换（DetachedBindMode）
      },
      hit: (p: PickRay) => {
        const t = ref.current;
        if (!t) return null;
        t.computeBoundingSphere();
        return surfaceHit(p, t);
      },
    }),
    [c.ref.name, skeleton, margin],
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

/** 蒙皮网格的线框，使用同一份蒙皮。 */
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
    <skinnedMesh ref={ref} geometry={geometry} renderOrder={ORDER.body} frustumCulled={false}>
      {material}
    </skinnedMesh>
  );
}

/** 无法在 GPU 上蒙皮（每点超过四个关节）的蒙皮角色网格：使用服务器逐帧的求值结果。 */
function EvaluatedBody({ c, mesh, frame, o, pickKey, through, person }: { c: CharacterData; mesh: CharacterMeshData; frame: number; o: ViewOptions; pickKey: string; through: boolean; person: number | undefined }) {
  const model = useMemo<ModelData>(
    () => ({ ref: { name: c.ref.name, path: c.ref.path, frames: c.ref.frames, faces: mesh.ref.faces, vertices: mesh.ref.vertices, per_frame: true, uv: mesh.ref.uv }, faces: mesh.faces, world: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), points: null, samples: mesh.samples, uv: mesh.uv }),
    [c, mesh],
  );
  return <ModelMesh m={model} frame={frame} o={o} pickKey={pickKey} through={through} version={mesh.samples.size} person={person} />;
}

/** 蒙皮角色（蒙皮网格）或骨架动画（没有网格）：按各自的显示开关画网格与骨骼。 */
export function Character({ c, frame, o, pickKey, through, meshes, bones, person }: { c: CharacterData; frame: number; o: ViewOptions; pickKey: string; through: boolean; meshes: boolean; bones: boolean; person: number | undefined }) {
  return (
    <>
      {meshes &&
        c.meshes.map((m, k) =>
          m.jointIndices && !m.ref.per_frame ? (
            <SkinnedBody key={k} c={c} mesh={m} frame={frame} o={o} pickKey={`${pickKey}/mesh/${k}`} through={through} person={person} />
          ) : (
            <EvaluatedBody key={k} c={c} mesh={m} frame={frame} o={o} pickKey={`${pickKey}/mesh/${k}`} through={through} person={person} />
          ),
        )}
      {bones && <Bones c={c} frame={frame} o={o} pickKey={`${pickKey}/skeleton`} />}
    </>
  );
}

// ------------------------------------------------------------------ 骨架的画法（骨与骨点）

/** 骨架的尺度：骨长的第 90 百分位（人形约是大腿、脊柱那一档），比中位数稳——手指、扭转骨这类短骨多的角色中位数会
 * 小得离谱；比最长骨稳——根到髋那根「假骨」常常是最长的。骨点与骨的上限都按它。 */
const REF_PERCENTILE = 0.9;
/** 骨点半径（「骨点自适应」开）= 与它相连的最短骨 × 这个比例，再不超过与它相连的最长骨 × JOINT_OF_LONGEST：手指、脊椎这类
 * 密集处骨点小、躯干大，层级一眼看清；0.125 让相邻两个骨点之间留出大半根骨的空。相连的骨里长度不到尺度 2% 的不算
 * （AccuRIG / CC 的扭转骨、Maya 的辅助关节常常与父关节重合，按它们算骨点就成了零）。 */
const JOINT_OF_SHORTEST = 0.125;
const JOINT_OF_LONGEST = 0.05;
/** 骨点半径（自适应关）= 尺度 × 这个比例：整副一个大小，不同大小的角色仍成比例。 */
const JOINT_OF_SCALE = 0.025;
/** 骨（八面体）底面的半宽 = 这根骨长 × 这个比例（Blender 的八面体骨是 0.1；这里细一半，骨架叠在模型上不挡住模型），
 * 长过尺度的骨（根到髋）按尺度算，不会出现一根粗得盖住半个身子的骨。骨的粗细不跟骨点挂钩：关节旁有重合的辅助关节时骨也照样画得出来。 */
const BONE_OF_LENGTH = 0.05;
/** 与父关节重合（不到尺度 2%）的关节不画骨：那是辅助关节，画出来只是一个点。 */
const TRIVIAL_OF_SCALE = 0.02;

/** 八面体骨的单位几何：从父骨点（原点）指向子骨点（+Y 上的 1）；底面在 1/8 处、半宽 1，尖端在 1。画时按实例缩放、转向。 */
function boneGeometry(): THREE.BufferGeometry {
  const tail = [0, 0, 0], tip = [0, 1, 0];
  const b = [[1, 0.125, 0], [0, 0.125, 1], [-1, 0.125, 0], [0, 0.125, -1]];
  const tris: number[] = [];
  for (let k = 0; k < 4; k++) {
    const a = b[k], c = b[(k + 1) % 4];
    // 逆时针朝外（从外面看）：法线朝外，只画正面时看到的是外表面，打光是凸的。顺序反了就只剩远侧的内壁，看着像凹进去
    tris.push(...tail, ...a, ...c, ...c, ...a, ...tip);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(tris, 3));
  g.computeVertexNormals();
  return g;
}
let BONE: THREE.BufferGeometry | null = null;
let BALL: THREE.BufferGeometry | null = null;

/** 骨点画多大：选中的放大一档，且至少和父骨点一样大（与父关节重合的辅助关节自己的半径是 0，选中了也要看得见）。
 * 实体与线框两种画法都按它画。 */
const ballRadius = (sizes: { joint: Float32Array }, parents: number[], i: number, highlight: number): number =>
  i === highlight ? Math.max(sizes.joint[i], parents[i] >= 0 ? sizes.joint[parents[i]] : 0) * 1.5 : sizes.joint[i];

/** 一副骨架的尺度：骨长的第 REF_PERCENTILE 百分位（辅助骨多、长短悬殊时中位数不可靠）。骨点与骨按它定大小
 * （boneSizes），它为 0 的姿势（全在原点、从 0 放大的开头）就是退化的：判据只有这一个。 */
function boneLengths(points: ArrayLike<number>, parents: number[]): { len: Float32Array; scale: number } {
  const n = parents.length;
  const len = new Float32Array(n); // 到父骨点的长度（根为 0）
  const lengths: number[] = [];
  for (let i = 0; i < n; i++) {
    const p = parents[i];
    if (p < 0) continue;
    len[i] = Math.hypot(points[i * 3] - points[p * 3], points[i * 3 + 1] - points[p * 3 + 1], points[i * 3 + 2] - points[p * 3 + 2]);
    lengths.push(len[i]);
  }
  lengths.sort((a, b) => a - b);
  return { len, scale: lengths.length ? lengths[Math.min(lengths.length - 1, Math.floor(lengths.length * REF_PERCENTILE))] : 1 };
}
export const boneScale = (points: ArrayLike<number>, parents: number[]): number => boneLengths(points, parents).scale;

/** 骨点半径（0 = 不画：与父关节重合的辅助关节）与每根骨（子关节下标 → 底面半宽，0 = 不画这根骨）：规则见上面的常数。
 * `points` 每个关节 xyz。 */
export function boneSizes(points: ArrayLike<number>, parents: number[], o: Pick<ViewOptions, "boneWidth" | "jointSize" | "jointAdaptive">): { joint: Float32Array; bone: Float32Array } {
  const n = parents.length;
  const { len, scale } = boneLengths(points, parents);
  const trivial = scale * TRIVIAL_OF_SCALE;
  const shortest = new Float32Array(n).fill(Infinity);
  const longest = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = parents[i];
    if (p < 0 || len[i] <= trivial) continue;
    shortest[i] = Math.min(shortest[i], len[i]);
    shortest[p] = Math.min(shortest[p], len[i]);
    longest[i] = Math.max(longest[i], len[i]);
    longest[p] = Math.max(longest[p], len[i]);
  }
  const joint = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (parents[i] >= 0 && len[i] <= trivial) continue; // 与父关节重合：父关节的骨点就是它的，不另画一层
    const base = o.jointAdaptive && Number.isFinite(shortest[i])
      ? Math.min(shortest[i] * JOINT_OF_SHORTEST, Math.min(longest[i], scale) * JOINT_OF_LONGEST)
      : scale * JOINT_OF_SCALE;
    joint[i] = Math.max(1e-6, base * o.jointSize);
  }
  const bone = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (parents[i] >= 0 && len[i] > trivial) bone[i] = Math.min(len[i], scale) * BONE_OF_LENGTH * o.boneWidth;
  }
  return { joint, bone };
}

const UP = new THREE.Vector3(0, 1, 0);
const WHITE = new THREE.Color(0xffffff);
const RING_SEGMENTS = 16;
/** 八面体的 12 条棱：尾 → 底面四角、底面一圈、底面四角 → 尖（单位几何同 boneGeometry）。 */
const BONE_EDGES: [number[], number[]][] = (() => {
  const tail = [0, 0, 0], tip = [0, 1, 0];
  const b = [[1, 0.125, 0], [0, 0.125, 1], [-1, 0.125, 0], [0, 0.125, -1]];
  return b.flatMap((a, k): [number[], number[]][] => [[tail, a], [a, b[(k + 1) % 4]], [a, tip]]);
})();

/** 单位圆上一圈的 RING_SEGMENTS + 1 个点（cos, sin），骨点圆环按半径缩放平移它，不每次算三角函数。 */
const RING = Array.from({ length: RING_SEGMENTS + 1 }, (_, t) => [Math.cos((t / RING_SEGMENTS) * Math.PI * 2), Math.sin((t / RING_SEGMENTS) * Math.PI * 2)]);

type Wire = { segments: Float32Array; colors: Float32Array };

/** 线框骨架的线段：每根骨 12 条棱，每个骨点三个正交圆环（Maya 的骨点画法），并附每段两端的颜色。先数出线段数，
 * 写进 `into`（长度对得上就用它，否则新建一份）：大骨架每帧重算几十万个数，不经 JS 数组，也不每帧新分配。 */
function wireSegments(points: ArrayLike<number>, parents: number[], sizes: { joint: Float32Array; bone: Float32Array },
                      bone: THREE.Color, ball: THREE.Color, highlight: number, hi: THREE.Color, into: Wire | null,
                      tints: Tints | null = null): Wire {
  const n = parents.length;
  const radius = (i: number) => ballRadius(sizes, parents, i, highlight);
  let verts = 0;
  for (let i = 0; i < n; i++) {
    if (parents[i] >= 0 && sizes.bone[i] > 0) verts += BONE_EDGES.length * 2;
    if (radius(i) > 0) verts += 3 * RING_SEGMENTS * 2;
  }
  const fits = into && into.segments.length === verts * 3;
  const out = fits ? into.segments : new Float32Array(verts * 3), col = fits ? into.colors : new Float32Array(verts * 3);
  let w = 0;
  const put = (x: number, y: number, z: number, c: THREE.Color) => {
    out[w] = x; out[w + 1] = y; out[w + 2] = z;
    col[w] = c.r; col[w + 1] = c.g; col[w + 2] = c.b;
    w += 3;
  };
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), d = new THREE.Vector3(), at = new THREE.Vector3(), sc = new THREE.Vector3(), v = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    const p = parents[i];
    if (p < 0 || !(sizes.bone[i] > 0)) continue;
    at.set(points[p * 3], points[p * 3 + 1], points[p * 3 + 2]);
    d.set(points[i * 3] - at.x, points[i * 3 + 1] - at.y, points[i * 3 + 2] - at.z);
    const l = d.length();
    q.setFromUnitVectors(UP, l > 1e-9 ? d.normalize() : UP);
    m.compose(at, q, sc.set(sizes.bone[i], l, sizes.bone[i]));
    const c = tints?.bone[i] ?? bone;
    for (const [a, b] of BONE_EDGES) {
      v.set(a[0], a[1], a[2]).applyMatrix4(m); put(v.x, v.y, v.z, c);
      v.set(b[0], b[1], b[2]).applyMatrix4(m); put(v.x, v.y, v.z, c);
    }
  }
  for (let i = 0; i < n; i++) {
    const r = radius(i);
    if (!(r > 0)) continue;
    const c = i === highlight ? hi : tints?.ball[i] ?? ball;
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    for (let axis = 0; axis < 3; axis++) {
      for (let k = 0; k < RING_SEGMENTS; k++) {
        for (const t of [k, k + 1]) {
          const ca = RING[t][0] * r, sa = RING[t][1] * r;
          if (axis === 0) put(x, y + ca, z + sa, c);
          else if (axis === 1) put(x + ca, y, z + sa, c);
          else put(x + ca, y + sa, z, c);
        }
      }
    }
  }
  return { segments: out, colors: col };
}

/** 一副骨架：父到子一根八面体骨（只有父子都在的关节对才有骨，不画「根到原点」），每个关节一个骨点。「骨骼显示」
 * 实体 = 八面体 + 小球，各一个 InstancedMesh，关节动了只改实例矩阵；线框 = 八面体的棱 + Maya 那种三个正交圆环的
 * 骨点，一份 FatLines。`overlay`：画在一切之上（骨架总叠在模型上，透过相机看时也是）——实体时靠所有这类骨架之前
 * 清一次深度做到（overlayClear）：骨架内部与骨架之间仍按深度画，八面体才是凸的；关了深度画就会前后颠倒、看着像凹进去。`highlight`：这个关节画成
 * 选中色。骨点比骨稍亮一档，分得清层级。`names` 给了且「骨点名」开着时，在每个骨点旁写名字（JointLabels）。 */
export function BoneFigure({ points, sizedBy, parents, names, color, o, opacity = 0.95, overlay = true, highlight = -1, highlightColor = "#0a84ff", selected, tints, shown }: {
  points: ArrayLike<number>; parents: number[]; names?: string[]; color: string; o: ViewOptions;
  sizedBy?: ArrayLike<number>; // 按哪一份关节位置定骨点与骨的大小（动画角色：它的第一帧，播放时不每帧重算）；缺省按 points
  opacity?: number; overlay?: boolean; highlight?: number; highlightColor?: string;
  selected?: number[]; // 这些关节的骨点按 highlightColor 上色
  // 逐关节的颜色：这个关节的骨点和通到它的那根骨用它（空：用 color）。双骨架编辑按状态上色（已配对 / 未配对）
  tints?: readonly (string | null | undefined)[];
  // 只画这些关节（true）的骨点和通到它们的骨，其余不画（缺省全画）。同一副骨架按状态分几份画（不同的透明度）、眼睛隐藏
  shown?: readonly boolean[];
}) {
  const from = sizedBy ?? points;
  const all = useMemo(() => boneSizes(from, parents, o), [from, parents, o.boneWidth, o.jointSize, o.jointAdaptive]); // eslint-disable-line react-hooks/exhaustive-deps
  const sizes = useMemo(() => {
    if (!shown) return all;
    const joint = all.joint.slice(), bone = all.bone.slice();
    for (let i = 0; i < parents.length; i++) if (!shown[i]) joint[i] = bone[i] = 0;
    return { joint, bone };
  }, [all, shown, parents.length]);
  const colors = useMemo((): Tints | null => {
    if (!tints) return null;
    const bone = parents.map((_, i) => (tints[i] ? new THREE.Color(tints[i]!) : null));
    return { bone, ball: bone.map((c) => c && c.clone().offsetHSL(0, 0, 0.15)) };
  }, [tints, parents]);
  const labelNames = names && shown ? names.map((n, i) => (shown[i] ? n : "")) : names;
  const labels = labelNames && o.jointNames ? <JointLabels points={points} names={labelNames} sizes={sizes.joint} px={o.jointNamePx} color={color} /> : null;
  if (o.boneStyle === "wire") return <><WireFigure points={points} parents={parents} sizes={sizes} color={color} o={o} opacity={opacity} overlay={overlay} highlight={highlight} highlightColor={highlightColor} selected={selected} tints={colors} />{labels}</>;
  return <><SolidFigure points={points} parents={parents} sizes={sizes} color={color} opacity={opacity} overlay={overlay} highlight={highlight} highlightColor={highlightColor} selected={selected} tints={colors} />{labels}</>;
}

/** 逐关节的颜色（BoneFigure `tints` 换成 three 的颜色）：骨用 `bone[i]`，骨点亮一档用 `ball[i]`；null 用整副的颜色。 */
type Tints = { bone: (THREE.Color | null)[]; ball: (THREE.Color | null)[] };

function WireFigure({ points, parents, sizes, color, o, opacity, overlay, highlight, highlightColor, selected, tints }: {
  points: ArrayLike<number>; parents: number[]; sizes: { joint: Float32Array; bone: Float32Array }; color: string; o: ViewOptions;
  opacity: number; overlay: boolean; highlight: number; highlightColor: string; selected?: number[]; tints: Tints | null;
}) {
  // 两份缓冲轮流写：每次给 FatLines 的是另一份（它按引用认新数据，再拷进自己的缓冲），不每帧新分配
  const bufs = useRef<[Wire | null, Wire | null]>([null, null]);
  const turn = useRef(0);
  const { segments, colors } = useMemo(() => {
    const base = new THREE.Color(color);
    const k = (turn.current ^= 1);
    const hi = selected && selected.length ? selected[0] : highlight; // wire 模式高亮选中的第一个（多选少见）
    return (bufs.current[k] = wireSegments(points, parents, sizes, base, base.clone().offsetHSL(0, 0, 0.15), hi, new THREE.Color(highlightColor), bufs.current[k], tints));
  }, [points, parents, sizes, color, highlight, highlightColor, selected, tints]);
  if (!segments.length) return null;
  return <FatLines segments={segments} colors={colors} color={color} width={o.lineWidth} opacity={opacity} overlay={overlay} renderOrder={overlay ? ORDER.overlayBones : undefined} />;
}

function SolidFigure({ points, parents, sizes, color, opacity, overlay, highlight, highlightColor, selected, tints }: {
  points: ArrayLike<number>; parents: number[]; sizes: { joint: Float32Array; bone: Float32Array }; color: string;
  opacity: number; overlay: boolean; highlight: number; highlightColor: string; selected?: number[]; tints: Tints | null;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const n = parents.length;
  const bones = useMemo(() => parents.flatMap((p, i) => (p >= 0 ? [i] : [])), [parents]);
  const boneMesh = useMemo(() => {
    BONE ??= boneGeometry();
    const m = new THREE.InstancedMesh(BONE, new THREE.MeshLambertMaterial({ flatShading: true }), Math.max(1, bones.length));
    m.frustumCulled = false;
    return m;
  }, [bones.length]);
  const ballMesh = useMemo(() => {
    BALL ??= new THREE.SphereGeometry(1, 12, 8);
    const m = new THREE.InstancedMesh(BALL, new THREE.MeshLambertMaterial(), Math.max(1, n));
    m.frustumCulled = false;
    return m;
  }, [n]);
  const clear = useMemo(overlayClear, []);
  useEffect(() => () => {
    (clear.material as THREE.Material).dispose();
    clear.geometry.dispose();
  }, [clear]);
  useEffect(() => () => {
    boneMesh.material instanceof THREE.Material && boneMesh.material.dispose();
    ballMesh.material instanceof THREE.Material && ballMesh.material.dispose();
    boneMesh.dispose();
    ballMesh.dispose();
  }, [boneMesh, ballMesh]);
  useLayoutEffect(() => {
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), d = new THREE.Vector3(), at = new THREE.Vector3(), sc = new THREE.Vector3();
    bones.forEach((i, k) => {
      const p = parents[i];
      at.set(points[p * 3], points[p * 3 + 1], points[p * 3 + 2]);
      d.set(points[i * 3] - at.x, points[i * 3 + 1] - at.y, points[i * 3 + 2] - at.z);
      const l = sizes.bone[i] > 0 ? d.length() : 0; // 重合的辅助关节：不画骨（长度 0）
      q.setFromUnitVectors(UP, l > 1e-9 ? d.normalize() : UP);
      sc.set(sizes.bone[i], l, sizes.bone[i]);
      boneMesh.setMatrixAt(k, m.compose(at, q, sc));
    });
    boneMesh.count = bones.length;
    boneMesh.instanceMatrix.needsUpdate = true;
    for (let i = 0; i < n; i++) {
      at.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
      const r = ballRadius(sizes, parents, i, highlight) || 0;
      ballMesh.setMatrixAt(i, m.compose(at, q.identity(), sc.set(r, r, r)));
    }
    ballMesh.count = n;
    ballMesh.instanceMatrix.needsUpdate = true;
    invalidate();
  }, [points, parents, bones, n, boneMesh, ballMesh, sizes, highlight, invalidate]);
  useLayoutEffect(() => {
    const base = new THREE.Color(color);
    const bright = base.clone().offsetHSL(0, 0, 0.15);
    // 逐关节上色时骨的颜色也在 instanceColor 里（材质色为白，同骨点）；不上色时 instanceColor 回到白，颜色只在材质上
    for (const [mesh, c, order] of [[boneMesh, tints ? WHITE : base, ORDER.overlayBones], [ballMesh, WHITE, ORDER.overlayBalls]] as const) {
      const mat = mesh.material as THREE.MeshLambertMaterial;
      mat.color.copy(c);
      mat.transparent = overlay || opacity < 1; // 叠在上面的一律在透明那一批里画：排在清深度（overlayClear）之后
      mat.opacity = opacity;
      mat.depthTest = true;
      mat.depthWrite = true;
      mesh.renderOrder = overlay ? order : 0;
    }
    // 骨点的颜色全在 instanceColor 里、材质色为白：three 把两者相乘，材质色带骨色时选中色会被乘暗
    const hi = new THREE.Color(highlightColor);
    const sel = selected && selected.length ? new Set(selected) : null;
    for (let i = 0; i < n; i++) ballMesh.setColorAt(i, (sel ? sel.has(i) : i === highlight) ? hi : tints?.ball[i] ?? bright);
    if (ballMesh.instanceColor) ballMesh.instanceColor.needsUpdate = true;
    if (tints || boneMesh.instanceColor) {
      bones.forEach((i, k) => boneMesh.setColorAt(k, tints ? tints.bone[i] ?? base : WHITE));
      if (boneMesh.instanceColor) boneMesh.instanceColor.needsUpdate = true;
    }
    invalidate();
  }, [color, opacity, overlay, highlight, highlightColor, selected, tints, boneMesh, ballMesh, bones, n, invalidate]);
  return (
    <>
      {overlay && <primitive object={clear} />}
      <primitive object={boneMesh} />
      <primitive object={ballMesh} />
    </>
  );
}

/** 叠在一切之上的骨架的清深度：场景里所有这类骨架排在同一段顺序里——先清深度（9），再画所有骨（10），再画所有骨点
 * （11）。所以骨架之间、骨与骨点之间都按深度画（八面体是凸的，别人的骨挡得住我的骨点），而模型挡不住骨架。每副骨架
 * 带一个（一个三维物体只能挂在一处），几个都排在 9、都在任何骨之前，清几次都等于清一次。它什么都不画。 */
function overlayClear(): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, colorWrite: false }));
  m.frustumCulled = false;
  m.renderOrder = ORDER.overlayClear;
  m.onBeforeRender = (renderer) => renderer.clearDepth();
  return m;
}

/** 骨点名：画布上方一层 HTML（drei 的 Html 挂在画布旁边，一层放全部名字，不是一个关节一层），每帧把关节投影到屏幕、
 * 把名字挪到骨点右侧（在骨点半径之外），投影用舞台那一份（view/stageState.ts project）；在镜头后面的不显示。直接改 DOM 的 transform，不经 React 重绘
 * （几百个关节每帧写一次样式没有开销）。这一层的锚点放在骨架中心：中心跑到镜头背后时整层隐藏。 */
function JointLabels({ points, names, sizes, px, color }: { points: ArrayLike<number>; names: string[]; sizes: Float32Array; px: number; color: string }) {
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const box = useRef<HTMLDivElement>(null);
  const centre = useMemo(() => {
    const c = new THREE.Vector3();
    const n = names.length;
    for (let i = 0; i < n; i++) c.x += points[i * 3], c.y += points[i * 3 + 1], c.z += points[i * 3 + 2];
    return n ? c.divideScalar(n) : c;
  }, [points, names.length]);
  useFrame(() => {
    const el = box.current;
    if (!el) return;
    const v = new THREE.Vector3(), right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    const view = { camera, size };
    const kids = el.children;
    for (let i = 0; i < names.length && i < kids.length; i++) {
      const span = kids[i] as HTMLElement;
      const at = project(v.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]), view);
      if (!at) { span.style.display = "none"; continue; }
      // 骨点半径在屏幕上有多宽：半径沿相机右方向偏移后再投影，两点的距离就是
      const edge = project(v.addScaledVector(right, sizes[i] || 0), view);
      const x = at.x, y = at.y;
      const dx = Math.max(4, (edge ? Math.abs(edge.x - x) : 0) + 3);
      span.style.display = "";
      span.style.transform = `translate(${(x + dx).toFixed(1)}px, ${y.toFixed(1)}px) translateY(-50%)`;
    }
  });
  return (
    <Html position={centre} calculatePosition={() => [0, 0]} zIndexRange={[2, 2]} style={{ pointerEvents: "none" }}>
      <div ref={box} className="joint-labels" style={{ fontSize: `${px}px`, color }}>
        {names.map((name, i) => <span key={i}>{name}</span>)}
      </div>
    </Html>
  );
}

/** 某一帧的骨架骨骼：从每个关节画到它的父关节。 */
function Bones({ c, frame, o, pickKey }: { c: CharacterData; frame: number; o: ViewOptions; pickKey: string }) {
  const i = sampleAt(c.ref.frames, frame);
  const lang = useLang((s) => s.lang); // the pick label is a word
  // 骨段只给拾取与框显用：要时才算（播放时每帧不另分配一份）
  const pickable = useMemo(() => {
    let made: Float32Array | null = null;
    const segs = () => (made ??= boneSegments(c.anim, c.ref.parents, i));
    return {
      label: t("ui.view.skeleton_of", { name: c.ref.name }),
      order: ORDER.overlayBones, // drawn over everything (BoneFigure overlay): picked over the mesh around it
      bounds: () => { const segments = segs(); return segments.length ? new THREE.Box3().setFromArray(segments) : null; },
      hit: (p: PickRay) => segmentsHit(segs(), p), // anywhere along a bone, not only at its joints
    };
  }, [c, i, lang]);
  usePickable(pickKey, pickable);
  const points = useMemo(() => jointPoints(c.anim, c.ref.parents.length, i), [c, i]);
  // 骨长在动画里不变：大小按一个不退化的姿势定一次（boneScale > 0，与 boneSizes 同一个判据），播放时只挪位置。
  // 不扫全部采样：先看第 0 个，退化（前导的全在原点的帧、从 0 放大的开头）就用画到的第一个不退化的帧，还没有就按当前帧
  const sized = useRef<{ c: CharacterData; points: Float32Array } | null>(null);
  if (sized.current?.c !== c) {
    const zero = jointPoints(c.anim, c.ref.parents.length, 0);
    sized.current = boneScale(zero, c.ref.parents) > 0 ? { c, points: zero } : null;
  }
  if (!sized.current && boneScale(points, c.ref.parents) > 0) sized.current = { c, points };
  const first = sized.current?.points ?? null;
  if (!points.length) return null;
  return <BoneFigure points={points} sizedBy={first ?? undefined} parents={c.ref.parents} names={c.ref.joints} color={o.boneColor} o={o} />;
}

/** 每个关节在这个采样上的世界位置（xyz）。 */
function jointPoints(anim: Float32Array, joints: number, sample: number): Float32Array {
  if (sample < 0) return new Float32Array(0); // no sample (sampleAt -1): no joints
  const out = new Float32Array(joints * 3);
  for (let j = 0; j < joints; j++) for (let c = 0; c < 3; c++) out[j * 3 + c] = anim[(sample * joints + j) * 12 + c * 4 + 3];
  return out;
}
