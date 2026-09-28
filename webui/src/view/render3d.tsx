import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { GizmoHelper, GizmoViewport, Grid } from "@react-three/drei";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { FXAAPass } from "three/examples/jsm/postprocessing/FXAAPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { SMAAPass } from "three/examples/jsm/postprocessing/SMAAPass.js";
import type { ViewOptions } from "../model/viewOptions";

/** How the 3D stage becomes a picture: anti-aliasing (none, MSAA, FXAA or SMAA), the background, the lights, the
 * ground grid and the axes. The picture is always the screen's own resolution. */

/** Draws every frame: straight to the screen, or through the anti-aliasing passes. */
export function Pipeline({ o }: { o: ViewOptions }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const size = useThree((s) => s.size);
  const dpr = useThree((s) => s.viewport.dpr); // the canvas's dpr: the screen's own pixel ratio

  const passes = useMemo(() => {
    if (o.antialias === "off") return null;
    const target = new THREE.WebGLRenderTarget(2, 2, { type: THREE.HalfFloatType, samples: o.antialias === "msaa" ? 4 : 0 });
    const composer = new EffectComposer(gl, target);
    const render = new RenderPass(scene, new THREE.PerspectiveCamera());
    composer.addPass(render);
    if (o.antialias === "smaa") composer.addPass(new SMAAPass());
    composer.addPass(new OutputPass());
    if (o.antialias === "fxaa") composer.addPass(new FXAAPass()); // on the finished (display) picture
    return { composer, render };
  }, [o.antialias, gl, scene]);
  useEffect(() => {
    if (!passes) return;
    passes.composer.setPixelRatio(dpr);
    passes.composer.setSize(size.width, size.height);
  }, [passes, dpr, size.width, size.height]);
  // the composer frees its own two targets only; each pass holds GPU resources of its own (SMAA's edge and weight
  // targets and textures, the shader materials), freed with it when the anti-aliasing changes or the view closes
  useEffect(() => () => {
    if (!passes) return;
    for (const pass of passes.composer.passes) pass.dispose();
    passes.composer.dispose();
  }, [passes]);

  useFrame((state) => {
    if (passes) {
      passes.render.camera = state.camera;
      passes.composer.render();
    } else {
      gl.render(scene, state.camera);
    }
  }, 1);
  return null;
}

/** A vertical gradient from a lighter top to the chosen colour at the bottom (Houdini's viewport background). */
function gradient(color: string): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = 2;
  c.height = 256;
  const g = c.getContext("2d")!;
  const top = new THREE.Color(color).lerp(new THREE.Color("#454a55"), 0.55);
  const grad = g.createLinearGradient(0, 0, 0, c.height);
  grad.addColorStop(0, `#${top.getHexString()}`);
  grad.addColorStop(1, color);
  g.fillStyle = grad;
  g.fillRect(0, 0, c.width, c.height);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Behind the scene: a solid colour or a gradient (seen through a camera, the plate is an image plane in its gate). */
export function Background({ o }: { o: ViewOptions }) {
  const scene = useThree((s) => s.scene);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    let tex: THREE.Texture | null = null;
    if (o.background === "solid") scene.background = new THREE.Color(o.backgroundColor);
    else scene.background = tex = gradient(o.backgroundColor);
    invalidate();
    return () => {
      tex?.dispose();
      scene.background = null;
    };
  }, [o.background, o.backgroundColor, scene, invalidate]);
  return null;
}

/** A camera's plate behind everything, filling exactly its gate (as Maya's image plane): a picture at a fixed distance
 * in front of the camera, as wide as the camera sees at that distance, drawn first and never in front of anything. */
/** The plate as an image plane filling the camera's gate. `exact`: never filtered (the 2D stage's rule: nearest
 * texels at any zoom, so at 1:1 every plate pixel is one screen pixel, and edges can be judged). */
/** 背板：所绘制的图像由调用方提供（一张已解码的图），本组件只负责贴图。
 *
 * 不接收地址自行获取：否则背板不在取帧账本中（transfer/frames.ts 的 windows），播放时询问「下一帧是否已到达」
 * 总是得到肯定答复，低速网络下背板跟不上而出现闪烁，且没有窗口、不会取消、不受预算约束，开始播放即请求整段。
 * 背板与二维舞台使用同一条路径（useFrame → windows → 预算 + 取消），账本中有它，播放便会等待它。 */
export function ImagePlane({ image, pose, fovV, aspect, exact = false }: { image: ImageBitmap | null; pose: THREE.Matrix4; fovV: number; aspect: number; exact?: boolean }) {
  const invalidate = useThree((s) => s.invalidate);
  const mesh = useMemo(() => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ depthTest: false, depthWrite: false, toneMapped: false }));
    m.renderOrder = -10;
    m.frustumCulled = false;
    m.matrixAutoUpdate = false;
    return m;
  }, []);
  useEffect(
    () => () => {
      mesh.geometry.dispose();
      mesh.material.dispose();
    },
    [mesh],
  );
  useEffect(() => {
    if (!image) return;
    const tex = new THREE.Texture(image);
    tex.colorSpace = THREE.SRGBColorSpace;
    // ImageBitmap 须自行翻转 V。three.js（0.186，WebGLTextures.js）在纹理为 ImageBitmap 时
    // 完全跳过 `UNPACK_FLIP_Y_WEBGL`，因此 `flipY` 设置不生效。若不自行翻转，背板将上下颠倒。
    tex.flipY = false;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.repeat.set(1, -1);
    tex.offset.set(0, 1);
    if (exact) {
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.NearestFilter;
      tex.generateMipmaps = false;
    }
    tex.needsUpdate = true;
    const old = mesh.material.map;
    mesh.material.map = tex;
    mesh.material.needsUpdate = true;
    if (old && old !== tex) old.dispose();
    invalidate();
    // 位图本身由缓存管理（transfer/cache.ts 负责关闭），此处只释放本组件创建的纹理
    return () => void tex.dispose();
  }, [image, mesh, invalidate, exact]);
  useLayoutEffect(() => {
    const distance = 10;
    const h = 2 * distance * Math.tan((fovV * Math.PI) / 360);
    mesh.matrix.copy(pose).multiply(new THREE.Matrix4().makeTranslation(0, 0, -distance)).multiply(new THREE.Matrix4().makeScale(h * aspect, h, 1));
    mesh.matrixWorldNeedsUpdate = true;
    invalidate();
  }, [mesh, pose, fovV, aspect, invalidate]);
  return <primitive object={mesh} />;
}

/** The lights: a light that follows the camera (headlight), or a fixed three-light rig; exposure in stops. */
export function Lights({ o }: { o: ViewOptions }) {
  const k = Math.pow(2, o.exposure);
  const head = useRef<THREE.DirectionalLight>(null);
  const forward = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera }) => {
    const l = head.current;
    if (!l) return;
    camera.getWorldDirection(forward);
    l.position.copy(camera.position);
    l.target.position.copy(camera.position).add(forward);
    l.target.updateMatrixWorld();
  }, -2);
  if (o.lighting === "headlight")
    return (
      <>
        <ambientLight intensity={0.35 * k} />
        <directionalLight ref={head} intensity={2.4 * k} />
      </>
    );
  return (
    <>
      <hemisphereLight args={["#dfe6ff", "#1a1a1d", 0.7 * k]} />
      <directionalLight position={[260, 480, 220]} intensity={1.7 * k} />
      <directionalLight position={[-320, 180, -260]} intensity={0.45 * k} color="#9fb4ff" />
    </>
  );
}

/** The ground grid: lines every `gridSpacing`, a stronger one every tenth, unbounded or `gridSize` across. Drawn after
 * everything else without hiding what is below the ground (points below it show through, with the lines over them). */
export function GroundGrid({ o, through }: { o: ViewOptions; through: boolean }) {
  const dpr = useThree((s) => s.viewport.dpr);
  const ref = useRef<THREE.Mesh>(null);
  const finite = o.gridSize > 0;
  useEffect(() => {
    const m = ref.current?.material as THREE.Material | undefined;
    if (m) m.depthWrite = false;
  }, [finite, o.grid]);
  if (!o.grid) return null;
  // the grid lines follow the line width moderately: a large line width must not turn the ground into a checkerboard
  const section = 1 + Math.max(0, o.lineWidth - 1) * 0.4;
  return (
    <Grid
      ref={ref}
      renderOrder={10}
      key={finite ? "finite" : "infinite"}
      infiniteGrid={!finite}
      args={finite ? [o.gridSize, o.gridSize] : undefined}
      cellSize={o.gridSpacing}
      sectionSize={o.gridSpacing * 10}
      cellThickness={section * 0.55 * dpr}
      sectionThickness={section * dpr}
      cellColor={through ? "#9aa0a6" : "#2c2c32"}
      sectionColor={through ? "#e6e8ea" : "#4a4a52"}
      fadeDistance={finite ? o.gridSize * 4 : o.gridSpacing * (through ? 250 : 400)}
      fadeStrength={finite ? 0 : 1.6}
      followCamera={false}
    />
  );
}

/** The world axes in a corner (click one to look along it), `size` pixels across. */
export function Axes({ size }: { size: number }) {
  return (
    <GizmoHelper alignment="bottom-left" margin={[size * 0.78, size * 1.2]} renderPriority={2}>
      <GizmoViewport axisColors={["#ff4d5e", "#46d27a", "#3d8bff"]} labelColor="#0c0c0e" axisHeadScale={0.9} scale={size / 2} />
    </GizmoHelper>
  );
}
