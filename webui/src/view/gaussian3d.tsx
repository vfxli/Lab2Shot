/** Native anisotropic splats: project covariance, evaluate SH, sort back to front.
 * Shares scene samples and the stage's camera; no external renderer/runtime. */
import { useEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { CloudData } from "./sceneData";
import { heldSample } from "../model/viewFormat";
import { sampleAt } from "../model/timelineMath";
import { matrixAt } from "./matrix3d";
import { project, usePickable, type PickRay } from "./stageState";
import { useStageNotes } from "../state/viewer";
import { msg, textOf } from "../messages/message";
import { tipOf } from "../platform/tips";

const vertex = `
precision highp float;
attribute float splatId;
uniform sampler2D uData;
uniform sampler2D uSH;
uniform float uWidth;
uniform float uCoeffs;
uniform vec2 uViewport;
uniform vec3 uEye;
varying vec2 vEllipse;
varying vec4 vColor;
vec4 data(float n) { return texture2D(uData, (vec2(mod(n,uWidth),floor(n/uWidth))+.5)/vec2(uWidth,float(textureSize(uData,0).y))); }
vec3 sh(float id,float k) { float n=id*uCoeffs+k; return texture2D(uSH,(vec2(mod(n,uWidth),floor(n/uWidth))+.5)/vec2(uWidth,float(textureSize(uSH,0).y))).rgb; }
void main() {
  float id=splatId;
  vec4 p=data(id*3.);
  vec4 a=data(id*3.+1.);
  vec4 b=data(id*3.+2.);
  vec4 view=modelViewMatrix*vec4(p.xyz,1.);
  vec4 clip=projectionMatrix*view;
  if (view.z>=-0.001 && projectionMatrix[3][3]==0.) { gl_Position=vec4(2.,2.,2.,1.); vColor=vec4(0.); return; }
  mat3 cov=mat3(a.x,a.y,a.z,a.y,a.w,b.x,a.z,b.x,b.y);
  mat3 mv=mat3(modelViewMatrix);
  cov=mv*cov*transpose(mv);
  float z=max(-view.z,.001);
  float fx=projectionMatrix[0][0]*uViewport.x*.5;
  float fy=projectionMatrix[1][1]*uViewport.y*.5;
  vec3 jx=projectionMatrix[3][3]==0.?vec3(fx/z,0.,fx*view.x/(z*z)):vec3(fx,0.,0.);
  vec3 jy=projectionMatrix[3][3]==0.?vec3(0.,fy/z,fy*view.y/(z*z)):vec3(0.,fy,0.);
  float xx=dot(jx,cov*jx)+.3;
  float xy=dot(jx,cov*jy);
  float yy=dot(jy,cov*jy)+.3;
  float mid=.5*(xx+yy), gap=length(vec2(.5*(xx-yy),xy));
  float l1=max(mid+gap,.01), l2=max(mid-gap,.01);
  vec2 axis=abs(xy)>1.e-6?normalize(vec2(xy,l1-xx)):(xx>=yy?vec2(1.,0.):vec2(0.,1.));
  vec2 offset=3.*(position.x*sqrt(l1)*axis+position.y*sqrt(l2)*vec2(-axis.y,axis.x));
  clip.xy+=offset/uViewport*2.*clip.w;
  gl_Position=clip;
  vEllipse=position.xy*3.;
  vec3 dir=normalize(p.xyz-uEye); float x=dir.x,y=dir.y,zs=dir.z;
  vec3 color=.28209479177387814*sh(id,0.)+vec3(.5);
  if(uCoeffs>1.) color+=-.4886025119029199*y*sh(id,1.)+.4886025119029199*zs*sh(id,2.)-.4886025119029199*x*sh(id,3.);
  if(uCoeffs>4.) color+=1.0925484305920792*x*y*sh(id,4.)-1.0925484305920792*y*zs*sh(id,5.)+.31539156525252005*(2.*zs*zs-x*x-y*y)*sh(id,6.)-1.0925484305920792*x*zs*sh(id,7.)+.5462742152960396*(x*x-y*y)*sh(id,8.);
  if(uCoeffs>9.) color+=-.5900435899266435*y*(3.*x*x-y*y)*sh(id,9.)+2.890611442640554*x*y*zs*sh(id,10.)-.4570457994644658*y*(4.*zs*zs-x*x-y*y)*sh(id,11.)+.3731763325901154*zs*(2.*zs*zs-3.*x*x-3.*y*y)*sh(id,12.)-.4570457994644658*x*(4.*zs*zs-x*x-y*y)*sh(id,13.)+1.445305721320277*zs*(x*x-y*y)*sh(id,14.)-.5900435899266435*x*(x*x-3.*y*y)*sh(id,15.);
  vColor=vec4(max(color,vec3(0.)),p.w);
}`;
const fragment = `
precision highp float;
varying vec2 vEllipse;
varying vec4 vColor;
void main() {
 float power=-.5*dot(vEllipse,vEllipse);
 float alpha=min(.99,vColor.a*exp(power));
 if(alpha<1./255.) discard;
 gl_FragColor=vec4(vColor.rgb,alpha);
 #include <tonemapping_fragment>
 #include <colorspace_fragment>
}`;

function texture(values: Float32Array, width: number): THREE.DataTexture {
  const height = Math.max(1, Math.ceil(values.length / 4 / width));
  const padded = new Float32Array(width * height * 4);
  padded.set(values);
  const t = new THREE.DataTexture(padded, width, height, THREE.RGBAFormat, THREE.FloatType);
  t.needsUpdate = true;
  return t;
}

export function Gaussian({ src, frame, pickKey, version }: { src: CloudData; frame: number; pickKey: string; version: number }) {
  const { gl, invalidate } = useThree();
  const i = sampleAt(src.frames, frame);
  const sample = src.still ?? heldSample(src.samples, i)?.sample ?? null;
  const world = useMemo(() => matrixAt(src.world, i), [src.world, i]);
  const picking = useMemo(() => {
    if (!sample?.points || !sample.covariance) return null;
    const box = new THREE.Box3();
    const p = new THREE.Vector3(), extent = new THREE.Vector3();
    for (let k = 0; k < sample.count; k++) {
      p.fromArray(sample.points, k * 3);
      extent.set(3 * Math.sqrt(sample.covariance[k * 6]), 3 * Math.sqrt(sample.covariance[k * 6 + 3]), 3 * Math.sqrt(sample.covariance[k * 6 + 5]));
      box.expandByPoint(p.clone().sub(extent)); box.expandByPoint(p.add(extent));
    }
    box.applyMatrix4(world);
    return { label: src.name, bounds: () => box, hit: (ray: PickRay) => {
      let best: number | null = null;
      for (let k = 0; k < sample.count; k += Math.max(1, Math.ceil(sample.count / 200_000))) {
        const at = project(p.fromArray(sample.points!, k * 3).applyMatrix4(world), ray);
        if (at && Math.hypot(at.x - ray.px.x, at.y - ray.px.y) < 8 && (best === null || at.depth < best)) best = at.depth;
      }
      return best;
    } };
  }, [sample, world, src.name]);
  usePickable(pickKey, picking);
  const width = Math.min(2048, gl.capabilities.maxTextureSize);
  // 超出显卡单张纹理的容量：不画它，说明原因（进统一通知区），舞台其余照常
  const tooBig = !!sample && Math.ceil(sample.count * Math.max(3, src.ref.sh_coefficients ?? 1) / width) > gl.capabilities.maxTextureSize;
  const note = useStageNotes((n) => n.put);
  useEffect(() => {
    if (!tooBig) return;
    // 「有一份内容画不出来」（model/viewNotices.ts absent）：只在自己说了的时候说、走时收回
    note("absent", { text: textOf(msg("N-VIEW-GAUSSIANTOOBIG", { name: src.name })),
                     tip: tipOf("error", textOf(msg("I-VIEW-GAUSSIANTOOBIGWHY", { count: sample?.count ?? 0 }))) });
    return () => note("absent", null);
  }, [tooBig, src.name, sample?.count, note]);
  const built = useMemo(() => {
    if (tooBig || !sample?.points || !sample.covariance || !sample.opacity || !sample.sh) return null;
    const n = sample.count, coeffs = src.ref.sh_coefficients ?? 1;
    const values = new Float32Array(n * 12), sh = new Float32Array(n * coeffs * 4);
    for (let k = 0; k < n; k++) {
      values.set(sample.points.subarray(k * 3, k * 3 + 3), k * 12);
      values[k * 12 + 3] = sample.opacity[k];
      values.set(sample.covariance.subarray(k * 6, k * 6 + 6), k * 12 + 4);
      for (let c = 0; c < coeffs; c++) sh.set(sample.sh.subarray((k * coeffs + c) * 3, (k * coeffs + c + 1) * 3), (k * coeffs + c) * 4);
    }
    const dataTex = texture(values, width), shTex = texture(sh, width);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-1,-1,0, 1,-1,0, 1,1,0, -1,1,0]), 3));
    geometry.setIndex([0,1,2,0,2,3]);
    const ids = new Float32Array(n);
    geometry.setAttribute("splatId", new THREE.InstancedBufferAttribute(ids, 1));
    geometry.instanceCount = n;
    const material = new THREE.ShaderMaterial({ vertexShader: vertex, fragmentShader: fragment, transparent: true,
      depthWrite: false, side: THREE.DoubleSide, uniforms: { uData: { value: dataTex }, uSH: { value: shTex }, uWidth: { value: width },
        uCoeffs: { value: coeffs }, uViewport: { value: new THREE.Vector2() }, uEye: { value: new THREE.Vector3() } } });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false; mesh.frustumCulled = false;
    return { mesh, dataTex, shTex, ids, depths: new Float32Array(n), order: new Uint32Array(n),
      previous: new THREE.Matrix4().makeScale(0,0,0), mv: new THREE.Matrix4() };
  }, [sample, src, gl, width, tooBig]);
  useEffect(() => { invalidate(); return () => { if (built) { built.mesh.geometry.dispose(); built.mesh.material.dispose(); built.dataTex.dispose(); built.shTex.dispose(); } }; }, [built, invalidate]);
  useFrame(({ camera }) => {
    if (!built || !sample?.points) return;
    built.mesh.matrix.copy(world);
    built.mesh.updateMatrixWorld(true);
    const mv = built.mv.multiplyMatrices(camera.matrixWorldInverse, built.mesh.matrixWorld);
    const uniforms = built.mesh.material.uniforms;
    gl.getDrawingBufferSize(uniforms.uViewport.value);
    uniforms.uEye.value.setFromMatrixPosition(camera.matrixWorld).applyMatrix4(new THREE.Matrix4().copy(built.mesh.matrixWorld).invert());
    if (mv.equals(built.previous)) return;
    built.previous.copy(mv);
    const e = mv.elements;
    for (let k = 0; k < built.ids.length; k++) {
      built.depths[k] = e[2] * sample.points[k * 3] + e[6] * sample.points[k * 3 + 1] + e[10] * sample.points[k * 3 + 2] + e[14];
      built.order[k] = k;
    }
    built.order.sort((a, b) => built.depths[a] - built.depths[b]);
    for (let k = 0; k < built.ids.length; k++) built.ids[k] = built.order[k];
    built.mesh.geometry.attributes.splatId.needsUpdate = true;
  });
  void version;
  return built ? <primitive object={built.mesh} /> : null;
}
