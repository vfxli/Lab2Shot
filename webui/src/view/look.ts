import { mixOf, tintAlpha, tintLut, type Op, type Tint } from "../model/view2d";
import { isPlane, type Pixels, type Plane } from "../transfer/plane";

/** 一块画面在画布中的位置（x, y, 宽, 高）：取包自带的 data_window，每一路据此回到原位。 */
export type Box = [number, number, number, number];

/** 一侧的显示方式：选取的通道、黑白点、着色，以及第四条通道的用途。 */
export type Side = SideLook;

/** 中间的一次运算：加或乘、强度、乘的对象。 */
export interface Merge {
  width: number;
  height: number;
  left: Box;
  right: Box;
  op: Op;
  mix: number;
}

/** 显示层：每一帧在 GPU 上实时计算，不存储任何处理后的图像。
 *
 * 不采用异步生成处理后的图像并存入缓存的原因：每调整一次控件就多一张图，内存紧张时全部丢弃，丢弃后只能静默绘制原图；
 * 在途的图像到达时当前显示方式可能已改变；且同一逻辑会在 worker 和服务器各有一份实现。
 * 因此数据仍存于缓存中，显示方式一律实时计算：切换着色、拖动黑白点不产生任何缓存项，也不发任何请求。
 *
 * 一侧的数据有两种来源，此处同等处理：
 * - `picture`：服务器生成的显示图（RGBA，8 位有损 WebP 代理，已完成显示变换）。彩色画面走此路径（规则见
 *   transfer/route.ts）：一张图是三条颜色通道的最小载体；
 * - `planes`：一到三条通道的数据本身（transfer/plane.ts）。值为数据自身的值，
 *   范围映射（`range`）、黑白点、着色、合成均在此计算。数值图、alpha、视频的单条通道均走此路径，
 *   按需发送所需的通道。
 *
 * 两条路径的计算结果像素差 ≤ 1/255 即可，不要求逐位一致。 */

interface SideLook {
  index: number | null; // 选取的通道序号（null：多条一起查看，颜色即数据本身）。走 planes 路径时不使用
  black: number;
  white: number;
  tint: Tint;
  top?: number; // 「编号」档的最大编号（包的 range 上界）
  alpha: "data" | "weight"; // 第四条通道表示该数据自身的透明度，还是本次运算的强度
}

/** 一路待绘制的内容：数据（两种来源之一）+ 显示方式 + 在画布上的位置。 */
export interface SideSource {
  box: Box;
  side: SideLook;
  /** 图片路径：服务器生成的显示图。与 `planes` 二选一。 */
  picture?: ImageBitmap | null;
  /** 按通道取数路径：1 条（用户选中的通道）或 2–3 条（数值图「整体」：每条通道一种颜色）。 */
  planes?: (Plane | null)[] | null;
  /** 有效像素标记（包中额外写入的 valid 通道）。缺省表示所有像素均有值。 */
  valid?: Plane | null;
  /** 该数据自身的显示范围（数值图的 `meta.range`）。缺省为 0..1。仅对 `planes` 路径有意义。 */
  range?: readonly [number, number] | null;
}

interface LookAsk {
  width: number; // 画布（画面框）的尺寸
  height: number;
  left: SideSource;
  right?: SideSource | null;
  merge?: { op: Op; mix: number } | null;
}

const VERT = `#version 300 es
in vec2 p;
out vec2 uv;
void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;

// 单侧的计算：（图片：选取通道）或（通道：按范围映射 × 有效标记）→ 黑白点 → 查色表 → 强度
const FRAG = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 colour;

uniform sampler2D left, right, leftLut, rightLut;
uniform sampler2D leftP1, leftP2, leftValid, rightP1, rightP2, rightValid;
uniform vec4 leftBox, rightBox;   // 该侧的画面框（x, y, w, h），单位为画布像素
uniform vec2 canvas;              // 画布尺寸
uniform int leftIndex, rightIndex;    // -1：整体（仅对图片路径有意义）
uniform int leftPlanes, rightPlanes;  // 0：图片路径；1：单条通道；2、3：数值图整体
uniform vec2 leftRange, rightRange;   // 该数据自身的显示范围（通道路径）
uniform int leftHasValid, rightHasValid;
uniform vec2 leftGrade, rightGrade;   // (black, white)
uniform float leftSolid, rightSolid;  // 1：纯色（浓淡随值变化），0：色标（颜色自带数值信息）
uniform int leftAlpha, rightAlpha;    // 1：该通道的值作为运算强度
uniform int hasRight, op;             // op 0 加 · 1 乘 Alpha · 2 乘 RGBA（model/view2d.ts Op）
uniform float mix_;

float graded(float v, vec2 g) { return (v - g.x) / max(g.y - g.x, 1e-6); }

// 一条通道的值 → 该数据自身范围内的 0..1（与服务器生成显示图时的表达式完全一致：
// lab2shot/view/frames.py _map_rgb 的 clip((v - lo) / span)，范围反向时斜坡随之翻转）
float mapped(sampler2D tex, vec2 q, vec2 range) {
  float span = abs(range.y - range.x) > 1e-6 ? (range.y - range.x) : 1e-6;
  return clamp((texture(tex, q).r - range.x) / span, 0.0, 1.0);
}

// pix：以左上角为原点的画布像素坐标。WebGL 画布原点位于左下角，而图像、画面框（data_window）及
// 整条二维链的坐标均以左上角为原点，不统一会导致上下颠倒（三维背板的取样同理）。
vec4 side(sampler2D tex, sampler2D p1, sampler2D p2, sampler2D validTex, sampler2D lut,
          vec2 pix, vec4 box, int planes, vec2 range, int hasValid,
          int index, vec2 grade, float solid, int asWeight) {
  vec2 q = (pix - box.xy) / max(box.zw, vec2(1.0));
  if (q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) return vec4(0.0);
  float raw;
  if (planes == 0) {                     // 图片路径：已完成显示变换，在此选取通道
    vec4 px = texture(tex, q);
    if (index < 0) {                     // 多条一起查看：只应用黑白点，不着色
      return vec4(graded(px.r, grade), graded(px.g, grade), graded(px.b, grade), px.a);
    }
    raw = index == 0 ? px.r : index == 1 ? px.g : index == 2 ? px.b : px.a;
  } else {
    // 无值像素绘制为黑色（服务器端为 np.where(valid, scaled, 0)）
    float ok = (hasValid == 1 && texture(validTex, q).r <= 0.0) ? 0.0 : 1.0;
    if (planes > 1) {                    // 数值图「整体」：每条通道一种颜色，缺少第三条时为 0，不着色
      return vec4(graded(mapped(tex, q, range) * ok, grade),
                  graded(mapped(p1, q, range) * ok, grade),
                  planes > 2 ? graded(mapped(p2, q, range) * ok, grade) : graded(0.0, grade),
                  1.0);
    }
    raw = mapped(tex, q, range) * ok;
  }
  float t = clamp(graded(raw, grade), 0.0, 1.0);
  // 取第 round(t*255) 档，不在两档之间插值：与 model/view2d.ts 的 tintLut(...)[Math.round(t*255)]
  // 等价。（「编号」档尤其不得插值：插值得到的颜色不属于任何类别。）
  vec3 rgb = texture(lut, vec2((t * 255.0 + 0.5) / 256.0, 0.5)).rgb;
  float a = asWeight == 1 ? t : mix(1.0, t, solid);   // tintAlpha：纯色的浓淡即为数值，色标自带数值信息
  return vec4(rgb, a);
}

void main() {
  vec2 pix = vec2(uv.x, 1.0 - uv.y) * canvas;   // 以左上角为原点（见 side 上方的注释）
  vec4 l = side(left, leftP1, leftP2, leftValid, leftLut, pix, leftBox, leftPlanes, leftRange,
                leftHasValid, leftIndex, leftGrade, leftSolid, leftAlpha);
  vec4 out_ = l;
  if (hasRight == 1) {
    vec4 r = side(right, rightP1, rightP2, rightValid, rightLut, pix, rightBox, rightPlanes, rightRange,
                  rightHasValid, rightIndex, rightGrade, rightSolid, rightAlpha);
    if (op == 0) {                          // 加：右侧按其自身强度叠加，透明度不变
      float k = r.a * mix_;
      out_ = vec4(l.rgb + r.rgb * k, l.a);
    } else {                                // 乘：mix 将乘数向 1 拉回
      float k = 1.0 - mix_ + mix_ * r.a;
      out_ = op == 1 ? vec4(l.rgb, l.a * k) : vec4(l.rgb * k, l.a * k);
    }
  }
  // 上下文使用 premultipliedAlpha: false，因此颜色按未预乘输出（再乘一次会变暗）
  colour = out_;
}`;

/** 每一路在 GPU 上占用的纹理单元。编号固定，不得两处共用同一单元：
 * 创建查色表时若不指定单元，会替换画面一路的纹理，绘制出的将是查色表本身。`SCRATCH` 专用于创建查色表。 */
const UNIT = {
  left: 0, right: 1, leftLut: 2, rightLut: 3,
  leftP1: 4, leftP2: 5, leftValid: 6,
  rightP1: 8, rightP2: 9, rightValid: 10,
} as const;
const SCRATCH = 7;

type Slot = keyof typeof UNIT;
const PIXEL_SLOTS: Slot[] = ["left", "right", "leftP1", "leftP2", "leftValid", "rightP1", "rightP2", "rightValid"];

interface Gl {
  canvas: HTMLCanvasElement;
  gl: WebGL2RenderingContext;
  program: WebGLProgram;
  at: Record<string, WebGLUniformLocation | null>;
  luts: Map<string, WebGLTexture>;
  tex: Record<Slot, WebGLTexture>;
  // 每个单元当前纹理里装的是哪一份数据（图片或通道对象本身）：同一份不再上传。拖动手柄、鼠标悬停都会重画，
  // 若每次都重新上传，一张 4K 浮点图每次移动鼠标要传 30 多 MB 给显卡
  holds: Partial<Record<Slot, ImageBitmap | Plane>>;
}

/** 本模块唯一的可变状态（与 `transfer/frames.ts` 的 `Fetching` 做法相同）：已创建的 GPU 资源，
 * 以及「本机无法创建 WebGL2 上下文」这一事实（检测一次后记录，之后由 `Stage2D` 决定如何处理）。 */
const gpu: { own: Gl | null; broken: boolean } = { own: null, broken: false };

function make(): Gl | null {
  if (gpu.broken) return null;
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", { premultipliedAlpha: false, preserveDrawingBuffer: true, antialias: false });
  if (!gl) {
    gpu.broken = true;
    return null;
  }
  const compile = (kind: number, src: string) => {
    const sh = gl.createShader(kind)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? "shader");
    return sh;
  };
  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "link");
  gl.useProgram(program);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const p = gl.getAttribLocation(program, "p");
  gl.enableVertexAttribArray(p);
  gl.vertexAttribPointer(p, 2, gl.FLOAT, false, 0, 0);
  const names = ["leftBox", "rightBox", "canvas", "leftIndex", "rightIndex", "leftGrade", "rightGrade",
                 "leftSolid", "rightSolid", "leftAlpha", "rightAlpha", "hasRight", "op", "mix_",
                 "leftPlanes", "rightPlanes", "leftRange", "rightRange", "leftHasValid", "rightHasValid",
                 ...Object.keys(UNIT)];
  const at: Gl["at"] = {};
  for (const n of names) at[n] = gl.getUniformLocation(program, n);
  const blank = () => {
    const t = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0 + SCRATCH);
    gl.bindTexture(gl.TEXTURE_2D, t);
    // 最近邻采样：任何缩放下都不模糊，才能判断抠像边缘（与 2D 画布的 imageSmoothingEnabled=false 含义相同）。
    //
    // 两侧尺寸不同时依赖此设置（视图使用本地原件时这是常态：一侧为本机原件 1920×1080、
    // 另一侧为服务器代理 512×288；不缩小本地数据迁就代理，也不放大代理迁就本地数据）。
    // 两侧各为一张纹理，各自按其尺寸采样到舞台的像素网格上（片段着色器中的 `q`），
    // 代理为等比缩放，因此覆盖的画面范围相同、UV 一致，无需先将一侧缩放到另一侧。
    //
    // 两侧都使用最近邻，而非「颜色用双线性、遮罩用最近邻」的原因：
    // 最近邻从不产生原数据中不存在的值。双线性会：遮罩和「编号」这类标签图在 1 和 2 之间插出 1.5，
    // 该值不属于任何类别；深度会在边缘产生不存在的中间深度。
    // 颜色放大超过代理档位时，最近邻呈现为硬像素块，这与 Nuke / RV 放大到 800% 时的效果相同，
    // 属于 DCC 惯例而非缺陷。因此两侧统一遵循一条规则：不插值。
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // 该路本轮无数据时也须绑定一张实际纹理（未绑定的采样单元在部分驱动上不绘制）
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));
    return t;
  };
  const tex = Object.fromEntries(PIXEL_SLOTS.map((k) => [k, blank()])) as Gl["tex"];
  // 查色表的两个单元也需要纹理对象（下方 lutTexture 按「着色 + 上界」各建一张，这两个单元只负责绑定）
  return { canvas, gl, program, at, luts: new Map(), tex: { ...tex, leftLut: blank(), rightLut: blank() }, holds: {} };
}

/** 一条色标的 256 档查色表，生成为 256×1 的纹理（颜色本身取自 model/view2d.ts，不另写一套）。 */
function lutTexture(g: Gl, tint: Tint, top: number): WebGLTexture {
  // only the id colouring depends on the largest id (tintLut); every other one is the same table whatever the data's
  // range, so one texture per tint, not one more for every range met
  const key = tint === "id" && top > 0 ? `${tint}|${top}` : tint;
  const got = g.luts.get(key);
  if (got) return got;
  const table = tintLut(tint, top);
  const px = new Uint8Array(256 * 4);
  for (let v = 0; v < 256; v++) {
    const [r, gg, b] = table[v];
    px.set([r, gg, b, 255], v * 4);
  }
  const t = g.gl.createTexture()!;
  // 在专用单元上创建：若不指定，会绑定到当时激活的单元，替换画面一路的纹理
  g.gl.activeTexture(g.gl.TEXTURE0 + SCRATCH);
  g.gl.bindTexture(g.gl.TEXTURE_2D, t);
  g.gl.texImage2D(g.gl.TEXTURE_2D, 0, g.gl.RGBA, 256, 1, 0, g.gl.RGBA, g.gl.UNSIGNED_BYTE, px);
  g.gl.texParameteri(g.gl.TEXTURE_2D, g.gl.TEXTURE_MIN_FILTER, g.gl.NEAREST);
  g.gl.texParameteri(g.gl.TEXTURE_2D, g.gl.TEXTURE_MAG_FILTER, g.gl.NEAREST);
  g.gl.texParameteri(g.gl.TEXTURE_2D, g.gl.TEXTURE_WRAP_S, g.gl.CLAMP_TO_EDGE);
  g.gl.texParameteri(g.gl.TEXTURE_2D, g.gl.TEXTURE_WRAP_T, g.gl.CLAMP_TO_EDGE);
  g.luts.set(key, t);
  return t;
}

/** 将 u16 档转换为 float32：GPU 的核心格式中没有 16 位归一化格式（属于 EXT_texture_norm16），
 * 半精度又无法表示 65535。数值不做任何修改（k/65535，与 lab2shot/view/channels.py `_fits` 对应），
 * 同一条通道只转换一次（转换结果缓存于此，播放时不会每帧重复计算）。 */
const asFloat = new WeakMap<Plane, Float32Array>();
function floats(p: Plane): Float32Array {
  let f = asFloat.get(p);
  if (!f) {
    const src = p.data as Uint16Array;
    f = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) f[i] = src[i] / 65535;
    asFloat.set(p, f);
  }
  return f;
}

/** 当前浏览器是否无法创建 WebGL2 上下文（尝试一次后才能得知）。调用方据此在通知区提示，不得静默降级。 */
export const noGpu = (): boolean => gpu.broken;

const solidOf = (tint: Tint): number => (tintAlpha(tint, 0.5) === 0.5 ? 1 : 0); // 纯色：浓淡随值变化

/** 一路是否有可绘制的内容。 */
const hasPixels = (s: SideSource | null | undefined): boolean =>
  !!s && (!!s.picture || !!s.planes?.some((p) => p));

/** 当前显示结果的图像（画布尺寸等于画面框）。null 表示本机无法创建 WebGL2 上下文，或左侧一路尚无数据；
 * 由调用方决定如何处理（不得静默绘制其他内容）。 */
export function drawLook(ask: LookAsk): HTMLCanvasElement | null {
  const g = (gpu.own ??= make());
  if (!g || !hasPixels(ask.left)) return null;
  const { gl, at } = g;
  const [w, h] = [Math.max(1, Math.round(ask.width)), Math.max(1, Math.round(ask.height))];
  if (g.canvas.width !== w || g.canvas.height !== h) {
    g.canvas.width = w;
    g.canvas.height = h;
  }
  gl.viewport(0, 0, w, h);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); // 一条通道的一行字节数不一定是 4 的倍数（R8 奇数宽度）
  const bind = (slot: Slot) => {
    gl.activeTexture(gl.TEXTURE0 + UNIT[slot]);
    gl.bindTexture(gl.TEXTURE_2D, g.tex[slot]);
    gl.uniform1i(at[slot]!, UNIT[slot]);
  };
  const putPicture = (slot: Slot, image: ImageBitmap) => {
    bind(slot);
    if (g.holds[slot] === image) return;
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    g.holds[slot] = image;
  };
  const putPlane = (slot: Slot, p: Plane | null | undefined) => {
    bind(slot);
    if (!p || g.holds[slot] === p) return;
    g.holds[slot] = p;
    const [internal, type, data] =
      p.format === 0 ? [gl.R8, gl.UNSIGNED_BYTE, p.data as Uint8Array]
      : p.format === 2 ? [gl.R16F, gl.HALF_FLOAT, p.data as Uint16Array]
      : p.format === 3 ? [gl.R32F, gl.FLOAT, p.data as Float32Array]
      : [gl.R32F, gl.FLOAT, floats(p)];
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, p.width, p.height, 0, gl.RED, type, data);
  };
  // 一路：数据（图片，或一到三条通道 + valid）+ 显示方式
  const put = (s: SideSource | null, prefix: "left" | "right") => {
    const on = !!s && hasPixels(s);
    const side = s?.side;
    // 先移除尚未到达的格：格数必须与实际绑定的纹理数一致。计为 3 条而第 0 格为空时，
    // 绘制结果会出现错乱（一条通道仍在传输时即会如此）。
    const ps = (on && !s!.picture ? s!.planes ?? [] : []).filter(Boolean) as Plane[];
    if (on && s!.picture) putPicture(prefix, s!.picture);
    else putPlane(prefix, ps[0] ?? null);
    putPlane(`${prefix}P1` as Slot, ps[1] ?? null);
    putPlane(`${prefix}P2` as Slot, ps[2] ?? null);
    putPlane(`${prefix}Valid` as Slot, ps.length ? s!.valid ?? null : null);
    const planes = ps.length;
    // 先创建查色表，再选择单元：`lutTexture` 创建时会将创建单元设为当前单元，
    // 顺序颠倒会使查色表绑定到该单元，而本单元仍保留上一次使用的色标。
    const lut = lutTexture(g, side?.tint ?? "grey", side?.top ?? 0);
    gl.activeTexture(gl.TEXTURE0 + UNIT[`${prefix}Lut` as Slot]);
    gl.bindTexture(gl.TEXTURE_2D, lut);
    gl.uniform1i(at[`${prefix}Lut`]!, UNIT[`${prefix}Lut` as Slot]);
    const box = s?.box ?? [0, 0, 1, 1];
    gl.uniform4f(at[`${prefix}Box`]!, box[0], box[1], box[2], box[3]);
    gl.uniform1i(at[`${prefix}Index`]!, side?.index ?? -1);
    gl.uniform1i(at[`${prefix}Planes`]!, planes);
    const range = s?.range ?? [0, 1];
    gl.uniform2f(at[`${prefix}Range`]!, range[0], range[1]);
    gl.uniform1i(at[`${prefix}HasValid`]!, planes && s!.valid ? 1 : 0);
    gl.uniform2f(at[`${prefix}Grade`]!, side?.black ?? 0, side?.white ?? 1);
    gl.uniform1f(at[`${prefix}Solid`]!, solidOf(side?.tint ?? "grey"));
    gl.uniform1i(at[`${prefix}Alpha`]!, side?.alpha === "weight" ? 1 : 0);
  };
  put(ask.left, "left");
  gl.uniform2f(at.canvas!, w, h);
  const r = ask.right && ask.merge && hasPixels(ask.right) ? ask.right : null;
  gl.uniform1i(at.hasRight!, r ? 1 : 0);
  put(r, "right");
  // 0 加 · 1 乘 Alpha · 2 乘 RGBA：与着色器中的两个 if 分支一一对应（三档名称见 model/view2d.ts Op）
  gl.uniform1i(at.op!, !r || ask.merge!.op === "add" ? 0 : ask.merge!.op === "mulAlpha" ? 1 : 2);
  gl.uniform1f(at.mix_!, r ? mixOf(ask.merge!.op, ask.merge!.mix) : 0);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  return g.canvas;
}

export { isPlane, type Pixels, type Plane };
