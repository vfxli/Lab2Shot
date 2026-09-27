import { MessageError } from "../messages/message";
import { workerAsks } from "../platform/work";
import { lutOfFile } from "./lut";

/** 用户所选 EXR 在上传期间由页面自行绘制（本机画面）。浏览器没有 OCIO：服务器把该文件的显示变换
 * （节点读取时所用的色彩空间、当前显示与视图）烘焙成一张以线性值 log2 为坐标的 3D 查找表（GET /api/view/lut），
 * 页面逐像素查表。结果接近服务器的显示效果，素材上传完成后由服务器的读取结果取代。 */

const ask = workerAsks<{ width: number; height: number; pixels: Uint8ClampedArray }>(() => new Worker(new URL("./exrWorker.ts", import.meta.url), { type: "module" }));

/** 按服务器的显示方式解码 EXR 文件（`space`：节点的色彩空间，空串表示按文件名规则判定）。
 *
 * `rules` 是请求查找表时所用的文件名，默认为该帧自身的文件名。整段序列必须传入同一个名字，
 * 否则每帧对应不同地址，48 帧的 EXR 序列将产生 48 次 `/api/view/lut` 请求，导致播放卡顿。
 * 查找表仅由色彩空间决定（服务器 `lab2shot/server/view.py`：`lut_for(space or colorspace_for_file(file))`，
 * 指定 `space` 时不使用文件名）；一段序列只有一个色彩空间（读取节点同样依据第一个文件判定，
 * 见 `lab2shot/nodes/core/input.py`），因此整段共用一张表是准确的，并非近似。 */
export async function decodeExr(file: File, space = "", rules = ""): Promise<ImageBitmap> {
  const [bytes, lut] = await Promise.all([file.arrayBuffer().then((b) => new Uint8Array(b)), lutOfFile(space, rules || file.name)]);
  const answer = await ask({ file: bytes, lut }, [bytes]).catch((e: unknown) => {
    throw e instanceof MessageError && e.said.code === "E-THREAD-FAILED" ? new MessageError("E-VIEW-EXRFAILED") : e;
  });
  return createImageBitmap(new ImageData(answer.pixels as Uint8ClampedArray<ArrayBuffer>, answer.width, answer.height),
                           { premultiplyAlpha: "none", colorSpaceConversion: "none" });  // 不预乘：GPU 路径需要原始值（transfer/frames.ts BITMAP）
}
