/// <reference lib="webworker" />
import { workerAnswers } from "../platform/work";
import { BadOp } from "../ops/run";
import { runRecipe, type Recipe } from "../ops/recipe";
import { toPixels } from "../ops/pixels";

/** 在 Web Worker 线程中计算积木节点在浏览器端的单帧结果（19 条硬要求第 2、11 条）。
 *
 * 执行的是 `ops/recipe.ts` 的 `runRecipe`，与页面主线程为同一段代码，仅运行线程不同：
 * 一次 1080p 三通道合成约需六百万次逐像素求值，在主线程执行会造成编辑器卡顿，
 * 而本层的目的是使参数修改后画面即时更新。
 *
 * 返回值为该帧的 RGBA 八位像素（页面可直接 `createImageBitmap`），不生成包、不进入缓存、
 * 不写入任何数据（架构文件「五之二」：浏览器端计算仅用于预览）。 */

workerAnswers<{ recipe: Recipe }>(({ recipe }) => {
  const made = runRecipe(recipe.steps);
  if (!made.pix) throw new BadOp("this recipe does not end in an image")  // 开发者错误而非用户消息，因此使用英文（webui/tests/messages.test.ts）;
  const { w, h, c, data } = made.pix;
  // 同时返回原始值与显示图：原始值供通道路径在 GPU 上计算范围映射、黑白点与着色；
  // 显示图供多条颜色通道合并查看时的图片路径使用，服务器上没有该图，因此显示变换在此完成。
  // 每帧仅计算一次（view/useLocal.ts frameOf），额外开销仅为最后一次取像素。
  return { w, h, c, values: data, pixels: toPixels(made.pix, recipe.show).pixels };
});
