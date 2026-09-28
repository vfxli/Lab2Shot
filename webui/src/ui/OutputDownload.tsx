import type { Output } from "../api/files";
import { downloadOutput } from "../files/outputs";
import { agoText, sizeText } from "../platform/format";
import { Button } from "./Button";

/** 「输出」's one control: 下载, the browser's own download of the zip its own cook packed (files/outputs.ts). The
 * node's body, the parameter panel and the queue draw this same button. It can be clicked only once there is a zip:
 * computing the nodes before it only puts their results on the server; computing the 「输出」 itself collects and
 * packs them, then this is enabled. An output gone with its task (任务保留天数) is greyed and says so. */

const NONE = "还没有打包好的结果：右键「计算」这个「输出」，它把接进来的结果整理成一个文件夹、再打包成一个 zip，好了这里就能下载";
const GONE = "这个任务已经过了保留天数，服务器上删掉了：再计算一次「输出」（上游有缓存的话很快）";

export function outputTip(o: Output): string {
  const when = o.finished ? ` · ${agoText(o.finished)}打包` : "";
  return `${o.name}\n${sizeText(o.bytes)} · ${o.count} 个文件${when}\n\n` +
    "浏览器自己下载，看得到进度，断了能接着下。解压出来是一个和它同名的文件夹，里面每个输出设置一个子文件夹。" +
    "DCC 插件可以只取其中几个文件";
}

export function OutputDownload({ output, size = "xs" }: { output: Output | undefined | null; size?: "xs" | "sm" }) {
  const ready = !!output && !output.gone;
  return (
    <Button tip={!output ? NONE : output.gone ? GONE : outputTip(output)} tone="ghost" size={size} disabled={!ready}
            onClick={(e) => (e.stopPropagation(), ready && downloadOutput(output))}>
      {ready ? `下载 · ${sizeText(output.bytes)}` : "下载"}
    </Button>
  );
}
