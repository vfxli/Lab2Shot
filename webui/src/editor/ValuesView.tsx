import { useEffect, useState } from "react";
import { api, type CurvesData, type Manifest } from "../api";
import { useTrustedResults } from "../state/results";
import type { DisplayPlan, ViewItem } from "../view/plan";
import { CurvesView } from "./CurvesView";
import { manifestOf } from "../transfer/frames";

/** The node's basic values (浮点, 整数, 向量 ...: the "value" viewer role): one value shows as itself, one per frame as
 * its curve (a vector: X Y Z). `board`: the whole stage, for a node that gives only values (「浮点」, 「拆分相机」);
 * otherwise a strip under the stage, next to the node's other results (AnyCalib's ST-maps and its Focal Length). The values are
 * the server's words for them (graph status), known before cooking for a constant. */
export function ValuesView({ plan, board }: { plan: DisplayPlan; board: boolean }) {
  const said = useTrustedResults();
  // 有值时显示值，无值时显示「—」，计算前同样显示「—」；无值的原因（未连出或尚未计算）在「数据信息」面板中说明，此处不说明
  const shown = plan.values.map((it) => ({ it, text: said[it.nodeId]?.values?.[it.port] ?? "" }));
  const curves = shown.filter((v) => v.text.includes("（逐帧") && v.it.fp && !v.text.includes("不变"));
  // 节点声明要显示的参数（NodeDef.strip：「LensDistortion」所用镜头的 Focal Length / Filmback / 镜头模型）；
  // 值为服务器给出的当前文字（填写值或连线值），不是输出口，也不进入账本
  const params = plan.node ? (said[plan.node.id]?.strip ?? []) : [];
  return (
    <div className={board ? "values-board" : "values-strip"}>
      <div className="values-list">
        {params.map((p) => (
          <div key={`param:${p.label}`} className={`value-card${p.text ? "" : " off"}`} data-tip={`${p.label}：${p.text || "—"}`}>
            <span className="value-label">{p.label}</span>
            <span className="value-text" data-user-data={p.text ? true : undefined}>{p.text || "—"}</span>
          </div>
        ))}
        {shown.map(({ it, text }) => (
          <ValueCard key={it.key} item={it} text={text} />
        ))}
      </div>
      {curves.map(({ it }) => (
        <ValueCurve key={it.key} item={it} />
      ))}
    </div>
  );
}

/** One value: the server's words for that output, else (for a packet the status has no words about, such as one item of a
 * list being shown) what the packet itself says it holds (its manifest: the value and its unit). */
function ValueCard({ item, text }: { item: ViewItem; text: string }) {
  const [held, setHeld] = useState<{ fp: string; m: Manifest } | null>(null);
  const fp = !text && item.fp ? item.fp : null;
  useEffect(() => {
    if (!fp) return;
    let alive = true;
    void manifestOf(fp).then((m) => alive && setHeld({ fp, m }), () => {});
    return () => {
      alive = false;
    };
  }, [fp]);
  const shown = text || (held?.fp === fp ? valueOf(held.m) : "");
  const tip = shown ? `${item.label}：${shown}` : `${item.label}：—`;
  return (
    <div className={`value-card${shown ? "" : " off"}`} data-tip={tip}>
      <span className="value-label">{item.label}</span>
      <span className="value-text" data-user-data={shown ? true : undefined}>{shown || "—"}</span>
    </div>
  );
}

/** What a value packet says it holds (data/values.py value_meta): its value and unit ("" for a per-frame one: the
 * curve below says it). */
function valueOf(m: Manifest): string {
  const v = (m.meta as { value?: unknown }).value;
  const unit = String((m.meta as { unit?: unknown }).unit ?? "");
  if (v === undefined || v === null) return "";
  // 镜头内参（value.lens）：清单中为 {model, params}，此处只显示模型名称；系数在数据信息中查看
  if (typeof v === "object" && !Array.isArray(v) && "model" in (v as object)) return String((v as { model: unknown }).model);
  // 保留五位有效数字：解算出的 Focal Length 是浮点数（如 18.037566222907067），原样显示一格放不下，且第 15 位没有参考意义
  const one = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? String(Number(x.toPrecision(5))) : String(x));
  return `${Array.isArray(v) ? v.map(one).join(" ") : one(v)}${unit ? ` ${unit}` : ""}`;
}

function ValueCurve({ item }: { item: ViewItem }) {
  const [data, setData] = useState<{ fp: string; value: CurvesData } | null>(null);
  useEffect(() => {
    let alive = true;
    void api.curves(item.fp!).then((value) => alive && setData({ fp: item.fp!, value }), () => {});
    return () => {
      alive = false;
    };
  }, [item.fp]);
  return (
    <div className="value-curve">
      <div className="value-label">{item.label} · 每帧</div>
      {data?.fp === item.fp ? <CurvesView data={data.value} /> : <div className="empty">读取…</div>}
    </div>
  );
}
