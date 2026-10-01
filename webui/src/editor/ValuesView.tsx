import type { CurvesData, Manifest } from "../api";
import { useTrustedResults } from "../state/results";
import type { DisplayPlan, ViewItem } from "../view/plan";
import { CurvesView } from "./CurvesView";
import { useDescribed } from "../transfer/described";

/** 节点基本数值（浮点、整数、向量……：视图角色 "value"）的显示：单个值显示其本身，逐帧的值显示为曲线（向量：X Y Z）。
 * `board`：占满整个舞台，用于只给出数值的节点（「浮点」「拆分相机」）；否则为舞台下方的一条，与节点的其他结果并列
 * （AnyCalib 的 ST-map 及其 Focal Length）。数值为服务器给出的文字（graph status），常量在计算前即已知。 */
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

/** 一个值：服务器对该输出的文字；status 中没有文字的数据包（例如正在显示的列表中的一项）则取数据包自身声明的内容
 * （其 manifest：值与单位）。 */
function ValueCard({ item, text }: { item: ViewItem; text: string }) {
  const fp = !text && item.fp ? item.fp : null;
  const held = useDescribed<Manifest>("manifest", fp ? [fp] : [])[0];
  const shown = text || (held ? valueOf(held) : "");
  const tip = shown ? `${item.label}：${shown}` : `${item.label}：—`;
  return (
    <div className={`value-card${shown ? "" : " off"}`} data-tip={tip}>
      <span className="value-label">{item.label}</span>
      <span className="value-text" data-user-data={shown ? true : undefined}>{shown || "—"}</span>
    </div>
  );
}

/** 数值数据包自身声明的内容（data/values.py value_meta）：值与单位（逐帧的值返回 ""：由下方曲线表达）。 */
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
  const data = useDescribed<CurvesData>("curves", [item.fp!])[0];
  return (
    <div className="value-curve">
      <div className="value-label">{item.label} · 每帧</div>
      {data ? <CurvesView data={data} /> : <div className="empty">读取…</div>}
    </div>
  );
}
