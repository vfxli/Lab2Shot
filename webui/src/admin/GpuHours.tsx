/** 每小时平均显卡使用率。不单独读取显卡：该值来自队列已有的 nvidia-smi 读取结果，
 *  见 lab2shot/farm/scheduler/inventory.py `_note_hour`。
 *
 *  每张卡一行，最近 24 小时每小时一根柱，柱高为该小时的平均占用率（0–100%）。
 *  尚未结束的当前小时绘制为空心并注明「这一小时还在走」，因为不足一小时的平均值与整小时平均值不可比。 */

import { useRef, type RefObject } from "react";
import type { CardHour, CardRow } from "../api/cards";
import { useRefSize } from "../platform/size";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

const HOURS_SHOWN = 24;
const H = 64; // 单行图表高度。
const PAD = 10; // 顶部为「100%」参考线预留的空间。
const GUTTER = 40; // 左侧刻度栏宽度。
const NAME = 20; // 卡名位于图表上方（置于下方会与小时刻度重叠）。
const AXIS = 30;  // 底部小时刻度的高度，含与下一张图之间的间距（避免下一行卡名紧贴刻度）。

function useWidth(): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  return [ref, useRefSize(ref).w];
}

/** 小时标签：按本机时区格式化为「14 时」。 */
const hourLabel = (hour: number): string => t("ui.admin.gpu_hours.hour", { hour: new Date(hour * 3600 * 1000).getHours() });

export function GpuHours({ hourly, cards }: { hourly: CardHour[]; cards: CardRow[] }) {
  const [ref, width] = useWidth();
  // 格子固定排满 24 小时，而非按数据条数排列：否则数据仅有一小时时，单根柱位于图表正中，两侧空白，
  // 易被误认为异常。补齐后柱子从右向左增长，位置固定，可直观看出记录刚开始。
  // 无数据的格子不绘制。
  const now = Math.floor(Date.now() / 3600_000);
  const have = new Map(hourly.map((h) => [h.hour, h]));
  const hours: (CardHour | null)[] = Array.from({ length: HOURS_SHOWN }, (_, i) =>
    have.get(now - (HOURS_SHOWN - 1 - i)) ?? null);
  if (!hourly.length) {
    // 空状态说明原因及后续条件，不留空白区域。
    return <p className="adm-lede">{t("ui.admin.gpu_hours.empty")}</p>;
  }
  const plot = Math.max(40, width - GUTTER - 6);
  const band = plot / Math.max(hours.length, 1);
  const barW = Math.max(1, Math.min(22, band * 0.68));
  const rowH = NAME + H + AXIS;
  const height = cards.length * rowH;
  const x = (i: number) => GUTTER + i * band + (band - barW) / 2;
  return (
    <div ref={ref} className="u-chart">
      {width > 0 && (
        <svg className="u-svg" width={width} height={height} role="img" aria-label={t("ui.admin.gpu_hours.label")}>
          {cards.map((card, row) => {
            const top = row * rowH + NAME;  // 图表顶部（卡名位于其上方）。
            const base = top + H;
            return (
              <g key={card.uuid}>
                {[0, 0.5, 1].map((f) => (
                  <g key={f}>
                    <line x1={GUTTER} x2={width} y1={top + PAD + (1 - f) * (H - PAD)} y2={top + PAD + (1 - f) * (H - PAD)} className="u-grid" />
                    <text x={GUTTER - 6} y={top + PAD + (1 - f) * (H - PAD)} textAnchor="end" dominantBaseline="central" className="u-tick">
                      {Math.round(f * 100)}
                    </text>
                  </g>
                ))}
                <text x={GUTTER} y={top - 6} className="u-label">
                  {`GPU ${card.index} · ${card.model}`}
                </text>
                {hours.map((h, i) => {
                  const hour = h ? h.hour : now - (HOURS_SHOWN - 1 - i);
                  const pct = h ? Math.max(0, Math.min(100, h.average?.[card.uuid] ?? 0)) : null;
                  // 有采样但值为 0：绘制 2 px 的底线，以区别于「该小时无数据」。
                  const barH = pct === null ? 0 : Math.max(2, (pct / 100) * (H - PAD));
                  return (
                    <g key={hour}>
                      {pct !== null && (
                        <rect x={x(i)} y={base - barH} width={barW} height={barH} rx={2}
                              fill={h?.running ? "none" : "var(--accent)"}
                              stroke={h?.running ? "var(--accent)" : "none"}
                              strokeDasharray={h?.running ? "2 2" : undefined} />
                      )}
                      <rect x={GUTTER + i * band} y={top} width={band} height={H} fill="transparent"
                            {...tipAttrs(tipOf("value", pct === null ? t("ui.admin.gpu_hours.none", { hour: hourLabel(hour) })
                              : h?.running ? t("ui.admin.gpu_hours.average_running", { hour: hourLabel(hour), pct }) : t("ui.admin.gpu_hours.average", { hour: hourLabel(hour), pct })))} />
                    </g>
                  );
                })}
                {hours.map((h, i) =>
                  i % 6 === 0 || i === hours.length - 1 ? (
                    <text key={`t${i}`} x={GUTTER + (i + 0.5) * band} y={base + 16} textAnchor="middle" className="u-tick">
                      {hourLabel(h ? h.hour : now - (HOURS_SHOWN - 1 - i))}
                    </text>
                  ) : null,
                )}
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
