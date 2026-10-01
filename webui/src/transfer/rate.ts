/** 传输速度（字节 / 秒）：累计字节数随时间的记录，取最近 8 秒的平均，记录不足 1.5 秒时不下结论（沿用上一次的值）。
 * 这是项目里唯一的一份速度算法：上传进度（transfer/uploads.ts sendWith、transfer/planes.ts sendPlanes 写进任务的
 * `rate`）和顶栏的实时流量（editor/Chrome.tsx TransferRate：platform/traffic.ts 记的全部上下行累计字节）都用它。 */
export class Rate {
  private samples: [number, number][] = [];
  private last = 0;

  /** 断线后重新量：之前的记录和由它算出的速度都不再算数（重新量够 1.5 秒之前为 0）。 */
  reset(): void {
    this.samples.length = 0;
    this.last = 0;
  }

  /** 记下此刻累计到的字节数，返回现在的速度。 */
  add(total: number, now = Date.now()): number {
    this.samples.push([now, total]);
    while (this.samples.length > 2 && now - this.samples[0][0] > 8000) this.samples.shift();
    const [t0, s0] = this.samples[0];
    if (now - t0 > 1500) this.last = Math.max(0, ((total - s0) / (now - t0)) * 1000);
    return this.last;
  }
}
