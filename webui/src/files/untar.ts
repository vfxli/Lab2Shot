/** 在浏览器中解包 tar：所有交付模式均以整包传输，在本地解包。
 *
 * 在客户端解包的原因：交付到文件夹时若逐文件请求（`transfer.ts saveDelivery`），
 * 150 帧的 EXR 序列需要 150 次往返，往返次数是主要耗时。
 * 以单个请求获取整包并边下载边解包写入，只需一次往返。
 *
 * 不依赖第三方库：tar 不是压缩格式，由 512 字节的头和文件内容（补齐到 512 的整数倍）依次拼接而成；
 * tar.gz 外层的 gzip 由浏览器内置的 `DecompressionStream("gzip")` 解压。
 *
 * 服务器使用 `tarfile.PAX_FORMAT` 打包（`lab2shot/transfer/deliveries.py _header`），因此除普通成员
 * （typeflag "0" 或 "\0"）外还须识别 PAX 扩展头（typeflag "x"）：它是一个独立成员，内容为若干行
 * `<长度> <键>=<值>\n`，其中 `path=` 指定紧随其后的下一个成员的名称（名称含中文或超过 100 字节时
 * Python 采用此写法）。无法识别的 typeflag（目录 "5"、长名 "L" 等）一律跳过其内容，不作推断。
 *
 * 解包错误会向用户磁盘写入损坏数据，因此任何异常均立即抛错而不继续推断：头校验和不匹配、
 * 名称含 `..` 或绝对路径、包在中途截断。逐字节比对测试见 webui/tests/untar.test.ts
 * （使用服务器实际生成的包校验解出的每一个字节）。
 */

const BLOCK = 512;

/** 解包错误。`message` 为消息编号而非面向用户的文本（代码中只写编号，中文模板统一位于
 *  `lab2shot/messages/web.toml`）。`detail` 为面向开发者的英文补充信息，只写入日志，不显示给用户。 */
export class TarError extends Error {
  detail: string;
  constructor(code: string, detail = "") {
    super(code);
    this.name = "TarError";
    this.detail = detail;
  }
}

/** 解析头中的八进制数字段（末尾可能为空格或 \0）。同时支持 GNU 的 base-256 编码（首字节高位为 0x80），
 *  该编码仅用于超过 8 GB 的单个文件。 */
function octal(b: Uint8Array, at: number, len: number): number {
  const field = b.subarray(at, at + len);
  if (field[0] & 0x80) {
    let n = 0;
    for (let i = 1; i < field.length; i++) n = n * 256 + field[i];
    return n;
  }
  let text = "";
  for (const c of field) {
    if (c === 0 || c === 32) break;
    text += String.fromCharCode(c);
  }
  return text ? parseInt(text, 8) : 0;
}

const text = (b: Uint8Array, at: number, len: number): string => {
  const field = b.subarray(at, at + len);
  const end = field.indexOf(0);
  return new TextDecoder().decode(end < 0 ? field : field.subarray(0, end));
};

/** 校验头的校验和，计算时将校验和字段的 8 个字节视为空格。不匹配表示这 512 字节不是有效的 tar 头。 */
function checksumOk(b: Uint8Array): boolean {
  const said = octal(b, 148, 8);
  let sum = 0;
  for (let i = 0; i < BLOCK; i++) sum += i >= 148 && i < 156 ? 32 : b[i];
  return sum === said;
}

/** 解包输出目标：每个成员调用一次 `open` 获取写入流，按块写入后 close。 */
export interface TarSink {
  open(name: string, size: number): Promise<WritableStream<Uint8Array>>;
}

/** 校验成员名称，禁止写到用户所选文件夹之外（`..`、绝对路径、Windows 盘符）。 */
function safeName(name: string): string {
  const clean = name.replace(/\\/g, "/").replace(/^\/+/, "");
  if (!clean || clean.startsWith("../") || clean.includes("/../") || clean.endsWith("/..") || /^[a-zA-Z]:/.test(clean))
    throw new TarError("E-DELIVER-TARUNSAFE", `unsafe member name: ${name}`);
  return clean;
}

/** 从流中按需读取指定字节数的缓冲读取器（不足时等待下一块；流结束仍不足时返回已读取的部分）。 */
class Bytes {
  private buf: Uint8Array[] = [];
  private held = 0;
  private done = false;
  private reader: ReadableStreamDefaultReader<Uint8Array>;
  // 使用显式赋值而非参数属性：Node 测试运行器以 strip-only 模式直接加载 TypeScript，
  // 不支持需要生成代码的语法。
  constructor(reader: ReadableStreamDefaultReader<Uint8Array>) {
    this.reader = reader;
  }

  /** 缓冲至少 n 个字节；流结束时可能少于 n。 */
  private async fill(n: number): Promise<void> {
    while (this.held < n && !this.done) {
      const { value, done } = await this.reader.read();
      if (done) this.done = true;
      else if (value.byteLength) {
        this.buf.push(value);
        this.held += value.byteLength;
      }
    }
  }

  /** 取出恰好 n 个字节；不足（包中途截断）时返回 null。 */
  async take(n: number): Promise<Uint8Array | null> {
    await this.fill(n);
    if (this.held < n) return null;
    const out = new Uint8Array(n);
    let at = 0;
    while (at < n) {
      const head = this.buf[0];
      const want = Math.min(head.byteLength, n - at);
      out.set(head.subarray(0, want), at);
      at += want;
      if (want === head.byteLength) this.buf.shift();
      else this.buf[0] = head.subarray(want);
      this.held -= want;
    }
    return out;
  }

  /** 取出至多 n 个字节，返回当前可用的部分（直接返回原缓冲区的视图，不额外复制）。 */
  async some(n: number): Promise<Uint8Array | null> {
    await this.fill(1);
    const head = this.buf[0];
    if (!head) return null;
    const want = Math.min(head.byteLength, n);
    if (want === head.byteLength) this.buf.shift();
    else this.buf[0] = head.subarray(want);
    this.held -= want;
    return head.subarray(0, want);
  }
}

/** 为本机 tar 包建立成员索引（名称 → 偏移和大小）。
 *
 * 交付保存为 tar 时，用户本机上的副本即为该包；视图绘制其中一帧时，
 * 须能单独取出该成员（`File.slice()`），而不必为一帧读取整包（EXR 序列可达数 GB）。
 * tar 不是压缩格式，仅读取各 512 字节的头即可依次跳过（一百帧只需一百次小读取）。
 *
 * tar.gz 不支持此方式：gzip 必须从头解压才能定位成员，等同于读取整包，
 * 因此该模式回落到服务器代理。
 *
 * 头的解析规则与 `untarInto` 相同（PAX 扩展头的 `path=` 作用于下一个成员），tar 格式规则仅在本文件中实现。
 * 无法继续读取（非 tar 或中途截断）时返回已识别的成员而不抛错：找不到不视为错误，
 * 调用方回落到代理（见 `files/localDirs.ts` 的模块说明）。 */
export async function indexTar(file: File): Promise<Map<string, { at: number; size: number }>> {
  const out = new Map<string, { at: number; size: number }>();
  const head = async (at: number) => new Uint8Array(await file.slice(at, at + BLOCK).arrayBuffer());
  let at = 0;
  let nextName = "";
  while (at + BLOCK <= file.size) {
    const h = await head(at);
    if (h.every((b) => b === 0)) break;
    if (!checksumOk(h)) break;
    at += BLOCK;
    const size = octal(h, 124, 12);
    const padded = Math.ceil(size / BLOCK) * BLOCK;
    const flag = String.fromCharCode(h[156] || 48);
    if (flag === "x" || flag === "X" || flag === "g") {
      const raw = new Uint8Array(await file.slice(at, at + size).arrayBuffer());
      for (const line of new TextDecoder().decode(raw).split("\n")) {
        const eq = line.indexOf("=");
        if ((eq < 0 ? "" : line.slice(0, eq).split(" ").pop()) === "path") nextName = line.slice(eq + 1);
      }
      at += padded;
      continue;
    }
    const prefix = text(h, 345, 155);
    const plain = text(h, 0, 100);
    const name = nextName || (prefix ? `${prefix}/${plain}` : plain);
    nextName = "";
    if (flag === "0" || flag === "\0" || flag === "7") {
      try {
        out.set(safeName(name), { at, size });
      } catch {
        break; // 名称不安全：停止解析，返回已识别的成员
      }
    }
    at += padded;
  }
  return out;
}

/** 流式解包 tar，将每个成员写入 `sink` 提供的流。`onBytes` 接收已写入的内容字节数（不含头和补齐）。
 *  `gz` 表示外层有 gzip（tar.gz），由浏览器内置的解压流处理。 */
export async function untarInto(body: ReadableStream<Uint8Array>, sink: TarSink,
                                onBytes?: (n: number) => void, gz = false): Promise<string[]> {
  // DecompressionStream 的 TS 类型为 BufferSource 输入、Uint8Array 输出，与 ReadableStream<Uint8Array>
  // 的泛型参数不一致，但运行时完全兼容，因此仅在此处做一次类型转换。
  const stream = gz
    ? (body as unknown as ReadableStream<Uint8Array>).pipeThrough(
        new DecompressionStream("gzip") as unknown as ReadableWritablePair<Uint8Array, Uint8Array>)
    : body;
  const src = new Bytes(stream.getReader() as ReadableStreamDefaultReader<Uint8Array>);
  const written: string[] = [];
  let bytes = 0;
  let nextName = "";  // 上一个 PAX 头中的 path=，作用于下一个成员
  for (;;) {
    const head = await src.take(BLOCK);
    if (!head) throw new TarError("E-DELIVER-TARBROKEN", "short read: header");
    if (head.every((b) => b === 0)) break;  // 全零块表示归档结束
    if (!checksumOk(head)) throw new TarError("E-DELIVER-TARBAD", "header checksum mismatch");
    const size = octal(head, 124, 12);
    const flag = String.fromCharCode(head[156] || 48);
    const prefix = text(head, 345, 155);
    const plain = text(head, 0, 100);
    const name = nextName || (prefix ? `${prefix}/${plain}` : plain);
    nextName = "";
    // 补齐到 512 的整数倍。不得写成 `size + (-size) % BLOCK`：JavaScript 的 % 保留被除数符号
    // （`-100 % 512` 为 -100，而 Python 为 412），会导致补齐量恒为 0，后续头错位。
    const padded = Math.ceil(size / BLOCK) * BLOCK;

    if (flag === "x" || flag === "X" || flag === "g") {  // PAX 扩展头：仅解析 path=
      const raw = await src.take(padded);
      if (!raw) throw new TarError("E-DELIVER-TARBROKEN", "short read: PAX header");
      for (const line of new TextDecoder().decode(raw.subarray(0, size)).split("\n")) {
        const eq = line.indexOf("=");
        const key = eq < 0 ? "" : line.slice(0, eq).split(" ").pop();
        if (key === "path") nextName = line.slice(eq + 1);
      }
      continue;
    }
    if (flag !== "0" && flag !== "\0" && flag !== "7") {  // 目录、链接及其他类型：跳过内容
      if (padded && !(await src.take(padded))) throw new TarError("E-DELIVER-TARBROKEN", "short read: skipped member");
      continue;
    }

    const to = await sink.open(safeName(name), size);
    const writer = to.getWriter();
    let left = size;
    try {
      while (left > 0) {
        const chunk = await src.some(left);
        if (!chunk) throw new TarError("E-DELIVER-TARBROKEN", `short read: ${name} needs ${left} more bytes`);
        await writer.write(chunk);
        left -= chunk.byteLength;
        bytes += chunk.byteLength;
        onBytes?.(bytes);
      }
      await writer.close();
    } catch (e) {
      await writer.abort(e).catch(() => {});
      throw e;
    }
    written.push(safeName(name));
    const pad = padded - size;
    if (pad && !(await src.take(pad))) throw new TarError("E-DELIVER-TARBROKEN", "short read: padding");
  }
  return written;
}
