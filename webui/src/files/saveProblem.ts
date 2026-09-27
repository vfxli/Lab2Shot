/** 将保存文件失败时浏览器抛出的异常分类为消息编号。
 *
 * 浏览器的原始错误文本（例如重启后出现的「An operation that depends on state cached in an interface object was made
 * but the state had changed since it was read from disk.」）对用户不可读，也未给出处理方法，因此可识别的几类
 * 各映射到一个消息编号。代码中只包含编号和参数，中文模板统一位于消息目录（lab2shot/messages/web.toml）。
 *
 * 本文件独立且不导入任何模块，以便通过 node --test 单独测试。 */
export function saveProblem(e: unknown): { code: string; reason: string } {
  const name = (e as { name?: string } | null)?.name ?? "";
  const said = e instanceof Error ? e.message : String(e);
  // 解包模块抛出的异常（files/untar.ts TarError）：message 即为消息编号，英文详情放入 reason 供日志使用。
  if (name === "TarError" && said.startsWith("E-DELIVER-"))
    return { code: said, reason: (e as { detail?: string }).detail || said };
  const low = said.toLowerCase();
  // 所选位置已失效：浏览器保存的文件句柄与磁盘状态不一致（重启、文件被移动或重命名、授权失效）。
  if (name === "InvalidStateError" || low.includes("state had changed since it was read from disk"))
    return { code: "E-DELIVER-STALEPLACE", reason: said };
  if (name === "NotAllowedError" || name === "SecurityError") return { code: "E-DELIVER-NOPERMISSION", reason: said };
  if (name === "NotFoundError") return { code: "E-DELIVER-PLACEGONE", reason: said };
  if (name === "NoModificationAllowedError") return { code: "E-DELIVER-LOCKED", reason: said };
  if (name === "QuotaExceededError") return { code: "E-DELIVER-DISKFULL", reason: said };
  return { code: "E-DELIVER-SAVEFAILED", reason: said };
}
