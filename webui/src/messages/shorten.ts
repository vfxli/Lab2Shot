/** 正文里去掉和上面那行重复的名字：「节点名」整段，或项目名（节点名开头那一段）。
 *
 * 消息按节点分组，组上已经写着节点名，正文里再写一遍就成了「「序列图输出设置」「序列图输出设置」没有用上……」。
 *
 * 扩展的消息模板以项目名开头是对的：安装和环境那些消息显示在扩展卡片上，那里没有节点名，项目名必须留着。
 * 所以去重只发生在有节点名的地方（页面日志 state/say.ts），不改模板。
 *
 * 自己一个文件、不 import 任何东西。 */
export function shorten(text: string, named: string): string {
  if (!named) return text;
  const quoted = `\u300c${named}\u300d`;
  if (text.startsWith(quoted)) return text.slice(quoted.length);
  const head = text.split(/[\s\uff0c\u3002\uff1a]/, 1)[0];
  return head.length >= 2 && named.startsWith(head) ? text.slice(head.length).replace(/^\s+/, "") : text;
}
