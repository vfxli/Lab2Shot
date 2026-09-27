import type { ReactNode } from "react";
import type { Message } from "../messages/message";
import { shorten } from "../messages/shorten";
import { LEVEL_TABLE } from "../messages/levels";
import "./message.css";

/** One message as displayed by the page: the level's square with its letter, the text, the code at the end, and any
 * content the caller appends (the button that fixes it). With `onGo` it is a list row that navigates to its subject;
 * without it, a sentence displayed in place (the information card's 提醒). The level's colour comes from the single
 * table (messages/levels.ts); red is used only for production risk. */
export function MessageRow({ message, onGo, tip, after, named = "" }: {
  message: Message;
  onGo?: () => void;
  tip?: string;
  after?: ReactNode;
  named?: string; // 该行上方已显示的节点名：正文若也以其开头，则不再重复
}) {
  // 名称只显示一次：消息按节点分组，组标题已显示节点名，正文若再写一遍就会出现
  // 「「序列图输出设置」「序列图输出设置」没有用上「图像」」这类重复。此处去除两种重复：
  //   ① 核心消息以「{node}」开头（带书名号）；
  //   ② 扩展消息以项目名开头（「MonST3R 在第 120–145 帧…」），而节点名为
  //      「MonST3R 相机解算」，项目名是节点名的开头部分，同样构成重复。
  //      安装和环境类消息显示在扩展卡片上，没有节点名，因此只在此处（有节点名时）去除，不修改模板。
  const text = shorten(message.text, named);
  return (
    <div className={`msg-row ${message.level}${onGo ? " go" : ""}`} data-code={message.code} data-level={message.level} onClick={onGo} data-tip={tip}>
      <i className="msg-dot" data-tip={LEVEL_TABLE[message.level].tip}>
        {message.level}
      </i>
      <span className="msg-text">
        {text}
        <span className="msg-code">{message.code}</span>
      </span>
      {after}
    </div>
  );
}
