// 画面通知区的通知种类。ui/ViewNotices.tsx 负责绘制，state/viewTools.ts 保存舞台提交的通知。
// 类型定义放在 model 中供两侧引用，避免 state 依赖 ui（由 tests/layers.test.ts 检查）。

/** 画面上的所有文字均由此处定义：统一通知区位于画面左上角，仅显示文字，不使用浮动圆角框，各条对齐。
 * 本机画面、手柄提示、点云代理、三维错误、相机标签均在此显示，其他位置不得在画面上绘制文字。
 *
 *   local    本机画面（仍在上传或服务器尚未读取）
 *   proxy    点云以代理方式显示，并非全部点
 *   camera   当前透过的相机及其焦距（毫米）
 *   error    三维显示错误
 *   hint     当前手柄的用法
 *   partial  边算边看：计算尚未完成
 *   did      视图刚自动执行的操作（4 秒后自动消失） */
export type NoticeKind =
  | "local"
  | "looked"   // 当前浏览器无法创建 WebGL2 上下文：画面按原样显示
  | "proxy"
  | "camera"
  | "error"
  | "hint"
  | "partial"
  | "did";

