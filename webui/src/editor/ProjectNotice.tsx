/** The project's word to the research authors, with the address to write to about a licence: always in sight on the
 * node graph — its bottom left corner, or under the welcome card while the graph is empty (editor/Welcome.tsx), where
 * the corner would run under the card. Plain text over everything, taking no pointer: every click, drag and wheel
 * goes through to what is under it. */
export function ProjectNotice({ className }: { className: string }) {
  return (
    <p className={className}>
      本站的算法均来自公开发表的研究，成果归原作者所有。如涉及授权问题，恳请联系 vfx_dli@foxmail.com，我会及时下架相应算法。
      <br />
      Every method here comes from published research and belongs to its original authors. For any licensing concern,
      please contact vfx_dli@foxmail.com and I will take it down promptly.
    </p>
  );
}
