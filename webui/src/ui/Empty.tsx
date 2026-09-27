import "./empty.css";

/** What a place shows when it has nothing in it yet: what the state is, and the next step — never a
 * blank area. `title` says what is missing, `hint` what to do about it. */
export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty" role="status">
      <div>
        <div className="empty-title">{title}</div>
        {hint && <div className="empty-hint">{hint}</div>}
      </div>
    </div>
  );
}
