import "./Swatches.css";

/** Pick one colour of a few (the 3D view's colours, a network box's colour). `disabled`: why it cannot be
 * picked now (the control never disappears: it greys and says why); an empty string or false means it can. */
export function Swatches({ value, colors, onChange, label, size = "md", layout, disabled }:
  { value: string; colors: readonly string[]; onChange: (c: string) => void; label: string; size?: "md" | "sm"; layout?: string; disabled?: string | false }) {
  return (
    <div className={["swatches", size === "sm" && "sm", layout].filter(Boolean).join(" ")} role="group" aria-label={label}>
      {colors.map((c) => (
        <button
          key={c}
          type="button"
          className={`swatch${c.toLowerCase() === value.toLowerCase() ? " on" : ""}`}
          style={{ ["--c" as string]: c }}
          aria-label={c}
          aria-pressed={c.toLowerCase() === value.toLowerCase()}
          disabled={!!disabled}
          data-tip={disabled || `${label}：${c}`}
          onClick={() => onChange(c)}
        />
      ))}
    </div>
  );
}
