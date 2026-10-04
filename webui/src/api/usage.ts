// Usage statistics (GET /api/admin/usage, lab2shot/farm/usage.py): per project, node, department and person.
// Re-exported by api/index.ts.

export interface UsageCounts {
  runs: number; // it computed
  reuses: number; // answered without computing: a cached result, or a worker's raw results from before
  seconds: number; // computing
  gpu_seconds: number; // the part of it on a GPU
  frames: number; // processed by its runs
  users: string[]; // who asked
  last: number | null; // last used since the last reset, in the range or before it
}

interface UsageNode extends UsageCounts {
  id: string;
  subtitle: string; // the node's subtitle
}

export interface UsageProject extends UsageCounts {
  name: string; // the extension's name; "core" for the core's nodes
  title: string;
  core: boolean;
  installed: boolean; // false: used in the range but no longer installed
  daily: { runs: number[]; reuses: number[]; seconds: number[] }; // per day of `days`
  nodes: UsageNode[];
}

/** A project in someone's usage (a department's person, a person): its counts. */
export interface UsageShare extends UsageCounts {
  name: string;
  title: string;
}

/** A person as a department sees them, with their projects. */
interface UsageMember extends UsageCounts {
  name: string;
  projects: UsageShare[];
}

export interface UsageDepartment extends UsageCounts {
  name: string;
  listed: boolean; // in the settings' list (false: a department taken off it since, or 未分环节)
  daily: UsageProject["daily"];
  people: UsageMember[];
}

export interface UsagePerson extends UsageCounts {
  name: string;
  departments: string[];
  daily: UsageProject["daily"];
  projects: UsageShare[];
}

export interface UsageStats {
  since: number | null; // what the range really starts at (never before the last reset)
  until: number | null;
  start: number | null; // the last reset (null: counting since the first job)
  undo: boolean; // a reset can be undone
  days: string[]; // YYYY-MM-DD in the viewer's time zone
  projects: UsageProject[]; // third-party projects, the most used first, then the core
  departments: UsageDepartment[]; // every listed department, the busiest first; others after
  people: UsagePerson[]; // who submitted in the range, the most compute first
  nobody: string; // the row of accounts without a department (未分环节)
}


/** The server's usage answer filled in to what the page reads directly (in this one place, so the page never checks
 * for absence piecemeal): a missing list as empty, each project's three per-day columns padded with 0 to the length of
 * `days` (one missing day would make a total NaN). */
export function usageFilled(u: UsageStats): UsageStats {
  const n = (u.days ?? []).length;
  const col = (xs: number[] | undefined) => Array.from({ length: n }, (_, i) => xs?.[i] ?? 0);
  return {
    ...u,
    days: u.days ?? [],
    projects: (u.projects ?? []).map((p) => ({ ...p, users: p.users ?? [], daily: { runs: col(p.daily?.runs), reuses: col(p.daily?.reuses), seconds: col(p.daily?.seconds) } })),
    departments: (u.departments ?? []).map((d) => ({ ...d, daily: { runs: col(d.daily?.runs), reuses: col(d.daily?.reuses), seconds: col(d.daily?.seconds) } })),
    people: (u.people ?? []).map((q) => ({ ...q, daily: { runs: col(q.daily?.runs), reuses: col(q.daily?.reuses), seconds: col(q.daily?.seconds) } })),
  };
}
