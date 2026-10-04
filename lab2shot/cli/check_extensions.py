"""`lab2shot check extensions`: what every extension must state, and the rules of a base and what runs in it.

- generative: every extension states whether it runs a generative diffusion model (Extension.generative, True or False:
  no default to forget it by); the loader stamps the gate 生成式扩散 on its nodes from it.
- runs_in: an extension running in another's environment names a loaded base (an extension with no nodes, itself
  running in none), and declares nothing of the code or environment (no source, env, requirements or lock of its
  own: one copy, the base's).
- a base (no nodes) is there for something: at least one extension runs in it.
"""

from __future__ import annotations

from .. import i18n

OWN_ENV_FILES = ("requirements.txt", "requirements.lock")


def check_extensions(r) -> None:
    from ..extensions.registry import extensions

    exts = extensions()
    bad = 0
    for name, ext in sorted(exts.items()):
        cls = type(ext)
        if not isinstance(ext.generative, bool):
            r.bad(i18n.t("cli.check.extensions.generative", name=name))
            bad += 1
        if ext.runs_in:
            host = exts.get(ext.runs_in)
            if host is None or not host.is_base or host.runs_in:
                r.bad(i18n.t("cli.check.extensions.host", name=name, host=ext.runs_in))
                bad += 1
            own = [w for w in ("source", "env") if w in cls.__dict__]
            own += [f for f in OWN_ENV_FILES if (ext.adapter_dir / f).exists()]
            if own:
                r.bad(i18n.t("cli.check.extensions.own_env", name=name, host=ext.runs_in, what=i18n.separator().join(own)))
                bad += 1
        if ext.is_base and not any(e.runs_in == name for e in exts.values()):
            r.bad(i18n.t("cli.check.extensions.lone_base", name=name))
            bad += 1
    if not bad:
        r.ok(i18n.t("cli.check.extensions.ok", count=len(exts),
                    bases=i18n.separator().join(n for n, e in sorted(exts.items()) if e.is_base) or "-"))
