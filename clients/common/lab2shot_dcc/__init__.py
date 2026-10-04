"""Lab2Shot DCC plugin framework: everything a DCC plugin does that does not depend on the DCC.

Layers (design: 设计_DCC插件_Maya.md §12):
  backend    lab2shot/server/...           tools of four sources, typed signatures, formats, jobs, uploads, outputs
  transport  lab2shot_client.py            HTTP, login and token, upload, submit, poll, cancel, download
  framework  lab2shot_dcc (this package)   connection, tool list and ranking, the panel (view_model: what it shows,
                                           no Qt; ui: the Qt panel, the same in every DCC), job
                                           lifecycle on a background thread, guards, names and paths, result
                                           versions, delivery manifest, format preferences
  host       clients/<dcc>/                a lab2shot_dcc.host.Host for that DCC, and how the DCC loads the plugin

Nothing in this package imports a DCC module. The user interface depends on Qt only (lab2shot_dcc.qt: PySide6 or
PySide2, whichever the DCC brings).
"""

__all__ = ["paths", "log", "guard", "safety", "host", "connection", "catalog", "contract", "results", "jobs", "nodes",
           "view_model"]
