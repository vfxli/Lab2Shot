# Lab2Shot clients

DCC plugins are thin clients of a running Lab2Shot service (`lab2shot ui`).
All the heavy work — models, GPU, cache — stays on the Lab2Shot machine.
Files stay on the artist's side: the client uploads the inputs it is given
(the same bytes only once) and writes the outputs to the paths it is given.
The server never reads or writes the artist's paths.

| Folder | What |
| --- | --- |
| `../lab2shot/client.py` | One-file client, Python standard library only (3.7+), nothing else from lab2shot. Every plugin copies it as `lab2shot_client.py`; `lab2shot cook` uses it too. |
| `houdini/` | (planned) Shelf tool / HDA: pick a template, fill its exposed parameters, import the USD / Alembic it writes. |
| `maya/` | (planned) Menu + dialog built from the same exposed parameters. |
| `nuke/` | (planned) Nodes for plates, mattes, depth and ST-maps. |

## How it fits together

- **Templates are the tools.** Every template in `templates/` lists its
  *exposed* parameters. `GET /api/tools` returns
  them with their type, default and choices, so a plugin can build its
  dialog without knowing anything about the node graph.
- **Files are the exchange format.** A run writes USD / Alembic / EXR to the
  paths you give (local or a shared drive, any OS); the plugin imports those
  files. Nothing is streamed into the DCC.
- **Same cache as the web UI.** Unchanged steps are reused, so re-running a
  tool after tweaking one parameter only recomputes what changed.
- **Same queue as everyone.** A run is a job in the server's queue; the
  client sends its computer name, OS user and application, so the queue
  (web 队列 window, admin page) shows whose job is whose.

## HTTP API and command line

Every endpoint is described, generated from the code, in the OpenAPI document
`GET /api/admin/openapi.json` (administrator login; `lab2shot --help` lists the
commands).
