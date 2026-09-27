"""What the layers above hand to nodes and the engine: the one place a lower layer names something only a higher one
can do, as a declaration it reads, never an import upwards (the layering has no exceptions).

    ProjectFacts   a node type's project, stamped on its class by the extension loader (lab2shot/adapters.py): its
                   title, licence class, the tags its extension brings, whether it is usable now, and the extension
                   spec its worker runs from. Nodes, tags and the engine read `cls.project`, never the registry.
    PlanEnv        what planning needs from outside a cook: the file an upload reference names, and a quick worker job
                   run outside any cook (an import node's listing).
    DeliverySink   where 「输出」 hands its files over: built per run by whoever submits the cook (the farm), given only to
                   a node that delivers (CookContext.delivery).
    Services       the extension nodes and the plan environment of this process, installed once by the top layer
                   (lab2shot/catalog.py install(): the server, the command line, the tests and tools call it before
                   they read a node type). Reading them before that is an error that says so, never core nodes only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..messages import Msg
from .tags import BASIC, RESEARCH

if TYPE_CHECKING:
    from ..data.packet import Packet
    from ..extensions.spec import Extension
    from .base import NodeDef


@dataclass(frozen=True)
class ProjectFacts:
    """A node type's project, as the extension loader knows it (lab2shot/adapters.py project_of)."""

    title: str  # "Lab2Shot", or the project as people know it ("ViPE")
    licence: str  # its extension's licence class (nodes/tags.py LICENCES); a node's own Licence overrides it
    tags: frozenset[str]  # the other tags its extension brings (such as 需注册)
    available: Callable[[], Msg | None]  # None: usable now (its extension installed, ready, built from this code); else why not
    extension: Extension | None = None  # the spec its worker runs from (engine/external.py); None for the core
    # What this extension's results depend on: repository commit + environment (python / torch / dependencies / build
    # scripts) + the sha256 of every declared weight. It enters the node fingerprint (engine/evaluation.py), so after
    # reinstalling the extension, changing weights or changing the torch version, old results no longer hit the cache;
    # a manually maintained node version number cannot detect changed weights, and users would receive results from
    # an old model believing them new. Computed and stamped where extensions are loaded (adapters.py project_facts):
    # the engine only reads this string and does not depend upwards on the installer layer.
    result_identity: str = ""


CORE_PROJECT = ProjectFacts("Lab2Shot", BASIC, frozenset(), lambda: None)


def unloaded_project(runtime: str) -> ProjectFacts:
    """The project of a node whose extension the loader never stamped (not loaded): the strictest licence, never
    usable, so it is never offered as usable by mistake."""
    return ProjectFacts(runtime, RESEARCH, frozenset(), lambda: Msg("E-EXT-NOTINSTALLED", title=runtime, name=runtime))


class ExtensionNodes(Protocol):
    """The node types of the extensions that loaded, and why one is not there (lab2shot/adapters.py Adapters)."""

    nodes: Mapping[str, type[NodeDef]]

    def why_missing(self, type_id: str) -> Msg: ...


class PlanEnv(Protocol):
    """What planning a graph needs from outside a cook (lab2shot/catalog.py gives the real one)."""

    def upload(self, ref: str) -> Path:
        """Where the nodes read an upload reference (lab2shot/transfer/uploads.py resolve)."""
        ...

    def declared_upload(self, ref: str) -> dict | None:
        """For source bytes not yet uploaded, the description read from the file header at declaration
        (`{"of": first file, "layers": {...}, "frames": [...], "size": (width, height)}`; None when not declared or
        the header has not been read yet).

        A reader node's output ports come from opening the file, but a selected file should not be uploaded
        immediately; therefore only the first few KB of the first frame are sent on selection, and the server reads the
        layers with its own `describe_file` (transfer/uploads.py). This is the only way nodes obtain it.

        Stereo files (left and right eye) are grouped by the file's first view at declaration: reading without a view
        mixes the channels of both eyes, so the declared port count would differ from the count after the upload, and
        the ports would change without any message (see `nodes/core/input.py _view`).

        It is used only to create ports and fill options, never for computation: an actual cook requires all bytes and
        goes through `upload()` as before."""
        ...

    def declared_layers(self, path: Path) -> dict | None:
        """When this file in the upload folder links a channel subset, all layers of its original file (the declared
        ones; `transfer/uploads.py declared_layers_for`); None when it is not a subset, in which case counting from the
        opened file is correct. Reader node ports follow it (`nodes/core/input.py _layers_of`): the subset contains only
        the wired channels, and the ports must not shrink with it."""
        ...

    def channel_plan(self, ref: str, wanted: list[str]) -> dict | None:
        """The channels of this upload to transfer for this cook:
`{"take": [names in the file], "write": [names in the subset]}`;
        None = transfer the whole file (all channels needed, not a declared picture, multi-part EXR). See
        `transfer/uploads.py channel_plan`. `wanted` are the channel names the node states
        (`ReadSequence.upload_channels` derives them from the wired ports)."""
        ...

    def content_id(self, path: Path) -> str:
        """The content identity of this file, part of the node fingerprint (engine/evaluation.py).

        Uploaded sources are content-addressed (transfer/uploads.py: the manifest records each file's sha256), so this
        does not read the file. Only paths that are not uploads (datasets, benchmark sources on the server) are hashed.

        Size and modification time are not sufficient: CG production often replaces files in place, and some
        applications write data without changing the modification time, so edited sources would still hit an old cache."""
        ...

    def ask_worker(self, node_type: type[NodeDef], params: dict, inputs: dict[str, Path]) -> Path:
        """Run the node type's worker job outside any cook (engine/external.py ask_worker); its raw folder."""
        ...


class DeliverySink(Protocol):
    """Where 「输出」 hands its files over (lab2shot/transfer/deliveries.py Sink): it makes the delivery, reports it as the
    run's "output" event and returns its record (commercial, files, bytes ...)."""

    def deliver(self, node_id: str, label: str, outputs: list[Packet], name: str, mode: str,
                item: str = "") -> dict[str, Any]: ...  # `item`: the instance's item path ("" outside every block)


@dataclass(frozen=True)
class Services:
    extensions: Callable[[], ExtensionNodes]  # called when the node types are first read (loading is not free)
    plan: PlanEnv


_installed: Services | None = None


def install(services: Services) -> None:
    """Hand this process's services to the lower layers (lab2shot/catalog.py install)."""
    global _installed
    _installed = services


def services() -> Services:
    if _installed is None:
        raise RuntimeError("no Lab2Shot services installed: call lab2shot.catalog.install() first (the server, the command line, "
                           "the tests and the tool scripts do it when they start)")
    return _installed
