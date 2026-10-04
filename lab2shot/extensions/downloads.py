"""Generic tools an extension declares its downloads with: no file is named here.

Download is one pinned file (its URL and sha256; hf() builds one for a Hugging Face file at a pinned revision), and
Download.weight turns it into one of the extension's weights (its own key, where its worker reads it). pip_git and
archive_zip turn a pinned GitSource into a pip requirement or the URL of its source archive.

Every extension pins the files and sources it uses in its own declaration (adapters/<name>/extension.py), even when
another extension uses the same file: a new algorithm never needs a core edit. The installer stores identical files
once (by sha256), so the same pin written in two extensions costs no second download."""

from __future__ import annotations

from dataclasses import dataclass

from .spec import GitSource, Weight


@dataclass(frozen=True)
class Download:
    source: str  # the URL (a Hugging Face file at a pinned revision: hf())
    sha256: str  # of the file (kind "url") or of the archive (kind "zip")
    kind: str = "url"
    repo: str = ""  # a Hugging Face file: its repository, revision and path in it (tables that list them apart)
    revision: str = ""
    filename: str = ""

    def weight(self, *, key: str, dest: str, **kw) -> Weight:
        """This file as one extension's weight: its own key, where its worker reads it, what it is for there."""
        return Weight(key=key, kind=self.kind, source=self.source, dest=dest, sha256=self.sha256, **kw)


def hf(repo: str, revision: str, filename: str, sha256: str) -> Download:
    """A Hugging Face file at a pinned commit (the file's LFS sha256)."""
    return Download(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", sha256, repo=repo, revision=revision,
                    filename=filename)


# ------------------------------------------------------------------ pinned git sources

def pip_git(name: str, source: GitSource) -> str:
    """A pip requirement installing `name` from a pinned git source."""
    return f"{name} @ git+{source.url}@{source.commit}"


def archive_zip(source: GitSource) -> str:
    """A GitHub source's archive of its pinned commit (a zip download of the code)."""
    return f"{source.url.removesuffix('.git')}/archive/{source.commit}.zip"

