# third_party

Paper repositories, model weights and their isolated Python environments live
here. Nothing in this folder except this README is part of Lab2Shot's source
tree: each package is downloaded from its original authors by

    lab2shot ext install <name>

and keeps its own license. Lab2Shot never modifies the files in `repo/`.

Layout per package:

    third_party/<name>/
        repo/                 original repository, pinned commit, untouched
        weights/              model weights
        .venv/                isolated environment (own torch / CUDA build)
        install_state.json    what has been installed, versions, weight status
