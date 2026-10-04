# Lab2Shot

[中文](README.md) | **English**

<p align="center"><img src="docs/images/welcome.png" alt="Lab2Shot" width="900"></p>

Lab2Shot brings machine-learning research into computer-graphics production. It packages published research code — camera solving, matting, body and face capture, lighting estimation and related methods — as nodes that film and game artists can use directly. All results are delivered in standard formats: USD for 3D data (Y-up, centimetres) and EXR / PNG / JPG for images, color-managed with OCIO, ready for import into Houdini, Maya and Nuke.

The aim is simple: to let more artists on the production floor work with the latest research. Every method here is the work of its original authors; Lab2Shot only connects it to production and leaves their code as it is.

## Features

Lab2Shot connects third-party research projects through adapters and organizes them into nodes by production task. For shots and scenes, it solves the camera and lens distortion from live-action plates, reconstructs the scene's 3D geometry, and estimates depth and normals. For plates, it does matting, masks, tracking, optical flow and alignment. For people, it captures body motion and facial expressions from video and edits motion. It also estimates the scene's lighting and materials. Every result is delivered in the same formats.

The nodes, inputs and outputs, VRAM needs and license notes of each adapter are in `adapters/<project>/docs.md`.

## Installation

Requires Linux or WSL2, an NVIDIA GPU and driver, and Node.js 20.19+ or 22.12+ (Python is installed automatically by [uv](https://docs.astral.sh/uv/)). In the repository root, run:

```bash
./setup.sh
```

Follow the menu through the "First Installation Wizard", the "Extension Build and Download Settings Wizard" and the "Extension Installation Wizard"; each step explains what it does before it runs. Installation, the service, settings and the database are all managed from this menu.

A few models must be downloaded by hand or need access requested first; see [docs/manual-downloads.md](docs/manual-downloads.md).

## Upgrading

Back up the work folder, stop the service in the old version (optionally waiting for running jobs to finish), pull the code by hand, then run the new version's upgrade:

```bash
./setup.sh stop
git pull --ff-only
./setup.sh update
```

Commit or stash local code changes before pulling, and resolve any conflicts Git reports. The upgrade backs up the settings, locally managed files and the database, then syncs dependencies, builds the web page, runs the checks, migrates the database and starts the service. It does not run the Git pull. If it fails after the environment sync, it restores the previously deployed code, database and environment and starts the old version; the first upgrade takes the version to return to from the pull you just made. After a successful upgrade, reinstall the extensions it lists as needing it.

## Usage

Open the service address in a browser and log in, create a node graph from a template, choose the footage, adjust parameters and submit the job; jobs are queued and cooked on the GPU, and results are downloaded when they finish.

## Architecture

- The core (`lab2shot/`) imports no third-party research project; it only handles the interface, scheduling, node graphs and unified input and output.
- Each research project has one folder, `adapters/<project>/`, and is cooked by a worker process in its own isolated environment, exchanging only files and messages with the core.
- Third-party repositories are fetched as is, at locked versions, into `third_party/`, without any modification.

## License

- Lab2Shot is licensed under the GNU Affero General Public License version 3 or (at your option) any later version (AGPL-3.0-or-later); the full text is in [LICENSE](LICENSE).
- [LICENSE-EXCEPTION.md](LICENSE-EXCEPTION.md) is an additional permission granted under Section 7 of the AGPL: it allows Lab2Shot to be combined at run time with the third-party projects the installer fetches (including running as a network service), without extending the AGPL to those third-party projects.
- Third-party projects are fetched by the installer from their original sources on the user's own computer, each under its own license; Lab2Shot distributes no code or model weights of any third-party project. Each project's license can be viewed with `uv run lab2shot ext info <project>`.
- Some third-party projects are for non-commercial use only. Using the nodes such a project provides in a commercial project is subject to that project's license.

## Acknowledgements

Lab2Shot is built on a large body of published research; our thanks go to every author who released their code and models. The source, authors and license of each third-party project are in the notes in its adapter folder.
