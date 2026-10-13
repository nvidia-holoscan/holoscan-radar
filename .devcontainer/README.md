# Dev container

A ready-to-use standalone development container for `holoscan-radar`. It reuses
the module's root [`Dockerfile`](../Dockerfile) and mirrors the `holoscan-cli`
in-container layout.

## What it does

- **Source-mounts the working directory.** The repo is bind-mounted at
  `/workspace/holoscan-radar`, so every edit in the container is written back to
  your local checkout.
- **Exposes the GPUs.** The dev container passes `--gpus=all` (requires the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).
  A CUDA-capable NVIDIA GPU is required to build and run the operator suite.
- **Runs as a non-root `dev` user** (team convention). The devcontainer runtime
  updates its UID/GID to match your host account so the bind-mounted source stays
  writable.

## The `dev` user

You work inside the container as `dev` (passwordless `sudo` available), not root.
The image creates `dev` at UID/GID `1000` (reclaiming those ids from the base
image's `ubuntu` account), and the devcontainer runtime updates them to match
your host user if it isn't `1000`. That keeps files written through the source
mount owned by your host account.

Passwordless `sudo` is enabled for `dev`, matching common dev-container practice.
Treat the container as a development environment: code and tools you run inside it
can escalate to root within the container.

This non-root workflow is specific to the dev container. The
`./holohub run-container` path intentionally keeps the Dockerfile's default root
user for compatibility with the CLI and packaging flows, so files created through
that path may have different ownership behavior on bind mounts.

If your host UID/GID change, rebuild the container (*Dev Containers: Rebuild
Container*) so the `dev` user is recreated with the new ids.

## Opening it

- **VS Code / Cursor:** *Dev Containers: Reopen in Container* from the command palette.
- **CLI:** `devcontainer up --workspace-folder .`

## Host display (Holoviz)

Startup forwards the host X11 display into the container so apps like
`siso` with `visualize: true` can open a window:

- `initializeCommand` runs `xhost +local:docker` (or `+local:`) on the **host**
- `DISPLAY` is copied from the host via `containerEnv`
- `/tmp/.X11-unix` is bind-mounted

Rebuild/reopen the container after pulling these settings. On a headless host
(or without `xhost`), startup still succeeds; visualization simply won't work.
If Holoviz still can't authenticate after a rebuild, also mount the host
Xauthority cookie (not enabled by default — paths vary by desktop).

## Data Directory

The in-repo `./data` folder is available at `/workspace/holoscan-radar/data`,
the same path the Holoscan CLI uses for this repository's `HOLOSCAN_CLI_DATA_DIR`.

## SDK base image

The dev container uses the HSDK 5 staging image. Set your **user org** in NGC
Catalog to `nvstaging`, then authenticate to NGC and pull the image before
opening the container:

```bash
docker login nvcr.io
docker pull nvcr.io/nvstaging/holoscan/holoscan:v5.0.0.7-cuda13
```

The image version, Holoscan CLI commands, and current packaging limitation are
documented in
[`SDK_DEVELOPMENT.md`](../SDK_DEVELOPMENT.md).
