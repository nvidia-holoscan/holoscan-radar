# Dev container

A ready-to-use development container for `holoscan-radar`. It reuses the module's
root [`Dockerfile`](../Dockerfile) — the same image `./holohub run-container` builds —
and mirrors the `holoscan-cli` in-container layout so paths are identical whether you
work here or through the `holohub` wrapper.

## What it does

- **Source-mounts the working directory.** The repo is bind-mounted at
  `/workspace/holoscan-radar` via Docker Compose, so every edit in the container is
  written back to your local checkout.
- **Exposes the GPUs.** Compose sets `gpus: all` (requires the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).
  On a GPU-less host, remove the `gpus:` line from `docker-compose.yml`.

## Opening it

- **VS Code / Cursor:** *Dev Containers: Reopen in Container* from the command palette.
- **CLI:** `devcontainer up --workspace-folder .`

## Mapping your own storage / data directory

By default the in-repo `./data` folder is mounted at `/workspace/holoscan-radar/data`
(the same path the `holohub` wrapper uses for `HOLOSCAN_CLI_DATA_DIR`).

To point at a different local location — a large scratch disk, a shared captures
mount, etc. — set the `HOLOSCAN_RADAR_DATA_DIR` environment variable **on the host**
before opening the container:

```bash
export HOLOSCAN_RADAR_DATA_DIR=/mnt/scratch/radar-data
```

Docker Compose reads that variable when creating the bind mount (see
[`docker-compose.yml`](docker-compose.yml)). The directory is exposed inside the
container at `/workspace/holoscan-radar/data`, so both your code and the `holohub`
CLI see data at the conventional path. The variable is per-developer and never
committed — leave it unset to fall back to `./data`.

To make the setting persistent, add the `export` line to your shell profile
(`~/.bashrc`, `~/.zshenv`, …). VS Code / Cursor may need a restart to pick up a
newly exported variable.

## Overriding the base image

The container is pinned to the SDK version in the [`Dockerfile`](../Dockerfile)
(`BASE_IMAGE`). To build against a different Holoscan SDK image, export
`HOLOSCAN_RADAR_BASE_IMAGE` on the host before opening the container:

```bash
export HOLOSCAN_RADAR_BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v4.2.0-cuda13
```

Docker Compose forwards it as the `BASE_IMAGE` build arg (see
[`docker-compose.yml`](docker-compose.yml)); unset falls back to the Dockerfile default.
