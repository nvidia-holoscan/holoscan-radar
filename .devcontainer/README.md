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
  On a GPU-less host, remove `runArgs` from `devcontainer.json`.
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
`fmcw_siso` with `visualize: true` can open a window:

- `initializeCommand` runs `xhost +local:docker` (or `+local:`) on the **host**
- `DISPLAY` is copied from the host via `containerEnv`
- `/tmp/.X11-unix` is bind-mounted

Rebuild/reopen the container after pulling these settings. On a headless host
(or without `xhost`), startup still succeeds; visualization simply won't work.
If Holoviz still can't authenticate after a rebuild, also mount the host
Xauthority cookie (not enabled by default — paths vary by desktop).

## Data Directory

The in-repo `./data` folder is available at `/workspace/holoscan-radar/data`,
the same path the `holohub` wrapper uses for `HOLOSCAN_CLI_DATA_DIR`.

## [EA HSDK5 ONLY] Build the SDK base image

This temporary workflow creates a devcontainer with HSDK 5 installed. It does
not build or update the radar module, which is expected to remain incompatible
until it is ported from HSDK 4.3.

On the host, from the root of this repository, run:

```bash
./.devcontainer/build-hsdk5-ea-image.sh
```

The script clones the internal HSDK repository at `v5.0.0.1` into the ignored
`.cache/` directory, builds the SDK, and creates the local image already named
in [`devcontainer.json`](devcontainer.json). It requires internal GitLab SSH
access and Docker.

After it completes, rebuild and reopen the devcontainer. Re-run the script only
when the pinned HSDK build needs to be recreated.

### Using `./holohub`

The devcontainer configuration does not change the image selected by
`./holohub`, which continues to default to HSDK 4.3. To launch its container
against the EA image from the host, run:

```bash
./holohub run-container \
  --base-img holoscan-radar-hsdk5-ea:v5.0.0.1
```

Inside the EA devcontainer, use local execution so `holohub` uses the HSDK 5
installation already present instead of starting another 4.3 container:

```bash
./holohub build fmcw_siso --local
./holohub run fmcw_siso --local
./holohub test --local
```

### [EA HSDK5 ONLY] Removal before GA

All temporary EA integration is marked `[EA HSDK5 ONLY]`. Before merging the
HSDK 5 port into `main` for the public release:

1. Delete `build-hsdk5-ea-image.sh` and `sdk-installed.Dockerfile`.
2. Remove the marked `build.args.BASE_IMAGE` block from `devcontainer.json`.
3. Remove this marked section from the devcontainer README.
4. Search the repository for `[EA HSDK5 ONLY]` and remove any remaining
   temporary paths.
