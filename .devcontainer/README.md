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

This temporary workflow creates a devcontainer with HSDK 5 installed; it does
not build the radar module. The module is **partially** ported: the allow-listed
operators and their C++ tests build and run, while the FMCW range/Doppler and
MIMO marshaling operators, the Python bindings, and the demo applications are
still gated off. See [`../docs/planning/HSDK5_PORT.md`](../docs/planning/HSDK5_PORT.md)
for the current state.

On the host, from the root of this repository, run:

```bash
./.devcontainer/build-hsdk5-ea-image.sh
```

The script clones the internal HSDK repository into the ignored `.cache/`
directory, builds the SDK, and creates a local image. It requires internal
GitLab SSH access and Docker.

The default pin is the branch `deustice/5x/optional-port-contracts`, which
carries the optional-unwired-input feature this module needs in order to retire
its `expect_*` parameters ([`../docs/planning/HSDK5_PORT.md`](../docs/planning/HSDK5_PORT.md)
§11), and [`devcontainer.json`](devcontainer.json) names the image it produces.
The module no longer builds against the `v5.0.0.1` EA tag: it uses that
branch's Tensor spec API, and its operators declare optional `[live]` inputs
the tag rejects as unconnected. Because a branch pin's image tag carries its
short sha, re-run with `--print-pin` and update `BASE_IMAGE` whenever the
branch moves.

After it completes, rebuild and reopen the devcontainer. Re-run the script only
when the pinned HSDK build needs to be recreated.

### Changing the SDK pin

`--pin` accepts a tag, a branch, or a commit sha (`$HSDK_PIN` sets the same
thing). Each pin gets its own `.cache/` checkout and its own image tag, so
several pins can coexist:

```bash
./.devcontainer/build-hsdk5-ea-image.sh --pin v5.0.0.1
./.devcontainer/build-hsdk5-ea-image.sh --pin 9bd905908
./.devcontainer/build-hsdk5-ea-image.sh --pin deustice/5x/optional-port-contracts
```

A tag or sha pin is immutable, and the script verifies the cached checkout
against it without touching the network. A branch pin is resolved with
`git ls-remote` and must sit exactly on the current tip — when the branch moves,
the script refuses to build the stale checkout and tells you to remove it or pin
the sha instead. Because a branch is mutable, its image tag carries the short
sha (`holoscan-radar-hsdk5-ea:deustice-5x-optional-port-contracts-9bd905908`).

`--print-pin` resolves and verifies the pin, prints the commit and the image
name, and exits without building — use it to read off the value to record in
`devcontainer.json`'s `BASE_IMAGE`, which the script does not edit. A pin other
than the default therefore takes two steps: build the image, then point
`BASE_IMAGE` at it before reopening the container.

### Using `./holohub`

The devcontainer configuration does not change the image selected by
`./holohub`, which continues to default to HSDK 4.3. To launch its container
against the EA image from the host, run:

```bash
./holohub run-container \
  --base-img holoscan-radar-hsdk5-ea:deustice-5x-optional-port-contracts-4fd17e7
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
