# Releasing Holoscan Radar

This document describes how Holoscan Radar branches, versions, and releases are managed.

## Branches

| Branch | Purpose | Merged by |
| --- | --- | --- |
| `main` | All development. Every change lands here first. | Code owners, after review |
| `release/vX.Y` | Stabilization and patch releases for minor version X.Y | Release managers |

- **Fix forward.** Bug and security fixes merge to `main` first and are then backported to the active release branch. Changes made only on a release branch are limited to release mechanics such as version metadata and release notes.
- **Release branches** are created from `main` at feature freeze for each new minor version. When the next minor version is released, the previous release branch is frozen (read-only) and kept for reference.

## Support policy

Only the latest minor release receives fixes. While a new minor release is being stabilized, critical fixes may also be backported to the current release. Users of older releases should upgrade. See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Contributing changes

- Open pull requests against `main`. Pull requests are squash-merged, so each one becomes a single commit with an `Area: Summary` subject (for example, `Operators: Add CFAR detector`). Commits must be signed off; see [CONTRIBUTING.md](CONTRIBUTING.md).
- To request a backport, ask in the pull request or add the `backport release/vX.Y` label. Backports are cherry-picked with `git cherry-pick -x`, one commit per original pull request, so every release commit records the `main` commit it came from.
- `main` and `release/**` are protected: changes require a pull request, an approving review from a code owner, and passing checks. Force-pushes and branch deletion are not allowed.

## Versions and tags

Holoscan Radar follows [Semantic Versioning](https://semver.org/).
The upcoming public release uses `1.0.0` for the module, native packages,
Python wheel, operators, and applications. Its annotated release tag is
`v1.0.0` on `release/v1.0`.

Package versions come from checked-in files; the tag does not rewrite them.
Before building release artifacts, synchronize:

- `CMakeLists.txt`: `project(... VERSION X.Y.Z)`;
- `pyproject.toml`: `[project].version`;
- `metadata.json`: `module.version`;
- each operator/application manifest's current `version`; and
- `uv.lock`: refresh with `uv lock` after changing the Python project version.

Update versioned package examples at the same time. Dependency versions,
wire-schema versions, and historical changelog entries have their own meaning
and do not change merely because a release number changes.

Release-validation builds use the final package version, `1.0.0` for this
release. Keep them in private validation storage until approved; do not publish
successive validation builds under the same version to a public package index.

Release tags are created only by release managers on `release/vX.Y`.
Tags are annotated and immutable. A published tag or artifact is never moved,
replaced, or reused; a subsequent change requires a new patch version.

## Release process

1. **Feature freeze.** A release manager creates `release/vX.Y` from a passing `main` commit. After freeze, only bug, security, documentation, and packaging fixes are backported; other changes require release-manager approval.
2. **Prepare the version.** Set the checked-in versions to `X.Y.Z` and update package examples and release notes. For the upcoming release, use `1.0.0`.
3. **Build and validate.** Build the native DEB/TGZ and Python wheel from the intended release commit. Record that commit, the SDK/container environment, artifact checksums, and validation results. Run the [full GPU and packaging suite](DEVELOPER.md#from-the-host-full-test-suite), the [pinned lint suite](DEVELOPER.md#pinned-lint-workflow), and the release security review. Keep artifacts private until approval. If a change is needed, build and validate again from the updated commit.
4. **General availability.** After approval, create `vX.Y.Z` on the validated commit, publish the validated artifacts, and create a GitHub Release with release notes. Publish the same artifact bytes that passed validation. Release notes identify supported platforms/SDK versions, available artifacts, and known limitations.
5. **Patch releases.** Fixes land on `main`, are backported to the active release branch, and repeat steps 2–4 with an incremented patch version.

There is no fixed release schedule; releases are made as features and fixes are ready.
These instructions prepare a release; changing version files does not publish one.

## Release artifacts

Release managers publish the following build artifacts after validation.

| Artifact | Package or filename | Distribution |
| --- | --- | --- |
| Source | `vX.Y.Z` tag | GitHub tag and GitHub Release |
| Debian package | `holoscan-radar`, containing native libraries and development files | NVIDIA package repository when published |
| Native TGZ archive | `holoscan-radar-linux-<arch>-X.Y.Z.tar.gz`; `<arch>` is `x86_64` or `sbsa` | GitHub Release asset |
| Python wheel | `holoscan-radar` | PyPI when published |

The Debian package includes both native libraries and development files.
The wheel requires the matching HSDK 5/CUDA 13
runtime and does not install the SDK as a Python dependency. See
[package contents and consumption](pkg/holoscan-radar/README.md) and the
[Python provider guide](DEVELOPER.md#python-provider).

## Roles

- **Release managers:** JC Fillion-Robin, Tom Birdsong, Julien Jomier
- **Code owners:** Dylan Eustice, Dan Campbell, Tom Birdsong, Wenqi Li
