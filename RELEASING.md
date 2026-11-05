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

| Tag | Meaning | Python version |
| --- | --- | --- |
| `vX.Y.Z-rcN` | Release candidate | `X.Y.ZrcN` |
| `vX.Y.Z` | General availability release | `X.Y.Z` |

- Release tags are created only on `release/vX.Y` branches, by release managers.
- Tags are annotated and immutable; a published tag is never moved or reused.
- A GA tag points to the same commit as the release candidate it promotes.
- Package versions are derived from the checked-in version.

## Release process

1. **Feature freeze.** A release manager creates `release/vX.Y` from a passing `main` commit. After freeze, only bug, security, documentation, and packaging fixes are backported; other changes require release-manager approval.
2. **Release candidates.** A release manager tags `vX.Y.Z-rcN`. Release candidates are built, validated, and security-reviewed before promotion.
3. **General availability.** A release manager tags `vX.Y.Z` on the approved release candidate commit, publishes packages, and creates a GitHub Release with release notes.
4. **Patch releases.** Fixes land on `main`, are backported to the active `release/vX.Y`, and follow steps 2–3 with an incremented patch version.

There is no fixed release schedule; releases are made as features and fixes are ready.

## Release artifacts

| Artifact | Distribution |
| --- | --- |
| Source | GitHub tag and GitHub Release |
| Debian packages (`libholoscan-radar-cuda-13`, `libholoscan-radar-dev-cuda-13`) | NVIDIA package repositories |
| Python wheels (`holoscan-radar`, `holoscan-radar-cu13`) | PyPI |

## Roles

- **Release managers:** JC Fillion-Robin, Tom Birdsong, Julien Jomier
- **Code owners:** Dylan Eustice, Dan Campbell, Tom Birdsong, Wenqi Li
