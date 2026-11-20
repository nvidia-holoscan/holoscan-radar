# Security Policy: Holoscan Radar

## Reporting a Vulnerability

If you discover a potential security vulnerability in Holoscan Radar, please **do not open a public issue** or discuss it in public forums.

Report it through one of the following channels:

- **NVIDIA Vulnerability Disclosure Program** (preferred): [NVIDIA Product Security](https://www.nvidia.com/en-us/security/)
- **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  We encourage you to encrypt sensitive details using the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
- **GitHub:** On [nvidia-holoscan/holoscan-radar](https://github.com/nvidia-holoscan/holoscan-radar), use **Security** → **Report a vulnerability** to submit a private advisory.

Please include:

- Product name (**Holoscan Radar**), version or branch (see `pyproject.toml` / CMake `project(... VERSION ...)` when reporting from source), and how it was obtained (e.g., built from source, wheel, or container).
- Type of vulnerability (e.g., memory corruption, unsafe defaults, logic flaw).
- Step-by-step reproduction instructions.
- Proof-of-concept code or minimal reproducer, if available.
- Impact assessment (confidentiality, integrity, availability, scope).

Detailed reports help NVIDIA PSIRT validate severity, develop fixes, and publish security bulletins when appropriate. PSIRT will acknowledge receipt and work with engineering on remediation.

## Security Architecture & Context

**Holoscan Radar** is a Holoscan Module: a C++/CUDA (MatX) and Python (`holoscan.radar`) library of radar signal-processing operators plus example applications and visualization helpers under `applications/`.

**Classification:** Library / SDK (Holoscan Module), with optional local demo applications.

**Repository Exposure Classification:** Public.
Basis: origin remote is github.com; confirmed as publicly readable; SECURITY.md written to public-safe detail.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: externally distributed Apache-2.0 Holoscan module with customer/partner-facing operator APIs and packaging (CMake / Python wheel); no in-repo network service, but intended for redistributable use in Holoscan deployments.

**Primary security responsibilities:**

- **Safe composition of native GPU code:** Robustness of operators and Python bindings so application graphs and parameters do not cause exploitable memory corruption or undefined behavior beyond what the host platform already exposes.
- **Controlled handling of configuration and tensor inputs:** Predictable behavior when YAML configs and Holoscan `from_config()` paths, waveform descriptors, and device/host tensors are supplied by the application.
- **Honest treatment of sensor-derived outputs:** Detection, clustering, and report/visualization paths process operational radar data when connected to real sensors; this module does not implement access control around those outputs.

**Key boundaries and interfaces:**

- **Operator APIs:** C++ (`holoscan::radar`) and Python (`holoscan.radar`) operators consume application-supplied tensors and parameters in the same process as the Holoscan runtime.
- **Application configuration:** Demo apps load YAML via Holoscan `config_io` / `yaml.safe_load` and decode operator parameters with Holoscan `params::decode` (e.g., under `applications/fmcw_*/`); operator parameter blocks and graph shape can change based on those files.
- **Visualization and reporting:** Optional Holoviz paths (`HolovizOp`, `radar_app` viz helpers) and report operators (e.g., `DetectionReportOp`, `ClusterReportOp`) render or log detection/cluster metadata locally.
- **Build and packaging:** CMake against Holoscan SDK / CUDA / MatX; optional container based on a published Holoscan NGC image; wheel builds via scikit-build-core.
- **Auxiliary tooling:** Checkout data/diagnostic tools beside their owners and CI helper scripts under `.github/workflows/scripts/`.

This repository does **not** implement network listeners, authentication, or remote APIs. Any network or multi-node exposure comes from the surrounding Holoscan application and platform, not from operators in this module alone.

### Threat Model

The following scenarios reflect the main security concerns for this codebase, including auxiliary paths:

1. **Malicious or untrusted application configuration (YAML):** Attacker-controlled config files consumed by demo loaders (Holoscan `config_io` / `yaml.safe_load`) or Holoscan `params::decode` can change operator parameters, graph wiring (e.g., `peak_only` gating clustering), and visualization settings, leading to denial of service, unexpected processing behavior, or unsafe resource use if writers of those files are not controlled.

2. **Untrusted tensor / parameter inputs to GPU operators:** Malformed shapes, sizes, or values on ports into FMCW, CFAR, clustering, or angle-estimation operators (MatX/CUDA paths under `operators/`) can cause process crashes, GPU faults, or resource exhaustion; impact is generally bounded by the host process unless the platform provides stronger isolation.

3. **Same-process execution of application and extension code:** User operators, Python code, and this module’s bindings share the Holoscan process trust domain. There is no sandbox between untrusted application logic and radar operators; compromise of one implies compromise of the other.

4. **Disclosure of operational radar outputs via reports and visualization:** When connected to real sensors, report operators and Holoviz overlays can expose detections, ranges, velocities, and angles on local displays or in logs. Misconfigured logging or shared displays in multi-user environments can leak operationally sensitive situational data.

5. **Supply chain and build-time trust:** Builds depend on Holoscan SDK, CUDA Toolkit, MatX (via the SDK), container base images, and optional pip installs (e.g., developer CLI tools in the Dockerfile). Compromised mirrors, tampered packages, or unpinned tooling increase exposure if builds are not verified and run in trusted environments.

6. **Auxiliary scripts invoking shells:** CI helpers (e.g., subprocess usage under `.github/workflows/scripts/`) can amplify risk if run on untrusted inputs or in shared CI contexts without the intended trusted-operator assumptions.

7. **Native / GPU faults affecting isolation:** Bugs in CUDA kernels, MatX plans, or the C++/Python boundary can corrupt memory or destabilize the process; isolation relies on the OS and deployment (containers, users, devices), not on in-module sandboxing.

### Critical Security Assumptions

- **Trusted application authors and operators:** Holoscan Radar does not sandbox caller code or Python; deployers must treat application graphs and operators as trusted or isolate them externally (containers, separate hosts, least privilege).
- **Trusted configuration and artifacts:** YAML configs, waveform descriptors, schedules, and shared libraries are assumed to come from trusted sources or to be integrity-checked by the deployer.
- **No network security surface in this module:** The library assumes any network, distributed execution, or remote I/O is provided and hardened by Holoscan SDK features or external infrastructure outside this repository’s defaults.
- **Correct and patched platform stack:** Assumes a patched OS, NVIDIA GPU drivers, CUDA, Holoscan SDK, MatX, and related runtimes as supplied by the platform or container maintainer.
- **Host OS enforcement:** Process isolation and file permissions are assumed to prevent unauthorized modification of binaries, configs, and models/data used by production pipelines.
- **Deployer responsibility for sensor-data sensitivity:** Confidentiality of real radar ADC and detection outputs is assumed to be enforced by deployment policy (access control, logging retention, display placement); this module does not encrypt or access-control those streams.
- **Physical and administrative security:** Edge or embedded devices running radar pipelines are assumed to be physically and administratively controlled where device loss implies compromise of local data regardless of library behavior.

---

For general bug reports and feature requests that are **not** security-sensitive, use the project’s public issue tracker as described in the README.
