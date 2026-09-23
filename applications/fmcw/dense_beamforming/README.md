# Dense beamforming

Beamform-then-detect FMCW example. This implementation happens to use a SIMO
array, but the application is named for the processing approach it demonstrates. It forms the full range-Doppler-angle cube, applies N-D CFAR peak suppression, resolves physical coordinates, and reports the retained detections directly.

```mermaid
flowchart LR
    Source[Sim ADC source] --> Range[Range processor] --> Clutter[Clutter filter] --> Doppler[Doppler processor]
    Doppler --> Beamform[Phase-shift beamformer] --> CFAR[CFAR detector] --> Features[Feature assembly] --> Report[Detection report]
    Range --> Features
    Doppler --> Features
    Beamform --> Features
```

## Build and run

```bash
cmake -S . -B build -DBUILD_ALL=OFF -DAPP_dense_beamforming=ON
cmake --build build --target dense_beamforming
./build/applications/fmcw/dense_beamforming/cpp/dense_beamforming
```

The normal preset enables relative monostatic free-space range loss with a 100 m reference. The smoke preset intentionally omits it, uses the constant-amplitude analytic model, and emits a warning.

The executable defaults to `config/dense_beamforming.yaml`. Pass `config/dense_beamforming_smoke.yaml` as the first argument for the two-frame, headless single-target smoke scene.

Simulator-backed sources default `app.scene_visibility` to `front_half_space`, with radar-local `+x` as boresight. Select `frustum` with explicit horizontal/vertical FOVs in radians, or `all` to restore isotropic target synthesis. This binary coverage gate is separate from relative range loss and does not model antenna gain.
