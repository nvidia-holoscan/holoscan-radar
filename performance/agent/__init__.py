# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator performance-optimization harness (nsys attribution + diagnosis).

This package drives the deterministic measurement/analysis substrate the
``perf-analyze`` agent skill sits on top of:

* ``runner``   — build-aware; runs a ``perf_<op>`` binary clean (accurate
  latency) and under ``nsys`` (attribution), and exports the nsys stat tables.
* ``parse``    — folds the nsys tables + the schema-v2 perf JSON into one
  ``analysis.json`` per (operator, size, scenario).
* ``diagnose`` — encodes the ``docs/planning/OPERATOR_DEVELOPMENT.md`` perf
  rubric as rules over ``analysis.json`` and emits ranked findings.
* ``report``   — renders a per-operator markdown perf report.

Everything runs inside ``./holohub run-container`` (the host has no toolchain);
the tools assert that in preflight. See ``performance/agent/README.md``.
"""
