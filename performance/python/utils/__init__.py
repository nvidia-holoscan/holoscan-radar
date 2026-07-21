# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python performance harness utilities.

Keep this package ``__init__`` free of CuPy / Holoscan imports. Importing
``utils.bootstrap`` (or any ``utils.*`` submodule) must not pull CuPy before
``holoscan.radar`` — CuPy-first makes MatX eig/inv ``GetCache().Clear`` abort
with a double-free when MVDR retires solver plans (see
``operators/angle_estimation/mvdr_estimator_op/OPEN_ITEMS.md``).
"""
