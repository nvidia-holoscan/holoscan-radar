# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HSDK 5 visualization graph wiring for FMCW workflows."""

from typing import Any

import holoscan as hs


def add_dashboard(
    graph: hs.Graph,
    partition: Any,
    options: hs.CompileOptions,
    config: dict[str, Any],
    path: str,
    kind: str,
    title: str,
    inputs: dict[str, Any],
    *,
    clustered: bool,
    profiled: bool = False,
    profile_gap: int = 1,
) -> None:
    """Connect the native Radar compositor to the SDK Visualization provider."""

    viz = config["visualization"]
    width = viz["window_width"]
    height = viz["window_height"]
    display = graph.op(f"holoscan.radar._app.{kind}Dashboard", "display_compositor", path=path)
    bridge = graph.op(
        "holoscan.radar._app.ImageBridge", "dashboard_image", width=width, height=height
    )
    holoviz = graph.op(
        "holoscan.radar._app.DashboardViewer",
        "holoviz",
        width=width,
        height=height,
        title=f"{viz['window_title']} - {title}",
    )
    partition.add(display, bridge)
    graph.partition("visualization").add(holoviz)
    for operator_name, port_name in (
        ("display_compositor", "rgba"),
        ("dashboard_image", "image"),
    ):
        options.deployment.bind_tensor_output_device(
            operator_path=operator_name,
            output_port=port_name,
            device=hs.execution.DeviceId(0),
        )
    # Feed the compositor the detected cells and their physical coordinate axes.
    graph.add_flow(
        inputs["cfar_power"],
        display.cfar_power,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=4,
    )
    graph.add_flow(
        inputs["detection_idx"],
        display.detection_idx,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=4,
    )
    graph.add_flow(
        inputs["points"],
        display.points,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=4,
    )
    graph.add_flow(
        inputs["range_bins"],
        display.range_bins,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=4,
    )
    graph.add_flow(
        inputs["velocity_bins"],
        display.velocity_bins,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=4,
    )

    # Clustered dashboards also show membership and reduced representatives.
    if clustered:
        graph.add_flow(
            inputs["labels"],
            display.labels,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=4,
        )
        graph.add_flow(
            inputs["sizes"],
            display.sizes,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=4,
        )
        graph.add_flow(
            inputs["reduced_points"],
            display.reduced_points,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=4,
        )
    # Profiled dashboards display the waveform stamped onto this frame.
    if profiled:
        graph.add_flow(inputs["waveform"], display.waveform, queue_depth=profile_gap + 2)
    # Convert the rendered Tensor to an ImageT for the SDK viewer.
    graph.add_flow(
        display.rgba, bridge.rgba, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
    )
    graph.add_flow(
        bridge.image, holoviz.image, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
    )

    graph.add_flow(
        display.text, holoviz.text, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
    )
