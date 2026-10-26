# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HSDK 5 visualization graph wiring for FMCW workflows."""

from typing import Any

from holoscan_radar import _radar

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
    device: int,
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
    frame = graph.op(
        "holoscan.radar._app.FrameComposer", "dashboard_frame", width=width, height=height
    )
    holoviz = graph.op(
        "holoscan.radar._app.DashboardViewer",
        "holoviz",
        width=width,
        height=height,
        title=f"{viz['window_title']} - {title}",
    )
    partition.add(display, bridge)
    visualization = graph.partition("visualization")
    visualization.add(frame)
    composer_device = _radar._application_composer_device()
    for operator_name, port_name, ordinal in (
        ("display_compositor", "rgba", device),
        ("dashboard_image", "image", device),
        ("dashboard_frame", "output", composer_device),
    ):
        options.deployment.bind_tensor_output_device(
            operator_path=operator_name,
            output_port=port_name,
            device=hs.execution.DeviceId(ordinal),
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
    _connect_image_devices(
        graph,
        partition,
        visualization,
        options,
        bridge.image,
        frame.image,
        "dashboard_composer",
        width,
        height,
        device,
        composer_device,
    )
    graph.add_flow(display.text, frame.text, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)

    presentation = graph.partition("presentation")
    presentation.add(holoviz)
    _connect_image_devices(
        graph,
        visualization,
        presentation,
        options,
        frame.output,
        holoviz.image,
        "dashboard",
        width,
        height,
        composer_device,
        _radar._application_display_device(),
    )


def _connect_image_devices(
    graph,
    source_partition,
    target_partition,
    options,
    source,
    target,
    prefix,
    width,
    height,
    source_device,
    target_device,
):
    if source_device == target_device:
        graph.add_flow(source, target, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)
        return
    download = graph.op(
        "holoscan.radar._app.ImageDownload", f"{prefix}_download", width=width, height=height
    )
    upload = graph.op(
        "holoscan.radar._app.ImageUpload", f"{prefix}_upload", width=width, height=height
    )
    source_partition.add(download)
    target_partition.add(upload)
    options.deployment.bind_tensor_output_device(
        operator_path=f"{prefix}_upload",
        output_port="image_out",
        device=hs.execution.DeviceId(target_device),
    )
    graph.add_flow(
        source, download.image_in, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
    )
    graph.add_flow(
        download.image_out, upload.image_in, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
    )
    graph.add_flow(upload.image_out, target, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1)
