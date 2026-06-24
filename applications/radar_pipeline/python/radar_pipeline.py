# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Python pipeline example for radar.
# TODO: replace the stub with your actual pipeline topology.
#
# Run with:
#   HOLOSCAN_RADAR_BUILD_DIR=build \
#   PYTHONPATH=build/python${PYTHONPATH:+:$PYTHONPATH} \
#   python applications/radar_pipeline/radar_pipeline.py

import logging

from holoscan.core import Application
from holoscan.radar import RadarOp

logging.basicConfig(level=logging.INFO)


class RadarPipelineApp(Application):
    def compose(self) -> None:
        # TODO: instantiate and connect your operators, e.g.:
        op = RadarOp(self, name="radar_op")  # noqa: F841 (wire up via add_flow below)
        # self.add_flow(source, op, {("out", "in")})

        logging.info("TODO: RadarPipelineApp is not yet implemented")


if __name__ == "__main__":
    app = RadarPipelineApp()
    app.run()
