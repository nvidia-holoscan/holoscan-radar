// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "radar_app/sar/mosaic_op.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_schemas/sample_geometry_bridge.hpp"
#include "utils/fixture_completion.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::SampleGeometry;
using holoscan::radar::SampleGeometrySpec;
using holoscan::radar::app::sar::SarMosaicOp;

SampleGeometry lattice(int64_t nv, int64_t nu, double v = 0, double u = 0) {
  return SampleGeometry(SampleGeometrySpec{.origin = {u, v, 0},
                                           .axis_u = {1, 0, 0},
                                           .axis_v = {0, 1, 0},
                                           .spacing = {1, 1},
                                           .counts = {nv, nu}});
}

template <typename T>
class SarMosaicBounds : public ::testing::Test {};
using Precisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SarMosaicBounds, Precisions);

TYPED_TEST(SarMosaicBounds, RejectsWrappedCanvasAndCropBeforeAllocation) {
  using Op = SarMosaicOp<TypeParam>;
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  EXPECT_THROW((Op(lattice(1LL << 32, 1LL << 32), false)), std::invalid_argument);
  EXPECT_THROW((Op(lattice(1LL << 31, 1LL << 30), false)), std::invalid_argument);
  EXPECT_THROW((Op(lattice(2, 4), false, max, 2)), std::invalid_argument);
  EXPECT_THROW((Op(lattice(2, 4), false, 1, max)), std::invalid_argument);
  // A large representable canvas is allowed without reserving GPU memory.
  EXPECT_NO_THROW((Op(lattice(1, 1LL << 32), false, 0, 0, max)));
}

TYPED_TEST(SarMosaicBounds, CroppingPreservesPhaseAndDecimationPreservesPower) {
  using T = TypeParam;
  namespace testing = holoscan::radar::testing;
  for (const int64_t display : {0, 2}) {
    auto fixture = holoscan::testing::create_operator_test<SarMosaicOp<T>>(lattice(4, 6), false, 1,
                                                                           4, display);
    std::vector<std::complex<T>> samples(24, {T{3}, T{4}});
    const auto tensor =
        testing::to_device_tensor(samples, {4, 6}, holoscan::radar::complex_dtype<T>());
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::nanoseconds{1}, holoscan::testing::input("image", *tensor),
        holoscan::testing::input("sample_geometry", holoscan::radar::to_wire(lattice(4, 6)))));
    fixture.input_source_device("image", holoscan::DeviceId{0});
    fixture.output_device("image_out", holoscan::DeviceId{0});
    fixture.template capture_snapshot<holoscan::Tensor>("image_out", 1,
                                                        16 * sizeof(std::complex<T>));
    testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    const bool observed = completion.wait_for_output(running, "image_out", 1);
    running.request_stop();
    const auto result = running.finish();
    completion.check(result);
    ASSERT_TRUE(observed) << result.describe_failure();
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    const auto captures = result.template snapshots<holoscan::Tensor>("image_out");
    ASSERT_EQ(captures.size(), 1U);
    const auto output = testing::to_host(captures[0]);
    const int64_t side = display == 0 ? 4 : 2;
    ASSERT_EQ(output.shape, (std::vector<int64_t>{side, side}));
    for (int64_t i = 0; i < side * side; ++i) {
      const auto z = output.template as<std::complex<T>>()[i];
      EXPECT_NEAR(z.real(), display == 0 ? 3 : 5, 1e-6);
      EXPECT_NEAR(z.imag(), display == 0 ? 4 : 0, 1e-6);
    }
  }
}

TYPED_TEST(SarMosaicBounds, RejectsUnrepresentableAndOutsideTileCoordinates) {
  using T = TypeParam;
  namespace testing = holoscan::radar::testing;
  for (const double origin : {2.0, std::numeric_limits<double>::max()}) {
    auto fixture = holoscan::testing::create_operator_test<SarMosaicOp<T>>(lattice(2, 2), false);
    const auto tensor = testing::to_device_tensor(std::vector<std::complex<T>>(1, {T{1}, T{0}}),
                                                  {1, 1}, holoscan::radar::complex_dtype<T>());
    fixture.push_frame(holoscan::testing::frame_at(
        std::chrono::nanoseconds{1}, holoscan::testing::input("image", *tensor),
        holoscan::testing::input("sample_geometry",
                                 holoscan::radar::to_wire(lattice(1, 1, origin)))));
    fixture.input_source_device("image", holoscan::DeviceId{0});
    fixture.output_device("image_out", holoscan::DeviceId{0});
    fixture.template capture_snapshot<holoscan::Tensor>("image_out", 1,
                                                        4 * sizeof(std::complex<T>));
    testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    EXPECT_FALSE(completion.wait_for_output(running, "image_out", 1));
    running.request_stop();
    const auto result = running.finish();
    EXPECT_FALSE(result.ok());
    EXPECT_NE(result.describe_failure().find("window outside parent lattice"), std::string::npos);
  }
}
}  // namespace
