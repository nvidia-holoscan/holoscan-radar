// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <unistd.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "io/tensor_dump_op/npy_writer.hpp"
#include "io/tensor_dump_op/tensor_dump_op.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace holoscan::radar {

namespace {

namespace fs = std::filesystem;

// A directory of its own per test, so a leftover file from one cannot be
// mistaken for output of another.
class TempDir {
 public:
  explicit TempDir(const std::string& tag) {
    path_ = fs::temp_directory_path() /
            ("holoscan_radar_dump_" + tag + "_" +
             std::to_string(::getpid()) + "_" + std::to_string(counter_++));
    fs::remove_all(path_);
    fs::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  std::string pattern(const std::string& leaf) const {
    return (path_ / leaf).string();
  }
  const fs::path& path() const { return path_; }

  int file_count() const {
    int count = 0;
    for (const auto& entry : fs::directory_iterator(path_)) {
      count += entry.is_regular_file() ? 1 : 0;
    }
    return count;
  }

 private:
  fs::path path_;
  static inline int counter_ = 0;
};

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("test: cannot open " + path);
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                              std::istreambuf_iterator<char>());
}

// Split a .npy into its header text and the element bytes that follow. The
// header's own correctness is checked by NumPy in the readback test; here it
// is split only so the values can be compared.
struct NpyParts {
  std::string header;
  std::vector<uint8_t> data;
};

NpyParts split_npy(const std::vector<uint8_t>& bytes) {
  EXPECT_GE(bytes.size(), 10u);
  EXPECT_EQ(std::memcmp(bytes.data(), "\x93NUMPY", 6), 0);
  const size_t length = static_cast<size_t>(bytes[8]) |
                        (static_cast<size_t>(bytes[9]) << 8);
  NpyParts parts;
  parts.header.assign(reinterpret_cast<const char*>(bytes.data()) + 10,
                      length);
  parts.data.assign(bytes.begin() + 10 + length, bytes.end());
  return parts;
}

// The value both this test and the NumPy readback script compute, so the
// script asserts against a formula rather than against bytes we handed it.
float ramp(int64_t i) { return static_cast<float>(i) * 1.5f - 3.0f; }

std::vector<std::complex<float>> complex_ramp(int64_t count) {
  std::vector<std::complex<float>> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] =
        std::complex<float>(ramp(i), -ramp(i) * 0.25f);
  }
  return out;
}

// One frame per tick, all the same tensor: the tests here are about the sink's
// throttling and file format, not about the data varying.
std::vector<holoscan::TensorMap> repeat_frames(
    const std::shared_ptr<holoscan::Tensor>& tensor, int ticks) {
  std::vector<holoscan::TensorMap> frames;
  frames.reserve(static_cast<size_t>(ticks));
  for (int i = 0; i < ticks; ++i) {
    frames.push_back(testing::single_map(tensor, "tensor"));
  }
  return frames;
}

class DumpApp : public holoscan::Application {
 public:
  DumpApp(std::vector<holoscan::TensorMap> frames,
          std::vector<holoscan::Arg> args,
          std::vector<int64_t> every_n_overrides = {})
      : frames_(std::move(frames)),
        args_(std::move(args)),
        overrides_(std::move(every_n_overrides)) {}

  void compose() override {
    holoscan::ArgList arg_list;
    for (const holoscan::Arg& arg : args_) {
      arg_list.add(arg);
    }
    auto dump = make_operator<TensorDumpOp>("dump", arg_list);
    testing::wire_data_source(this, dump, "tensor", frames_);
    if (!overrides_.empty()) {
      testing::wire_leading_override<int64_t>(this, dump, "every_n",
                                              overrides_);
    }
  }

 private:
  std::vector<holoscan::TensorMap> frames_;
  std::vector<holoscan::Arg> args_;
  std::vector<int64_t> overrides_;
};

}  // namespace

// --- npy_writer, without a graph ------------------------------------------

TEST(NpyWriter, SpellsEveryDtypeTheModuleCanEmit) {
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLFloat, 32, 1}), "<f4");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLFloat, 64, 1}), "<f8");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLFloat, 16, 1}), "<f2");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLComplex, 64, 1}), "<c8");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLComplex, 128, 1}), "<c16");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLInt, 16, 1}), "<i2");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLInt, 64, 1}), "<i8");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLUInt, 32, 1}), "<u4");
  // Single-byte types take '|', not '<': numpy.lib.format rejects "<u1".
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLInt, 8, 1}), "|i1");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLUInt, 8, 1}), "|u1");
  EXPECT_EQ(npy::dtype_descr(DLDataType{kDLBool, 8, 1}), "|b1");
}

TEST(NpyWriter, RefusesDtypesTheFormatCannotExpress) {
  // bfloat16 has no NumPy equivalent, and a vector dtype is not an element.
  EXPECT_THROW(npy::dtype_descr(DLDataType{kDLBfloat, 16, 1}),
               std::invalid_argument);
  EXPECT_THROW(npy::dtype_descr(DLDataType{kDLFloat, 32, 4}),
               std::invalid_argument);
  EXPECT_THROW(npy::dtype_descr(DLDataType{kDLFloat, 8, 1}),
               std::invalid_argument);
}

TEST(NpyWriter, HeaderIsPaddedToASixtyFourByteBoundary) {
  for (const std::vector<int64_t>& shape :
       {std::vector<int64_t>{}, std::vector<int64_t>{7},
        std::vector<int64_t>{2, 3}, std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7}}) {
    const std::string header =
        npy::header(DLDataType{kDLComplex, 64, 1}, shape);
    EXPECT_EQ(header.size() % 64, 0u) << "rank " << shape.size();
    EXPECT_EQ(header.back(), '\n');
  }
}

TEST(NpyWriter, ShapeTupleFollowsPythonLiteralRules) {
  const DLDataType f32{kDLFloat, 32, 1};
  // A rank-0 dump is "()", and a rank-1 needs the trailing comma or Python
  // reads "(7)" as an integer rather than a tuple.
  EXPECT_NE(npy::header(f32, {}).find("'shape': (), "), std::string::npos);
  EXPECT_NE(npy::header(f32, {7}).find("'shape': (7, ), "), std::string::npos);
  EXPECT_NE(npy::header(f32, {2, 3}).find("'shape': (2, 3), "),
            std::string::npos);
}

// --- The operator ---------------------------------------------------------

TEST(TensorDumpOp, NpyRoundTripsShapeDtypeAndValues) {
  TempDir dir("npy_roundtrip");
  // Rank 4: nothing about the sink is rank-aware, and the shape it records
  // has to prove it.
  const std::vector<int64_t> shape{2, 3, 4, 5};
  const auto host = complex_ramp(2 * 3 * 4 * 5);
  auto tensor = testing::to_device_tensor(host, shape, complex_dtype<float>());

  DumpApp(repeat_frames(tensor, 1),
          {holoscan::Arg("path", dir.pattern("dump_{index}.npy"))})
      .run();

  const NpyParts parts = split_npy(read_file(dir.pattern("dump_0.npy")));
  EXPECT_NE(parts.header.find("'descr': '<c8'"), std::string::npos);
  EXPECT_NE(parts.header.find("'shape': (2, 3, 4, 5), "), std::string::npos);
  EXPECT_NE(parts.header.find("'fortran_order': False"), std::string::npos);
  ASSERT_EQ(parts.data.size(), host.size() * sizeof(std::complex<float>));
  EXPECT_EQ(std::memcmp(parts.data.data(), host.data(), parts.data.size()), 0);
}

TEST(TensorDumpOp, RawWritesTheElementBytesAndNothingElse) {
  TempDir dir("raw");
  const std::vector<int64_t> shape{4, 8};
  const auto host = complex_ramp(32);
  auto tensor = testing::to_device_tensor(host, shape, complex_dtype<float>());

  DumpApp(repeat_frames(tensor, 1),
          {holoscan::Arg("path", dir.pattern("dump_{index}.bin")),
           holoscan::Arg("format", TensorDumpFormat::kRaw)})
      .run();

  const std::vector<uint8_t> bytes = read_file(dir.pattern("dump_0.bin"));
  // Shape and dtype are lost, which is the format's whole trade: the file is
  // exactly the elements, so its size is the only thing left to check them by.
  ASSERT_EQ(bytes.size(), host.size() * sizeof(std::complex<float>));
  EXPECT_EQ(std::memcmp(bytes.data(), host.data(), bytes.size()), 0);
}

TEST(TensorDumpOp, RawDumpsADtypeNpyCannotSpell) {
  TempDir dir("bfloat");
  auto tensor = make_device_tensor({4, 4}, DLDataType{kDLBfloat, 16, 1});

  // npy has no bfloat16 type string, and the error says which format does.
  EXPECT_THROW(DumpApp(repeat_frames(tensor, 1),
                       {holoscan::Arg("path", dir.pattern("bf_{index}.npy"))})
                   .run(),
               std::exception);

  DumpApp(repeat_frames(tensor, 1),
          {holoscan::Arg("path", dir.pattern("bf_{index}.bin")),
           holoscan::Arg("format", TensorDumpFormat::kRaw)})
      .run();
  EXPECT_EQ(read_file(dir.pattern("bf_0.bin")).size(), 4u * 4u * 2u);
}

TEST(TensorDumpOp, IndexNamesTheTickNotThePositionInTheSeries) {
  TempDir dir("every_n");
  auto tensor = testing::to_device_tensor(complex_ramp(8), {8},
                                          complex_dtype<float>());

  DumpApp(repeat_frames(tensor, 7),
          {holoscan::Arg("path", dir.pattern("dump_{index}.npy")),
           holoscan::Arg("every_n", static_cast<int64_t>(3))})
      .run();

  // Ticks 0, 3 and 6 of seven, so the file name says which tick produced it
  // rather than counting the files.
  EXPECT_EQ(dir.file_count(), 3);
  EXPECT_TRUE(fs::exists(dir.pattern("dump_0.npy")));
  EXPECT_TRUE(fs::exists(dir.pattern("dump_3.npy")));
  EXPECT_TRUE(fs::exists(dir.pattern("dump_6.npy")));
}

TEST(TensorDumpOp, EveryNOverrideTakesEffectMidStreamAndSticks) {
  TempDir dir("override");
  auto tensor = testing::to_device_tensor(complex_ramp(8), {8},
                                          complex_dtype<float>());

  // Ticks 0-1 override to 1, tick 2 overrides to 4, and nothing arrives after
  // that: the sticky value is what governs ticks 3-9.
  DumpApp(repeat_frames(tensor, 10),
          {holoscan::Arg("path", dir.pattern("dump_{index}.npy")),
           holoscan::Arg("every_n", static_cast<int64_t>(1))},
          {1, 1, 4})
      .run();

  EXPECT_TRUE(fs::exists(dir.pattern("dump_0.npy")));
  EXPECT_TRUE(fs::exists(dir.pattern("dump_1.npy")));
  EXPECT_FALSE(fs::exists(dir.pattern("dump_2.npy")));
  EXPECT_FALSE(fs::exists(dir.pattern("dump_3.npy")));
  EXPECT_TRUE(fs::exists(dir.pattern("dump_4.npy")));
  EXPECT_TRUE(fs::exists(dir.pattern("dump_8.npy")));
  EXPECT_EQ(dir.file_count(), 4);
}

TEST(TensorDumpOp, RejectsEveryNBelowOneRatherThanCoercingIt) {
  TempDir dir("every_n_zero");
  auto tensor = testing::to_device_tensor(complex_ramp(4), {4},
                                          complex_dtype<float>());

  for (int64_t bad : {static_cast<int64_t>(0), static_cast<int64_t>(-2)}) {
    EXPECT_THROW(DumpApp(repeat_frames(tensor, 1),
                         {holoscan::Arg("path", dir.pattern("d_{index}.npy")),
                          holoscan::Arg("every_n", bad)})
                     .run(),
                 std::exception)
        << "every_n = " << bad;
  }
  EXPECT_EQ(dir.file_count(), 0);
}

TEST(TensorDumpOp, RejectsAnEveryNOverrideBelowOne) {
  TempDir dir("override_zero");
  auto tensor = testing::to_device_tensor(complex_ramp(4), {4},
                                          complex_dtype<float>());

  // The port is held to the same rule as the parameter: an override of zero
  // is as ambiguous mid-stream as it is at build.
  EXPECT_THROW(DumpApp(repeat_frames(tensor, 4),
                       {holoscan::Arg("path", dir.pattern("d_{index}.npy"))},
                       {0})
                   .run(),
               std::exception);
}

TEST(TensorDumpOp, MaxFilesStopsWritingWithoutStoppingTheRun) {
  TempDir dir("max_files");
  auto tensor = testing::to_device_tensor(complex_ramp(4), {4},
                                          complex_dtype<float>());

  // Five messages, two files: the later three are consumed and discarded, so
  // the graph runs to completion rather than stalling on a full sink.
  DumpApp(repeat_frames(tensor, 5),
          {holoscan::Arg("path", dir.pattern("dump_{index}.npy")),
           holoscan::Arg("max_files", static_cast<int64_t>(2))})
      .run();

  EXPECT_EQ(dir.file_count(), 2);
  EXPECT_TRUE(fs::exists(dir.pattern("dump_0.npy")));
  EXPECT_TRUE(fs::exists(dir.pattern("dump_1.npy")));
}

TEST(TensorDumpOp, MaxFilesZeroWritesNothing) {
  TempDir dir("max_files_zero");
  auto tensor = testing::to_device_tensor(complex_ramp(4), {4},
                                          complex_dtype<float>());

  // Legal, and the reason every_n = 0 is not: this is how "write nothing" is
  // spelled, so every_n never has to mean it.
  DumpApp(repeat_frames(tensor, 3),
          {holoscan::Arg("path", dir.pattern("dump_{index}.npy")),
           holoscan::Arg("max_files", static_cast<int64_t>(0))})
      .run();

  EXPECT_EQ(dir.file_count(), 0);
}

TEST(TensorDumpOp, RejectsANegativeMaxFiles) {
  TempDir dir("max_files_negative");
  auto tensor = testing::to_device_tensor(complex_ramp(4), {4},
                                          complex_dtype<float>());

  EXPECT_THROW(DumpApp(repeat_frames(tensor, 1),
                       {holoscan::Arg("path", dir.pattern("d_{index}.npy")),
                        holoscan::Arg("max_files", static_cast<int64_t>(-1))})
                   .run(),
               std::exception);
}

TEST(TensorDumpOp, APathWithoutTheIndexTokenOverwritesOneFile) {
  TempDir dir("no_token");
  auto tensor = testing::to_device_tensor(complex_ramp(4), {4},
                                          complex_dtype<float>());

  // Legal — dumping a single tensor needs no placeholder — and start() warns
  // that the dumps do not accumulate.
  DumpApp(repeat_frames(tensor, 3),
          {holoscan::Arg("path", dir.pattern("only.npy"))})
      .run();

  EXPECT_EQ(dir.file_count(), 1);
  EXPECT_TRUE(fs::exists(dir.pattern("only.npy")));
}

TEST(TensorDumpOp, CreatesTheOutputDirectory) {
  TempDir dir("mkdir");
  auto tensor = testing::to_device_tensor(complex_ramp(4), {4},
                                          complex_dtype<float>());

  DumpApp(repeat_frames(tensor, 1),
          {holoscan::Arg("path", dir.pattern("nested/deeper/d_{index}.npy"))})
      .run();

  EXPECT_TRUE(fs::exists(dir.pattern("nested/deeper/d_0.npy")));
}

TEST(TensorDumpOp, RejectsPageableHostInput) {
  TempDir dir("pageable");
  const auto host = complex_ramp(4);
  auto tensor = testing::to_pageable_host_tensor(host, {4},
                                                 complex_dtype<float>());

  // The module-wide input rule: CPHDReaderOp is the sanctioned exception,
  // and a debugging sink is not a second one.
  EXPECT_THROW(DumpApp(repeat_frames(tensor, 1),
                       {holoscan::Arg("path", dir.pattern("d_{index}.npy"))})
                   .run(),
               std::exception);
}

TEST(TensorDumpOp, DumpsEveryPrecisionAndRealAsWellAsComplex) {
  TempDir dir("dtypes");
  std::vector<double> real64(16);
  std::vector<std::complex<double>> complex128(16);
  std::vector<uint8_t> mask(16);
  for (size_t i = 0; i < 16; ++i) {
    real64[i] = static_cast<double>(i) * 0.5;
    complex128[i] = std::complex<double>(real64[i], -real64[i]);
    mask[i] = static_cast<uint8_t>(i % 2);
  }

  struct Case {
    std::shared_ptr<holoscan::Tensor> tensor;
    const char* leaf;
    const char* descr;
  };
  const std::vector<Case> cases{
      {testing::to_device_tensor(real64, {4, 4}, real_dtype<double>()),
       "f64.npy", "<f8"},
      {testing::to_device_tensor(complex128, {4, 4},
                                 complex_dtype<double>()),
       "c128.npy", "<c16"},
      {testing::to_device_uint8_tensor(mask, {16}), "u8.npy", "|u1"},
  };

  for (const Case& one : cases) {
    DumpApp(repeat_frames(one.tensor, 1),
            {holoscan::Arg("path", dir.pattern(one.leaf))})
        .run();
    const NpyParts parts = split_npy(read_file(dir.pattern(one.leaf)));
    EXPECT_NE(parts.header.find(std::string("'descr': '") + one.descr + "'"),
              std::string::npos)
        << one.leaf;
    EXPECT_EQ(parts.data.size(),
              static_cast<size_t>(one.tensor->nbytes())) << one.leaf;
  }
}

// Writes the artifacts the NumPy readback test loads. Split out because the
// point of that test is that something other than this code reads the files:
// checking our own writer with our own parser would prove only that they
// agree with each other.
TEST(TensorDumpOp, WritesArtifactsForTheNumpyReadbackTest) {
  const fs::path out(TENSOR_DUMP_ARTIFACT_DIR);
  std::error_code ec;
  fs::create_directories(out, ec);

  const auto c64 = complex_ramp(2 * 3 * 4);
  auto complex_tensor = testing::to_device_tensor(c64, {2, 3, 4},
                                                  complex_dtype<float>());
  DumpApp(repeat_frames(complex_tensor, 1),
          {holoscan::Arg("path", (out / "readback_c8_{index}.npy").string())})
      .run();

  std::vector<double> f64(10);
  for (size_t i = 0; i < f64.size(); ++i) {
    f64[i] = static_cast<double>(ramp(static_cast<int64_t>(i)));
  }
  auto real_tensor =
      testing::to_device_tensor(f64, {10}, real_dtype<double>());
  DumpApp(repeat_frames(real_tensor, 1),
          {holoscan::Arg("path", (out / "readback_f8.npy").string())})
      .run();

  DumpApp(repeat_frames(complex_tensor, 1),
          {holoscan::Arg("path", (out / "readback_c8.bin").string()),
           holoscan::Arg("format", TensorDumpFormat::kRaw)})
      .run();

  EXPECT_TRUE(fs::exists(out / "readback_c8_0.npy"));
  EXPECT_TRUE(fs::exists(out / "readback_f8.npy"));
  EXPECT_TRUE(fs::exists(out / "readback_c8.bin"));
}

}  // namespace holoscan::radar
