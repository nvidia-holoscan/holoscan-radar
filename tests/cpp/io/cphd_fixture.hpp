// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <fstream>
#include <string>
#include <vector>
#include <array>

namespace holoscan::radar::testing::cphd_fixture {
namespace fs = std::filesystem;
inline void write_be64(std::ofstream& out, double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int shift = 56; shift >= 0; shift -= 8)
    out.put(static_cast<char>(bits >> shift));
}
inline void write_be32(std::ofstream& out, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int shift = 24; shift >= 0; shift -= 8)
    out.put(static_cast<char>(bits >> shift));
}

// Standalone C++ CPHD 1.x fixture; CI needs no Python or external collection.
inline fs::path make_fixture(const fs::path& path, double sc0_per_pulse = 0.0,
                             double scss_per_pulse = 0.0, double iarp_x = 6378000.0,
                             bool wide_swath = false, int64_t pulses = 5) {
  if (pulses <= 0 || pulses * 104 > 8192 - 4096)
    throw std::invalid_argument("CPHD fixture: pulse count exceeds the fixed PVP block");
  const int64_t kPulses = pulses;
  constexpr int64_t kSamples = 8;
  constexpr int64_t kXmlOffset = 1024;
  constexpr int64_t kPvpOffset = 4096;
  constexpr int64_t kSignalOffset = 8192;
  std::string xml =
      R"(<CPHD><Data><SignalArrayFormat>CF8</SignalArrayFormat><NumBytesPVP>104</NumBytesPVP><Channel><Identifier>test</Identifier><NumVectors>5</NumVectors><NumSamples>8</NumSamples><SignalArrayByteOffset>0</SignalArrayByteOffset><PVPArrayByteOffset>0</PVPArrayByteOffset></Channel></Data><Global><DomainType>FX</DomainType><SGN>-1</SGN></Global><SceneCoordinates><IARP><ECF><X>6378000</X><Y>0</Y><Z>0</Z></ECF></IARP><ReferenceSurface><Planar><uIAX><X>0</X><Y>1</Y><Z>0</Z></uIAX><uIAY><X>0</X><Y>0</Y><Z>1</Z></uIAY></Planar></ReferenceSurface><ImageArea><X1Y1><X>-10</X><Y>-10</Y></X1Y1><X2Y2><X>10</X><Y>10</Y></X2Y2></ImageArea></SceneCoordinates><ReferenceGeometry><Monostatic><GrazeAngle>45</GrazeAngle><SlantRange>1000</SlantRange><AzimuthAngle>0</AzimuthAngle></Monostatic></ReferenceGeometry><Channel><Parameters><FxC>9004000000</FxC><FxBW>8000000</FxBW><TOASaved>0.0000001</TOASaved></Parameters></Channel><Antenna><AntPattern><GainZero>20</GainZero></AntPattern></Antenna><PVP>)";
  const std::string vector_count = "<NumVectors>5</NumVectors>";
  const auto vector_pos = xml.find(vector_count);
  if (vector_pos == std::string::npos)
    throw std::runtime_error("CPHD fixture: NumVectors element not found");
  xml.replace(vector_pos, vector_count.size(),
              "<NumVectors>" + std::to_string(pulses) + "</NumVectors>");
  if (wide_swath) {
    const std::string original = "<TOASaved>0.0000001</TOASaved>";
    const auto pos = xml.find(original);
    if (pos == std::string::npos)
      throw std::runtime_error("CPHD fixture: TOASaved element not found");
    xml.replace(pos, original.size(), "<TOASaved>0.0000010</TOASaved>");
  }
  if (iarp_x != 6378000.0) {
    const std::string original = "<IARP><ECF><X>6378000</X>";
    const auto pos = xml.find(original);
    if (pos == std::string::npos)
      throw std::runtime_error("CPHD fixture: IARP element not found");
    xml.replace(pos, original.size(), "<IARP><ECF><X>" + std::to_string(iarp_x) + "</X>");
  }
  const std::array<std::pair<const char*, int>, 7> fields{{{"TxTime", 0},
                                                           {"TxPos", 1},
                                                           {"RcvPos", 4},
                                                           {"SRPPos", 7},
                                                           {"SC0", 10},
                                                           {"SCSS", 11},
                                                           {"SIGNAL", 12}}};
  for (const auto& [name, offset] : fields) {
    const int size = (std::string(name) == "TxTime" || std::string(name) == "SC0" ||
                      std::string(name) == "SCSS" || std::string(name) == "SIGNAL")
                         ? 1
                         : 3;
    xml += "<" + std::string(name) + "><Offset>" + std::to_string(offset) + "</Offset><Size>" +
           std::to_string(size) + "</Size><Format>F8</Format></" + name + ">";
  }
  xml += "</PVP></CPHD>";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << "CPHD/1.1.0\n"
      << "XML_BLOCK_BYTE_OFFSET := " << kXmlOffset << "\n"
      << "XML_BLOCK_SIZE := " << xml.size() << "\n"
      << "PVP_BLOCK_BYTE_OFFSET := " << kPvpOffset << "\n"
      << "SIGNAL_BLOCK_BYTE_OFFSET := " << kSignalOffset << "\n\f\n";
  out.seekp(kXmlOffset);
  out.write(xml.data(), static_cast<std::streamsize>(xml.size()));
  out.seekp(kPvpOffset);
  for (int64_t pulse = 0; pulse < kPulses; ++pulse) {
    for (double value :
         {pulse * 0.01, 6378000.0, static_cast<double>(pulse), 1000.0, 6378000.0,
          static_cast<double>(pulse), 1000.0, 6378000.0, static_cast<double>(pulse * 100), 0.0,
          9.0e9 + pulse * sc0_per_pulse, 1.0e6 + pulse * scss_per_pulse, 1.0}) {
      write_be64(out, value);
    }
  }
  out.seekp(kSignalOffset);
  for (int64_t pulse = 0; pulse < kPulses; ++pulse) {
    for (int64_t sample = 0; sample < kSamples; ++sample) {
      const float phase = static_cast<float>(0.1 * pulse + 0.2 * sample);
      write_be32(out, std::cos(phase));
      write_be32(out, std::sin(phase));
    }
  }
  return path;
}

inline void write_truth(const fs::path& cphd) {
  const fs::path sidecar = cphd.parent_path() / (cphd.stem().string() + "_truth.json");
  std::ofstream out(sidecar);
  out << R"({
  "schema": "holoscan-radar/sar-sim-truth/1",
  "sample_geometry": {
    "origin": [6378000.0, -2.0, 0.0],
    "axis_u": [0.0, 1.0, 0.0],
    "axis_v": [0.0, 0.0, 1.0],
    "spacing": [1.0, 1.0],
    "counts": [5, 5]
  },
  "collection": {
    "fx_min_hz": 9000000000.0,
    "fx_spacing_hz": 1000000.0,
    "num_samples": 8
  }
})";
}
}  // namespace holoscan::radar::testing::cphd_fixture
