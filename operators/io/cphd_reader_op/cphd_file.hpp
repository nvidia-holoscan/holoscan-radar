// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Host-side CPHD 1.x reading: the ASCII header, the XML metadata block, the
// per-vector parameter (PVP) block, and the signal block.
//
// Deliberately separate from the operator. Everything here is plain host
// code with no Holoscan or CUDA in it, which is what lets the file-format
// work be tested against a fixture without standing up a graph -- and file
// parsing is where the fiddly, high-bug-density work in this operator is.
//
// All three CPHD 1.x sample formats decode, and a collection that scales its
// samples per vector (`AmpSF`) is scaled on the way out. What is still
// deliberately absent is signal compression, which no surveyed collection
// uses and which the header would have to declare.
namespace holoscan::radar::cphd {

// Sample storage format, from Data/SignalArrayFormat: 8-byte complex float,
// or complex pairs of 16- and 8-bit signed integers.
enum class SampleFormat { kCF8, kCI4, kCI2 };

// Fast-time domain of the stored samples, from Global/DomainType. Decides
// whether the reader's sample axis is frequency (Hz) or time of arrival (s).
enum class SampleDomain { kFX, kTOA };

const char* to_string(SampleFormat format);
const char* to_string(SampleDomain domain);

// One PVP field's placement within a vector's parameter record, as declared
// in the XML rather than assumed: the field set and the record size both
// vary between collections.
struct PvpField {
  int64_t offset_words{0};  // 8-byte words from the start of the record
  int64_t size_words{0};
  std::string format;
};

// A span of per-vector parameter records held in host memory, so that a
// tick's several fields are picked out of one read rather than costing a read
// each. Borrows its layout from the CphdFile that produced it and must not
// outlive it.
class PvpBlock {
 public:
  int64_t count() const { return count_; }
  bool has(const std::string& name) const;

  // `components` 8-byte words of `name` for every vector in the span, as
  // doubles, row-major (count, components). A scalar field is 1 component
  // and a position is 3. Throws if the field is absent or narrower than
  // asked for.
  std::vector<double> field(const std::string& name, int64_t components) const;

 private:
  friend class CphdFile;
  PvpBlock(std::vector<char> bytes, int64_t count, int64_t record_bytes,
           const std::map<std::string, PvpField>* fields, const std::string* path)
  : bytes_(std::move(bytes)),
    count_(count),
    record_bytes_(record_bytes),
    fields_(fields),
    path_(path) {}

  std::vector<char> bytes_;
  int64_t count_{0};
  int64_t record_bytes_{0};
  const std::map<std::string, PvpField>* fields_{nullptr};
  const std::string* path_{nullptr};
};

struct ChannelInfo {
  std::string identifier;
  int64_t num_vectors{0};
  int64_t num_samples{0};
  int64_t signal_byte_offset{0};  // from the start of the signal block
  int64_t pvp_byte_offset{0};     // from the start of the PVP block
};

// A CPHD file opened for reading. Construction parses the header and the
// XML; the signal and PVP blocks are read on demand, so opening a 28 GiB
// collection costs only its metadata.
class CphdFile {
 public:
  explicit CphdFile(const std::string& path, int64_t io_chunk_bytes = 64 * 1024 * 1024);
  ~CphdFile();
  CphdFile(const CphdFile&) = delete;
  CphdFile& operator=(const CphdFile&) = delete;
  CphdFile(CphdFile&&) = delete;
  CphdFile& operator=(CphdFile&&) = delete;

  const std::string& path() const { return path_; }
  const std::string& version() const { return version_; }
  SampleFormat sample_format() const { return sample_format_; }
  SampleDomain sample_domain() const { return sample_domain_; }
  int fourier_sign() const { return fourier_sign_; }
  int64_t num_channels() const { return static_cast<int64_t>(channels_.size()); }
  const ChannelInfo& channel(int64_t index) const;
  bool has_pvp_field(const std::string& name) const { return pvp_fields_.count(name) != 0; }

  // Whether the stored samples are per-vector scaled. When true, reading the
  // signal without supplying the covering PvpBlock is an error rather than a
  // quietly unscaled read -- see read_signal_into.
  bool has_amp_sf() const { return has_pvp_field("AmpSF"); }

  // Per-vector AmpSF for a span, or empty when the file has none. The
  // operator applies these on the GPU CF8 path; read_signal_into applies
  // them on the host for integer formats and for tools.
  std::vector<double> amplitude_scale(const PvpBlock* pvp, int64_t count) const;

  // SceneCoordinates/IARP/ECF when the XML declares it. Spotlight
  // collections stare here; the independent images lattice is centred on this point.
  bool has_iarp() const { return has_iarp_; }
  const std::array<double, 3>& iarp_ecef() const;

  // SceneCoordinates ImageArea in the planar IA frame (metres from
  // IARP, along uIAX / uIAY). Explicit lattice.counts still uses
  // this; the independent images default is collection_physics() instead.
  struct ImageArea {
    std::array<double, 3> u_iax{};
    std::array<double, 3> u_iay{};
    double x1{0.0};
    double y1{0.0};
    double x2{0.0};
    double y2{0.0};
  };
  bool has_image_area() const { return has_image_area_; }
  const ImageArea& image_area() const;

  // Side of the largest east-north square, centred at the IARP,
  // that fits in a range x azimuth rectangle of those extents
  // whose range axis is `heading_cw_from_north_rad` from north.
  static double xy_square_in_range_azimuth(double range_swath_m, double azimuth_swath_m,
                                           double heading_cw_from_north_rad);

  // Ground-range / ground-azimuth support and the demo lattice:
  // XY-aligned square at the coarser of the two GSDs. Throws if
  // the XML/PVP fields it needs are missing.
  struct CollectionPhysics {
    double range_swath_m{0.0};
    double azimuth_swath_m{0.0};
    double gsd_range_m{0.0};
    double gsd_azimuth_m{0.0};
    double heading_cw_rad{0.0};
    double xy_square_m{0.0};
    double gsd_m{0.0};
    int64_t native_count{0};
  };
  bool has_collection_physics() const;
  CollectionPhysics collection_physics() const;

  // The per-vector parameter records for `count` vectors starting at
  // `first`, read in one go. Prefer this wherever more than one field is
  // wanted from the same span, which is the usual case.
  PvpBlock read_pvp_block(int64_t channel_index, int64_t first, int64_t count) const;

  // One field of one span. A convenience over read_pvp_block for the
  // single-field case; reading several fields this way re-reads the span
  // once per field.
  std::vector<double> read_pvp(int64_t channel_index, const std::string& name, int64_t first,
                               int64_t count, int64_t components) const;

  // Signal samples for `count` vectors starting at `first`, row-major
  // (count, num_samples), decoded to the caller's precision and scaled by the
  // file's per-vector amplitude factors where it has them. `out` must have
  // room for count * num_samples elements.
  //
  // Writes into the caller's buffer rather than returning one so that no
  // intermediate copy exists: for `T = float` the stored CF8 samples are
  // bit-for-bit big-endian `std::complex<float>`, so the bytes land directly
  // in `out` and are byte-swapped in place.
  //
  // `pvp` must cover exactly this span when has_amp_sf(); it is where the
  // scale factors are read from, and the caller already holds it. Required
  // rather than fetched internally so that omitting it is a loud error
  // instead of a silent factor-of-AmpSF error in the image -- and so that a
  // tick still costs one parameter-block read.
  template <typename T>
  void read_signal_into(int64_t channel_index, int64_t first, int64_t count, std::complex<T>* out,
                        const PvpBlock* pvp = nullptr, char* scratch = nullptr,
                        size_t scratch_bytes = 0) const;

  // Convenience wrapper allocating its own buffer, and reading the parameter
  // block itself where scale factors are needed. Used by tests and tools; the
  // operator uses read_signal_bytes (CF8) or read_signal_into (integer).
  std::vector<std::complex<float>> read_signal(int64_t channel_index, int64_t first,
                                               int64_t count) const;

  // Stored-form bytes for `count` vectors starting at `first`, no endian
  // swap and no widen. `dest` must hold at least
  // count * num_samples * stored-bytes-per-sample. The operator H2Ds this
  // span and decodes CF8 on the device; tests and tools still go through
  // read_signal_into for a host decode.
  void read_signal_bytes(int64_t channel_index, int64_t first, int64_t count, char* dest,
                         size_t dest_bytes) const;

 private:
  void parse_ascii_header(std::ifstream* stream);
  void parse_xml(const std::string& xml);
  int64_t signal_bytes_per_sample() const;
  // Fill `bytes` at `offset` via pread, or throw naming `what`. Spans
  // at or above 1 MiB are split across a few threads: one outstanding
  // sequential read is ~2 GB/s on NVMe here, four-way ~4.4 GB/s.
  void read_at(int64_t offset, char* dest, int64_t bytes, const char* what) const;
  // Validate before any allocation, not just when the read is issued.
  void check_span(int64_t offset, int64_t bytes, const char* what) const;
  int64_t checked_block_span(int64_t block_offset, int64_t block_size, int64_t relative,
                             int64_t bytes, const char* what) const;
  // Shared validation for the signal entry points.
  const ChannelInfo& check_signal_request(int64_t channel_index, int64_t first,
                                          int64_t count) const;
  // Stored-form scratch for the host widen path (double CF8, integer).
  // `new char[n]` is default-init, not zeroed; vector<char>(n) would be.
  void ensure_decode_scratch(size_t bytes) const;

  std::string path_;
  int64_t io_chunk_bytes_;
  std::string version_;
  std::map<std::string, std::string> header_;
  std::vector<ChannelInfo> channels_;
  std::map<std::string, PvpField> pvp_fields_;

  SampleFormat sample_format_{SampleFormat::kCF8};
  SampleDomain sample_domain_{SampleDomain::kFX};
  int fourier_sign_{-1};
  int64_t num_bytes_pvp_{0};
  int64_t xml_block_offset_{0};
  int64_t xml_block_size_{0};
  int64_t pvp_block_offset_{0};
  int64_t pvp_block_size_{-1};
  int64_t signal_block_offset_{0};
  int64_t signal_block_size_{-1};
  bool has_iarp_{false};
  std::array<double, 3> iarp_ecef_{};
  bool has_image_area_{false};
  ImageArea image_area_{};
  bool has_physics_xml_{false};
  double graze_deg_{0.0};
  double slant_range_m_{0.0};
  double azimuth_deg_{0.0};
  double fx_c_hz_{0.0};
  double fx_bw_hz_{0.0};
  double toa_saved_s_{0.0};
  double antenna_gain_db_{0.0};

  int fd_{-1};
  mutable std::unique_ptr<char[]> decode_scratch_;
  mutable size_t decode_scratch_bytes_{0};
};

}  // namespace holoscan::radar::cphd
