// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>
#include <cstdint>
#include <fstream>
#include <map>
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
// Scope is what the mode 1 collection needs (SIGPROC_PLAN.md's
// minimum-per-demo rule): one channel, `CF8` samples, no `AmpSF`. The paths
// this does not implement throw rather than guess; mode 3 adds `CI4`,
// `AmpSF`, and the second CPHD minor version.
namespace holoscan::radar::cphd {

// Sample storage format, from Data/SignalArrayFormat. Only kCF8 decodes
// today; the others are named so the error can say what it found.
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
  std::vector<double> field(const std::string& name,
                            int64_t components) const;

 private:
  friend class CphdFile;
  PvpBlock(std::vector<char> bytes, int64_t count, int64_t record_bytes,
           const std::map<std::string, PvpField>* fields,
           const std::string* path)
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
  explicit CphdFile(const std::string& path);

  const std::string& path() const { return path_; }
  const std::string& version() const { return version_; }
  SampleFormat sample_format() const { return sample_format_; }
  SampleDomain sample_domain() const { return sample_domain_; }
  int fourier_sign() const { return fourier_sign_; }
  int64_t num_channels() const {
    return static_cast<int64_t>(channels_.size());
  }
  const ChannelInfo& channel(int64_t index) const;
  bool has_pvp_field(const std::string& name) const {
    return pvp_fields_.count(name) != 0;
  }

  // The per-vector parameter records for `count` vectors starting at
  // `first`, read in one go. Prefer this wherever more than one field is
  // wanted from the same span, which is the usual case.
  PvpBlock read_pvp_block(int64_t channel_index, int64_t first,
                          int64_t count) const;

  // One field of one span. A convenience over read_pvp_block for the
  // single-field case; reading several fields this way re-reads the span
  // once per field.
  std::vector<double> read_pvp(int64_t channel_index, const std::string& name,
                               int64_t first, int64_t count,
                               int64_t components) const;

  // Signal samples for `count` vectors starting at `first`, row-major
  // (count, num_samples), in host byte order and at the caller's precision.
  // `out` must have room for count * num_samples elements.
  //
  // Writes into the caller's buffer rather than returning one so that no
  // intermediate copy exists: for `T = float` the stored CF8 samples are
  // bit-for-bit big-endian `std::complex<float>`, so the bytes land directly
  // in `out` and are byte-swapped in place.
  template <typename T>
  void read_signal_into(int64_t channel_index, int64_t first, int64_t count,
                        std::complex<T>* out) const;

  // Convenience wrapper allocating its own buffer. Used by tests; the
  // operator uses read_signal_into to avoid the copy.
  std::vector<std::complex<float>> read_signal(int64_t channel_index,
                                               int64_t first,
                                               int64_t count) const;

 private:
  void parse_ascii_header(std::ifstream* stream);
  void parse_xml(const std::string& xml);
  int64_t signal_bytes_per_sample() const;
  // Seek and fill `bytes` at `offset`, or throw naming `what`.
  void read_at(int64_t offset, char* dest, int64_t bytes,
               const char* what) const;
  // Shared validation for the two signal entry points.
  const ChannelInfo& check_signal_request(int64_t channel_index, int64_t first,
                                          int64_t count) const;

  std::string path_;
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
  int64_t signal_block_offset_{0};

  mutable std::ifstream stream_;
};

}  // namespace holoscan::radar::cphd
