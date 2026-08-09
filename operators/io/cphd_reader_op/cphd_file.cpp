// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/cphd_reader_op/cphd_file.hpp"

#include <tinyxml2.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace holoscan::radar::cphd {

namespace {

// CPHD stores every numeric field big-endian regardless of the host, so the
// reads below always swap.
//
// That the host is little-endian is a precondition, not a guess: invariant 8
// makes a CUDA-capable NVIDIA GPU a hard dependency, and every target NVIDIA
// ships CUDA for is little-endian (x86_64, aarch64 Linux, ppc64*le*). It is
// asserted rather than detected at runtime because the alternative is a
// big-endian branch that no machine we can build on would ever execute, and
// so no test could ever cover -- the same reasoning invariant 8 applies to
// CPU fallback paths. A big-endian host fails to compile, which is the right
// failure for an assumption this load-bearing.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "CphdFile byte-swaps unconditionally; a big-endian host needs "
              "the swaps made conditional first");

uint64_t byteswap64(uint64_t value) {
  return ((value & 0x00000000000000ffULL) << 56) |
         ((value & 0x000000000000ff00ULL) << 40) |
         ((value & 0x0000000000ff0000ULL) << 24) |
         ((value & 0x00000000ff000000ULL) << 8) |
         ((value & 0x000000ff00000000ULL) >> 8) |
         ((value & 0x0000ff0000000000ULL) >> 24) |
         ((value & 0x00ff000000000000ULL) >> 40) |
         ((value & 0xff00000000000000ULL) >> 56);
}

uint32_t byteswap32(uint32_t value) {
  return ((value & 0x000000ffU) << 24) | ((value & 0x0000ff00U) << 8) |
         ((value & 0x00ff0000U) >> 8) | ((value & 0xff000000U) >> 24);
}

double read_be_double(const char* bytes) {
  uint64_t raw = 0;
  std::memcpy(&raw, bytes, sizeof(raw));
  raw = byteswap64(raw);
  double value = 0.0;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

int64_t read_be_int64(const char* bytes) {
  uint64_t raw = 0;
  std::memcpy(&raw, bytes, sizeof(raw));
  raw = byteswap64(raw);
  int64_t value = 0;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

// The declared format of one PVP word. CPHD writes each field's type in the
// XML, and the same eight bytes mean different numbers depending on it: the
// integer 1 read as a double is 5e-324, not 1.0.
enum class WordType { kFloat, kSignedInt, kUnsignedInt };

// `format` is the XML's Format text -- "F8", "I8", "U8", or a compound like
// "X=F8;Y=F8;Z=F8;". Compound forms are homogeneous in every collection seen,
// so the first component decides.
WordType word_type_of(const std::string& format) {
  const size_t equals = format.find('=');
  const size_t start = equals == std::string::npos ? 0 : equals + 1;
  const char code = start < format.size() ? format[start] : 'F';
  if (code == 'I') return WordType::kSignedInt;
  if (code == 'U') return WordType::kUnsignedInt;
  return WordType::kFloat;
}

float read_be_float(const char* bytes) {
  uint32_t raw = 0;
  std::memcpy(&raw, bytes, sizeof(raw));
  raw = byteswap32(raw);
  float value = 0.0F;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

// tinyxml2 has no namespace handling, so a document using a prefix reports
// element names as "cphd:Data". Matching on the local name accepts both that
// and the default-namespace form CPHD files normally use.
bool local_name_is(const tinyxml2::XMLElement* element, const char* name) {
  const char* value = element->Name();
  if (value == nullptr) {
    return false;
  }
  const char* colon = std::strrchr(value, ':');
  const char* local = (colon == nullptr) ? value : colon + 1;
  return std::strcmp(local, name) == 0;
}

const tinyxml2::XMLElement* first_child(const tinyxml2::XMLElement* parent,
                                        const char* name) {
  if (parent == nullptr) {
    return nullptr;
  }
  for (const tinyxml2::XMLElement* child = parent->FirstChildElement();
       child != nullptr; child = child->NextSiblingElement()) {
    if (local_name_is(child, name)) {
      return child;
    }
  }
  return nullptr;
}

// Walk a slash-separated path of local names, e.g. "Data/SignalArrayFormat".
const tinyxml2::XMLElement* find_path(const tinyxml2::XMLElement* root,
                                      const std::string& path) {
  const tinyxml2::XMLElement* node = root;
  std::stringstream parts(path);
  std::string part;
  while (std::getline(parts, part, '/')) {
    node = first_child(node, part.c_str());
    if (node == nullptr) {
      return nullptr;
    }
  }
  return node;
}

std::string required_text(const tinyxml2::XMLElement* root,
                          const std::string& path, const std::string& file) {
  const tinyxml2::XMLElement* node = find_path(root, path);
  if (node == nullptr || node->GetText() == nullptr) {
    throw std::runtime_error("CphdFile: " + file + ": missing XML element '" +
                             path + "'");
  }
  std::string text = node->GetText();
  const size_t begin = text.find_first_not_of(" \t\r\n");
  const size_t end = text.find_last_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    throw std::runtime_error("CphdFile: " + file + ": XML element '" + path +
                             "' is empty");
  }
  return text.substr(begin, end - begin + 1);
}

int64_t to_int(const std::string& text, const std::string& what,
               const std::string& file) {
  try {
    size_t consumed = 0;
    const int64_t value = std::stoll(text, &consumed);
    if (consumed != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("CphdFile: " + file + ": " + what +
                             " is not an integer: '" + text + "'");
  }
}

}  // namespace

const char* to_string(SampleFormat format) {
  switch (format) {
    case SampleFormat::kCF8:
      return "CF8";
    case SampleFormat::kCI4:
      return "CI4";
    case SampleFormat::kCI2:
      return "CI2";
  }
  return "unknown";
}

const char* to_string(SampleDomain domain) {
  return domain == SampleDomain::kFX ? "FX" : "TOA";
}

CphdFile::CphdFile(const std::string& path) : path_(path) {
  stream_.open(path, std::ios::binary);
  if (!stream_) {
    throw std::runtime_error("CphdFile: cannot open '" + path + "'");
  }

  parse_ascii_header(&stream_);

  if (xml_block_size_ <= 0) {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": header declares no XML block");
  }
  std::string xml(static_cast<size_t>(xml_block_size_), '\0');
  stream_.seekg(xml_block_offset_);
  stream_.read(xml.data(), xml_block_size_);
  if (stream_.gcount() != xml_block_size_) {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": truncated XML block (wanted " +
                             std::to_string(xml_block_size_) + " bytes, got " +
                             std::to_string(stream_.gcount()) + ")");
  }
  parse_xml(xml);
}

// The CPHD ASCII header is "KEY := VALUE" lines terminated by a form feed
// and a newline. The first line is the version, e.g. "CPHD/1.1.0".
void CphdFile::parse_ascii_header(std::ifstream* stream) {
  std::string line;
  if (!std::getline(*stream, line)) {
    throw std::runtime_error("CphdFile: " + path_ + ": file is empty");
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  if (line.rfind("CPHD/", 0) != 0) {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": not a CPHD file (first line is '" + line +
                             "')");
  }
  version_ = line.substr(std::strlen("CPHD/"));

  // Bound the scan so a binary file that happens to start with the magic
  // cannot spin through gigabytes looking for a terminator.
  constexpr int kMaxHeaderLines = 512;
  bool terminated = false;
  for (int i = 0; i < kMaxHeaderLines; ++i) {
    if (!std::getline(*stream, line)) {
      break;
    }
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty() && line[0] == '\f') {
      terminated = true;
      break;
    }
    const size_t separator = line.find(":=");
    if (separator == std::string::npos) {
      continue;
    }
    std::string key = line.substr(0, separator);
    std::string value = line.substr(separator + 2);
    const auto trim = [](std::string* text) {
      const size_t begin = text->find_first_not_of(" \t");
      const size_t end = text->find_last_not_of(" \t");
      *text = (begin == std::string::npos)
                  ? std::string()
                  : text->substr(begin, end - begin + 1);
    };
    trim(&key);
    trim(&value);
    header_[key] = value;
  }
  if (!terminated) {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": ASCII header has no form-feed terminator");
  }

  const auto required = [&](const char* key) {
    const auto found = header_.find(key);
    if (found == header_.end()) {
      throw std::runtime_error("CphdFile: " + path_ +
                               ": header is missing '" + key + "'");
    }
    return to_int(found->second, key, path_);
  };
  xml_block_offset_ = required("XML_BLOCK_BYTE_OFFSET");
  xml_block_size_ = required("XML_BLOCK_SIZE");
  pvp_block_offset_ = required("PVP_BLOCK_BYTE_OFFSET");
  signal_block_offset_ = required("SIGNAL_BLOCK_BYTE_OFFSET");
}

void CphdFile::parse_xml(const std::string& xml) {
  tinyxml2::XMLDocument document;
  if (document.Parse(xml.c_str(), xml.size()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": XML metadata does not parse (" +
                             document.ErrorStr() + ")");
  }
  const tinyxml2::XMLElement* root = document.RootElement();
  if (root == nullptr) {
    throw std::runtime_error("CphdFile: " + path_ + ": XML has no root");
  }

  const std::string format = required_text(root, "Data/SignalArrayFormat",
                                           path_);
  if (format == "CF8") {
    sample_format_ = SampleFormat::kCF8;
  } else if (format == "CI4") {
    sample_format_ = SampleFormat::kCI4;
  } else if (format == "CI2") {
    sample_format_ = SampleFormat::kCI2;
  } else {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": unknown SignalArrayFormat '" + format + "'");
  }

  const std::string domain = required_text(root, "Global/DomainType", path_);
  if (domain == "FX") {
    sample_domain_ = SampleDomain::kFX;
  } else if (domain == "TOA") {
    sample_domain_ = SampleDomain::kTOA;
  } else {
    throw std::runtime_error("CphdFile: " + path_ + ": unknown DomainType '" +
                             domain + "'");
  }

  fourier_sign_ = static_cast<int>(
      to_int(required_text(root, "Global/SGN", path_), "Global/SGN", path_));
  num_bytes_pvp_ = to_int(required_text(root, "Data/NumBytesPVP", path_),
                          "Data/NumBytesPVP", path_);

  const tinyxml2::XMLElement* data = first_child(root, "Data");
  for (const tinyxml2::XMLElement* node = data->FirstChildElement();
       node != nullptr; node = node->NextSiblingElement()) {
    if (!local_name_is(node, "Channel")) {
      continue;
    }
    ChannelInfo info;
    info.identifier = required_text(node, "Identifier", path_);
    info.num_vectors =
        to_int(required_text(node, "NumVectors", path_), "NumVectors", path_);
    info.num_samples =
        to_int(required_text(node, "NumSamples", path_), "NumSamples", path_);
    info.signal_byte_offset =
        to_int(required_text(node, "SignalArrayByteOffset", path_),
               "SignalArrayByteOffset", path_);
    info.pvp_byte_offset = to_int(
        required_text(node, "PVPArrayByteOffset", path_), "PVPArrayByteOffset",
        path_);
    channels_.push_back(info);
  }
  if (channels_.empty()) {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": XML declares no Data/Channel");
  }

  // The PVP record layout is declared, not assumed: both the field set and
  // NumBytesPVP vary between collections.
  const tinyxml2::XMLElement* pvp = first_child(root, "PVP");
  if (pvp == nullptr) {
    throw std::runtime_error("CphdFile: " + path_ + ": XML has no PVP block");
  }
  for (const tinyxml2::XMLElement* node = pvp->FirstChildElement();
       node != nullptr; node = node->NextSiblingElement()) {
    const tinyxml2::XMLElement* offset = first_child(node, "Offset");
    const tinyxml2::XMLElement* size = first_child(node, "Size");
    if (offset == nullptr || size == nullptr) {
      continue;  // AddedPVP wrappers and similar; not a field record.
    }
    PvpField field;
    field.offset_words = to_int(offset->GetText(), "PVP Offset", path_);
    field.size_words = to_int(size->GetText(), "PVP Size", path_);
    const tinyxml2::XMLElement* format_node = first_child(node, "Format");
    if (format_node != nullptr && format_node->GetText() != nullptr) {
      field.format = format_node->GetText();
    }
    const char* name = node->Name();
    const char* colon = std::strrchr(name, ':');
    pvp_fields_[(colon == nullptr) ? name : colon + 1] = field;
  }
  if (pvp_fields_.empty()) {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": PVP block declares no fields");
  }
}

void CphdFile::read_at(int64_t offset, char* dest, int64_t bytes,
                       const char* what) const {
  stream_.clear();
  stream_.seekg(offset);
  stream_.read(dest, static_cast<std::streamsize>(bytes));
  if (stream_.gcount() != static_cast<std::streamsize>(bytes)) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": truncated " + what + " (wanted " +
        std::to_string(bytes) + " bytes at " + std::to_string(offset) +
        ", got " + std::to_string(stream_.gcount()) + ")");
  }
}

const ChannelInfo& CphdFile::channel(int64_t index) const {
  if (index < 0 || index >= static_cast<int64_t>(channels_.size())) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": channel " + std::to_string(index) +
        " out of range (file has " + std::to_string(channels_.size()) + ")");
  }
  return channels_[static_cast<size_t>(index)];
}

int64_t CphdFile::signal_bytes_per_sample() const {
  switch (sample_format_) {
    case SampleFormat::kCF8:
      return 8;
    case SampleFormat::kCI4:
      return 4;
    case SampleFormat::kCI2:
      return 2;
  }
  return 0;
}

bool PvpBlock::has(const std::string& name) const {
  return fields_->count(name) != 0;
}

std::vector<double> PvpBlock::field(const std::string& name,
                                    int64_t components) const {
  const auto found = fields_->find(name);
  if (found == fields_->end()) {
    throw std::runtime_error("CphdFile: " + *path_ + ": PVP field '" + name +
                             "' is not in this file");
  }
  const PvpField& layout = found->second;
  if (components > layout.size_words) {
    throw std::runtime_error(
        "CphdFile: " + *path_ + ": PVP field '" + name + "' has " +
        std::to_string(layout.size_words) + " words, asked for " +
        std::to_string(components));
  }

  // Decode by the declared format, not by assumption. Reading an I8 field as
  // a double reinterprets the bit pattern rather than converting it, so
  // SIGNAL's 1 becomes 5e-324 -- wrong in a way that looks like a tiny number
  // rather than like an error.
  const WordType type = word_type_of(layout.format);

  // Integers wider than the mantissa cannot round-trip through a double, so
  // this refuses rather than returning a quietly rounded value. No field in
  // the CPHD defined set can reach it (SIGNAL is the only integer, and it is
  // a 0/1 flag); an added field carrying, say, a 64-bit identifier could.
  constexpr int64_t kExactLimit = 1LL << 53;
  const auto checked = [&](int64_t raw) {
    if (raw > kExactLimit || raw < -kExactLimit) {
      throw std::runtime_error(
          "CphdFile: " + *path_ + ": PVP field '" + name + "' (format " +
          layout.format + ") holds " + std::to_string(raw) +
          ", which no double represents exactly; read it as an integer "
          "instead of through field()");
    }
    return static_cast<double>(raw);
  };

  std::vector<double> values(static_cast<size_t>(count_ * components));
  for (int64_t vector = 0; vector < count_; ++vector) {
    const char* record = bytes_.data() + vector * record_bytes_;
    for (int64_t component = 0; component < components; ++component) {
      const char* word =
          record + (layout.offset_words + component) * sizeof(double);
      double value = 0.0;
      switch (type) {
        case WordType::kFloat:
          value = read_be_double(word);
          break;
        case WordType::kSignedInt:
          value = checked(read_be_int64(word));
          break;
        case WordType::kUnsignedInt: {
          const int64_t raw = read_be_int64(word);
          if (raw < 0) {
            throw std::runtime_error(
                "CphdFile: " + *path_ + ": PVP field '" + name +
                "' is unsigned but its top bit is set, so it exceeds what a "
                "double represents exactly");
          }
          value = checked(raw);
          break;
        }
      }
      values[static_cast<size_t>(vector * components + component)] = value;
    }
  }
  return values;
}

PvpBlock CphdFile::read_pvp_block(int64_t channel_index, int64_t first,
                                  int64_t count) const {
  const ChannelInfo& info = channel(channel_index);
  if (first < 0 || count < 0 || first + count > info.num_vectors) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": PVP vectors [" + std::to_string(first) +
        ", " + std::to_string(first + count) + ") out of range (channel has " +
        std::to_string(info.num_vectors) + ")");
  }

  // One read for the whole span. Reading per vector would cost `count`
  // seeks, and reading per field would cost one pass over the same bytes for
  // each of the eight or so fields a tick wants.
  std::vector<char> block(static_cast<size_t>(count * num_bytes_pvp_));
  const int64_t base =
      pvp_block_offset_ + info.pvp_byte_offset + first * num_bytes_pvp_;
  read_at(base, block.data(), static_cast<int64_t>(block.size()),
          "PVP block");
  return PvpBlock(std::move(block), count, num_bytes_pvp_, &pvp_fields_,
                  &path_);
}

std::vector<double> CphdFile::read_pvp(int64_t channel_index,
                                       const std::string& name, int64_t first,
                                       int64_t count,
                                       int64_t components) const {
  return read_pvp_block(channel_index, first, count).field(name, components);
}

const ChannelInfo& CphdFile::check_signal_request(int64_t channel_index,
                                                  int64_t first,
                                                  int64_t count) const {
  const ChannelInfo& info = channel(channel_index);
  if (sample_format_ != SampleFormat::kCF8) {
    throw std::runtime_error(
        std::string("CphdFile: ") + path_ + ": sample format " +
        to_string(sample_format_) +
        " is not implemented yet (only CF8; CI4 and CI2 arrive with mode 3)");
  }
  if (first < 0 || count < 0 || first + count > info.num_vectors) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": signal vectors [" + std::to_string(first) +
        ", " + std::to_string(first + count) + ") out of range (channel has " +
        std::to_string(info.num_vectors) + ")");
  }
  return info;
}

template <typename T>
void CphdFile::read_signal_into(int64_t channel_index, int64_t first,
                                int64_t count, std::complex<T>* out) const {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "CphdFile reads into float or double precision only");
  const ChannelInfo& info = check_signal_request(channel_index, first, count);
  const int64_t row_bytes = info.num_samples * signal_bytes_per_sample();
  const int64_t bytes = count * row_bytes;
  const int64_t words = 2 * count * info.num_samples;  // one per real scalar
  const int64_t base =
      signal_block_offset_ + info.signal_byte_offset + first * row_bytes;

  if constexpr (std::is_same_v<T, float>) {
    // CF8 *is* std::complex<float>, big-endian: same size, same layout. So
    // the bytes go straight into the caller's buffer and the swap happens in
    // place, which is why a single-precision instance makes no copy at all.
    char* destination = reinterpret_cast<char*>(out);
    read_at(base, destination, bytes, "signal block");
    for (int64_t word = 0; word < words; ++word) {
      uint32_t raw = 0;
      std::memcpy(&raw, destination + word * sizeof(uint32_t), sizeof(raw));
      raw = byteswap32(raw);
      std::memcpy(destination + word * sizeof(uint32_t), &raw, sizeof(raw));
    }
  } else {
    // Widening to double needs somewhere to put the narrow input, so this
    // path keeps one scratch buffer. Same bit patterns, same values as the
    // float path -- asserted in the tests.
    std::vector<char> scratch(static_cast<size_t>(bytes));
    read_at(base, scratch.data(), bytes, "signal block");
    for (int64_t sample = 0; sample < count * info.num_samples; ++sample) {
      const char* pair = scratch.data() + sample * 2 * sizeof(float);
      out[sample] = std::complex<T>(read_be_float(pair),
                                    read_be_float(pair + sizeof(float)));
    }
  }
}

template void CphdFile::read_signal_into<float>(int64_t, int64_t, int64_t,
                                                std::complex<float>*) const;
template void CphdFile::read_signal_into<double>(int64_t, int64_t, int64_t,
                                                 std::complex<double>*) const;

std::vector<std::complex<float>> CphdFile::read_signal(int64_t channel_index,
                                                       int64_t first,
                                                       int64_t count) const {
  const ChannelInfo& info = check_signal_request(channel_index, first, count);
  std::vector<std::complex<float>> samples(
      static_cast<size_t>(count * info.num_samples));
  read_signal_into(channel_index, first, count, samples.data());
  return samples;
}

}  // namespace holoscan::radar::cphd
