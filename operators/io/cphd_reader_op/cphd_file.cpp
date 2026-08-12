// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "io/cphd_reader_op/cphd_file.hpp"

#include <tinyxml2.h>

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

int16_t read_be_int16(const char* bytes) {
  uint16_t raw = 0;
  std::memcpy(&raw, bytes, sizeof(raw));
  raw = static_cast<uint16_t>((raw << 8) | (raw >> 8));
  int16_t value = 0;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
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

  // The declared XML size sizes the string below and comes from an ASCII
  // header a malformed file is free to lie in, so it is bounded by the file's
  // real length first: a header claiming a 4 GiB block would otherwise have
  // it allocated before the read discovered the file is 20 kB long.
  stream_.clear();
  stream_.seekg(0, std::ios::end);
  const int64_t file_size = static_cast<int64_t>(stream_.tellg());
  if (file_size < 0) {
    throw std::runtime_error("CphdFile: " + path_ +
                             ": cannot determine the file size");
  }
  // Compared without forming offset + size, which overflows for a header
  // declaring values near the range of the type.
  if (xml_block_size_ <= 0 || xml_block_offset_ < 0 ||
      xml_block_size_ > file_size ||
      xml_block_offset_ > file_size - xml_block_size_) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": header declares a " +
        std::to_string(xml_block_size_) + "-byte XML block at offset " +
        std::to_string(xml_block_offset_) + ", which does not fit in the " +
        std::to_string(file_size) + "-byte file");
  }
  std::string xml(static_cast<size_t>(xml_block_size_), '\0');
  // Through read_at like every other read, so that resetting the stream state
  // before seeking happens in one place rather than depending on getline
  // having left eofbit rather than failbit here.
  read_at(xml_block_offset_, xml.data(), xml_block_size_, "XML block");
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
  // The PVP record stride sizes every allocation and every field read below,
  // so a nonsensical value has to fail here rather than as an out-of-bounds
  // read later: zero would alias every vector onto the first, and a negative
  // wraps when it reaches the unsigned allocation size.
  if (num_bytes_pvp_ <= 0 ||
      num_bytes_pvp_ % static_cast<int64_t>(sizeof(double)) != 0) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": Data/NumBytesPVP is " +
        std::to_string(num_bytes_pvp_) +
        "; expected a positive multiple of 8 (PVP words are 8 bytes)");
  }

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
    // These four size and locate every read of the channel. A non-positive
    // count reaches an empty vector that callers index unconditionally, and a
    // negative count or offset wraps at the cast to an unsigned size.
    if (info.num_vectors <= 0 || info.num_samples <= 0) {
      throw std::runtime_error(
          "CphdFile: " + path_ + ": channel '" + info.identifier + "' declares "
          "NumVectors " + std::to_string(info.num_vectors) + " and NumSamples "
          + std::to_string(info.num_samples) + "; both must be positive");
    }
    if (info.signal_byte_offset < 0 || info.pvp_byte_offset < 0) {
      throw std::runtime_error(
          "CphdFile: " + path_ + ": channel '" + info.identifier +
          "' declares a negative array byte offset");
    }
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
    // Through required_text rather than GetText(): an empty <Offset/> returns
    // a null pointer, and building the std::string parameter from it is
    // undefined behaviour rather than the parse error it should be.
    field.offset_words =
        to_int(required_text(node, "Offset", path_), "PVP Offset", path_);
    field.size_words =
        to_int(required_text(node, "Size", path_), "PVP Size", path_);
    // Bound the field against the record it is read from. field() checks the
    // requested component count against size_words but never the declared
    // offset, so without this a field placed past the end of the record is an
    // out-of-bounds read on every vector of every block.
    if (field.offset_words < 0 || field.size_words <= 0 ||
        (field.offset_words + field.size_words) *
                static_cast<int64_t>(sizeof(double)) >
            num_bytes_pvp_) {
      throw std::runtime_error(
          "CphdFile: " + path_ + ": PVP field '" + node->Name() +
          "' declares offset " + std::to_string(field.offset_words) +
          " size " + std::to_string(field.size_words) +
          " words, which does not fit in the " +
          std::to_string(num_bytes_pvp_) + "-byte record");
    }
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
  // Bounded without forming first + count anywhere, including in the message:
  // this is a public entry point taking caller-supplied values, and the sum
  // of two large ones overflows, which is undefined rather than merely wrong.
  if (first < 0 || count < 0 || count > info.num_vectors ||
      first > info.num_vectors - count) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": " + std::to_string(count) +
        " PVP vectors from " + std::to_string(first) +
        " are out of range (channel has " +
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
  // Ordered as in read_pvp_block, and for the same reason: first + count is
  // never formed, because a caller passing two large values would overflow it.
  if (first < 0 || count < 0 || count > info.num_vectors ||
      first > info.num_vectors - count) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": " + std::to_string(count) +
        " signal vectors from " + std::to_string(first) +
        " are out of range (channel has " +
        std::to_string(info.num_vectors) + ")");
  }
  return info;
}

std::vector<double> CphdFile::amplitude_scale(const PvpBlock* pvp,
                                              int64_t count) const {
  if (!has_amp_sf()) {
    return {};
  }
  // Refusing here is the point of taking the block as an argument: a
  // collection whose samples are stored as small integers is off by orders of
  // magnitude without its scale factors, and wrong by a smooth factor, which
  // forms a perfectly focused image at the wrong amplitude.
  if (pvp == nullptr) {
    throw std::runtime_error(
        "CphdFile: " + path_ +
        ": this collection scales its samples per vector (AmpSF), so reading "
        "the signal needs the PVP block covering the same vectors");
  }
  if (pvp->count() != count) {
    throw std::runtime_error(
        "CphdFile: " + path_ + ": the PVP block covers " +
        std::to_string(pvp->count()) + " vectors but " +
        std::to_string(count) + " were asked for; the AmpSF factors would be "
        "applied to the wrong pulses");
  }
  return pvp->field("AmpSF", 1);
}

template <typename T>
void CphdFile::read_signal_into(int64_t channel_index, int64_t first,
                                int64_t count, std::complex<T>* out,
                                const PvpBlock* pvp) const {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "CphdFile reads into float or double precision only");
  const ChannelInfo& info = check_signal_request(channel_index, first, count);
  const std::vector<double> amp = amplitude_scale(pvp, count);

  const int64_t row_bytes = info.num_samples * signal_bytes_per_sample();
  const int64_t bytes = count * row_bytes;
  const int64_t total = count * info.num_samples;
  const int64_t base =
      signal_block_offset_ + info.signal_byte_offset + first * row_bytes;

  bool decoded = false;
  if constexpr (std::is_same_v<T, float>) {
    if (sample_format_ == SampleFormat::kCF8) {
      // CF8 *is* std::complex<float>, big-endian: same size, same layout. So
      // the bytes go straight into the caller's buffer and the swap happens
      // in place, which is why this case makes no copy at all.
      char* destination = reinterpret_cast<char*>(out);
      read_at(base, destination, bytes, "signal block");
      for (int64_t word = 0; word < 2 * total; ++word) {
        uint32_t raw = 0;
        std::memcpy(&raw, destination + word * sizeof(uint32_t), sizeof(raw));
        raw = byteswap32(raw);
        std::memcpy(destination + word * sizeof(uint32_t), &raw, sizeof(raw));
      }
      decoded = true;
    }
  }

  if (!decoded) {
    // Anything narrower than the output, or wider, needs somewhere to put the
    // stored form. One scratch buffer, one pass. The integer formats are
    // plain signed pairs -- no offset, no normalisation -- so the conversion
    // is a cast, and any scaling a collection needs is what AmpSF carries.
    std::vector<char> scratch(static_cast<size_t>(bytes));
    read_at(base, scratch.data(), bytes, "signal block");
    const char* source = scratch.data();
    for (int64_t sample = 0; sample < total; ++sample) {
      switch (sample_format_) {
        case SampleFormat::kCF8: {
          const char* pair = source + sample * 2 * sizeof(float);
          out[sample] = std::complex<T>(read_be_float(pair),
                                        read_be_float(pair + sizeof(float)));
          break;
        }
        case SampleFormat::kCI4: {
          const char* pair = source + sample * 2 * sizeof(int16_t);
          out[sample] = std::complex<T>(
              static_cast<T>(read_be_int16(pair)),
              static_cast<T>(read_be_int16(pair + sizeof(int16_t))));
          break;
        }
        case SampleFormat::kCI2: {
          // Single bytes have no byte order to get wrong; the only trap is
          // signedness, since plain char is unsigned on aarch64.
          const char* pair = source + sample * 2;
          out[sample] = std::complex<T>(
              static_cast<T>(static_cast<int8_t>(pair[0])),
              static_cast<T>(static_cast<int8_t>(pair[1])));
          break;
        }
      }
    }
  }

  if (!amp.empty()) {
    for (int64_t pulse = 0; pulse < count; ++pulse) {
      const T scale = static_cast<T>(amp[static_cast<size_t>(pulse)]);
      std::complex<T>* row = out + pulse * info.num_samples;
      for (int64_t sample = 0; sample < info.num_samples; ++sample) {
        row[sample] *= scale;
      }
    }
  }
}

template void CphdFile::read_signal_into<float>(int64_t, int64_t, int64_t,
                                                std::complex<float>*,
                                                const PvpBlock*) const;
template void CphdFile::read_signal_into<double>(int64_t, int64_t, int64_t,
                                                 std::complex<double>*,
                                                 const PvpBlock*) const;

std::vector<std::complex<float>> CphdFile::read_signal(int64_t channel_index,
                                                       int64_t first,
                                                       int64_t count) const {
  const ChannelInfo& info = check_signal_request(channel_index, first, count);
  std::vector<std::complex<float>> samples(
      static_cast<size_t>(count * info.num_samples));
  if (has_amp_sf()) {
    const PvpBlock pvp = read_pvp_block(channel_index, first, count);
    read_signal_into(channel_index, first, count, samples.data(), &pvp);
  } else {
    read_signal_into(channel_index, first, count, samples.data());
  }
  return samples;
}

}  // namespace holoscan::radar::cphd
