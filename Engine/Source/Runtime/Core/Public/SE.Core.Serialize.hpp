#pragma once
#include <set>
#include <map>
#include <list>
#include <vector>
#include <string>
#include <cstdint>

namespace SIByL {
inline namespace utility {
struct DataStream;

struct ISerializable {
  virtual auto serialize(DataStream& stream) const noexcept -> void = 0;
  virtual auto deserialize(DataStream& stream) noexcept -> bool = 0;
};

struct DataStream {
  std::vector<char> buf;
  size_t pos = 0;
  // serialize data
  inline auto write(char const* data, int len) noexcept -> void;
  inline auto write(bool value) noexcept -> void;
  inline auto write(char value) noexcept -> void;
  inline auto write(int32_t value) noexcept -> void;
  inline auto write(int64_t value) noexcept -> void;
  inline auto write(uint32_t value) noexcept -> void;
  inline auto write(uint64_t value) noexcept -> void;
  inline auto write(float value) noexcept -> void;
  inline auto write(double value) noexcept -> void;
  inline auto write(char const* value) noexcept -> void;
  inline auto write(std::string const& value) noexcept -> void;
  inline auto write(ISerializable const& value) noexcept -> void;
  template <class T>
  inline auto write(std::vector<T> const& value) noexcept -> void;
  template <class T>
  inline auto write(std::list<T> const& value) noexcept -> void;
  template <class T>
  inline auto write(std::set<T> const& value) noexcept -> void;
  template <class K, class V>
  inline auto write(std::map<K, V> const& value) noexcept -> void;
  template <class T, class... Args>
  inline auto write_args(T const& value, Args const&... args) noexcept -> void;
  inline auto write_args() noexcept -> void;
  // deserialize data
  inline auto read(char* data, int len) noexcept -> void;
  inline auto read(bool& value) noexcept -> bool;
  inline auto read(char& value) noexcept -> bool;
  inline auto read(int32_t& value) noexcept -> bool;
  inline auto read(int64_t& value) noexcept -> bool;
  inline auto read(uint32_t& value) noexcept -> bool;
  inline auto read(uint64_t& value) noexcept -> bool;
  inline auto read(float& value) noexcept -> bool;
  inline auto read(double& value) noexcept -> bool;
  inline auto read(std::string& value) noexcept -> bool;
  inline auto read(ISerializable& value) noexcept -> bool;
  template <class T>
  inline auto read(std::vector<T>& value) noexcept -> bool;
  template <class T>
  inline auto read(std::list<T>& value) noexcept -> bool;
  template <class T>
  inline auto read(std::set<T>& value) noexcept -> bool;
  template <class K, class V>
  inline auto read(std::map<K, V>& value) noexcept -> bool;
  template <class T, class... Args>
  inline auto read_args(T& value, Args&... args) noexcept -> bool;
  inline auto read_args() noexcept -> bool;
  // stream serialize
  inline auto operator<<(bool value) -> DataStream&;
  inline auto operator<<(char value) -> DataStream&;
  inline auto operator<<(int32_t value) -> DataStream&;
  inline auto operator<<(int64_t value) -> DataStream&;
  inline auto operator<<(uint32_t value) -> DataStream&;
  inline auto operator<<(uint64_t value) -> DataStream&;
  inline auto operator<<(float value) -> DataStream&;
  inline auto operator<<(double value) -> DataStream&;
  inline auto operator<<(char const* value) -> DataStream&;
  inline auto operator<<(std::string const& value) -> DataStream&;
  inline auto operator<<(ISerializable const& value) -> DataStream&;
  template <class T>
  inline auto operator<<(std::vector<T>& value) noexcept -> DataStream&;
  template <class T>
  inline auto operator<<(std::list<T>& value) noexcept -> DataStream&;
  template <class T>
  inline auto operator<<(std::set<T>& value) noexcept -> DataStream&;
  template <class K, class V>
  inline auto operator<<(std::map<K, V>& value) noexcept -> DataStream&;
  // stream serialize
  inline auto operator>>(bool& value) -> DataStream&;
  inline auto operator>>(char& value) -> DataStream&;
  inline auto operator>>(int32_t& value) -> DataStream&;
  inline auto operator>>(int64_t& value) -> DataStream&;
  inline auto operator>>(uint32_t& value) -> DataStream&;
  inline auto operator>>(uint64_t& value) -> DataStream&;
  inline auto operator>>(float& value) -> DataStream&;
  inline auto operator>>(double& value) -> DataStream&;
  inline auto operator>>(std::string& value) -> DataStream&;
  inline auto operator>>(ISerializable& value) -> DataStream&;
  template <class T>
  inline auto operator>>(std::vector<T>& value) noexcept -> DataStream&;
  template <class T>
  inline auto operator>>(std::list<T>& value) noexcept -> DataStream&;
  template <class T>
  inline auto operator>>(std::set<T>& value) noexcept -> DataStream&;
  template <class K, class V>
  inline auto operator>>(std::map<K, V>& value) noexcept -> DataStream&;
  // data type definition
  enum struct DataType : uint8_t {
    BOOL,
    CHAR,
    INT32,
    INT64,
    UINT32,
    UINT64,
    FLOAT,
    DOUBLE,
    STRING,
    VECTOR,
    LIST,
    MAP,
    SET,
    CUSTOM
  };
};
}  // namespace utility
}  // namespace SIByL

namespace SIByL::Core::Impl {
inline auto reserve(DataStream* ds, int len) noexcept -> void {
  size_t const size = ds->buf.size();
  size_t cap = ds->buf.capacity();
  if (size + len > cap) {
    while (size + len > cap) cap = (cap == 0) ? 1 : cap << 1;
    ds->buf.reserve(cap);
  }
}
inline auto write_type(DataStream* ds, DataStream::DataType type) noexcept
    -> void {
  ds->write((char*)&type, sizeof(char));
}
}  // namespace SIByL::Core::Impl

#define SERIALIZE(...)                                                      \
  virtual auto serialize(DataStream& stream) const noexcept                 \
      -> void override {                                                    \
    Core::Impl::write_type(&stream, DataStream::DataType::CUSTOM);          \
    stream.write_args(__VA_ARGS__);                                         \
  }                                                                         \
  virtual auto deserialize(DataStream& stream) noexcept                     \
      -> bool override {                                                    \
    DataStream::DataType type;                                              \
    stream.read((char*)&type, sizeof(char));                                \
    if (type != DataStream::DataType::CUSTOM) {                             \
      return false;                                                         \
    }                                                                       \
    stream.read_args(__VA_ARGS__);                                          \
    return true;                                                            \
  }

namespace SIByL {
inline namespace utility {
inline auto DataStream::write(char const* data, int len) noexcept -> void {
  Core::Impl::reserve(this, len);
  size_t size = buf.size();
  buf.resize(size + len);
  std::memcpy(&buf[size], data, len);
}
inline auto DataStream::write(bool value) noexcept -> void {
  Core::Impl::write_type(this, DataType::BOOL);
  write((char*)&value, sizeof(char));
}
inline auto DataStream::write(char value) noexcept -> void {
  Core::Impl::write_type(this, DataType::CHAR);
  write((char*)&value, sizeof(char));
}
inline auto DataStream::write(int32_t value) noexcept -> void {
  Core::Impl::write_type(this, DataType::INT32);
  write((char*)&value, sizeof(int32_t));
}
inline auto DataStream::write(int64_t value) noexcept -> void {
  Core::Impl::write_type(this, DataType::INT64);
  write((char*)&value, sizeof(int64_t));
}
inline auto DataStream::write(uint32_t value) noexcept -> void {
  Core::Impl::write_type(this, DataType::UINT32);
  write((char*)&value, sizeof(uint32_t));
}
inline auto DataStream::write(uint64_t value) noexcept -> void {
  Core::Impl::write_type(this, DataType::UINT64);
  write((char*)&value, sizeof(uint64_t));
}
inline auto DataStream::write(float value) noexcept -> void {
  Core::Impl::write_type(this, DataType::FLOAT);
  write((char*)&value, sizeof(float));
}
inline auto DataStream::write(double value) noexcept -> void {
  Core::Impl::write_type(this, DataType::DOUBLE);
  write((char*)&value, sizeof(double));
}
inline auto DataStream::write(char const* value) noexcept -> void {
  Core::Impl::write_type(this, DataType::STRING);
  const size_t len = std::strlen(value);
  write(len); write(value, len);
}
inline auto DataStream::write(std::string const& value) noexcept -> void {
  Core::Impl::write_type(this, DataType::STRING);
  const size_t len = value.size();
  write(len); write(value.data(), len);
}
inline auto DataStream::write(ISerializable const& value) noexcept -> void {
  value.serialize(*this);
}
template <class T>
inline auto DataStream::write(std::vector<T> const& value) noexcept -> void {
  Core::Impl::write_type(this, DataType::VECTOR);
  size_t len = value.size(); write(len);
  for (auto iter = value.begin(); iter != value.end(); ++iter) write((*iter));
}
template <class T>
inline auto DataStream::write(std::list<T> const& value) noexcept -> void {
  Core::Impl::write_type(this, DataType::LIST);
  size_t len = value.size(); write(len);
  for (auto iter = value.begin(); iter != value.end(); ++iter) write((*iter));
}
template <class T>
inline auto DataStream::write(std::set<T> const& value) noexcept -> void {
  Core::Impl::write_type(this, DataType::SET);
  size_t len = value.size(); write(len);
  for (auto iter = value.begin(); iter != value.end(); ++iter) write((*iter));
}
template <class K, class V>
inline auto DataStream::write(std::map<K, V> const& value) noexcept -> void {
  Core::Impl::write_type(this, DataType::MAP);
  size_t len = value.size(); write(len);
  for (auto iter = value.begin(); iter != value.end(); ++iter) {
    write(iter->first);
    write(iter->second);
  }
}
template <class T, class... Args>
inline auto DataStream::write_args(T const& value, Args const&... args) noexcept -> void {
  write(value);
  write_args(args...);
}
inline auto DataStream::write_args() noexcept -> void {}

inline auto DataStream::operator<<(bool value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(char value) -> DataStream&  { write(value); return *this; }
inline auto DataStream::operator<<(int32_t value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(int64_t value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(uint32_t value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(uint64_t value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(float value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(double value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(char const* value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(std::string const& value) -> DataStream& { write(value); return *this; }
inline auto DataStream::operator<<(ISerializable const& value) -> DataStream& { write(value); return *this; }
template <class T> inline auto DataStream::operator<<(std::vector<T>& value) noexcept -> DataStream& { write(value); return *this; }
template <class T> inline auto DataStream::operator<<(std::list<T>& value) noexcept -> DataStream& { write(value); return *this; }
template <class T> inline auto DataStream::operator<<(std::set<T>& value) noexcept -> DataStream& { write(value); return *this; }
template <class K, class V> inline auto DataStream::operator<<(std::map<K, V>& value) noexcept -> DataStream& { write(value); return *this; }

inline auto DataStream::read(char* data, int len) noexcept -> void {
  std::memcpy(data, (char*)&buf[pos], len);
  pos += len; return;
}
inline auto DataStream::read(bool& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::BOOL) return false;
  ++pos; value = buf[pos]; ++pos; return true;
}
inline auto DataStream::read(char& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::CHAR) return false;
  ++pos; value = buf[pos]; ++pos; return true;
}
inline auto DataStream::read(int32_t& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::INT32) return false;
  value = *((int32_t*)(&buf[++pos])); pos += 4; return true;
}
inline auto DataStream::read(int64_t& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::INT64) return false;
  value = *((int64_t*)(&buf[++pos])); pos += 8; return true;
}
inline auto DataStream::read(uint32_t& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::UINT32) return false;
  value = *((uint32_t*)(&buf[++pos])); pos += 4; return true;
}
inline auto DataStream::read(uint64_t& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::UINT64) return false;
  value = *((uint64_t*)(&buf[++pos])); pos += 8; return true;
}
inline auto DataStream::read(float& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::FLOAT) return false;
  value = *((float*)(&buf[++pos])); pos += 4; return true;
}
inline auto DataStream::read(double& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::DOUBLE) return false;
  value = *((double*)(&buf[++pos])); pos += 8; return true;
}
inline auto DataStream::read(std::string& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::STRING) return false;
  ++pos; size_t len; read(len); value.assign((char*)&(buf[pos]), len);
  pos += len; return true;
}
inline auto DataStream::read(ISerializable& value) noexcept -> bool {
  return value.deserialize(*this);
}
template <class T>
inline auto DataStream::read(std::vector<T>& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::VECTOR) return false;
  ++pos; size_t len; read(len); value.resize(len);
  for (size_t i = 0; i < len; ++i) read(value[i]); return true;
}
template <class T>
inline auto DataStream::read(std::list<T>& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::LIST) return false;
  ++pos; size_t len; read(len); value.resize(len);
  for (size_t i = 0; i < len; ++i) read(value[i]); return true;
}
template <class T>
inline auto DataStream::read(std::set<T>& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::SET) return false;
  ++pos;  size_t len; read(len);
  for (size_t i = 0; i < len; ++i) {
    T v; read(v); value.insert(v);
  } return true;
}
template <class K, class V>
inline auto DataStream::read(std::map<K, V>& value) noexcept -> bool {
  if (DataType(buf[pos]) != DataType::MAP) return false;
  ++pos; size_t len; read(len);
  for (size_t i = 0; i < len; ++i) {
    K k; read(k);
    V v; read(v);
    value[k] = v;
  } return true;
}
template <class T, class... Args>
inline auto DataStream::read_args(T& value, Args&... args) noexcept -> bool {
  read(value);
  return read_args(args...);
}
inline auto DataStream::read_args() noexcept -> bool { return true; }

inline auto DataStream::operator>>(bool& value) -> DataStream& { read(value); return *this; }
inline auto DataStream::operator>>(char& value) -> DataStream& { read(value); return *this; }
inline auto DataStream::operator>>(int32_t& value) -> DataStream& { read(value); return *this; }
inline auto DataStream::operator>>(int64_t& value) -> DataStream&  { read(value); return *this; }
inline auto DataStream::operator>>(uint32_t& value) -> DataStream& { read(value); return *this; }
inline auto DataStream::operator>>(uint64_t& value) -> DataStream& { read(value); return *this; }
inline auto DataStream::operator>>(float& value) -> DataStream& { read(value); return *this; }
inline auto DataStream::operator>>(double& value) -> DataStream& { read(value); return *this; }
inline auto DataStream::operator>>(std::string& value) -> DataStream& { read(value); return *this; }
inline auto DataStream::operator>>(ISerializable& value) -> DataStream& { read(value); return *this; }
template <class T> inline auto DataStream::operator>>(std::vector<T>& value) noexcept -> DataStream& { read(value); return *this; }
template <class T> inline auto DataStream::operator>>(std::list<T>& value) noexcept -> DataStream& { read(value); return *this; }
template <class T> inline auto DataStream::operator>>(std::set<T>& value) noexcept -> DataStream& { read(value); return *this; }
template <class K, class V> inline auto DataStream::operator>>(std::map<K, V>& value) noexcept -> DataStream& { read(value); return *this; }
}  // namespace utility
}  // namespace SIByL