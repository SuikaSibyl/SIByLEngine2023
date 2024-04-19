#ifndef NODE_NODE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define NODE_NODE_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <stdexcept>
#include <string>

#include "yaml-cpp/dll.h"
#include "yaml-cpp/emitterstyle.h"
#include "yaml-cpp/mark.h"
#include "yaml-cpp/node/detail/iterator_fwd.h"
#include "yaml-cpp/node/ptr.h"
#include "yaml-cpp/node/type.h"

namespace YAML {
namespace detail {
class node;
class node_data;
struct iterator_value;
}  // namespace detail
}  // namespace YAML

namespace YAML {
class YAML_CPP_API NodeAoS {
 public:
  friend class NodeBuilder;
  friend class NodeEvents;
  friend struct detail::iterator_value;
  friend class detail::node;
  friend class detail::node_data;
  template <typename>
  friend class detail::iterator_base;
  template <typename T, typename S>
  friend struct as_if;

  using iterator = YAML::iterator;
  using const_iterator = YAML::const_iterator;

  NodeAoS();
  explicit NodeAoS(NodeType::value type);
  template <typename T>
  explicit NodeAoS(const T& rhs);
  explicit NodeAoS(const detail::iterator_value& rhs);
  NodeAoS(const NodeAoS& rhs);
  ~NodeAoS();

  YAML::Mark Mark() const;
  NodeType::value Type() const;
  bool IsDefined() const;
  bool IsNull() const { return Type() == NodeType::Null; }
  bool IsScalar() const { return Type() == NodeType::Scalar; }
  bool IsSequence() const { return Type() == NodeType::Sequence; }
  bool IsMap() const { return Type() == NodeType::Map; }

  // bool conversions
  explicit operator bool() const { return IsDefined(); }
  bool operator!() const { return !IsDefined(); }

  // access
  template <typename T>
  T as() const;
  template <typename T, typename S>
  T as(const S& fallback) const;
  const std::string& Scalar() const;

  const std::string& Tag() const;
  void SetTag(const std::string& tag);

  // style
  // WARNING: This API might change in future releases.
  EmitterStyle::value Style() const;
  void SetStyle(EmitterStyle::value style);

  // assignment
  bool is(const NodeAoS& rhs) const;
  template <typename T>
  NodeAoS& operator=(const T& rhs);
  NodeAoS& operator=(const NodeAoS& rhs);
  void reset(const NodeAoS& rhs = NodeAoS());

  // size/iterator
  std::size_t size() const;

  const_iterator begin() const;
  iterator begin();

  const_iterator end() const;
  iterator end();

  // sequence
  template <typename T>
  void push_back(const T& rhs);
  void push_back(const NodeAoS& rhs);

  // indexing
  template <typename Key>
  const NodeAoS operator[](const Key& key) const;
  template <typename Key>
  NodeAoS operator[](const Key& key);
  template <typename Key>
  bool remove(const Key& key);

  const NodeAoS operator[](const NodeAoS& key) const;
  NodeAoS operator[](const NodeAoS& key);
  bool remove(const NodeAoS& key);

  // map
  template <typename Key, typename Value>
  void force_insert(const Key& key, const Value& value);

 private:
  enum Zombie { ZombieNode };
  explicit NodeAoS(Zombie);
  explicit NodeAoS(Zombie, const std::string&);
  explicit NodeAoS(detail::node& node, detail::shared_memory_holder pMemory);

  void EnsureNodeExists() const;

  template <typename T>
  void Assign(const T& rhs);
  void Assign(const char* rhs);
  void Assign(char* rhs);

  void AssignData(const NodeAoS& rhs);
  void AssignNode(const NodeAoS& rhs);

 private:
  bool m_isValid;
  // String representation of invalid key, if the node is invalid.
  std::string m_invalidKey;
  mutable detail::shared_memory_holder m_pMemory;
  mutable detail::node* m_pNode;
};

YAML_CPP_API bool operator==(const NodeAoS& lhs, const NodeAoS& rhs);

YAML_CPP_API NodeAoS Clone(const NodeAoS& node);

template <typename T>
struct convert;
}

#endif  // NODE_NODE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
