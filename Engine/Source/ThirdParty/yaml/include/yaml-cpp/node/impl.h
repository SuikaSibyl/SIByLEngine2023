#ifndef NODE_IMPL_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define NODE_IMPL_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/exceptions.h"
#include "yaml-cpp/node/detail/memory.h"
#include "yaml-cpp/node/detail/node.h"
#include "yaml-cpp/node/iterator.h"
#include "yaml-cpp/node/node.h"
#include <sstream>
#include <string>

namespace YAML {
inline NodeAoS::NodeAoS()
    : m_isValid(true), m_invalidKey{}, m_pMemory(nullptr), m_pNode(nullptr) {}

inline NodeAoS::NodeAoS(NodeType::value type)
    : m_isValid(true),
      m_invalidKey{},
      m_pMemory(new detail::memory_holder),
      m_pNode(&m_pMemory->create_node()) {
  m_pNode->set_type(type);
}

template <typename T>
inline NodeAoS::NodeAoS(const T& rhs)
    : m_isValid(true),
      m_invalidKey{},
      m_pMemory(new detail::memory_holder),
      m_pNode(&m_pMemory->create_node()) {
  Assign(rhs);
}

inline NodeAoS::NodeAoS(const detail::iterator_value& rhs)
    : m_isValid(rhs.m_isValid),
      m_invalidKey(rhs.m_invalidKey),
      m_pMemory(rhs.m_pMemory),
      m_pNode(rhs.m_pNode) {}

inline NodeAoS::NodeAoS(const NodeAoS& rhs) = default;

inline NodeAoS::NodeAoS(Zombie)
    : m_isValid(false), m_invalidKey{}, m_pMemory{}, m_pNode(nullptr) {}

inline NodeAoS::NodeAoS(Zombie, const std::string& key)
    : m_isValid(false), m_invalidKey(key), m_pMemory{}, m_pNode(nullptr) {}

inline NodeAoS::NodeAoS(detail::node& node, detail::shared_memory_holder pMemory)
    : m_isValid(true), m_invalidKey{}, m_pMemory(pMemory), m_pNode(&node) {}

inline NodeAoS::~NodeAoS() = default;

inline void NodeAoS::EnsureNodeExists() const {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  if (!m_pNode) {
    m_pMemory.reset(new detail::memory_holder);
    m_pNode = &m_pMemory->create_node();
    m_pNode->set_null();
  }
}

inline bool NodeAoS::IsDefined() const {
  if (!m_isValid) {
    return false;
  }
  return m_pNode ? m_pNode->is_defined() : true;
}

inline Mark NodeAoS::Mark() const {
  if (!m_isValid) {
    throw InvalidNode(m_invalidKey);
  }
  return m_pNode ? m_pNode->mark() : Mark::null_mark();
}

inline NodeType::value NodeAoS::Type() const {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  return m_pNode ? m_pNode->type() : NodeType::Null;
}

// access

// template helpers
template <typename T, typename S>
struct as_if {
  explicit as_if(const NodeAoS& node_) : node(node_) {}
  const NodeAoS& node;

  T operator()(const S& fallback) const {
    if (!node.m_pNode)
      return fallback;

    T t;
    if (convert<T>::decode(node, t))
      return t;
    return fallback;
  }
};

template <typename S>
struct as_if<std::string, S> {
  explicit as_if(const NodeAoS& node_) : node(node_) {}
  const NodeAoS& node;

  std::string operator()(const S& fallback) const {
    if (node.Type() == NodeType::Null)
      return "null";
    if (node.Type() != NodeType::Scalar)
      return fallback;
    return node.Scalar();
  }
};

template <typename T>
struct as_if<T, void> {
  explicit as_if(const NodeAoS& node_) : node(node_) {}
  const NodeAoS& node;

  T operator()() const {
    if (!node.m_pNode)
      throw TypedBadConversion<T>(node.Mark());

    T t;
    if (convert<T>::decode(node, t))
      return t;
    throw TypedBadConversion<T>(node.Mark());
  }
};

template <>
struct as_if<std::string, void> {
  explicit as_if(const NodeAoS& node_) : node(node_) {}
  const NodeAoS& node;

  std::string operator()() const {
    if (node.Type() == NodeType::Null)
      return "null";
    if (node.Type() != NodeType::Scalar)
      throw TypedBadConversion<std::string>(node.Mark());
    return node.Scalar();
  }
};

// access functions
template <typename T>
inline T NodeAoS::as() const {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  return as_if<T, void>(*this)();
}

template <typename T, typename S>
inline T NodeAoS::as(const S& fallback) const {
  if (!m_isValid)
    return fallback;
  return as_if<T, S>(*this)(fallback);
}

inline const std::string& NodeAoS::Scalar() const {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  return m_pNode ? m_pNode->scalar() : detail::node_data::empty_scalar();
}

inline const std::string& NodeAoS::Tag() const {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  return m_pNode ? m_pNode->tag() : detail::node_data::empty_scalar();
}

inline void NodeAoS::SetTag(const std::string& tag) {
  EnsureNodeExists();
  m_pNode->set_tag(tag);
}

inline EmitterStyle::value NodeAoS::Style() const {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  return m_pNode ? m_pNode->style() : EmitterStyle::Default;
}

inline void NodeAoS::SetStyle(EmitterStyle::value style) {
  EnsureNodeExists();
  m_pNode->set_style(style);
}

// assignment
inline bool NodeAoS::is(const NodeAoS& rhs) const {
  if (!m_isValid || !rhs.m_isValid)
    throw InvalidNode(m_invalidKey);
  if (!m_pNode || !rhs.m_pNode)
    return false;
  return m_pNode->is(*rhs.m_pNode);
}

template <typename T>
inline NodeAoS& NodeAoS::operator=(const T& rhs) {
  Assign(rhs);
  return *this;
}

inline NodeAoS& NodeAoS::operator=(const NodeAoS& rhs) {
  if (is(rhs))
    return *this;
  AssignNode(rhs);
  return *this;
}

inline void NodeAoS::reset(const YAML::NodeAoS& rhs) {
  if (!m_isValid || !rhs.m_isValid)
    throw InvalidNode(m_invalidKey);
  m_pMemory = rhs.m_pMemory;
  m_pNode = rhs.m_pNode;
}

template <typename T>
inline void NodeAoS::Assign(const T& rhs) {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  AssignData(convert<T>::encode(rhs));
}

template <>
inline void NodeAoS::Assign(const std::string& rhs) {
  EnsureNodeExists();
  m_pNode->set_scalar(rhs);
}

inline void NodeAoS::Assign(const char* rhs) {
  EnsureNodeExists();
  m_pNode->set_scalar(rhs);
}

inline void NodeAoS::Assign(char* rhs) {
  EnsureNodeExists();
  m_pNode->set_scalar(rhs);
}

inline void NodeAoS::AssignData(const NodeAoS& rhs) {
  EnsureNodeExists();
  rhs.EnsureNodeExists();

  m_pNode->set_data(*rhs.m_pNode);
  m_pMemory->merge(*rhs.m_pMemory);
}

inline void NodeAoS::AssignNode(const NodeAoS& rhs) {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  rhs.EnsureNodeExists();

  if (!m_pNode) {
    m_pNode = rhs.m_pNode;
    m_pMemory = rhs.m_pMemory;
    return;
  }

  m_pNode->set_ref(*rhs.m_pNode);
  m_pMemory->merge(*rhs.m_pMemory);
  m_pNode = rhs.m_pNode;
}

// size/iterator
inline std::size_t NodeAoS::size() const {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  return m_pNode ? m_pNode->size() : 0;
}

inline const_iterator NodeAoS::begin() const {
  if (!m_isValid)
    return const_iterator();
  return m_pNode ? const_iterator(m_pNode->begin(), m_pMemory)
                 : const_iterator();
}

inline iterator NodeAoS::begin() {
  if (!m_isValid)
    return iterator();
  return m_pNode ? iterator(m_pNode->begin(), m_pMemory) : iterator();
}

inline const_iterator NodeAoS::end() const {
  if (!m_isValid)
    return const_iterator();
  return m_pNode ? const_iterator(m_pNode->end(), m_pMemory) : const_iterator();
}

inline iterator NodeAoS::end() {
  if (!m_isValid)
    return iterator();
  return m_pNode ? iterator(m_pNode->end(), m_pMemory) : iterator();
}

// sequence
template <typename T>
inline void NodeAoS::push_back(const T& rhs) {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  push_back(NodeAoS(rhs));
}

inline void NodeAoS::push_back(const NodeAoS& rhs) {
  EnsureNodeExists();
  rhs.EnsureNodeExists();

  m_pNode->push_back(*rhs.m_pNode, m_pMemory);
  m_pMemory->merge(*rhs.m_pMemory);
}

template<typename Key>
std::string key_to_string(const Key& key) {
  return streamable_to_string<Key, is_streamable<std::stringstream, Key>::value>().impl(key);
}

// indexing
template <typename Key>
inline const NodeAoS NodeAoS::operator[](const Key& key) const {
  EnsureNodeExists();
  detail::node* value =
      static_cast<const detail::node&>(*m_pNode).get(key, m_pMemory);
  if (!value) {
    return NodeAoS(ZombieNode, key_to_string(key));
  }
  return NodeAoS(*value, m_pMemory);
}

template <typename Key>
inline NodeAoS NodeAoS::operator[](const Key& key) {
  EnsureNodeExists();
  detail::node& value = m_pNode->get(key, m_pMemory);
  return NodeAoS(value, m_pMemory);
}

template <typename Key>
inline bool NodeAoS::remove(const Key& key) {
  EnsureNodeExists();
  return m_pNode->remove(key, m_pMemory);
}

inline const NodeAoS NodeAoS::operator[](const NodeAoS& key) const {
  EnsureNodeExists();
  key.EnsureNodeExists();
  m_pMemory->merge(*key.m_pMemory);
  detail::node* value =
      static_cast<const detail::node&>(*m_pNode).get(*key.m_pNode, m_pMemory);
  if (!value) {
    return NodeAoS(ZombieNode, key_to_string(key));
  }
  return NodeAoS(*value, m_pMemory);
}

inline NodeAoS NodeAoS::operator[](const NodeAoS& key) {
  EnsureNodeExists();
  key.EnsureNodeExists();
  m_pMemory->merge(*key.m_pMemory);
  detail::node& value = m_pNode->get(*key.m_pNode, m_pMemory);
  return NodeAoS(value, m_pMemory);
}

inline bool NodeAoS::remove(const NodeAoS& key) {
  EnsureNodeExists();
  key.EnsureNodeExists();
  return m_pNode->remove(*key.m_pNode, m_pMemory);
}

// map
template <typename Key, typename Value>
inline void NodeAoS::force_insert(const Key& key, const Value& value) {
  EnsureNodeExists();
  m_pNode->force_insert(key, value, m_pMemory);
}

// free functions
inline bool operator==(const NodeAoS& lhs, const NodeAoS& rhs) { return lhs.is(rhs); }
}  // namespace YAML

#endif  // NODE_IMPL_H_62B23520_7C8E_11DE_8A39_0800200C9A66
