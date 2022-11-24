#ifndef VALUE_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define VALUE_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/dll.h"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/node/detail/iterator_fwd.h"
#include "yaml-cpp/node/detail/iterator.h"
#include <list>
#include <utility>
#include <vector>

namespace YAML {
namespace detail {
struct iterator_value : public NodeAoS, std::pair<NodeAoS, NodeAoS> {
  iterator_value() = default;
  explicit iterator_value(const NodeAoS& rhs)
      : NodeAoS(rhs),
        std::pair<NodeAoS, NodeAoS>(NodeAoS(NodeAoS::ZombieNode), NodeAoS(NodeAoS::ZombieNode)) {}
  explicit iterator_value(const NodeAoS& key, const NodeAoS& value)
      : NodeAoS(NodeAoS::ZombieNode), std::pair<NodeAoS, NodeAoS>(key, value) {}
};
}
}

#endif  // VALUE_ITERATOR_H_62B23520_7C8E_11DE_8A39_0800200C9A66
