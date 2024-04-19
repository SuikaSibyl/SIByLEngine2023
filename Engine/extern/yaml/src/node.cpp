#include "yaml-cpp/node/node.h"
#include "nodebuilder.h"
#include "nodeevents.h"

namespace YAML {
NodeAoS Clone(const NodeAoS& node) {
  NodeEvents events(node);
  NodeBuilder builder;
  events.Emit(builder);
  return builder.Root();
}
}  // namespace YAML
