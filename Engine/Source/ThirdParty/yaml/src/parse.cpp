#include "yaml-cpp/node/parse.h"

#include <fstream>
#include <sstream>

#include "nodebuilder.h"
#include "yaml-cpp/node/impl.h"
#include "yaml-cpp/node/node.h"
#include "yaml-cpp/parser.h"

namespace YAML {
NodeAoS Load(const std::string& input) {
  std::stringstream stream(input);
  return Load(stream);
}

NodeAoS Load(const char* input) {
  std::stringstream stream(input);
  return Load(stream);
}

NodeAoS Load(std::istream& input) {
  Parser parser(input);
  NodeBuilder builder;
  if (!parser.HandleNextDocument(builder)) {
    return NodeAoS();
  }

  return builder.Root();
}

NodeAoS LoadFile(const std::string& filename) {
  std::ifstream fin(filename);
  if (!fin) {
    throw BadFile(filename);
  }
  return Load(fin);
}

std::vector<NodeAoS> LoadAll(const std::string& input) {
  std::stringstream stream(input);
  return LoadAll(stream);
}

std::vector<NodeAoS> LoadAll(const char* input) {
  std::stringstream stream(input);
  return LoadAll(stream);
}

std::vector<NodeAoS> LoadAll(std::istream& input) {
  std::vector<NodeAoS> docs;

  Parser parser(input);
  while (true) {
    NodeBuilder builder;
    if (!parser.HandleNextDocument(builder)) {
      break;
    }
    docs.push_back(builder.Root());
  }

  return docs;
}

std::vector<NodeAoS> LoadAllFromFile(const std::string& filename) {
  std::ifstream fin(filename);
  if (!fin) {
    throw BadFile(filename);
  }
  return LoadAll(fin);
}
}  // namespace YAML
