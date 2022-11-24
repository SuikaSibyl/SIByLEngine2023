#ifndef VALUE_PARSE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define VALUE_PARSE_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <iosfwd>
#include <string>
#include <vector>

#include "yaml-cpp/dll.h"

namespace YAML {
class NodeAoS;

/**
 * Loads the input string as a single YAML document.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API NodeAoS Load(const std::string& input);

/**
 * Loads the input string as a single YAML document.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API NodeAoS Load(const char* input);

/**
 * Loads the input stream as a single YAML document.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API NodeAoS Load(std::istream& input);

/**
 * Loads the input file as a single YAML document.
 *
 * @throws {@link ParserException} if it is malformed.
 * @throws {@link BadFile} if the file cannot be loaded.
 */
YAML_CPP_API NodeAoS LoadFile(const std::string& filename);

/**
 * Loads the input string as a list of YAML documents.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API std::vector<NodeAoS> LoadAll(const std::string& input);

/**
 * Loads the input string as a list of YAML documents.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API std::vector<NodeAoS> LoadAll(const char* input);

/**
 * Loads the input stream as a list of YAML documents.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API std::vector<NodeAoS> LoadAll(std::istream& input);

/**
 * Loads the input file as a list of YAML documents.
 *
 * @throws {@link ParserException} if it is malformed.
 * @throws {@link BadFile} if the file cannot be loaded.
 */
YAML_CPP_API std::vector<NodeAoS> LoadAllFromFile(const std::string& filename);
}  // namespace YAML

#endif  // VALUE_PARSE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
