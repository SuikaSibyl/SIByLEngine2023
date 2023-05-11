#include <SE.GFX-GFXConfig.hpp>

namespace SIByL::GFX {
/** static member of GFXConfig,
 * it is managed by GFXManager,
 * do not manually create it */
GFXConfig* GFXConfig::globalConfig = nullptr;
}  // namespace SIByL::GFX