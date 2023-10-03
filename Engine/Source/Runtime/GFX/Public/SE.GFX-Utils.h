#pragma once
#include "SE.GFX-Main.hpp"
#include <SE.RHI-Interface.hpp>

namespace SIByL::GFX {
SE_EXPORT auto CaptureImage(GFX::Texture* tex, std::string const& path) noexcept -> void;
}