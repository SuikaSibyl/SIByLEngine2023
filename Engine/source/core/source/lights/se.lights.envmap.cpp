#define DLIB_EXPORT
#include <lights/se.lights.envmap.hpp>
#undef DLIB_EXPORT
#include <se.image.hpp>
#include <se.math.hpp>

namespace se {
  EnvmapLight::EnvmapLight(std::string const& path, ImportanceType type) {
	texture = gfx::GFXContext::create_texture_file(path);
	auto host_image = image::load_image(path);
	std::span<float> image = host_image->buffer.as_span<float>();
	int width = host_image->extend.width;
	int height = host_image->extend.height;
	std::vector<float> importance(width * height);
	float r_sum = 0.f;
	float g_sum = 0.f;
	float b_sum = 0.f;
	for (int j = 0; j < height; j++) {
	  for (int i = 0; i < width; i++) {
		  const int index = i + j * width;
		  const float r = image[index * 4 + 0];
		  const float g = image[index * 4 + 1];
		  const float b = image[index * 4 + 2];
		  const float luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
		  importance[index] = luminance;

		  r_sum += r / width;
		  g_sum += g / width;
		  b_sum += b / width;
	  }
	}

	r_sum /= height;
	g_sum /= height;
	b_sum /= height;
	size = { width, height };
	rgb_int = { r_sum, g_sum, b_sum };
	distribution = gfx::PMFConstructor::build_piecewise_constant_2d(importance, width, height, vec2{ 0 }, vec2{ 1 });
  }

  auto EnvmapLight::width() noexcept -> int { return size.x; }
  auto EnvmapLight::height() noexcept -> int { return size.y; }
  auto EnvmapLight::rgb_integrated() noexcept -> vec3 { return rgb_int; }
  auto EnvmapLight::get_texture() noexcept -> gfx::TextureHandle { return texture; }
  auto EnvmapLight::condition_offset() noexcept -> int { return distribution.condition_offset; }
  auto EnvmapLight::marginal_offset() noexcept -> int { return distribution.marginal_offset; }
}