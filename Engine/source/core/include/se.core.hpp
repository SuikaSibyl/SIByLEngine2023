#ifndef SIByL_CORE_MODULE
#define SIByL_CORE_MODULE

#include <se.utils.hpp>
#include <span>
#include <vector>
#include <string>
#include <type_traits>
#include <functional>
#include <filesystem>

namespace se {
struct SIByL_API root {
  struct SIByL_API print {
	/** until to log a debug print */
	static auto debug(std::string const& s) noexcept -> void;
	/** until to log a log print */
	static auto log(std::string const& s) noexcept -> void;
	/** until to log a warning print */
	static auto warning(std::string const& s) noexcept -> void;
	/** until to log a error print */
	static auto error(std::string const& s) noexcept -> void;
	/** until to log a custom print */
	static auto correct(std::string const& s) noexcept -> void;
	/** a callback func for editor to get logged info */
	static ex::delegate<void(std::string const&, int type)> callbacks;
  };
  /** A general memory allocator, probably faster than new char[].
   * An easier way to use the allocator is by creating se::buffer. */
  struct SIByL_API memory {
	/** allocate size bytes of the memory */
	static auto allocate(size_t size) noexcept -> void*;
	/** free the allocated memory given pointer and size */
	static auto free(void* p, size_t size) noexcept -> void;
  };
  /** A general resource management. */
  struct SIByL_API resource {
	/** add a path to the resource find path */
	static auto add_path(std::filesystem::path const& path) noexcept -> void;
	/** find a full path given a partial one */
	static auto find_path(std::filesystem::path const& path) noexcept -> std::filesystem::path;
	/** query a Resource UID */
	static auto queryRUID(std::string info="") noexcept -> RUID;
  };
};

struct SIByL_API platform {
  static auto get_num_system_cores() noexcept -> int;
  static auto string_cast(const std::string& utf8str) noexcept -> std::wstring;
  static auto string_cast(const std::wstring& utf16str) noexcept -> std::string;
  static auto openFile(const char* filter, void* window_handle) noexcept -> std::string;
  static auto saveFile(const char* filter, std::string const& name, void* window_handle) noexcept -> std::string;
};

struct SIByL_API buffer {
  /** various constructors*/
  buffer(); buffer(size_t size);
  buffer(buffer const& b); buffer(buffer&& b);
  buffer(void* data, size_t size);
  /** release the buffer and dctor*/
  ~buffer(); auto release() noexcept -> void;
  /** cast the buffer as a span */
  template<class T> auto as_span() -> std::span<T> {
    return std::span<T>((T*)data, size / sizeof(T)); }
  /** the deep copy / moved copy of the buffer */
  auto operator=(buffer const& b)->buffer&;
  auto operator=(buffer&& b)->buffer&;
  /** the data pointer and size of the allocated buffer */
  void* data = nullptr; size_t size = 0;
  /** identify whether the buffer is a reference */
  bool isReference = false;
};

auto SIByL_API syncReadFile(char const* path, se::buffer& buffer) noexcept -> bool;
auto SIByL_API syncWriteFile(char const* path, se::buffer& buffer) noexcept -> bool;
auto SIByL_API syncWriteFile(char const* path, std::vector<se::buffer*> const& buffers) noexcept -> bool;

template <class... T>
struct SIByL_API signal {
  using slot = std::function<void(T...)>;
  auto connect(slot const& new_slot) noexcept -> void {
	connectedSlots.push_back(new_slot); }
  template <class... U>
  auto emit(U&&... args) noexcept -> void {
	for (auto& slot_iter : connectedSlots) slot_iter(std::forward<U>(args)...); }
 private: std::vector<slot> connectedSlots;
};

struct SIByL_API input {
  struct SIByL_API CodeEnum { int GLFWCode; int WinCode;};
  virtual auto isKeyPressed(CodeEnum const& keycode) noexcept -> bool = 0;
  virtual auto isMouseButtonPressed(CodeEnum const& button) noexcept -> bool = 0;
  virtual auto getMousePosition(int button) noexcept -> std::pair<float, float> = 0;
  virtual auto getMouseX() noexcept -> float = 0;
  virtual auto getMouseY() noexcept -> float = 0;
  virtual auto getMouseScrollX() noexcept -> float = 0;
  virtual auto getMouseScrollY() noexcept -> float = 0;
  virtual auto disableCursor() noexcept -> void = 0;
  virtual auto enableCursor() noexcept -> void = 0;
  virtual auto decodeCodeEnum(CodeEnum const& code) noexcept -> int = 0;
  // input code enum defined statically
  static CodeEnum key_unkown;
  static CodeEnum key_space;
  static CodeEnum key_apostrophe;
  static CodeEnum key_comma;
  static CodeEnum key_minus;
  static CodeEnum key_period;
  static CodeEnum key_slash;
  static CodeEnum key_0;
  static CodeEnum key_1;
  static CodeEnum key_2;
  static CodeEnum key_3;
  static CodeEnum key_4;
  static CodeEnum key_5;
  static CodeEnum key_6;
  static CodeEnum key_7;
  static CodeEnum key_8;
  static CodeEnum key_9;
  static CodeEnum key_semicolon;
  static CodeEnum key_equal;
  static CodeEnum key_a;
  static CodeEnum key_b;
  static CodeEnum key_c;
  static CodeEnum key_d;
  static CodeEnum key_e;
  static CodeEnum key_f;
  static CodeEnum key_g;
  static CodeEnum key_h;
  static CodeEnum key_i;
  static CodeEnum key_j;
  static CodeEnum key_k;
  static CodeEnum key_l;
  static CodeEnum key_m;
  static CodeEnum key_n;
  static CodeEnum key_o;
  static CodeEnum key_p;
  static CodeEnum key_q;
  static CodeEnum key_r;
  static CodeEnum key_s;
  static CodeEnum key_t;
  static CodeEnum key_u;
  static CodeEnum key_v;
  static CodeEnum key_w;
  static CodeEnum key_x;
  static CodeEnum key_y;
  static CodeEnum key_z;
  static CodeEnum key_left_bracket;
  static CodeEnum key_backslash;
  static CodeEnum key_right_bracket;
  static CodeEnum key_grave_accent;
  static CodeEnum key_world_1;
  static CodeEnum key_world_2;
  static CodeEnum key_escape;
  static CodeEnum key_enter;
  static CodeEnum key_tab;
  static CodeEnum key_backspace;
  static CodeEnum key_insert;
  static CodeEnum key_delete;
  static CodeEnum key_right;
  static CodeEnum key_left;
  static CodeEnum key_down;
  static CodeEnum key_up;
  static CodeEnum key_page_up;
  static CodeEnum key_page_down;
  static CodeEnum key_home;
  static CodeEnum key_end;
  static CodeEnum key_caps_lock;
  static CodeEnum key_scroll_lock;
  static CodeEnum key_num_lock;
  static CodeEnum key_print_screen;
  static CodeEnum key_pause;
  static CodeEnum key_f1;
  static CodeEnum key_f2;
  static CodeEnum key_f3;
  static CodeEnum key_f4;
  static CodeEnum key_f5;
  static CodeEnum key_f6;
  static CodeEnum key_f7;
  static CodeEnum key_f8;
  static CodeEnum key_f9;
  static CodeEnum key_f10;
  static CodeEnum key_f11;
  static CodeEnum key_f12;
  static CodeEnum key_f13;
  static CodeEnum key_f14;
  static CodeEnum key_f15;
  static CodeEnum key_f16;
  static CodeEnum key_f17;
  static CodeEnum key_f18;
  static CodeEnum key_f19;
  static CodeEnum key_f20;
  static CodeEnum key_f21;
  static CodeEnum key_f22;
  static CodeEnum key_f23;
  static CodeEnum key_f24;
  static CodeEnum key_f25;
  static CodeEnum key_kp_0;
  static CodeEnum key_kp_1;
  static CodeEnum key_kp_2;
  static CodeEnum key_kp_3;
  static CodeEnum key_kp_4;
  static CodeEnum key_kp_5;
  static CodeEnum key_kp_6;
  static CodeEnum key_kp_7;
  static CodeEnum key_kp_8;
  static CodeEnum key_kp_9;
  static CodeEnum key_kp_decimal;
  static CodeEnum key_kp_divide;
  static CodeEnum key_kp_multiply;
  static CodeEnum key_kp_subtract;
  static CodeEnum key_kp_add;
  static CodeEnum key_kp_enter;
  static CodeEnum key_kp_equal;
  static CodeEnum key_left_shift;
  static CodeEnum key_left_control;
  static CodeEnum key_left_alt;
  static CodeEnum key_left_super;
  static CodeEnum key_right_shift;
  static CodeEnum key_right_control;
  static CodeEnum key_right_alt;
  static CodeEnum key_right_super;
  static CodeEnum key_menu;
  static CodeEnum key_last;
  static CodeEnum mouse_button_1;
  static CodeEnum mouse_button_2;
  static CodeEnum mouse_button_3;
  static CodeEnum mouse_button_4;
  static CodeEnum mouse_button_5;
  static CodeEnum mouse_button_6;
  static CodeEnum mouse_button_7;
  static CodeEnum mouse_button_8;
  static CodeEnum mouse_button_last;
  static CodeEnum mouse_button_left;
  static CodeEnum mouse_button_right;
  static CodeEnum mouse_button_middle;
};

struct SIByL_API window {
  virtual ~window() = default;
  enum struct Vendor { GLFW, WIN_64, };
  enum struct Properties { NONE = 0 << 0, OPENGL_CONTEX = 1 << 0, VULKAN_CONTEX = 1 << 1, };
  struct SIByL_API WindowOptions {
	Vendor vendor = Vendor::GLFW;
    std::wstring title = L"SIByL Application";
    uint32_t width = 720, height = 480;
	Properties properties = Properties::NONE; };
  /** create a window with options */
  static auto create(WindowOptions const& options) noexcept -> std::unique_ptr<window>;
  // ---------------------------------
  // Life Cycle
  // ---------------------------------
  /** intialize created window */
  virtual auto init() noexcept -> bool = 0;
  /** return whether the window is still runniong or has been closed */
  virtual auto isRunning() noexcept -> bool = 0;
  /** fetch window events */
  virtual auto fetchEvents() noexcept -> int = 0;
  /** flush window contents immediately */
  virtual auto invalid() noexcept -> void = 0;
  /** should be called when frame ends */
  virtual auto endFrame() noexcept -> void = 0;
  /** destroy window */
  virtual auto destroy() noexcept -> void = 0;
  // ---------------------------------
  // Event Based Behaviors
  // ---------------------------------
  /** resizie the window */
  virtual auto resize(size_t x, size_t y) noexcept -> void = 0;
  /** bind a block of CPU bitmap data to be drawn on the window */
  virtual auto bindPaintingBitmapRGB8(size_t width, size_t height, char* data) noexcept -> void = 0;
  /** connect resize signal events */
  virtual auto connectResizeEvent(std::function<void(size_t, size_t)> const& func) noexcept -> void = 0;
  // ---------------------------------
  // Fetch Properties
  // ---------------------------------
  /** return the high DPI value */
  virtual auto getHighDPI() noexcept -> float = 0;
  /** return vendor */
  virtual auto getVendor() noexcept -> Vendor = 0;
  /** return window handle */
  virtual auto getHandle() noexcept -> void* = 0;
  /** return window framebuffer size */
  virtual auto getFramebufferSize(int* width, int* height) noexcept -> void = 0;
  /** return window input */
  virtual auto getInput() noexcept -> input* = 0;
  // ---------------------------------
  // System Functional
  // ---------------------------------
  /** open a local file using browser */
  virtual auto openFile(const char* filter) noexcept -> std::string = 0;
  /** save a local file using browser */
  virtual auto saveFile(const char* filter, std::string const& name = {}) noexcept -> std::string = 0;
};
}

#endif