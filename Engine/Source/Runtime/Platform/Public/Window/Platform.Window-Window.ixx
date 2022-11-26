module;
#include <string>
#include <functional>
#include <Windows.h>
export module Platform.Window:Window;
import Core.Memory;

namespace SIByL::Platform
{
	export enum struct WindowVendor {
		GLFW,
		WIN_64,
	};

	namespace FlagEnum {
		export enum WindowProperties {
			OPENGL_CONTEX = 1 << 0,
			VULKAN_CONTEX = 1 << 1,
		};
	}
	export using WindowProperties = FlagEnum::WindowProperties;

	export struct WindowOptions {
		WindowVendor vendor;
		std::wstring title;
		uint32_t width, height;
		WindowProperties properties = static_cast<WindowProperties>(0);
	};
	
	export struct CodeEnum {
		int GLFWCode;
		int WinCode;
	};

	export struct Input {
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
	};

	export struct Window
	{
		/** create a window with options */
		static auto create(WindowOptions const& options) noexcept -> Scope<Window>;

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
		virtual auto getVendor() noexcept -> WindowVendor = 0;
		/** return window handle */
		virtual auto getHandle() noexcept -> void* = 0;
		/** return window framebuffer size */
		virtual auto getFramebufferSize(int* width, int* height) noexcept -> void = 0;
		/** return window input */
		virtual auto getInput() noexcept -> Input* = 0;

		// ---------------------------------
		// System Functional
		// ---------------------------------
		/** open a local file using browser */
		virtual auto openFile(const char* filter) noexcept -> std::string = 0;
		/** save a local file using browser */
		virtual auto saveFile(const char* filter, std::string const& name = {}) noexcept -> std::string = 0;
	};

	/* The unknown key */
	export inline CodeEnum SIByL_KEY_UNKNOWN = (CodeEnum(-1, -1));

	/* Printable keys */
	export inline CodeEnum SIByL_KEY_SPACE = (CodeEnum(32, VK_SPACE));
	export inline CodeEnum SIByL_KEY_APOSTROPHE = (CodeEnum(39, VK_OEM_7));	/* ' */
	export inline CodeEnum SIByL_KEY_COMMA = (CodeEnum(44, VK_OEM_COMMA));	/* , */
	export inline CodeEnum SIByL_KEY_MINUS = (CodeEnum(45, VK_OEM_MINUS));	/* - */
	export inline CodeEnum SIByL_KEY_PERIOD = (CodeEnum(46, VK_OEM_2));	/* . */
	export inline CodeEnum SIByL_KEY_SLASH = (CodeEnum(47, 0x30));	/* / */
	export inline CodeEnum SIByL_KEY_0 = (CodeEnum(48, 0x30));
	export inline CodeEnum SIByL_KEY_1 = (CodeEnum(49, 0x31));
	export inline CodeEnum SIByL_KEY_2 = (CodeEnum(50, 0x32));
	export inline CodeEnum SIByL_KEY_3 = (CodeEnum(51, 0x33));
	export inline CodeEnum SIByL_KEY_4 = (CodeEnum(52, 0x34));
	export inline CodeEnum SIByL_KEY_5 = (CodeEnum(53, 0x35));
	export inline CodeEnum SIByL_KEY_6 = (CodeEnum(54, 0x36));
	export inline CodeEnum SIByL_KEY_7 = (CodeEnum(55, 0x37));
	export inline CodeEnum SIByL_KEY_8 = (CodeEnum(56, 0x38));
	export inline CodeEnum SIByL_KEY_9 = (CodeEnum(57, 0x39));
	export inline CodeEnum SIByL_KEY_SEMICOLON = (CodeEnum(59, VK_OEM_1));	 /* ; */
	export inline CodeEnum SIByL_KEY_EQUAL = (CodeEnum(61, VK_OEM_PLUS));		 /* = */
	export inline CodeEnum SIByL_KEY_A = (CodeEnum(65, 0x41));
	export inline CodeEnum SIByL_KEY_B = (CodeEnum(66, 0x42));
	export inline CodeEnum SIByL_KEY_C = (CodeEnum(67, 0x43));
	export inline CodeEnum SIByL_KEY_D = (CodeEnum(68, 0x44));
	export inline CodeEnum SIByL_KEY_E = (CodeEnum(69, 0x45));
	export inline CodeEnum SIByL_KEY_F = (CodeEnum(70, 0x46));
	export inline CodeEnum SIByL_KEY_G = (CodeEnum(71, 0x47));
	export inline CodeEnum SIByL_KEY_H = (CodeEnum(72, 0x48));
	export inline CodeEnum SIByL_KEY_I = (CodeEnum(73, 0x49));
	export inline CodeEnum SIByL_KEY_J = (CodeEnum(74, 0x4A));
	export inline CodeEnum SIByL_KEY_K = (CodeEnum(75, 0x4B));
	export inline CodeEnum SIByL_KEY_L = (CodeEnum(76, 0x4C));
	export inline CodeEnum SIByL_KEY_M = (CodeEnum(77, 0x4D));
	export inline CodeEnum SIByL_KEY_N = (CodeEnum(78, 0x4E));
	export inline CodeEnum SIByL_KEY_O = (CodeEnum(79, 0x4F));
	export inline CodeEnum SIByL_KEY_P = (CodeEnum(80, 0x50));
	export inline CodeEnum SIByL_KEY_Q = (CodeEnum(81, 0x51));
	export inline CodeEnum SIByL_KEY_R = (CodeEnum(82, 0x52));
	export inline CodeEnum SIByL_KEY_S = (CodeEnum(83, 0x53));
	export inline CodeEnum SIByL_KEY_T = (CodeEnum(84, 0x54));
	export inline CodeEnum SIByL_KEY_U = (CodeEnum(85, 0x55));
	export inline CodeEnum SIByL_KEY_V = (CodeEnum(86, 0x56));
	export inline CodeEnum SIByL_KEY_W = (CodeEnum(87, 0x57));
	export inline CodeEnum SIByL_KEY_X = (CodeEnum(88, 0x58));
	export inline CodeEnum SIByL_KEY_Y = (CodeEnum(89, 0x59));
	export inline CodeEnum SIByL_KEY_Z = (CodeEnum(90, 0x5A));
	export inline CodeEnum SIByL_KEY_LEFT_BRACKET = (CodeEnum(91, VK_OEM_4));	 /* [ */
	export inline CodeEnum SIByL_KEY_BACKSLASH = (CodeEnum(92, VK_OEM_5));	 /* \ */
	export inline CodeEnum SIByL_KEY_RIGHT_BRACKET = (CodeEnum(93, VK_OEM_6));	 /* ] */
	export inline CodeEnum SIByL_KEY_GRAVE_ACCENT = (CodeEnum(96, VK_OEM_3));	 /* ` */
	export inline CodeEnum SIByL_KEY_WORLD_1 = (CodeEnum(161, VK_SPACE));	 /* non-US #1 */
	export inline CodeEnum SIByL_KEY_WORLD_2 = (CodeEnum(162, VK_SPACE));	 /* non-US #2 */

	/* Function keys */
	export inline CodeEnum SIByL_KEY_ESCAPE = (CodeEnum(256, VK_SPACE));
	export inline CodeEnum SIByL_KEY_ENTER = (CodeEnum(257, VK_RETURN));
	export inline CodeEnum SIByL_KEY_TAB = (CodeEnum(258, VK_TAB));
	export inline CodeEnum SIByL_KEY_BACKSPACE = (CodeEnum(259, VK_BACK));
	export inline CodeEnum SIByL_KEY_INSERT = (CodeEnum(260, VK_INSERT));
	export inline CodeEnum SIByL_KEY_DELETE = (CodeEnum(261, VK_DELETE));
	export inline CodeEnum SIByL_KEY_RIGHT = (CodeEnum(262, VK_RIGHT));
	export inline CodeEnum SIByL_KEY_LEFT = (CodeEnum(263, VK_LEFT));
	export inline CodeEnum SIByL_KEY_DOWN = (CodeEnum(264, VK_DOWN));
	export inline CodeEnum SIByL_KEY_UP = (CodeEnum(265, VK_UP));
	export inline CodeEnum SIByL_KEY_PAGE_UP = (CodeEnum(266, VK_PRIOR));
	export inline CodeEnum SIByL_KEY_PAGE_DOWN = (CodeEnum(267, VK_NEXT));
	export inline CodeEnum SIByL_KEY_HOME = (CodeEnum(268, VK_HOME));
	export inline CodeEnum SIByL_KEY_END = (CodeEnum(269, VK_END));
	export inline CodeEnum SIByL_KEY_CAPS_LOCK = (CodeEnum(280, VK_CAPITAL));
	export inline CodeEnum SIByL_KEY_SCROLL_LOCK = (CodeEnum(281, VK_SCROLL));
	export inline CodeEnum SIByL_KEY_NUM_LOCK = (CodeEnum(282, VK_NUMLOCK));
	export inline CodeEnum SIByL_KEY_PRINT_SCREEN = (CodeEnum(283, VK_SNAPSHOT));
	export inline CodeEnum SIByL_KEY_PAUSE = (CodeEnum(284, VK_PAUSE));
	export inline CodeEnum SIByL_KEY_F1 = (CodeEnum(290, VK_F1));
	export inline CodeEnum SIByL_KEY_F2 = (CodeEnum(291, VK_F2));
	export inline CodeEnum SIByL_KEY_F3 = (CodeEnum(292, VK_F3));
	export inline CodeEnum SIByL_KEY_F4 = (CodeEnum(293, VK_F4));
	export inline CodeEnum SIByL_KEY_F5 = (CodeEnum(294, VK_F5));
	export inline CodeEnum SIByL_KEY_F6 = (CodeEnum(295, VK_F6));
	export inline CodeEnum SIByL_KEY_F7 = (CodeEnum(296, VK_F7));
	export inline CodeEnum SIByL_KEY_F8 = (CodeEnum(297, VK_F8));
	export inline CodeEnum SIByL_KEY_F9 = (CodeEnum(298, VK_F9));
	export inline CodeEnum SIByL_KEY_F10 = (CodeEnum(299, VK_F10));
	export inline CodeEnum SIByL_KEY_F11 = (CodeEnum(300, VK_F11));
	export inline CodeEnum SIByL_KEY_F12 = (CodeEnum(301, VK_F12));
	export inline CodeEnum SIByL_KEY_F13 = (CodeEnum(302, VK_F13));
	export inline CodeEnum SIByL_KEY_F14 = (CodeEnum(303, VK_F14));
	export inline CodeEnum SIByL_KEY_F15 = (CodeEnum(304, VK_F15));
	export inline CodeEnum SIByL_KEY_F16 = (CodeEnum(305, VK_F16));
	export inline CodeEnum SIByL_KEY_F17 = (CodeEnum(306, VK_F17));
	export inline CodeEnum SIByL_KEY_F18 = (CodeEnum(307, VK_F18));
	export inline CodeEnum SIByL_KEY_F19 = (CodeEnum(308, VK_F19));
	export inline CodeEnum SIByL_KEY_F20 = (CodeEnum(309, VK_F20));
	export inline CodeEnum SIByL_KEY_F21 = (CodeEnum(310, VK_F21));
	export inline CodeEnum SIByL_KEY_F22 = (CodeEnum(311, VK_F22));
	export inline CodeEnum SIByL_KEY_F23 = (CodeEnum(312, VK_F23));
	export inline CodeEnum SIByL_KEY_F24 = (CodeEnum(313, VK_F24));
	export inline CodeEnum SIByL_KEY_F25 = (CodeEnum(314, 0x88));
	export inline CodeEnum SIByL_KEY_KP_0 = (CodeEnum(320, VK_NUMPAD0));
	export inline CodeEnum SIByL_KEY_KP_1 = (CodeEnum(321, VK_NUMPAD1));
	export inline CodeEnum SIByL_KEY_KP_2 = (CodeEnum(322, VK_NUMPAD2));
	export inline CodeEnum SIByL_KEY_KP_3 = (CodeEnum(323, VK_NUMPAD3));
	export inline CodeEnum SIByL_KEY_KP_4 = (CodeEnum(324, VK_NUMPAD4));
	export inline CodeEnum SIByL_KEY_KP_5 = (CodeEnum(325, VK_NUMPAD5));
	export inline CodeEnum SIByL_KEY_KP_6 = (CodeEnum(326, VK_NUMPAD6));
	export inline CodeEnum SIByL_KEY_KP_7 = (CodeEnum(327, VK_NUMPAD7));
	export inline CodeEnum SIByL_KEY_KP_8 = (CodeEnum(328, VK_NUMPAD8));
	export inline CodeEnum SIByL_KEY_KP_9 = (CodeEnum(329, VK_NUMPAD9));
	export inline CodeEnum SIByL_KEY_KP_DECIMAL = (CodeEnum(330, VK_DECIMAL));
	export inline CodeEnum SIByL_KEY_KP_DIVIDE = (CodeEnum(331, VK_DIVIDE));
	export inline CodeEnum SIByL_KEY_KP_MULTIPLY = (CodeEnum(332, VK_MULTIPLY));
	export inline CodeEnum SIByL_KEY_KP_SUBTRACT = (CodeEnum(333, VK_SUBTRACT));
	export inline CodeEnum SIByL_KEY_KP_ADD = (CodeEnum(334, VK_ADD));
	export inline CodeEnum SIByL_KEY_KP_ENTER = (CodeEnum(335, VK_SPACE));
	export inline CodeEnum SIByL_KEY_KP_EQUAL = (CodeEnum(336, VK_SPACE));
	export inline CodeEnum SIByL_KEY_LEFT_SHIFT = (CodeEnum(340, VK_LSHIFT));
	export inline CodeEnum SIByL_KEY_LEFT_CONTROL = (CodeEnum(341, VK_LCONTROL));
	export inline CodeEnum SIByL_KEY_LEFT_ALT = (CodeEnum(342, VK_LMENU));
	export inline CodeEnum SIByL_KEY_LEFT_SUPER = (CodeEnum(343, VK_SPACE));
	export inline CodeEnum SIByL_KEY_RIGHT_SHIFT = (CodeEnum(344, VK_RSHIFT));
	export inline CodeEnum SIByL_KEY_RIGHT_CONTROL = (CodeEnum(345, VK_RCONTROL));
	export inline CodeEnum SIByL_KEY_RIGHT_ALT = (CodeEnum(346, VK_RMENU));
	export inline CodeEnum SIByL_KEY_RIGHT_SUPER = (CodeEnum(347, VK_SPACE));
	export inline CodeEnum SIByL_KEY_MENU = (CodeEnum(348, VK_MENU));

	export inline CodeEnum SIByL_KEY_LAST = SIByL_KEY_MENU;

	export inline CodeEnum SIByL_MOUSE_BUTTON_1 = (CodeEnum(0, VK_LBUTTON));
	export inline CodeEnum SIByL_MOUSE_BUTTON_2 = (CodeEnum(1, VK_RBUTTON));
	export inline CodeEnum SIByL_MOUSE_BUTTON_3 = (CodeEnum(2, VK_MBUTTON));
	export inline CodeEnum SIByL_MOUSE_BUTTON_4 = (CodeEnum(3, VK_XBUTTON1));
	export inline CodeEnum SIByL_MOUSE_BUTTON_5 = (CodeEnum(4, VK_XBUTTON2));
	export inline CodeEnum SIByL_MOUSE_BUTTON_6 = (CodeEnum(5, 0x07));
	export inline CodeEnum SIByL_MOUSE_BUTTON_7 = (CodeEnum(6, 0x07));
	export inline CodeEnum SIByL_MOUSE_BUTTON_8 = (CodeEnum(7, 0x07));
	export inline CodeEnum SIByL_MOUSE_BUTTON_LAST = SIByL_MOUSE_BUTTON_8;
	export inline CodeEnum SIByL_MOUSE_BUTTON_LEFT = SIByL_MOUSE_BUTTON_1;
	export inline CodeEnum SIByL_MOUSE_BUTTON_RIGHT = SIByL_MOUSE_BUTTON_2;
	export inline CodeEnum SIByL_MOUSE_BUTTON_MIDDLE = SIByL_MOUSE_BUTTON_3;
}