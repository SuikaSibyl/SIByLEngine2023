#define DLIB_EXPORT
#include <se.core.hpp>
#undef DLIB_EXPORT
#ifdef _WIN32
#include <Windows.h>

namespace se {
auto platform::get_num_system_cores() noexcept -> int {
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  return sysInfo.dwNumberOfProcessors;
}

auto platform::string_cast(const std::string& utf8str) noexcept -> std::wstring {
  if (utf8str.empty()) return L"";
  int char_count = MultiByteToWideChar(CP_UTF8, 0, &utf8str[0], (int)utf8str.size(), NULL, 0);
  std::wstring conv(char_count, 0);
  MultiByteToWideChar(CP_UTF8, 0, &utf8str[0], (int)utf8str.size(), &conv[0], char_count);
  return conv;
}

auto platform::string_cast(const std::wstring& utf16str) noexcept -> std::string {
  if (utf16str.empty()) return "";
  int char_count = WideCharToMultiByte(CP_UTF8, 0, &utf16str[0], (int)utf16str.size(), NULL, 0, NULL, NULL);
  std::string conv(char_count, 0);
  WideCharToMultiByte(CP_UTF8, 0, &utf16str[0], (int)utf16str.size(), &conv[0], char_count, NULL, NULL);
  return conv;
}

auto platform::openFile(const char* filter, void* window_handle) noexcept -> std::string {
  OPENFILENAMEA ofn;
  CHAR szFile[260] = { 0 };
  ZeroMemory(&ofn, sizeof(OPENFILENAME));
  ofn.lStructSize = sizeof(OPENFILENAME);
  ofn.hwndOwner = *((HWND*)window_handle);
  ofn.lpstrFile = szFile;
  ofn.nMaxFile = sizeof(szFile);
  ofn.lpstrFilter = filter;
  ofn.nFilterIndex = 1;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
  if (GetOpenFileNameA(&ofn) == TRUE) {
  	return ofn.lpstrFile;
  }
  return std::string();
}

auto platform::saveFile(const char* filter, std::string const& name, void* window_handle) noexcept -> std::string {
  OPENFILENAMEA ofn;
  CHAR szFile[260] = { 0 };
  memcpy(szFile, name.c_str(), name.size() + 1);
  ZeroMemory(&ofn, sizeof(OPENFILENAME));
  ofn.lStructSize = sizeof(OPENFILENAME);
  ofn.hwndOwner = *((HWND*)window_handle);
  ofn.lpstrFile = szFile;
  ofn.nMaxFile = sizeof(szFile);
  ofn.lpstrFilter = filter;
  ofn.nFilterIndex = 1;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
  if (GetSaveFileNameA(&ofn) == TRUE) {
  	return ofn.lpstrFile;
  }
  return std::string();
}

// Unknown Keys Code Enums
input::CodeEnum input::key_unkown = input::CodeEnum(-1, -1);
// Printable Keys Code Enums
input::CodeEnum input::key_space = (input::CodeEnum(32, VK_SPACE));
input::CodeEnum input::key_apostrophe = (input::CodeEnum(39, VK_OEM_7));	/* ' */
input::CodeEnum input::key_comma = (input::CodeEnum(44, VK_OEM_COMMA));	/* , */
input::CodeEnum input::key_minus = (input::CodeEnum(45, VK_OEM_MINUS));	/* - */
input::CodeEnum input::key_period = (input::CodeEnum(46, VK_OEM_2));	/* . */
input::CodeEnum input::key_slash = (input::CodeEnum(47, 0x30));	/* / */
input::CodeEnum input::key_0 = (input::CodeEnum(48, 0x30));
input::CodeEnum input::key_1 = (input::CodeEnum(49, 0x31));
input::CodeEnum input::key_2 = (input::CodeEnum(50, 0x32));
input::CodeEnum input::key_3 = (input::CodeEnum(51, 0x33));
input::CodeEnum input::key_4 = (input::CodeEnum(52, 0x34));
input::CodeEnum input::key_5 = (input::CodeEnum(53, 0x35));
input::CodeEnum input::key_6 = (input::CodeEnum(54, 0x36));
input::CodeEnum input::key_7 = (input::CodeEnum(55, 0x37));
input::CodeEnum input::key_8 = (input::CodeEnum(56, 0x38));
input::CodeEnum input::key_9 = (input::CodeEnum(57, 0x39));
input::CodeEnum input::key_semicolon = (input::CodeEnum(59, VK_OEM_1));	 /* ; */
input::CodeEnum input::key_equal = (input::CodeEnum(61, VK_OEM_PLUS));		 /* = */
input::CodeEnum input::key_a = (input::CodeEnum(65, 0x41));
input::CodeEnum input::key_b = (input::CodeEnum(66, 0x42));
input::CodeEnum input::key_c = (input::CodeEnum(67, 0x43));
input::CodeEnum input::key_d = (input::CodeEnum(68, 0x44));
input::CodeEnum input::key_e = (input::CodeEnum(69, 0x45));
input::CodeEnum input::key_f = (input::CodeEnum(70, 0x46));
input::CodeEnum input::key_g = (input::CodeEnum(71, 0x47));
input::CodeEnum input::key_h = (input::CodeEnum(72, 0x48));
input::CodeEnum input::key_i = (input::CodeEnum(73, 0x49));
input::CodeEnum input::key_j = (input::CodeEnum(74, 0x4A));
input::CodeEnum input::key_k = (input::CodeEnum(75, 0x4B));
input::CodeEnum input::key_l = (input::CodeEnum(76, 0x4C));
input::CodeEnum input::key_m = (input::CodeEnum(77, 0x4D));
input::CodeEnum input::key_n = (input::CodeEnum(78, 0x4E));
input::CodeEnum input::key_o = (input::CodeEnum(79, 0x4F));
input::CodeEnum input::key_p = (input::CodeEnum(80, 0x50));
input::CodeEnum input::key_q = (input::CodeEnum(81, 0x51));
input::CodeEnum input::key_r = (input::CodeEnum(82, 0x52));
input::CodeEnum input::key_s = (input::CodeEnum(83, 0x53));
input::CodeEnum input::key_t = (input::CodeEnum(84, 0x54));
input::CodeEnum input::key_u = (input::CodeEnum(85, 0x55));
input::CodeEnum input::key_v = (input::CodeEnum(86, 0x56));
input::CodeEnum input::key_w = (input::CodeEnum(87, 0x57));
input::CodeEnum input::key_x = (input::CodeEnum(88, 0x58));
input::CodeEnum input::key_y = (input::CodeEnum(89, 0x59));
input::CodeEnum input::key_z = (input::CodeEnum(90, 0x5A));
input::CodeEnum input::key_left_bracket = (input::CodeEnum(91, VK_OEM_4));	/* [ */
input::CodeEnum input::key_backslash = (input::CodeEnum(92, VK_OEM_5));		/* \ */
input::CodeEnum input::key_right_bracket = (input::CodeEnum(93, VK_OEM_6));	/* ] */
input::CodeEnum input::key_grave_accent = (input::CodeEnum(96, VK_OEM_3));	/* ` */
input::CodeEnum input::key_world_1 = (input::CodeEnum(161, VK_SPACE));		/* non-US #1 */
input::CodeEnum input::key_world_2 = (input::CodeEnum(162, VK_SPACE));		/* non-US #2 */
// Function Keys Code Enums
input::CodeEnum input::key_escape = (input::CodeEnum(256, VK_SPACE));
input::CodeEnum input::key_enter = (input::CodeEnum(257, VK_RETURN));
input::CodeEnum input::key_tab = (input::CodeEnum(258, VK_TAB));
input::CodeEnum input::key_backspace = (input::CodeEnum(259, VK_BACK));
input::CodeEnum input::key_insert = (input::CodeEnum(260, VK_INSERT));
input::CodeEnum input::key_delete = (input::CodeEnum(261, VK_DELETE));
input::CodeEnum input::key_right = (input::CodeEnum(262, VK_RIGHT));
input::CodeEnum input::key_left = (input::CodeEnum(263, VK_LEFT));
input::CodeEnum input::key_down = (input::CodeEnum(264, VK_DOWN));
input::CodeEnum input::key_up = (input::CodeEnum(265, VK_UP));
input::CodeEnum input::key_page_up = (input::CodeEnum(266, VK_PRIOR));
input::CodeEnum input::key_page_down = (input::CodeEnum(267, VK_NEXT));
input::CodeEnum input::key_home = (input::CodeEnum(268, VK_HOME));
input::CodeEnum input::key_end = (input::CodeEnum(269, VK_END));
input::CodeEnum input::key_caps_lock = (input::CodeEnum(280, VK_CAPITAL));
input::CodeEnum input::key_scroll_lock = (input::CodeEnum(281, VK_SCROLL));
input::CodeEnum input::key_num_lock = (input::CodeEnum(282, VK_NUMLOCK));
input::CodeEnum input::key_print_screen = (input::CodeEnum(283, VK_SNAPSHOT));
input::CodeEnum input::key_pause = (input::CodeEnum(284, VK_PAUSE));
input::CodeEnum input::key_f1 = (input::CodeEnum(290, VK_F1));
input::CodeEnum input::key_f2 = (input::CodeEnum(291, VK_F2));
input::CodeEnum input::key_f3 = (input::CodeEnum(292, VK_F3));
input::CodeEnum input::key_f4 = (input::CodeEnum(293, VK_F4));
input::CodeEnum input::key_f5 = (input::CodeEnum(294, VK_F5));
input::CodeEnum input::key_f6 = (input::CodeEnum(295, VK_F6));
input::CodeEnum input::key_f7 = (input::CodeEnum(296, VK_F7));
input::CodeEnum input::key_f8 = (input::CodeEnum(297, VK_F8));
input::CodeEnum input::key_f9 = (input::CodeEnum(298, VK_F9));
input::CodeEnum input::key_f10 = (input::CodeEnum(299, VK_F10));
input::CodeEnum input::key_f11 = (input::CodeEnum(300, VK_F11));
input::CodeEnum input::key_f12 = (input::CodeEnum(301, VK_F12));
input::CodeEnum input::key_f13 = (input::CodeEnum(302, VK_F13));
input::CodeEnum input::key_f14 = (input::CodeEnum(303, VK_F14));
input::CodeEnum input::key_f15 = (input::CodeEnum(304, VK_F15));
input::CodeEnum input::key_f16 = (input::CodeEnum(305, VK_F16));
input::CodeEnum input::key_f17 = (input::CodeEnum(306, VK_F17));
input::CodeEnum input::key_f18 = (input::CodeEnum(307, VK_F18));
input::CodeEnum input::key_f19 = (input::CodeEnum(308, VK_F19));
input::CodeEnum input::key_f20 = (input::CodeEnum(309, VK_F20));
input::CodeEnum input::key_f21 = (input::CodeEnum(310, VK_F21));
input::CodeEnum input::key_f22 = (input::CodeEnum(311, VK_F22));
input::CodeEnum input::key_f23 = (input::CodeEnum(312, VK_F23));
input::CodeEnum input::key_f24 = (input::CodeEnum(313, VK_F24));
input::CodeEnum input::key_f25 = (input::CodeEnum(314, 0x88));
input::CodeEnum input::key_kp_0 = (input::CodeEnum(320, VK_NUMPAD0));
input::CodeEnum input::key_kp_1 = (input::CodeEnum(321, VK_NUMPAD1));
input::CodeEnum input::key_kp_2 = (input::CodeEnum(322, VK_NUMPAD2));
input::CodeEnum input::key_kp_3 = (input::CodeEnum(323, VK_NUMPAD3));
input::CodeEnum input::key_kp_4 = (input::CodeEnum(324, VK_NUMPAD4));
input::CodeEnum input::key_kp_5 = (input::CodeEnum(325, VK_NUMPAD5));
input::CodeEnum input::key_kp_6 = (input::CodeEnum(326, VK_NUMPAD6));
input::CodeEnum input::key_kp_7 = (input::CodeEnum(327, VK_NUMPAD7));
input::CodeEnum input::key_kp_8 = (input::CodeEnum(328, VK_NUMPAD8));
input::CodeEnum input::key_kp_9 = (input::CodeEnum(329, VK_NUMPAD9));
input::CodeEnum input::key_kp_decimal = (input::CodeEnum(330, VK_DECIMAL));
input::CodeEnum input::key_kp_divide = (input::CodeEnum(331, VK_DIVIDE));
input::CodeEnum input::key_kp_multiply = (input::CodeEnum(332, VK_MULTIPLY));
input::CodeEnum input::key_kp_subtract = (input::CodeEnum(333, VK_SUBTRACT));
input::CodeEnum input::key_kp_add = (input::CodeEnum(334, VK_ADD));
input::CodeEnum input::key_kp_enter = (input::CodeEnum(335, VK_SPACE));
input::CodeEnum input::key_kp_equal = (input::CodeEnum(336, VK_SPACE));
input::CodeEnum input::key_left_shift = (input::CodeEnum(340, VK_LSHIFT));
input::CodeEnum input::key_left_control = (input::CodeEnum(341, VK_LCONTROL));
input::CodeEnum input::key_left_alt = (input::CodeEnum(342, VK_LMENU));
input::CodeEnum input::key_left_super = (input::CodeEnum(343, VK_SPACE));
input::CodeEnum input::key_right_shift = (input::CodeEnum(344, VK_RSHIFT));
input::CodeEnum input::key_right_control = (input::CodeEnum(345, VK_RCONTROL));
input::CodeEnum input::key_right_alt = (input::CodeEnum(346, VK_RMENU));
input::CodeEnum input::key_right_super = (input::CodeEnum(347, VK_SPACE));
input::CodeEnum input::key_menu = (input::CodeEnum(348, VK_MENU));
input::CodeEnum input::key_last			= input::key_menu;
// Mouse Code Enums
input::CodeEnum input::mouse_button_1 = (input::CodeEnum(0, VK_LBUTTON));
input::CodeEnum input::mouse_button_2 = (input::CodeEnum(1, VK_RBUTTON));
input::CodeEnum input::mouse_button_3 = (input::CodeEnum(2, VK_MBUTTON));
input::CodeEnum input::mouse_button_4 = (input::CodeEnum(3, VK_XBUTTON1));
input::CodeEnum input::mouse_button_5 = (input::CodeEnum(4, VK_XBUTTON2));
input::CodeEnum input::mouse_button_6 = (input::CodeEnum(5, 0x07));
input::CodeEnum input::mouse_button_7 = (input::CodeEnum(6, 0x07));
input::CodeEnum input::mouse_button_8 = (input::CodeEnum(7, 0x07));
input::CodeEnum input::mouse_button_last = input::mouse_button_8;
input::CodeEnum input::mouse_button_left = input::mouse_button_1;
input::CodeEnum input::mouse_button_right = input::mouse_button_2;
input::CodeEnum input::mouse_button_middle = input::mouse_button_3;
}

#endif