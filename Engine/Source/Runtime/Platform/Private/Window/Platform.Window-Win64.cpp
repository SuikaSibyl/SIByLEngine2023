module;
#include <string>
#include <functional>
#include <windows.h>
#include <WinUser.h>
module Platform.Window:WindowWin64;
import Platform.Window;
import SE.Utility;
import Core.Log;

namespace SIByL::Platform
{
	/**
	* A static general function for window message, which dispatch the message
	* to the specific Window_Win64 struct
	* @see Window_Win64
	*/
	LRESULT CALLBACK StaticWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
	{    
		// Store instance pointer while handling the first message
		if (msg == WM_NCCREATE)
		{
			CREATESTRUCT* pCS = reinterpret_cast<CREATESTRUCT*>(lParam);
			LPVOID pThis = pCS->lpCreateParams;
			SetWindowLongPtrW(hwnd, 0, reinterpret_cast<LONG_PTR>(pThis));
		}

		// At this point the instance pointer will always be available
		Window_Win64* pWnd = reinterpret_cast<Window_Win64*>(GetWindowLongPtrW(hwnd, 0));
		// see Note 1a below
		return pWnd->wndProc(hwnd, msg, wParam, lParam);
	}

	/**
	* Bind OpenGL context for Win32 window.
	* Ref: https://github.com/gszauer/GamePhysicsCookbook/blob/master/Code/main-win32.cpp
	*/
	auto bindOpenGLContext(HDC hdc) noexcept -> HGLRC {
		PIXELFORMATDESCRIPTOR pfd;
		ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

		pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
		pfd.nVersion = 1;
		pfd.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
		pfd.iPixelType = PFD_TYPE_RGBA;
		pfd.cColorBits = 24;
		pfd.cDepthBits = 32;
		pfd.cStencilBits = 8;
		pfd.iLayerType = PFD_MAIN_PLANE;

		int pixelFormat = ChoosePixelFormat(hdc, &pfd);
		SetPixelFormat(hdc, pixelFormat, &pfd);

		HGLRC context = wglCreateContext(hdc);
		wglMakeCurrent(hdc, context);

		return context;
	}

	Window_Win64::Window_Win64(WindowOptions const& option)
		:uniName(option.title), width(option.width), height(option.height), properties(option.properties)
	{ init(); }

	auto Window_Win64::init() noexcept -> bool {
		// Always set the window to be DPI awared, to aligned to GLFW ones
		SetProcessDPIAware();

		WNDCLASS wc;
		wc.style = CS_HREDRAW | CS_VREDRAW;
		wc.lpfnWndProc = StaticWndProc;
		wc.cbClsExtra = 0;
		wc.cbWndExtra = sizeof(Window_Win64*);
		wc.hInstance = instanceHandle;
		wc.hIcon = LoadIcon(0, IDI_APPLICATION);
		wc.hCursor = LoadCursor(0, IDC_ARROW);
		wc.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);
		wc.lpszMenuName = 0;
		wc.lpszClassName = uniName.c_str();

		if (!RegisterClass(&wc)) {
			MessageBox(0, L"RegisterClass Failed.", 0, 0);
			return false;
		}

		// Compute window rectangle dimensions based on requested client area dimensions.
		RECT R = { 0, 0, width, height };
		AdjustWindowRect(&R, WS_OVERLAPPEDWINDOW, false);
		int width = R.right - R.left;
		int height = R.bottom - R.top;

		FLOAT dpiX, dpiY;
		HDC screen = GetDC(0);
		dpiX = static_cast<FLOAT>(GetDeviceCaps(screen, LOGPIXELSX));
		dpiY = static_cast<FLOAT>(GetDeviceCaps(screen, LOGPIXELSY));
		ReleaseDC(0, screen);

		wndHandle = CreateWindowEx(
			NULL,
			uniName.c_str(),
			uniName.c_str(),
			WS_OVERLAPPEDWINDOW,
			CW_USEDEFAULT,
			CW_USEDEFAULT,
			static_cast<INT>(width),
			static_cast<INT>(height), 
			0,
			0,
			instanceHandle,
			static_cast<LPVOID>(this)
		);

		if (!wndHandle) {
			MessageBox(0, L"CreateWindow Failed.", 0, 0);
			return false;
		}

		ShowWindow(wndHandle, SW_SHOW);
		UpdateWindow(wndHandle);

		if (hasBit(properties, WindowProperties::OPENGL_CONTEX)) {
			HDC hdc = GetDC(wndHandle);
			HGLRC hglrc = bindOpenGLContext(hdc);
		}
		return true;
	}

	auto Window_Win64::fetchEvents() noexcept -> int
	{
		MSG msg = { 0 };
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
			if (msg.message == WM_QUIT) {
				shouldQuit = true;
			}
		}
		return (int)msg.wParam;
	}
	
	auto Window_Win64::invalid() noexcept -> void
	{		
		InvalidateRect(wndHandle, NULL, TRUE);
		UpdateWindow(wndHandle);
	}
	
	auto Window_Win64::endFrame() noexcept -> void {
		if (hasBit(properties, WindowProperties::OPENGL_CONTEX))
			SwapBuffers(GetDC(wndHandle));
		else {

		}
	}
	
	auto Window_Win64::getInput() noexcept -> Input* {
		return nullptr; // TODO
	}

	auto Window_Win64::wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)noexcept -> LRESULT {
		switch (msg)
		{
		case WM_CREATE:
			//std::fill(arr, arr + 240 * 120, RGB(255, 0, 0));
			//hBitmap = CreateBitmap(240, 120, 1, sizeof(COLORREF) * 8, (void*)arr);
			UpdateWindow(hwnd);
			break;

		case WM_ACTIVATE:
			return 0;
		case WM_SIZE:
			return 0;
		case WM_ENTERSIZEMOVE:
			return 0;
		case WM_EXITSIZEMOVE:
			return 0;
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
		case WM_MENUCHAR:
			return MAKELRESULT(0, MNC_CLOSE);
		case WM_GETMINMAXINFO:
			return 0;
		case WM_LBUTTONDOWN:
		case WM_MBUTTONDOWN:
		case WM_RBUTTONDOWN:
			return 0;
		case WM_LBUTTONUP:
		case WM_MBUTTONUP:
		case WM_RBUTTONUP:
			return 0;
		case WM_MOUSEMOVE:
			return 0;
		case WM_KEYUP:
			return 0;
		case WM_PAINT:
		{
			PAINTSTRUCT ps;
			HDC hdc = BeginPaint(hwnd, &ps);
			onPaintSignal.emit(hdc);
			EndPaint(hwnd, &ps);
			return 0;
		}
		}
		return DefWindowProc(hwnd, msg, wParam, lParam);
	}

	auto Window_Win64::destroy() noexcept -> void {
		DestroyWindow(wndHandle);
		UnregisterClass(uniName.c_str(), instanceHandle);
	}

	auto Window_Win64::isRunning() noexcept -> bool {
		return !shouldQuit;
	}

	auto Window_Win64::getHighDPI() noexcept -> float {
		return 1. * GetDpiForWindow(wndHandle) / 96;
	}
	
	auto Window_Win64::resize(size_t x, size_t y) noexcept -> void
	{
		RECT rcClient, rcWind;
		POINT ptDiff;
		GetClientRect(wndHandle, &rcClient);
		GetWindowRect(wndHandle, &rcWind);
		ptDiff.x = (rcWind.right - rcWind.left) - rcClient.right;
		ptDiff.y = (rcWind.bottom - rcWind.top) - rcClient.bottom;
		MoveWindow(wndHandle, rcWind.left, rcWind.top, x + ptDiff.x, y + ptDiff.y, TRUE);
	}

	auto Window_Win64::bindPaintingBitmapRGB8(size_t width, size_t height, char* data) noexcept -> void {
		auto paintbitmap = std::bind(paintRGB8Bitmap, std::placeholders::_1, width, height, data);
		onPaintSignal.connect(paintbitmap);
	}
	
	auto Window_Win64::connectResizeEvent(std::function<void(size_t, size_t)> const& func) noexcept -> void {
		Core::LogManager::Error("Error|TODO :: Window_Win64 does not support func { connectResizeEvent } for now!");
	}

	auto Window_Win64::getFramebufferSize(int* w, int* h) noexcept -> void {
		*w = width;
		*h = height;
	}

	auto Window_Win64::openFile(const char* filter) noexcept -> std::string {
		OPENFILENAMEA ofn;
		CHAR szFile[260] = { 0 };
		ZeroMemory(&ofn, sizeof(OPENFILENAME));
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.hwndOwner = wndHandle;
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
	
	auto Window_Win64::saveFile(const char* filter, std::string const& name) noexcept -> std::string {
		OPENFILENAMEA ofn;
		CHAR szFile[260] = { 0 };
		memcpy(szFile, name.c_str(), name.size() + 1);
		ZeroMemory(&ofn, sizeof(OPENFILENAME));
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.hwndOwner = wndHandle;
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

	auto paintRGB8Bitmap(HDC& hdc, size_t width, size_t height, char* data) -> void
	{
		BITMAPINFO bminfo;
		bminfo.bmiHeader.biSize = sizeof(bminfo.bmiHeader);
		bminfo.bmiHeader.biWidth = width;
		bminfo.bmiHeader.biHeight = height;
		bminfo.bmiHeader.biPlanes = 1;
		bminfo.bmiHeader.biBitCount = 24;
		bminfo.bmiHeader.biCompression = BI_RGB;
		bminfo.bmiHeader.biSizeImage = 0;
		bminfo.bmiHeader.biXPelsPerMeter = 1;
		bminfo.bmiHeader.biYPelsPerMeter = 1;
		bminfo.bmiHeader.biClrUsed = 0;
		bminfo.bmiHeader.biClrImportant = 0;

		int result = SetDIBitsToDevice(hdc, 0, 0,
			bminfo.bmiHeader.biWidth,
			bminfo.bmiHeader.biHeight,
			0, 0, 0,
			bminfo.bmiHeader.biHeight,
			data,
			reinterpret_cast<BITMAPINFO*>(&bminfo),
			DIB_RGB_COLORS);
	}
}