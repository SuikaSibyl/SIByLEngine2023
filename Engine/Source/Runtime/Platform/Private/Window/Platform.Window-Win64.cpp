module;
#include <string>
#include <functional>
#include <windows.h>
module Platform.Window:WindowWin64;
import Platform.Window;

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

	Window_Win64::Window_Win64(std::wstring const& unique_name)
		:uniName(unique_name) {}

	auto Window_Win64::create() noexcept -> bool {
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

		if (!RegisterClass(&wc))
		{
			MessageBox(0, L"RegisterClass Failed.", 0, 0);
			return false;
		}

		// Compute window rectangle dimensions based on requested client area dimensions.
		RECT R = { 0, 0, 800, 600 };
		AdjustWindowRect(&R, WS_OVERLAPPEDWINDOW, false);
		int width = R.right - R.left;
		int height = R.bottom - R.top;

		wndHandle = CreateWindowEx(
			NULL,
			uniName.c_str(),
			L"d3d App",
			WS_OVERLAPPEDWINDOW,
			CW_USEDEFAULT,
			CW_USEDEFAULT,
			width,
			height,
			0,
			0,
			instanceHandle,
			static_cast<LPVOID>(this)
		);

		if (!wndHandle)
		{
			MessageBox(0, L"CreateWindow Failed.", 0, 0);
			return false;
		}

		ShowWindow(wndHandle, SW_SHOW);
		UpdateWindow(wndHandle);

		return true;
	}

	auto Window_Win64::run() noexcept -> int
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