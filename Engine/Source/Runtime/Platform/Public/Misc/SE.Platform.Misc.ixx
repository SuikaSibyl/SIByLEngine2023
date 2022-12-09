/**
* Platform Module
* ------------------------------
* Provide platform-independent window class / query functions.
*/
export module SE.Platform.Misc;

#ifdef _WIN32
export import :Func.Win64;
#endif