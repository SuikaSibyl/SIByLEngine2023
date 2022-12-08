/**
* Platform : System Module
* ------------------------------
* Provide platform independent query.
*/
export module Platform.System;

#ifdef _WIN32
export import :Win64;
#endif