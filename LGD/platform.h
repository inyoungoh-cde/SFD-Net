// platform.h
// Small shims that let the same sources build with Visual Studio on Windows
// and with GCC/Clang on Linux. Everything OS- or toolchain-specific lives
// here; the rest of the code stays portable.
#ifndef LGD_PLATFORM_H
#define LGD_PLATFORM_H

#include <string>
#include <ctime>

// --- std::filesystem -------------------------------------------------------
// Visual Studio 2017 (toolset v141) with its default /std:c++14 only ships the
// pre-standard experimental version. C++17 toolchains provide the real one.
#if defined(_MSC_VER) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)
#include <filesystem>
namespace fs = std::experimental::filesystem;
#elif defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 8)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

// --- OS headers ------------------------------------------------------------
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>   // GetModuleFileNameA
#else
#include <unistd.h>    // readlink
#include <limits.h>    // PATH_MAX
#endif

// Full path of the running executable ("" if it cannot be determined)
inline std::string executablePath() {
#if defined(_WIN32)
	char buf[MAX_PATH] = { 0 };
	DWORD n = GetModuleFileNameA(NULL, buf, MAX_PATH);
	return (n > 0) ? std::string(buf, n) : std::string();
#else
	char buf[PATH_MAX] = { 0 };
	ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	return (n > 0) ? std::string(buf, (size_t)n) : std::string();
#endif
}

// Current local time (the thread-safe variant differs between platforms)
inline std::tm localTimeNow() {
	std::time_t t = std::time(NULL);
	std::tm out = {};
#if defined(_WIN32)
	localtime_s(&out, &t);
#else
	localtime_r(&t, &out);
#endif
	return out;
}

#endif // LGD_PLATFORM_H
