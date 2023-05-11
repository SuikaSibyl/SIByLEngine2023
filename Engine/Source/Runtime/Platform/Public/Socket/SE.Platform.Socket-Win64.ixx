module;
#include <string>
#include <codecvt>
#include <Print/SE.Core.Log.hpp>
#include <Memory/SE.Core.Memory.hpp>
#ifdef _WIN32
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#ifndef _WINSOCK2API_
#include <WinSock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif // !_WINSOCK2API_
#endif // !__WIN32
export module SE.Platform.Socket:Win64;

#ifdef _WIN32
namespace SIByL::Platform
{
    export struct Socket {

        Socket();
        ~Socket();

        auto bind_client(uint16_t port) noexcept -> void;
        auto bind_server(uint16_t port) noexcept -> void;

        auto connect() noexcept -> bool;

        auto send(Core::Buffer const& buffer) noexcept -> void;
        auto recv(Core::Buffer& buffer) noexcept -> void;

        SOCKET mSocket;

        struct sockaddr_in addr;
        struct sockaddr_in server_addr;
    };

    inline auto initWSA() noexcept -> bool {
        WORD w_req = MAKEWORD(2, 2);    // version
        WSADATA wsadata;
        if (WSAStartup(w_req, &wsadata) != 0) {
            Core::LogManager::Error("Platform :: Socket :: WSAStartup failed!");
            return false;
        }
        else
            Core::LogManager::Log("Platform :: Socket :: WSAStartup succeed!");
        return true;
    }

    Socket::Socket() {
        // Initialize WSA
        static bool wsa_inited = false;
        if (!wsa_inited)
            wsa_inited = initWSA();
        if (!wsa_inited)
            return;

        // Create empty socket
        mSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (mSocket == INVALID_SOCKET) {
            Core::LogManager::Error("Platform :: Socket :: Create socket failed!");
            WSACleanup();
            wsa_inited = false;
            return;
        }
        else
            Core::LogManager::Log("Platform :: Socket :: Create socket succeed!");
    }

    Socket::~Socket() {
        closesocket(mSocket);
        WSACleanup();
    }

    auto Socket::bind_client(uint16_t port) noexcept -> void {
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
    }

    auto Socket::bind_server(uint16_t port) noexcept -> void {
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        server_addr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
    }


    auto Socket::connect() noexcept -> bool {
        if (::connect(mSocket, (PSOCKADDR)&server_addr, sizeof(sockaddr_in)) == INVALID_SOCKET) {
            Core::LogManager::Error("Platform :: Socket :: Connect failed!");
            closesocket(mSocket);
            WSACleanup();
            return false;
        }
        else {
            Core::LogManager::Log("Platform :: Socket :: Connect succeed!");
            return true;
        }
    }

    auto Socket::send(Core::Buffer const& buffer) noexcept -> void {
        ::send(mSocket, static_cast<char const*>(buffer.data), int(buffer.size), 0);
    }

    auto Socket::recv(Core::Buffer& buffer) noexcept -> void {
        ::recv(mSocket, static_cast<char*>(buffer.data), int(buffer.size), 0);
    }
}
#endif