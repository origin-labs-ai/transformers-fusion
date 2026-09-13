#include "quant/production_internal.h"
#include "quant/production_socket.h"
#include "quant/sha1.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <ctime>
#include <thread>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <set>
#include <map>
#include <chrono>

#ifdef ERROR
#undef ERROR
#endif


namespace quant {

// ========================================================================
// Internal helpers
// ========================================================================
namespace {

    // SHA1_CTX now shared from include/quant/sha1.h
    using sha1::SHA1_CTX;

    thread_local std::string g_last_error;

} // anonymous namespace

// ========================================================================
// I6: WebSocket
// ========================================================================
WebSocketHandler::WebSocketHandler(int port) : port_(port) {}

WebSocketHandler::~WebSocketHandler() {
    stop();
}

void WebSocketHandler::start() {
    if (running_) return;
    running_ = true;
    accept_thread_ = std::thread(&WebSocketHandler::accept_loop, this);
}

void WebSocketHandler::stop() {
    running_ = false;
    // Close all client connections
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (int fd : clients_) {
            socket_helpers::close_socket(fd);
        }
        clients_.clear();
    }
    if (accept_thread_.joinable()) accept_thread_.join();
    for (auto& t : client_threads_) {
        if (t.joinable()) t.join();
    }
    client_threads_.clear();
}

void WebSocketHandler::remove_client(int client_fd) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = std::find(clients_.begin(), clients_.end(), client_fd);
    if (it != clients_.end()) {
        socket_helpers::close_socket(*it);
        clients_.erase(it);
    }
}

void WebSocketHandler::close_socket(int fd) {
    socket_helpers::close_socket(fd);
}

// Compute Sec-WebSocket-Accept value
std::string WebSocketHandler::compute_accept_key(const std::string& client_key) {
    static const char* magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string concat = client_key + magic;
    uint8_t hash[20];
    sha1((const uint8_t*)concat.data(), concat.size(), hash);
    return base64_encode(hash, 20);
}

std::string WebSocketHandler::base64_encode(const uint8_t* data, size_t len) {
    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "abcdefghijklmnopqrstuvwxyz"
                                "0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t val = (uint32_t)data[i] << 16;
        if (i + 1 < len) val |= (uint32_t)data[i+1] << 8;
        if (i + 2 < len) val |= (uint32_t)data[i+2];

        out += table[(val >> 18) & 0x3F];
        out += table[(val >> 12) & 0x3F];
        out += (i + 1 < len) ? table[(val >> 6) & 0x3F] : '=';
        out += (i + 2 < len) ? table[val & 0x3F] : '=';
    }
    return out;
}

void WebSocketHandler::sha1(const uint8_t* data, size_t len, uint8_t out[20]) {
    SHA1_CTX ctx;
    SHA1_CTX::init(&ctx);
    SHA1_CTX::update(&ctx, data, len);
    SHA1_CTX::final(&ctx, out);
}

std::vector<uint8_t> WebSocketHandler::create_frame(const std::string& data,
                                                      uint8_t opcode) {
    size_t len = data.size();
    std::vector<uint8_t> frame;

    // FIN + opcode
    frame.push_back((uint8_t)(0x80 | opcode));

    // Payload length
    if (len < 126) {
        frame.push_back((uint8_t)len);
    } else if (len < 65536) {
        frame.push_back(126);
        frame.push_back((uint8_t)(len >> 8));
        frame.push_back((uint8_t)(len & 0xFF));
    } else {
        frame.push_back(127);
        for (int i = 7; i >= 0; i--)
            frame.push_back((uint8_t)(len >> (i * 8)));
    }

    // Payload
    frame.insert(frame.end(), data.begin(), data.end());
    return frame;
}

bool WebSocketHandler::parse_frame(const uint8_t* buf, size_t len,
                                    std::string& payload, uint8_t& opcode,
                                    bool& fin) {
    if (len < 2) return false;

    fin = (buf[0] & 0x80) != 0;
    opcode = buf[0] & 0x0F;
    bool masked = (buf[1] & 0x80) != 0;
    uint64_t payload_len = buf[1] & 0x7F;

    size_t offset = 2;

    if (payload_len == 126) {
        if (len < 4) return false;
        payload_len = ((uint64_t)buf[2] << 8) | buf[3];
        offset = 4;
    } else if (payload_len == 127) {
        if (len < 10) return false;
        payload_len = 0;
        for (int i = 0; i < 8; i++)
            payload_len = (payload_len << 8) | buf[2 + i];
        offset = 10;
    }

    uint8_t mask_key[4] = {0, 0, 0, 0};
    if (masked) {
        if (len < offset + 4) return false;
        memcpy(mask_key, buf + offset, 4);
        offset += 4;
    }

    if (len < offset + payload_len) return false;

    payload.assign((const char*)(buf + offset), (size_t)payload_len);

    // Unmask if needed
    if (masked) {
        for (size_t i = 0; i < payload.size(); i++)
            payload[i] ^= mask_key[i & 3];
    }

    return true;
}

void WebSocketHandler::accept_loop() {
    if (!socket_helpers::platform_init()) return;

#ifdef _WIN32
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) { socket_helpers::platform_cleanup(); return; }
#else
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { socket_helpers::platform_cleanup(); return; }
#endif

    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
               (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((u_short)port_);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        closesocket(server_fd);
        socket_helpers::platform_cleanup();
        return;
    }

    if (listen(server_fd, 16) < 0) {
        closesocket(server_fd);
        socket_helpers::platform_cleanup();
        return;
    }

    fd_set read_fds;
    while (running_) {
        FD_ZERO(&read_fds);
#ifdef _WIN32
        FD_SET(server_fd, &read_fds);
#else
        FD_SET(server_fd, &read_fds);
#endif
        struct timeval tv = {1, 0};

        if (select((int)server_fd + 1, &read_fds, nullptr, nullptr, &tv) > 0) {
            sockaddr_in client_addr;
#ifdef _WIN32
            int addr_len = sizeof(client_addr);
#else
            socklen_t addr_len = sizeof(client_addr);
#endif
            int client_fd = (int)accept(server_fd, (sockaddr*)&client_addr, &addr_len);
            if (client_fd < 0) continue;

            // Read WebSocket upgrade request
            char buf[4096];
            int n = recv(client_fd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) {
                socket_helpers::close_socket(client_fd);
                continue;
            }
            buf[n] = '\0';
            std::string req(buf);

            if (req.find("Upgrade: websocket") == std::string::npos &&
                req.find("upgrade: websocket") == std::string::npos) {
                // Not a WebSocket request
                std::string resp = "HTTP/1.1 400 Bad Request\r\n\r\n";
                send(client_fd, resp.c_str(), (int)resp.size(), 0);
                socket_helpers::close_socket(client_fd);
                continue;
            }

            // Extract WebSocket key
            std::string ws_key;
            auto key_pos = req.find("Sec-WebSocket-Key:");
            if (key_pos == std::string::npos)
                key_pos = req.find("sec-websocket-key:");
            if (key_pos != std::string::npos) {
                auto val_start = req.find_first_not_of(" \t", key_pos + 18);
                if (val_start != std::string::npos) {
                    auto val_end = req.find("\r\n", val_start);
                    if (val_end != std::string::npos)
                        ws_key = req.substr(val_start, val_end - val_start);
                }
            }

            if (ws_key.empty()) {
                ws_key = "dGhlIHNhbXBsZSBub25jZQ==";
            }

            std::string accept = compute_accept_key(ws_key);

            std::string resp =
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: " + accept + "\r\n"
                "\r\n";

            send(client_fd, resp.c_str(), (int)resp.size(), 0);

            {
                std::lock_guard<std::mutex> lock(clients_mutex_);
                clients_.push_back(client_fd);
            }

            // Spawn client handler thread
            client_threads_.emplace_back(&WebSocketHandler::client_loop, this, client_fd);
        }
    }

    closesocket(server_fd);
    socket_helpers::platform_cleanup();
}

void WebSocketHandler::client_loop(int client_fd) {
    socket_helpers::set_timeout(client_fd, 30);
    uint8_t buf[4096];

    while (running_) {
#ifdef _WIN32
        int n = recv(client_fd, (char*)buf, sizeof(buf), 0);
#else
        int n = recv(client_fd, (char*)buf, sizeof(buf), 0);
#endif
        if (n <= 0) {
            // Client disconnected
            break;
        }

        std::string payload;
        uint8_t opcode = 0;
        bool fin = false;

        if (!parse_frame(buf, (size_t)n, payload, opcode, fin)) {
            break;
        }

        switch (opcode) {
            case 0x8: // Close frame
                {
                    auto close_frame = create_frame("", 0x8);
                    send(client_fd, (const char*)close_frame.data(),
                         (int)close_frame.size(), 0);
                }
                goto disconnect;

            case 0x9: // Ping
                {
                    auto pong = create_frame(payload, 0xA);
                    send(client_fd, (const char*)pong.data(),
                         (int)pong.size(), 0);
                }
                break;

            case 0xA: // Pong
                break;

            case 0x1: // Text frame
            case 0x2: // Binary frame
                // Echo back for now
                {
                    auto echo = create_frame(payload, opcode);
                    send(client_fd, (const char*)echo.data(),
                         (int)echo.size(), 0);
                }
                break;
        }
    }

disconnect:
    remove_client(client_fd);
}

void WebSocketHandler::broadcast(const std::string& msg) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto frame = create_frame(msg, 0x1); // Text opcode

    auto it = clients_.begin();
    while (it != clients_.end()) {
        int client = *it;
        int sent = send(client, (const char*)frame.data(),
                        (int)frame.size(), 0);
        if (sent < 0) {
            socket_helpers::close_socket(client);
            it = clients_.erase(it);
        } else {
            ++it;
        }
    }
}


} // namespace quant
