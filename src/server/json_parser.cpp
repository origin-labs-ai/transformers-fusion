#include "quant/json_parser.h"
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdio>

namespace quant {

// ===========================================================================
// JSON Parser — minimal hand-written recursive descent parser
// ===========================================================================
namespace {

class Parser {
public:
    Parser(const std::string& s, std::string* err)
        : src_(s), pos_(0), err_(err) {}

    JsonValue parse() {
        skip_ws();
        if (pos_ >= src_.size()) {
            if (err_) *err_ = "unexpected end of input";
            return JsonValue();
        }
        return parse_value();
    }

private:
    const std::string& src_;
    size_t pos_;
    std::string* err_;

    void set_error(const std::string& msg) {
        if (err_) *err_ = msg;
    }

    char peek() const {
        if (pos_ >= src_.size()) return '\0';
        return src_[pos_];
    }

    char advance() {
        if (pos_ >= src_.size()) return '\0';
        return src_[pos_++];
    }

    void expect(char c) {
        skip_ws();
        if (peek() != c) {
            set_error(std::string("expected '") + c + "' at position " + std::to_string(pos_));
            throw std::runtime_error("json parse error");
        }
        advance();
    }

    void skip_ws() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                pos_++;
            else
                break;
        }
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '"') return parse_string_value();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        set_error(std::string("unexpected character '") + c + "'");
        throw std::runtime_error("json parse error");
    }

    JsonValue parse_string_value() {
        return JsonValue(parse_string());
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (pos_ < src_.size()) {
            char c = advance();
            if (c == '"') break;
            if (c == '\\') {
                char esc = advance();
                switch (esc) {
                    case '"':  result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/':  result += '/'; break;
                    case 'b':  result += '\b'; break;
                    case 'f':  result += '\f'; break;
                    case 'n':  result += '\n'; break;
                    case 'r':  result += '\r'; break;
                    case 't':  result += '\t'; break;
                    case 'u': {
                        // Parse 4 hex digits
                        std::string hex;
                        for (int i = 0; i < 4 && pos_ < src_.size(); i++)
                            hex += advance();
                        uint32_t cp = (uint32_t)std::stoul(hex, nullptr, 16);
                        if (cp < 0x80) {
                            result += (char)cp;
                        } else if (cp < 0x800) {
                            result += (char)(0xC0 | (cp >> 6));
                            result += (char)(0x80 | (cp & 0x3F));
                        } else {
                            result += (char)(0xE0 | (cp >> 12));
                            result += (char)(0x80 | ((cp >> 6) & 0x3F));
                            result += (char)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: result += esc; break;
                }
            } else {
                result += c;
            }
        }
        return result;
    }

    JsonValue parse_number() {
        skip_ws();
        size_t start = pos_;
        bool is_float = false;

        if (peek() == '-') advance();
        while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9')
            advance();
        if (pos_ < src_.size() && src_[pos_] == '.') {
            is_float = true;
            advance();
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9')
                advance();
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            is_float = true;
            advance();
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-'))
                advance();
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9')
                advance();
        }

        std::string num_str = src_.substr(start, pos_ - start);
        if (is_float) {
            return JsonValue(std::strtod(num_str.c_str(), nullptr));
        } else {
            int64_t val = std::strtoll(num_str.c_str(), nullptr, 10);
            return JsonValue(val);
        }
    }

    JsonValue parse_bool() {
        if (src_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return JsonValue(true);
        }
        if (src_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return JsonValue(false);
        }
        set_error("expected boolean");
        throw std::runtime_error("json parse error");
    }

    JsonValue parse_null() {
        if (src_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return JsonValue();
        }
        set_error("expected null");
        throw std::runtime_error("json parse error");
    }

    JsonValue parse_object() {
        expect('{');
        std::unordered_map<std::string, JsonValue> result;
        skip_ws();
        if (peek() == '}') { advance(); return JsonValue(std::move(result)); }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            expect(':');
            JsonValue val = parse_value();
            result[std::move(key)] = std::move(val);
            skip_ws();
            char c = peek();
            if (c == '}') { advance(); break; }
            if (c == ',') { advance(); continue; }
            set_error("expected ',' or '}' in object");
            throw std::runtime_error("json parse error");
        }
        return JsonValue(std::move(result));
    }

    JsonValue parse_array() {
        expect('[');
        std::vector<JsonValue> result;
        skip_ws();
        if (peek() == ']') { advance(); return JsonValue(std::move(result)); }
        while (true) {
            result.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ']') { advance(); break; }
            if (c == ',') { advance(); continue; }
            set_error("expected ',' or ']' in array");
            throw std::runtime_error("json parse error");
        }
        return JsonValue(std::move(result));
    }
};

} // anonymous namespace

JsonValue JsonValue::parse(const std::string& json, std::string* error) {
    try {
        Parser p(json, error);
        return p.parse();
    } catch (...) {
        std::fprintf(stderr, "[WARN] Exception caught: %s (json parse error)\n", __func__);
        if (error && error->empty())
            *error = "json parse error";
        return JsonValue();
    }
}

std::string JsonValue::to_string() const {
    switch (type) {
        case NUL:    return "null";
        case BOOL:   return bool_val ? "true" : "false";
        case INT:    return std::to_string(int_val);
        case FLOAT: {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", float_val);
            return buf;
        }
        case STRING:  return "\"" + json_escape_string(str_val) + "\"";
        case ARRAY: {
            std::string s = "[";
            for (size_t i = 0; i < arr.size(); i++) {
                if (i) s += ",";
                s += arr[i].to_string();
            }
            s += "]";
            return s;
        }
        case OBJECT: {
            std::string s = "{";
            bool first = true;
            for (auto& [k, v] : obj) {
                if (!first) s += ",";
                s += "\"" + json_escape_string(k) + "\":" + v.to_string();
                first = false;
            }
            s += "}";
            return s;
        }
    }
    return "null";
}

std::string json_escape_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + s.size() / 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 32) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

} // namespace quant
