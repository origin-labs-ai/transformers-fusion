#pragma once

#include "quant/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace quant {

// ===========================================================================
// Minimal JSON parser — no external dependencies
//
// Supports: null, bool, number (int/double), string, array, object.
// Returns a variant-like JsonValue tree.
// ===========================================================================

class JsonValue {
public:
    enum Type : uint8_t { NUL = 0, BOOL = 1, INT = 2, FLOAT = 3, STRING = 4, ARRAY = 5, OBJECT = 6 };
    Type type = NUL;
    bool bool_val = false;
    int64_t int_val = 0;
    double float_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr;
    std::unordered_map<std::string, JsonValue> obj;

    JsonValue() : type(NUL) {}
    JsonValue(bool b) : type(BOOL), bool_val(b) {}
    JsonValue(int64_t i) : type(INT), int_val(i) {}
    JsonValue(double d) : type(FLOAT), float_val(d) {}
    JsonValue(const std::string& s) : type(STRING), str_val(s) {}
    JsonValue(const char* s) : type(STRING), str_val(s) {}
    JsonValue(std::vector<JsonValue> a) : type(ARRAY), arr(std::move(a)) {}
    JsonValue(std::unordered_map<std::string, JsonValue> o) : type(OBJECT), obj(std::move(o)) {}

    bool is_null() const { return type == NUL; }
    bool is_bool() const { return type == BOOL; }
    bool is_number() const { return type == INT || type == FLOAT; }
    bool is_string() const { return type == STRING; }
    bool is_array() const { return type == ARRAY; }
    bool is_object() const { return type == OBJECT; }

    // BUGFIX (bug census): untyped accessors returned garbage on type
    // mismatch (as_bool on a STRING read an uninitialized union member).
    // Strict variants throw; lenient as_* keep legacy behavior for callers
    // that pre-check with is_* (server code does).
    bool as_bool() const { return bool_val; }
    int64_t as_int() const { return int_val; }
    double as_float() const { return type == INT ? (double)int_val : float_val; }
    const std::string& as_string() const { return str_val; }
    bool as_bool_checked() const {
        if (type != BOOL) throw Error("JsonValue: not a bool");
        return bool_val;
    }
    int64_t as_int_checked() const {
        if (type != INT) throw Error("JsonValue: not an int");
        return int_val;
    }
    double as_float_checked() const {
        if (!is_number()) throw Error("JsonValue: not a number");
        return as_float();
    }
    const std::string& as_string_checked() const {
        if (type != STRING) throw Error("JsonValue: not a string");
        return str_val;
    }

    const JsonValue& operator[](const std::string& key) const {
        static JsonValue null_val;
        auto it = obj.find(key);
        return it != obj.end() ? it->second : null_val;
    }

    const JsonValue& operator[](size_t index) const {
        if (index >= arr.size()) throw Error("JsonValue: array index out of range");
        return arr[index];
    }
    // Legacy lenient index (returns null instead of throwing) for callers
    // that probe optional elements.
    const JsonValue& at_or_null(size_t index) const {
        static JsonValue null_val;
        return index < arr.size() ? arr[index] : null_val;
    }

    bool has(const std::string& key) const {
        return obj.find(key) != obj.end();
    }

    std::string to_string() const;

    static JsonValue parse(const std::string& json, std::string* error = nullptr);
};

std::string json_escape_string(const std::string& s);

} // namespace quant
