#pragma once
// detail/cli_parse.h — shared crash-proof CLI number parsing for tools.
//
// BUGFIX (bug census): every tool hand-rolled bare std::stoi/stof/stoll in
// parse_args — a typo'd flag value threw uncaught out of main (ugly
// terminate). These helpers print `flag: bad value 'X'` + usage hint and
// exit(2) instead. Header-only so tools/ (no lib) can use it.
#include <cstdlib>
#include <iostream>
#include <string>

namespace quant {
namespace cli_parse {

inline void bad_value(const char* flag, const char* val) {
    std::cerr << "Error: " << flag << " has invalid numeric value '" << val << "'\n";
    std::exit(2);
}

inline int parse_int(const char* flag, const char* val) {
    try {
        size_t pos = 0;
        int v = std::stoi(val, &pos);
        if (val[pos] != '\0') bad_value(flag, val);
        return v;
    } catch (const std::exception&) {
        bad_value(flag, val);
        return 0; // unreachable (bad_value exits)
    }
}

inline long long parse_ll(const char* flag, const char* val) {
    try {
        size_t pos = 0;
        long long v = std::stoll(val, &pos);
        if (val[pos] != '\0') bad_value(flag, val);
        return v;
    } catch (const std::exception&) {
        bad_value(flag, val);
        return 0;
    }
}

inline float parse_float(const char* flag, const char* val) {
    try {
        size_t pos = 0;
        float v = std::stof(val, &pos);
        if (val[pos] != '\0') bad_value(flag, val);
        return v;
    } catch (const std::exception&) {
        bad_value(flag, val);
        return 0.0f;
    }
}

inline double parse_double(const char* flag, const char* val) {
    try {
        size_t pos = 0;
        double v = std::stod(val, &pos);
        if (val[pos] != '\0') bad_value(flag, val);
        return v;
    } catch (const std::exception&) {
        bad_value(flag, val);
        return 0.0;
    }
}

} // namespace cli_parse
} // namespace quant
