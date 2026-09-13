#include "quant/bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <climits>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <unordered_set>
#include <queue>

namespace quant {

std::string UnicodeUtil::utf8_encode(int cp) {
    std::string result;
    if (cp < 0x80) {
        result += (char)cp;
    } else if (cp < 0x800) {
        result += (char)(0xC0 | (cp >> 6));
        result += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        result += (char)(0xE0 | (cp >> 12));
        result += (char)(0x80 | ((cp >> 6) & 0x3F));
        result += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x110000) {
        result += (char)(0xF0 | (cp >> 18));
        result += (char)(0x80 | ((cp >> 12) & 0x3F));
        result += (char)(0x80 | ((cp >> 6) & 0x3F));
        result += (char)(0x80 | (cp & 0x3F));
    }
    return result;
}

int UnicodeUtil::utf8_decode(const std::string& s, size_t& pos) {
    if (pos >= s.size()) return -1;
    unsigned char c = (unsigned char)s[pos];
    if (c < 0x80) {
        pos++;
        return c;
    } else if (c < 0xE0 && pos + 1 < s.size()) {
        int cp = ((c & 0x1F) << 6) | (s[pos+1] & 0x3F);
        pos += 2;
        return cp;
    } else if (c < 0xF0 && pos + 2 < s.size()) {
        int cp = ((c & 0x0F) << 12) | ((s[pos+1] & 0x3F) << 6) | (s[pos+2] & 0x3F);
        pos += 3;
        return cp;
    } else if (pos + 3 < s.size()) {
        int cp = ((c & 0x07) << 18) | ((s[pos+1] & 0x3F) << 12) | ((s[pos+2] & 0x3F) << 6) | (s[pos+3] & 0x3F);
        pos += 4;
        return cp;
    }
    pos++;
    return (int)c;
}

std::vector<int> UnicodeUtil::utf8_to_codepoints(const std::string& s) {
    std::vector<int> result;
    size_t pos = 0;
    while (pos < s.size()) {
        int cp = utf8_decode(s, pos);
        if (cp >= 0) result.push_back(cp);
    }
    return result;
}

std::string UnicodeUtil::codepoints_to_utf8(const std::vector<int>& cps) {
    std::string result;
    for (int cp : cps) result += utf8_encode(cp);
    return result;
}

bool UnicodeUtil::is_cjk(int cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x2E80 && cp <= 0x2EFF) ||
           (cp >= 0x3000 && cp <= 0x303F) ||
           (cp >= 0xFF00 && cp <= 0xFFEF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0xAC00 && cp <= 0xD7AF) ||
           (cp >= 0x1100 && cp <= 0x11FF) ||
           (cp >= 0x3130 && cp <= 0x318F) ||
           (cp >= 0xA000 && cp <= 0xA4CF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool UnicodeUtil::is_whitespace(int cp) {
    return cp == 0x0020 || cp == 0x0009 || cp == 0x000A || cp == 0x000D ||
           cp == 0x00A0 || cp == 0x1680 || cp == 0x2000 || cp == 0x2001 ||
           cp == 0x2002 || cp == 0x2003 || cp == 0x2004 || cp == 0x2005 ||
           cp == 0x2006 || cp == 0x2007 || cp == 0x2008 || cp == 0x2009 ||
           cp == 0x200A || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000 || cp == 0x200B || cp == 0xFEFF ||
           cp == 0x180E;
}

bool UnicodeUtil::is_punctuation(int cp) {
    return (cp >= 0x0021 && cp <= 0x002F) ||
           (cp >= 0x003A && cp <= 0x0040) ||
           (cp >= 0x005B && cp <= 0x0060) ||
           (cp >= 0x007B && cp <= 0x007E) ||
           (cp >= 0x2000 && cp <= 0x206F) ||
           (cp >= 0x3000 && cp <= 0x303F) ||
           cp == 0x00A1 || cp == 0x00A7 || cp == 0x00AB ||
           cp == 0x00B6 || cp == 0x00B7 || cp == 0x00BB || cp == 0x00BF ||
           cp == 0x037E || cp == 0x0387 ||
           cp == 0x055A || cp == 0x055B || cp == 0x055C || cp == 0x055D ||
           cp == 0x055E || cp == 0x055F ||
           (cp >= 0x0589 && cp <= 0x058A) ||
           cp == 0x05BE || cp == 0x05C0 || cp == 0x05C3 || cp == 0x05C6 ||
           cp == 0x05F3 || cp == 0x05F4 ||
           (cp >= 0x0609 && cp <= 0x060A) ||
           cp == 0x060C || cp == 0x060D || cp == 0x061B || cp == 0x061E ||
           cp == 0x061F || cp == 0x066A || cp == 0x066B || cp == 0x066C ||
           cp == 0x066D || cp == 0x06D4 ||
           cp == 0x0700 || cp == 0x0701 || cp == 0x0702 || cp == 0x0703 ||
           cp == 0x0704 || cp == 0x0705 || cp == 0x0706 || cp == 0x0707 ||
           cp == 0x0708 || cp == 0x0709 || cp == 0x070A || cp == 0x070B ||
           cp == 0x070C || cp == 0x070D ||
           cp == 0x07F7 || cp == 0x07F8 || cp == 0x07F9 ||
           (cp >= 0x0830 && cp <= 0x083E) ||
           cp == 0x085E ||
           (cp >= 0x0964 && cp <= 0x0965) ||
           cp == 0x0970 ||
           cp == 0x09FD ||
           cp == 0x0A76 ||
           cp == 0x0AF0 ||
           cp == 0x0C77 || cp == 0x0C78 || cp == 0x0C79 || cp == 0x0C7A ||
           cp == 0x0C7B || cp == 0x0C7C || cp == 0x0C7D || cp == 0x0C7E ||
           cp == 0x0C7F ||
           cp == 0x0DF4 ||
           cp == 0x0E4F || cp == 0x0E5A || cp == 0x0E5B ||
           cp == 0x0F04 || cp == 0x0F05 || cp == 0x0F06 || cp == 0x0F07 ||
           cp == 0x0F08 || cp == 0x0F09 || cp == 0x0F0A || cp == 0x0F0B ||
           cp == 0x0F0C || cp == 0x0F0D || cp == 0x0F0E || cp == 0x0F0F ||
           cp == 0x0F10 || cp == 0x0F11 || cp == 0x0F12 ||
           cp == 0x0F14 || cp == 0x0F3A || cp == 0x0F3B ||
           cp == 0x0F3C || cp == 0x0F3D || cp == 0x0F85 ||
           cp == 0x0FD0 || cp == 0x0FD1 || cp == 0x0FD2 || cp == 0x0FD3 ||
           cp == 0x0FD4 || cp == 0x0FD5 || cp == 0x0FD6 || cp == 0x0FD7 ||
           cp == 0x0FD8 || cp == 0x0FD9 ||
           cp == 0x104A || cp == 0x104B || cp == 0x104C || cp == 0x104D ||
           cp == 0x104E || cp == 0x104F ||
           cp == 0x10FB ||
           cp == 0x1361 || cp == 0x1362 || cp == 0x1363 || cp == 0x1364 ||
           cp == 0x1365 || cp == 0x1366 || cp == 0x1367 || cp == 0x1368;
}

bool UnicodeUtil::is_letter(int cp) {
    return (cp >= 0x0041 && cp <= 0x005A) ||
           (cp >= 0x0061 && cp <= 0x007A) ||
           (cp >= 0x00C0 && cp <= 0x024F) ||
           (cp >= 0x0370 && cp <= 0x03FF) ||
           (cp >= 0x0400 && cp <= 0x04FF) ||
           (cp >= 0x0500 && cp <= 0x052F) ||
           (cp >= 0x0530 && cp <= 0x058F) ||
           (cp >= 0x0590 && cp <= 0x05FF) ||
           (cp >= 0x0600 && cp <= 0x06FF) ||
           (cp >= 0x0700 && cp <= 0x074F) ||
           (cp >= 0x0750 && cp <= 0x077F) ||
           (cp >= 0x0780 && cp <= 0x07BF) ||
           (cp >= 0x0900 && cp <= 0x097F) ||
           (cp >= 0x0980 && cp <= 0x09FF) ||
           (cp >= 0x0A00 && cp <= 0x0A7F) ||
           (cp >= 0x0A80 && cp <= 0x0AFF) ||
           (cp >= 0x0B00 && cp <= 0x0B7F) ||
           (cp >= 0x0B80 && cp <= 0x0BFF) ||
           (cp >= 0x0C00 && cp <= 0x0C7F) ||
           (cp >= 0x0C80 && cp <= 0x0CFF) ||
           (cp >= 0x0D00 && cp <= 0x0D7F) ||
           (cp >= 0x0D80 && cp <= 0x0DFF) ||
           (cp >= 0x0E00 && cp <= 0x0E7F) ||
           (cp >= 0x0E80 && cp <= 0x0EFF) ||
           (cp >= 0x0F00 && cp <= 0x0FFF) ||
           (cp >= 0x1000 && cp <= 0x109F) ||
           (cp >= 0x10A0 && cp <= 0x10FF) ||
           (cp >= 0x1100 && cp <= 0x11FF) ||
           (cp >= 0x1200 && cp <= 0x137F) ||
           (cp >= 0x13A0 && cp <= 0x13FF) ||
           (cp >= 0x1400 && cp <= 0x167F) ||
           (cp >= 0x1680 && cp <= 0x169F) ||
           (cp >= 0x16A0 && cp <= 0x16FF) ||
           (cp >= 0x1700 && cp <= 0x171F) ||
           (cp >= 0x1720 && cp <= 0x173F) ||
           (cp >= 0x1740 && cp <= 0x175F) ||
           (cp >= 0x1760 && cp <= 0x177F) ||
           (cp >= 0x1780 && cp <= 0x17FF) ||
           (cp >= 0x1800 && cp <= 0x18AF) ||
           (cp >= 0x1900 && cp <= 0x194F) ||
           (cp >= 0x1950 && cp <= 0x197F) ||
           (cp >= 0x1980 && cp <= 0x19DF) ||
           (cp >= 0x1A00 && cp <= 0x1A1F) ||
           (cp >= 0x1B00 && cp <= 0x1B7F) ||
           (cp >= 0x1C00 && cp <= 0x1C4F) ||
           (cp >= 0x1C50 && cp <= 0x1C7F) ||
           (cp >= 0x1D00 && cp <= 0x1D7F) ||
           (cp >= 0x1E00 && cp <= 0x1EFF) ||
           (cp >= 0x2C00 && cp <= 0x2C5F) ||
           (cp >= 0x2D00 && cp <= 0x2D2F) ||
           (cp >= 0x2D30 && cp <= 0x2D7F) ||
           (cp >= 0x2DE0 && cp <= 0x2DFF) ||
           (cp >= 0xA000 && cp <= 0xA4CF) ||
           (cp >= 0xA4D0 && cp <= 0xA4FF) ||
           (cp >= 0xA500 && cp <= 0xA63F) ||
           (cp >= 0xA640 && cp <= 0xA69F) ||
           (cp >= 0xA700 && cp <= 0xA71F) ||
           (cp >= 0xA720 && cp <= 0xA7FF) ||
           (cp >= 0xA800 && cp <= 0xA82F) ||
           (cp >= 0xA840 && cp <= 0xA87F) ||
           (cp >= 0xA880 && cp <= 0xA8DF) ||
           (cp >= 0xA900 && cp <= 0xA92F) ||
           (cp >= 0xA930 && cp <= 0xA95F) ||
           (cp >= 0xAA00 && cp <= 0xAA5F) ||
           (cp >= 0xAA60 && cp <= 0xAA7F) ||
           (cp >= 0xAA80 && cp <= 0xAADF) ||
           (cp >= 0xAB00 && cp <= 0xAB2F) ||
           (cp >= 0xABC0 && cp <= 0xABFF) ||
           (cp >= 0xD7B0 && cp <= 0xD7FF) ||
           is_cjk(cp);
}

bool UnicodeUtil::is_digit(int cp) {
    return (cp >= 0x0030 && cp <= 0x0039) ||
           (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06F0 && cp <= 0x06F9) ||
           (cp >= 0x0966 && cp <= 0x096F) ||
           (cp >= 0x09E6 && cp <= 0x09EF) ||
           (cp >= 0x0A66 && cp <= 0x0A6F) ||
           (cp >= 0x0AE6 && cp <= 0x0AEF) ||
           (cp >= 0x0B66 && cp <= 0x0B6F) ||
           (cp >= 0x0BE6 && cp <= 0x0BEF) ||
           (cp >= 0x0C66 && cp <= 0x0C6F) ||
           (cp >= 0x0CE6 && cp <= 0x0CEF) ||
           (cp >= 0x0D66 && cp <= 0x0D6F) ||
           (cp >= 0x0E50 && cp <= 0x0E59) ||
           (cp >= 0x0ED0 && cp <= 0x0ED9) ||
           (cp >= 0x0F20 && cp <= 0x0F29) ||
           (cp >= 0x1040 && cp <= 0x1049) ||
           (cp >= 0x17E0 && cp <= 0x17E9) ||
           (cp >= 0x1810 && cp <= 0x1819) ||
           (cp >= 0x1946 && cp <= 0x194F) ||
           (cp >= 0x19D0 && cp <= 0x19D9) ||
           (cp >= 0x1A80 && cp <= 0x1A89) ||
           (cp >= 0x1A90 && cp <= 0x1A99) ||
           (cp >= 0x1B50 && cp <= 0x1B59) ||
           (cp >= 0x1BB0 && cp <= 0x1BB9) ||
           (cp >= 0x1C40 && cp <= 0x1C49) ||
           (cp >= 0x1C50 && cp <= 0x1C59) ||
           (cp >= 0xA620 && cp <= 0xA629) ||
           (cp >= 0xA8D0 && cp <= 0xA8D9) ||
           (cp >= 0xA900 && cp <= 0xA909) ||
           (cp >= 0xA9D0 && cp <= 0xA9D9) ||
           (cp >= 0xAA50 && cp <= 0xAA59) ||
           (cp >= 0xABF0 && cp <= 0xABF9) ||
           (cp >= 0xFF10 && cp <= 0xFF19);
}

bool UnicodeUtil::is_control(int cp) {
    return (cp < 0x0020) || (cp >= 0x007F && cp <= 0x009F);
}

bool UnicodeUtil::is_math_symbol(int cp) {
    return (cp >= 0x2200 && cp <= 0x22FF) ||
           (cp >= 0x2A00 && cp <= 0x2AFF) ||
           (cp >= 0x27C0 && cp <= 0x27EF) ||
           (cp >= 0x2980 && cp <= 0x29FF);
}

bool UnicodeUtil::is_currency(int cp) {
    return cp == 0x0024 || cp == 0x00A2 || cp == 0x00A3 || cp == 0x00A4 ||
           cp == 0x00A5 || cp == 0x058F || cp == 0x060B || cp == 0x07FE ||
           cp == 0x07FF || cp == 0x09F2 || cp == 0x09F3 || cp == 0x09FB ||
           cp == 0x0AF1 || cp == 0x0BF9 || cp == 0x0E3F || cp == 0x17DB ||
           cp == 0x20A0 || cp == 0x20A1 || cp == 0x20A2 || cp == 0x20A3 ||
           cp == 0x20A4 || cp == 0x20A5 || cp == 0x20A6 || cp == 0x20A7 ||
           cp == 0x20A8 || cp == 0x20A9 || cp == 0x20AA || cp == 0x20AB ||
           cp == 0x20AC || cp == 0x20AD || cp == 0x20AE || cp == 0x20AF ||
           cp == 0x20B0 || cp == 0x20B1 || cp == 0x20B2 || cp == 0x20B3 ||
           cp == 0x20B4 || cp == 0x20B5 || cp == 0x20B6 || cp == 0x20B7 ||
           cp == 0x20B8 || cp == 0x20B9 || cp == 0x20BA || cp == 0x20BB ||
           cp == 0x20BC || cp == 0x20BD || cp == 0x20BE || cp == 0x20BF ||
           cp == 0xA838 || cp == 0xFDFC || cp == 0xFE69 || cp == 0xFF04 ||
           cp == 0xFFE0 || cp == 0xFFE1 || cp == 0xFFE5 || cp == 0xFFE6;
}

bool UnicodeUtil::is_combining_mark(int cp) {
    return (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x1AB0 && cp <= 0x1AFF) ||
           (cp >= 0x1DC0 && cp <= 0x1DFF) ||
           (cp >= 0x20D0 && cp <= 0x20FF) ||
           (cp >= 0xFE20 && cp <= 0xFE2F) ||
           (cp >= 0x0483 && cp <= 0x0489) ||
           (cp >= 0x0591 && cp <= 0x05BD) ||
           cp == 0x05BF ||
           (cp >= 0x05C1 && cp <= 0x05C2) ||
           (cp >= 0x05C4 && cp <= 0x05C5) ||
           cp == 0x05C7 ||
           (cp >= 0x0610 && cp <= 0x061A) ||
           (cp >= 0x064B && cp <= 0x065F) ||
           cp == 0x0670 ||
           (cp >= 0x06D6 && cp <= 0x06DC) ||
           (cp >= 0x06DF && cp <= 0x06E4) ||
           (cp >= 0x06E7 && cp <= 0x06E8) ||
           (cp >= 0x06EA && cp <= 0x06ED) ||
           cp == 0x0711 ||
           (cp >= 0x0730 && cp <= 0x074A) ||
           (cp >= 0x07A6 && cp <= 0x07B0) ||
           (cp >= 0x0901 && cp <= 0x0903) || cp == 0x093C ||
           (cp >= 0x093E && cp <= 0x094D) ||
           (cp >= 0x0951 && cp <= 0x0954) ||
           (cp >= 0x0962 && cp <= 0x0963) ||
           (cp >= 0x0981 && cp <= 0x0983) || cp == 0x09BC ||
           cp == 0x09BE || cp == 0x09BF ||
           (cp >= 0x09C1 && cp <= 0x09C4) ||
           (cp >= 0x09CD && cp <= 0x09CD) ||
           (cp >= 0x0A01 && cp <= 0x0A03) || cp == 0x0A3C ||
           (cp >= 0x0A3E && cp <= 0x0A42) ||
           (cp >= 0x0A47 && cp <= 0x0A48) ||
           (cp >= 0x0A4B && cp <= 0x0A4D);
}

bool UnicodeUtil::is_variation_selector(int cp) {
    return (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0xE0100 && cp <= 0xE01EF);
}

bool UnicodeUtil::is_emoji(int cp) {
    return cp == 0x00A9 || cp == 0x00AE ||
           cp == 0x203C || cp == 0x2049 ||
           cp == 0x2122 || cp == 0x2139 ||
           (cp >= 0x2194 && cp <= 0x2199) ||
           (cp >= 0x21A9 && cp <= 0x21AA) ||
           (cp >= 0x231A && cp <= 0x231B) ||
           cp == 0x2328 || cp == 0x23CF ||
           (cp >= 0x23E9 && cp <= 0x23F3) ||
           (cp >= 0x23F8 && cp <= 0x23FA) ||
           cp == 0x24C2 || cp == 0x25AA || cp == 0x25AB ||
           cp == 0x25B6 || cp == 0x25C0 ||
           (cp >= 0x25FB && cp <= 0x25FE) ||
           (cp >= 0x2600 && cp <= 0x27BF) ||
           cp == 0x2934 || cp == 0x2935 ||
           (cp >= 0x2B00 && cp <= 0x2BFF) ||
           (cp >= 0x2E80 && cp <= 0x2FFF) ||
           (cp >= 0x3000 && cp <= 0x303F) ||
           (cp >= 0x3200 && cp <= 0x33FF) ||
           (cp >= 0x4DC0 && cp <= 0x4DFF) ||
           (cp >= 0xFE00 && cp <= 0xFE0F) ||
           cp == 0xFEFF ||
           (cp >= 0x1F000 && cp <= 0x1F9FF) ||
           (cp >= 0x20000 && cp <= 0x2FA1F);
}

bool UnicodeUtil::is_emoji_modifier(int cp) {
    return cp >= 0x1F3FB && cp <= 0x1F3FF;
}

int UnicodeUtil::width(int cp) {
    if (cp < 0x80) return 1;
    if (is_cjk(cp)) return 2;
    if (is_emoji(cp) && cp >= 0x1F000) return 2;
    return 1;
}

bool UnicodeUtil::is_regional_indicator(int cp) {
    return cp >= 0x1F1E6 && cp <= 0x1F1FF;
}

int UnicodeUtil::compose_pair([[maybe_unused]] int a, [[maybe_unused]] int b) {
    return -1;
}

std::string UnicodeUtil::nfkc_normalize(const std::string& s) {
    std::string result;
    size_t pos = 0;
    while (pos < s.size()) {
        int cp = utf8_decode(s, pos);
        if (cp < 0) continue;
        if (is_control(cp) && cp != 0x000A && cp != 0x000D && cp != 0x0009) continue;
        if (cp == 0x00B7) { result += utf8_encode(0x002E); continue; }
        if (cp == 0x2126) { result += utf8_encode(0x03A9); continue; }
        if (cp == 0x212B) { result += utf8_encode(0x00C5); continue; }
        if (cp == 0x00AA) { result += utf8_encode(0x0061); continue; }
        if (cp == 0x00BA) { result += utf8_encode(0x006F); continue; }
        if (cp == 0x212A) { result += utf8_encode(0x004B); continue; }
        result += utf8_encode(cp);
    }
    return result;
}

std::vector<std::string> UnicodeUtil::split_on_whitespace(const std::string& s) {
    std::vector<std::string> result;
    std::string current;
    size_t pos = 0;
    while (pos < s.size()) {
        int cp = utf8_decode(s, pos);
        if (cp < 0) break;
        if (is_whitespace(cp)) {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        } else {
            current += utf8_encode(cp);
        }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}

std::vector<std::string> UnicodeUtil::pretokenize(const std::string& s) {
    std::vector<std::string> result;
    std::string current;
    size_t pos = 0;
    while (pos < s.size()) {
        int cp = utf8_decode(s, pos);
        if (cp < 0) break;
        if (is_whitespace(cp)) {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        } else if (is_punctuation(cp)) {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
            result.push_back(utf8_encode(cp));
        } else {
            current += utf8_encode(cp);
        }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}

std::string UnicodeUtil::detect_language(const std::string& segment) {
    size_t cjk_count = 0;
    size_t latin_count = 0;
    size_t arabic_count = 0;
    size_t devanagari_count = 0;
    size_t korean_count = 0;
    size_t total = 0;
    size_t pos = 0;
    while (pos < segment.size()) {
        int cp = utf8_decode(segment, pos);
        if (cp < 0) continue;
        total++;
        if (cp >= 0x4E00 && cp <= 0x9FFF) cjk_count++;
        else if (cp >= 0xAC00 && cp <= 0xD7AF) korean_count++;
        else if (cp >= 0x0600 && cp <= 0x06FF) arabic_count++;
        else if (cp >= 0x0900 && cp <= 0x097F) devanagari_count++;
        else if ((cp >= 0x0041 && cp <= 0x005A) || (cp >= 0x0061 && cp <= 0x007A)) latin_count++;
    }
    if (total == 0) return "unknown";
    if (cjk_count > total / 2) {
        if (korean_count > cjk_count) return "ko";
        return "zh";
    }
    if (korean_count > total / 2) return "ko";
    if (arabic_count > total / 2) return "ar";
    if (devanagari_count > total / 2) return "hi";
    if (latin_count > total / 2) return "en";
    return "unknown";
}

std::string SentencePieceTrainer::normalize_nfkc(const std::string& text) {
    return UnicodeUtil::nfkc_normalize(text);
}

std::vector<std::string> SentencePieceTrainer::split_on_whitespace_and_punctuation(const std::string& text) {
    return UnicodeUtil::pretokenize(text);
}

std::vector<std::string> SentencePieceTrainer::pretokenize_default(const std::string& text) {
    return UnicodeUtil::pretokenize(text);
}

void SentencePieceTrainer::train(const std::vector<std::string>& corpus,
                                  const std::string& model_prefix,
                                  ModelType model_type,
                                  int vocab_size,
                                  float character_coverage,
                                  [[maybe_unused]] int num_threads,
                                  [[maybe_unused]] unsigned int seed) {

    std::vector<std::string> normalized;
    for (const auto& text : corpus) {
        normalized.push_back(normalize_nfkc(text));
    }

    std::vector<std::string> pretokenized;
    for (const auto& text : normalized) {
        auto pieces = pretokenize_default(text);
        pretokenized.insert(pretokenized.end(), pieces.begin(), pieces.end());
    }

    if (character_coverage < 1.0f) {
        std::unordered_set<int> unique_cps;
        for (const auto& text : normalized) {
            auto cps = UnicodeUtil::utf8_to_codepoints(text);
            unique_cps.insert(cps.begin(), cps.end());
        }

        std::vector<int> cps_sorted(unique_cps.begin(), unique_cps.end());
        std::sort(cps_sorted.begin(), cps_sorted.end());

        int target = (int)(cps_sorted.size() * character_coverage);
        if (target < (int)cps_sorted.size()) {
            cps_sorted.resize(target);
        }

        std::set<int> allowed(cps_sorted.begin(), cps_sorted.end());
        std::vector<std::string> filtered;
        for (const auto& text : normalized) {
            std::string filtered_text;
            auto cps = UnicodeUtil::utf8_to_codepoints(text);
            for (int cp : cps) {
                if (allowed.count(cp)) {
                    filtered_text += UnicodeUtil::utf8_encode(cp);
                }
            }
            filtered.push_back(filtered_text);
        }
        normalized = filtered;
    }

    switch (model_type) {
        case BPE: {
            BPETokenizer bpe;
            bpe.train(normalized, vocab_size);
            bpe.save(model_prefix + ".model");
            break;
        }
        case UNIGRAM: {
            UnigramTokenizer unigram;
            unigram.train(normalized, vocab_size);
            unigram.save(model_prefix + ".model");
            break;
        }
        case WORDPIECE: {
            WordPieceTokenizer wp;
            wp.train(normalized, vocab_size);
            wp.save(model_prefix + ".model");
            break;
        }
    }
}

} // namespace quant
