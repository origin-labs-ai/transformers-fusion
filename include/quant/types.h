#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cassert>
namespace quant {
enum class Activation : uint8_t { None, ReLU, GELU, SiLU, SwiGLU, GeGLU };
enum class RoPEScalingMode : uint8_t { None, Linear, NTK, YARN };
// ============================================================
// Format enum — Q-series quantization formats — v3 RE-ARRANGED 2026-08-23
// Rules: exact BPW (name == claimed BPW), 3 K variants per BPW (L/M/H),
// GRP exact at same BPW (no extra), half-BPW GRP as Q_GRP_X_Y,
// MXQ only as MXQ_(BPW)_GRP (4-variant mix, no TWI), no TWI_MIX.
// ============================================================
enum class Format : uint8_t {
    // --- Base integer BPW (10) ---
    Q1              = 0,   // 1.00
    Q2              = 1,   // 2.00
    Q3              = 2,   // 3.00
    Q4              = 3,   // 4.00
    Q6              = 4,   // 6.00
    Q8              = 5,   // 8.00
    Q12             = 6,   // 12.00
    Q16             = 7,   // 16.00
    Q24             = 8,   // 24.00
    Q32             = 9,   // 32.00
    // --- K variants integer BPW (30) ---
    Q1_K_L          = 10, Q1_K_M          = 11, Q1_K_H          = 12,
    Q2_K_L          = 13, Q2_K_M          = 14, Q2_K_H          = 15,
    Q3_K_L          = 16, Q3_K_M          = 17, Q3_K_H          = 18,
    Q4_K_L          = 19, Q4_K_M          = 20, Q4_K_H          = 21,
    Q6_K_L          = 22, Q6_K_M          = 23, Q6_K_H          = 24,
    Q8_K_L          = 25, Q8_K_M          = 26, Q8_K_H          = 27,
    Q12_K_L         = 28, Q12_K_M         = 29, Q12_K_H         = 30,
    Q16_K_L         = 31, Q16_K_M         = 32, Q16_K_H         = 33,
    Q24_K_L         = 34, Q24_K_M         = 35, Q24_K_H         = 36,
    Q32_K_L         = 37, Q32_K_M         = 38, Q32_K_H         = 39,
    // --- GRP exact integer BPW (10) ---
    Q1_GRP          = 40, Q2_GRP          = 41, Q3_GRP          = 42, Q4_GRP          = 43, Q6_GRP          = 44,
    Q8_GRP          = 45, Q12_GRP         = 46, Q16_GRP         = 47, Q24_GRP         = 48, Q32_GRP         = 49,
    // --- K_GRP exact integer BPW (30) ---
    Q1_K_L_GRP      = 50, Q1_K_M_GRP      = 51, Q1_K_H_GRP      = 52,
    Q2_K_L_GRP      = 53, Q2_K_M_GRP      = 54, Q2_K_H_GRP      = 55,
    Q3_K_L_GRP      = 56, Q3_K_M_GRP      = 57, Q3_K_H_GRP      = 58,
    Q4_K_L_GRP      = 59, Q4_K_M_GRP      = 60, Q4_K_H_GRP      = 61,
    Q6_K_L_GRP      = 62, Q6_K_M_GRP      = 63, Q6_K_H_GRP      = 64,
    Q8_K_L_GRP      = 65, Q8_K_M_GRP      = 66, Q8_K_H_GRP      = 67,
    Q12_K_L_GRP     = 68, Q12_K_M_GRP     = 69, Q12_K_H_GRP     = 70,
    Q16_K_L_GRP     = 71, Q16_K_M_GRP     = 72, Q16_K_H_GRP     = 73,
    Q24_K_L_GRP     = 74, Q24_K_M_GRP     = 75, Q24_K_H_GRP     = 76,
    Q32_K_L_GRP     = 77, Q32_K_M_GRP     = 78, Q32_K_H_GRP     = 79,
    // --- Half BPW GRP (9) ---
    Q_GRP_1_5       = 80, Q_GRP_2_5       = 81, Q_GRP_3_5       = 82, Q_GRP_4_5       = 83, Q_GRP_6_5       = 84,
    Q_GRP_8_5       = 85, Q_GRP_12_5      = 86, Q_GRP_16_5      = 87, Q_GRP_24_5      = 88,
    // --- MXQ mix 4-variant only as MXQ_(BPW)_GRP (7) ---
    MXQ_3_5_GRP     = 89, MXQ_4_5_GRP     = 90, MXQ_6_5_GRP     = 91, MXQ_8_5_GRP     = 92, MXQ_12_5_GRP    = 93, MXQ_16_5_GRP    = 94, MXQ_24_5_GRP    = 95,
};
constexpr int FORMAT_COUNT = 96;
inline const char* format_name(Format f) {
    switch(f) {
        case Format::Q1: return "Q1"; case Format::Q2: return "Q2"; case Format::Q3: return "Q3"; case Format::Q4: return "Q4"; case Format::Q6: return "Q6"; case Format::Q8: return "Q8"; case Format::Q12: return "Q12"; case Format::Q16: return "Q16"; case Format::Q24: return "Q24"; case Format::Q32: return "Q32";
        case Format::Q1_K_L: return "Q1_K_L"; case Format::Q1_K_M: return "Q1_K_M"; case Format::Q1_K_H: return "Q1_K_H";
        case Format::Q2_K_L: return "Q2_K_L"; case Format::Q2_K_M: return "Q2_K_M"; case Format::Q2_K_H: return "Q2_K_H";
        case Format::Q3_K_L: return "Q3_K_L"; case Format::Q3_K_M: return "Q3_K_M"; case Format::Q3_K_H: return "Q3_K_H";
        case Format::Q4_K_L: return "Q4_K_L"; case Format::Q4_K_M: return "Q4_K_M"; case Format::Q4_K_H: return "Q4_K_H";
        case Format::Q6_K_L: return "Q6_K_L"; case Format::Q6_K_M: return "Q6_K_M"; case Format::Q6_K_H: return "Q6_K_H";
        case Format::Q8_K_L: return "Q8_K_L"; case Format::Q8_K_M: return "Q8_K_M"; case Format::Q8_K_H: return "Q8_K_H";
        case Format::Q12_K_L: return "Q12_K_L"; case Format::Q12_K_M: return "Q12_K_M"; case Format::Q12_K_H: return "Q12_K_H";
        case Format::Q16_K_L: return "Q16_K_L"; case Format::Q16_K_M: return "Q16_K_M"; case Format::Q16_K_H: return "Q16_K_H";
        case Format::Q24_K_L: return "Q24_K_L"; case Format::Q24_K_M: return "Q24_K_M"; case Format::Q24_K_H: return "Q24_K_H";
        case Format::Q32_K_L: return "Q32_K_L"; case Format::Q32_K_M: return "Q32_K_M"; case Format::Q32_K_H: return "Q32_K_H";
        case Format::Q1_GRP: return "Q1_GRP"; case Format::Q2_GRP: return "Q2_GRP"; case Format::Q3_GRP: return "Q3_GRP"; case Format::Q4_GRP: return "Q4_GRP"; case Format::Q6_GRP: return "Q6_GRP"; case Format::Q8_GRP: return "Q8_GRP"; case Format::Q12_GRP: return "Q12_GRP"; case Format::Q16_GRP: return "Q16_GRP"; case Format::Q24_GRP: return "Q24_GRP"; case Format::Q32_GRP: return "Q32_GRP";
        case Format::Q1_K_L_GRP: return "Q1_K_L_GRP"; case Format::Q1_K_M_GRP: return "Q1_K_M_GRP"; case Format::Q1_K_H_GRP: return "Q1_K_H_GRP";
        case Format::Q2_K_L_GRP: return "Q2_K_L_GRP"; case Format::Q2_K_M_GRP: return "Q2_K_M_GRP"; case Format::Q2_K_H_GRP: return "Q2_K_H_GRP";
        case Format::Q3_K_L_GRP: return "Q3_K_L_GRP"; case Format::Q3_K_M_GRP: return "Q3_K_M_GRP"; case Format::Q3_K_H_GRP: return "Q3_K_H_GRP";
        case Format::Q4_K_L_GRP: return "Q4_K_L_GRP"; case Format::Q4_K_M_GRP: return "Q4_K_M_GRP"; case Format::Q4_K_H_GRP: return "Q4_K_H_GRP";
        case Format::Q6_K_L_GRP: return "Q6_K_L_GRP"; case Format::Q6_K_M_GRP: return "Q6_K_M_GRP"; case Format::Q6_K_H_GRP: return "Q6_K_H_GRP";
        case Format::Q8_K_L_GRP: return "Q8_K_L_GRP"; case Format::Q8_K_M_GRP: return "Q8_K_M_GRP"; case Format::Q8_K_H_GRP: return "Q8_K_H_GRP";
        case Format::Q12_K_L_GRP: return "Q12_K_L_GRP"; case Format::Q12_K_M_GRP: return "Q12_K_M_GRP"; case Format::Q12_K_H_GRP: return "Q12_K_H_GRP";
        case Format::Q16_K_L_GRP: return "Q16_K_L_GRP"; case Format::Q16_K_M_GRP: return "Q16_K_M_GRP"; case Format::Q16_K_H_GRP: return "Q16_K_H_GRP";
        case Format::Q24_K_L_GRP: return "Q24_K_L_GRP"; case Format::Q24_K_M_GRP: return "Q24_K_M_GRP"; case Format::Q24_K_H_GRP: return "Q24_K_H_GRP";
        case Format::Q32_K_L_GRP: return "Q32_K_L_GRP"; case Format::Q32_K_M_GRP: return "Q32_K_M_GRP"; case Format::Q32_K_H_GRP: return "Q32_K_H_GRP";
        case Format::Q_GRP_1_5: return "Q_GRP_1.5"; case Format::Q_GRP_2_5: return "Q_GRP_2.5"; case Format::Q_GRP_3_5: return "Q_GRP_3.5"; case Format::Q_GRP_4_5: return "Q_GRP_4.5"; case Format::Q_GRP_6_5: return "Q_GRP_6.5"; case Format::Q_GRP_8_5: return "Q_GRP_8.5"; case Format::Q_GRP_12_5: return "Q_GRP_12.5"; case Format::Q_GRP_16_5: return "Q_GRP_16.5"; case Format::Q_GRP_24_5: return "Q_GRP_24.5";
        case Format::MXQ_3_5_GRP: return "MXQ_3.5_GRP"; case Format::MXQ_4_5_GRP: return "MXQ_4.5_GRP"; case Format::MXQ_6_5_GRP: return "MXQ_6.5_GRP"; case Format::MXQ_8_5_GRP: return "MXQ_8.5_GRP"; case Format::MXQ_12_5_GRP: return "MXQ_12.5_GRP"; case Format::MXQ_16_5_GRP: return "MXQ_16.5_GRP"; case Format::MXQ_24_5_GRP: return "MXQ_24.5_GRP";
        default: return "unknown";
    }
}
inline float format_bpw(Format f) {
    switch(f) {
        case Format::Q1: return 1.0f; case Format::Q2: return 2.0f; case Format::Q3: return 3.0f; case Format::Q4: return 4.0f; case Format::Q6: return 6.0f; case Format::Q8: return 8.0f; case Format::Q12: return 12.0f; case Format::Q16: return 16.0f; case Format::Q24: return 24.0f; case Format::Q32: return 32.0f;
        case Format::Q1_K_L: case Format::Q1_K_M: case Format::Q1_K_H: return 1.0f;
        case Format::Q2_K_L: case Format::Q2_K_M: case Format::Q2_K_H: return 2.0f;
        case Format::Q3_K_L: case Format::Q3_K_M: case Format::Q3_K_H: return 3.0f;
        case Format::Q4_K_L: case Format::Q4_K_M: case Format::Q4_K_H: return 4.0f;
        case Format::Q6_K_L: case Format::Q6_K_M: case Format::Q6_K_H: return 6.0f;
        case Format::Q8_K_L: case Format::Q8_K_M: case Format::Q8_K_H: return 8.0f;
        case Format::Q12_K_L: case Format::Q12_K_M: case Format::Q12_K_H: return 12.0f;
        case Format::Q16_K_L: case Format::Q16_K_M: case Format::Q16_K_H: return 16.0f;
        case Format::Q24_K_L: case Format::Q24_K_M: case Format::Q24_K_H: return 24.0f;
        case Format::Q32_K_L: case Format::Q32_K_M: case Format::Q32_K_H: return 32.0f;
        case Format::Q1_GRP: return 1.0f; case Format::Q2_GRP: return 2.0f; case Format::Q3_GRP: return 3.0f; case Format::Q4_GRP: return 4.0f; case Format::Q6_GRP: return 6.0f; case Format::Q8_GRP: return 8.0f; case Format::Q12_GRP: return 12.0f; case Format::Q16_GRP: return 16.0f; case Format::Q24_GRP: return 24.0f; case Format::Q32_GRP: return 32.0f;
        case Format::Q1_K_L_GRP: case Format::Q1_K_M_GRP: case Format::Q1_K_H_GRP: return 1.0f;
        case Format::Q2_K_L_GRP: case Format::Q2_K_M_GRP: case Format::Q2_K_H_GRP: return 2.0f;
        case Format::Q3_K_L_GRP: case Format::Q3_K_M_GRP: case Format::Q3_K_H_GRP: return 3.0f;
        case Format::Q4_K_L_GRP: case Format::Q4_K_M_GRP: case Format::Q4_K_H_GRP: return 4.0f;
        case Format::Q6_K_L_GRP: case Format::Q6_K_M_GRP: case Format::Q6_K_H_GRP: return 6.0f;
        case Format::Q8_K_L_GRP: case Format::Q8_K_M_GRP: case Format::Q8_K_H_GRP: return 8.0f;
        case Format::Q12_K_L_GRP: case Format::Q12_K_M_GRP: case Format::Q12_K_H_GRP: return 12.0f;
        case Format::Q16_K_L_GRP: case Format::Q16_K_M_GRP: case Format::Q16_K_H_GRP: return 16.0f;
        case Format::Q24_K_L_GRP: case Format::Q24_K_M_GRP: case Format::Q24_K_H_GRP: return 24.0f;
        case Format::Q32_K_L_GRP: case Format::Q32_K_M_GRP: case Format::Q32_K_H_GRP: return 32.0f;
        case Format::Q_GRP_1_5: return 1.5f; case Format::Q_GRP_2_5: return 2.5f; case Format::Q_GRP_3_5: return 3.5f; case Format::Q_GRP_4_5: return 4.5f; case Format::Q_GRP_6_5: return 6.5f; case Format::Q_GRP_8_5: return 8.5f; case Format::Q_GRP_12_5: return 12.5f; case Format::Q_GRP_16_5: return 16.5f; case Format::Q_GRP_24_5: return 24.5f;
        case Format::MXQ_3_5_GRP: return 3.5f; case Format::MXQ_4_5_GRP: return 4.5f; case Format::MXQ_6_5_GRP: return 6.5f; case Format::MXQ_8_5_GRP: return 8.5f; case Format::MXQ_12_5_GRP: return 12.5f; case Format::MXQ_16_5_GRP: return 16.5f; case Format::MXQ_24_5_GRP: return 24.5f;
        default: return 0;
    }
}
inline bool format_is_base(Format f) { auto v=(int)f; return v<=9; }
inline bool format_is_k(Format f) { auto v=(int)f; return (v>=10&&v<=39)||(v>=50&&v<=79); }
inline bool format_is_grp(Format f) { auto v=(int)f; return (v>=40&&v<=88)||(v>=165&&v<=183); }
inline bool format_is_mx(Format f) { auto v=(int)f; return v>=89; }
inline bool format_is_mx_plain(Format f){ return false; }
inline bool format_is_mixed(Format f){ return format_is_mx(f); }
inline bool format_is_twi_mix(Format f){ return false; }
inline bool format_is_quad_mix(Format f){ return format_is_mx(f); }
inline int format_codebook_size(Format f){
    switch(f){
        case Format::Q1: case Format::Q1_K_L: case Format::Q1_K_M: case Format::Q1_K_H: case Format::Q1_GRP: case Format::Q1_K_L_GRP: case Format::Q1_K_M_GRP: case Format::Q1_K_H_GRP: return 1;
        case Format::Q2: case Format::Q2_K_L: case Format::Q2_K_M: case Format::Q2_K_H: case Format::Q2_GRP: case Format::Q2_K_L_GRP: case Format::Q2_K_M_GRP: case Format::Q2_K_H_GRP: return 4;
        case Format::Q3: case Format::Q3_K_L: case Format::Q3_K_M: case Format::Q3_K_H: case Format::Q3_GRP: case Format::Q3_K_L_GRP: case Format::Q3_K_M_GRP: case Format::Q3_K_H_GRP: return 8;
        case Format::Q4: case Format::Q4_K_L: case Format::Q4_K_M: case Format::Q4_K_H: case Format::Q4_GRP: case Format::Q4_K_L_GRP: case Format::Q4_K_M_GRP: case Format::Q4_K_H_GRP: return 16;
        case Format::Q6: case Format::Q6_K_L: case Format::Q6_K_M: case Format::Q6_K_H: case Format::Q6_GRP: case Format::Q6_K_L_GRP: case Format::Q6_K_M_GRP: case Format::Q6_K_H_GRP: return 64;
        case Format::Q8: case Format::Q8_K_L: case Format::Q8_K_M: case Format::Q8_K_H: case Format::Q8_GRP: case Format::Q8_K_L_GRP: case Format::Q8_K_M_GRP: case Format::Q8_K_H_GRP: return 256;
        case Format::Q12: case Format::Q12_K_L: case Format::Q12_K_M: case Format::Q12_K_H: case Format::Q12_GRP: case Format::Q12_K_L_GRP: case Format::Q12_K_M_GRP: case Format::Q12_K_H_GRP: return 4096;
        default: return 0;
    }
}
enum class DType : uint8_t { I64,I32,U8,U4,U16,F16,F32 };
inline size_t dtype_size(DType dt){ switch(dt){case DType::I64:return 8;case DType::I32:return 4;case DType::U8:return 1;case DType::U4:return 1;case DType::U16:return 2;case DType::F16:return 2;case DType::F32:return 4;default:return 0;}}
inline DType format_to_dtype(Format f){
    switch(f){
        case Format::Q1: case Format::Q1_K_L: case Format::Q1_K_M: case Format::Q1_K_H: case Format::Q1_GRP: case Format::Q1_K_L_GRP: case Format::Q1_K_M_GRP: case Format::Q1_K_H_GRP: return DType::U8;
        case Format::Q2: case Format::Q2_K_L: case Format::Q2_K_M: case Format::Q2_K_H: case Format::Q2_GRP: case Format::Q2_K_L_GRP: case Format::Q2_K_M_GRP: case Format::Q2_K_H_GRP: return DType::U8;
        case Format::Q3: case Format::Q3_K_L: case Format::Q3_K_M: case Format::Q3_K_H: case Format::Q3_GRP: case Format::Q3_K_L_GRP: case Format::Q3_K_M_GRP: case Format::Q3_K_H_GRP: return DType::U8;
        case Format::Q4: case Format::Q4_K_L: case Format::Q4_K_M: case Format::Q4_K_H: case Format::Q4_GRP: case Format::Q4_K_L_GRP: case Format::Q4_K_M_GRP: case Format::Q4_K_H_GRP: return DType::U4;
        case Format::Q6: case Format::Q6_K_L: case Format::Q6_K_M: case Format::Q6_K_H: case Format::Q6_GRP: case Format::Q6_K_L_GRP: case Format::Q6_K_M_GRP: case Format::Q6_K_H_GRP: return DType::U8;
        case Format::Q8: case Format::Q8_K_L: case Format::Q8_K_M: case Format::Q8_K_H: case Format::Q8_GRP: case Format::Q8_K_L_GRP: case Format::Q8_K_M_GRP: case Format::Q8_K_H_GRP: return DType::U8;
        case Format::Q12: case Format::Q12_K_L: case Format::Q12_K_M: case Format::Q12_K_H: case Format::Q12_GRP: case Format::Q12_K_L_GRP: case Format::Q12_K_M_GRP: case Format::Q12_K_H_GRP: return DType::U16;
        case Format::Q16: case Format::Q16_K_L: case Format::Q16_K_M: case Format::Q16_K_H: case Format::Q16_GRP: case Format::Q16_K_L_GRP: case Format::Q16_K_M_GRP: case Format::Q16_K_H_GRP: return DType::F16;
        case Format::Q24: case Format::Q24_K_L: case Format::Q24_K_M: case Format::Q24_K_H: case Format::Q24_GRP: case Format::Q24_K_L_GRP: case Format::Q24_K_M_GRP: case Format::Q24_K_H_GRP: return DType::U8;
        case Format::Q32: case Format::Q32_K_L: case Format::Q32_K_M: case Format::Q32_K_H: case Format::Q32_GRP: case Format::Q32_K_L_GRP: case Format::Q32_K_M_GRP: case Format::Q32_K_H_GRP: return DType::F32;
        default: return DType::U8;
    }
}
struct Shape{int64_t dims[8];int rank;Shape():rank(0){dims[0]=dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;}explicit Shape(int64_t d0):rank(1){dims[0]=d0;dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;}Shape(int64_t d0,int64_t d1):rank(2){dims[0]=d0;dims[1]=d1;dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;}Shape(int64_t d0,int64_t d1,int64_t d2):rank(3){dims[0]=d0;dims[1]=d1;dims[2]=d2;dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;}Shape(std::initializer_list<int64_t> l):rank((int)l.size()){dims[0]=dims[1]=dims[2]=dims[3]=dims[4]=dims[5]=dims[6]=dims[7]=0;if(rank>8)throw std::runtime_error("Shape: rank exceeds maximum of 8");int i=0;for(auto x:l)dims[i++]=x;}int64_t& operator[](int i){if(i<0||i>=rank)throw std::out_of_range("Shape index out of range");return dims[i];}const int64_t& operator[](int i) const{if(i<0||i>=rank)throw std::out_of_range("Shape index out of range");return dims[i];}int64_t numel() const{int64_t n=1;for(int i=0;i<rank;i++)n*=dims[i];return n;}bool operator==(const Shape& o) const{if(rank!=o.rank)return false;for(int i=0;i<rank;i++)if(dims[i]!=o.dims[i])return false;return true;}bool operator!=(const Shape& o) const{return !(*this==o);}std::string to_string() const{std::string s="[";for(int i=0;i<rank;i++){if(i)s+=",";s+=std::to_string(dims[i]);}s+="]";return s;}};
struct Status{bool ok;std::string msg;Status():ok(true){}Status(const std::string& e):ok(false),msg(e){}static Status error(const std::string& m){return Status(m);}static Status success(){return Status();}explicit operator bool() const{return ok;}};
struct Config{int num_threads=1;uint64_t seed=42;size_t pool_size=64*1024*1024;bool verbose=false;};
class Error:public std::runtime_error{public:explicit Error(const std::string& m):std::runtime_error(m){}};
#ifdef QUANT_THROW_ABORT
#define QUANT_CHECK(cond,msg) do{if(!(cond)){fprintf(stderr,"QUANT_CHECK FAIL: %s\n",std::string(msg).c_str());fflush(stderr);std::abort();}}while(0)
#else
#define QUANT_CHECK(cond,msg) do{if(!(cond))throw quant::Error(msg);}while(0)
#endif
} // namespace quant
