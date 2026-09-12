#include "quant/gpu_compute_cuda.h"
#include <string>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <cstring>
#include <cstdio>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace quant {
namespace gpu {

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

// BUGFIX (bug census, C4244 wave): PTX kernel args are 32-bit, but the API
// takes int64_t. Bare `uint32_t u = big` silently wraps past 4G. Central
// checked narrow: negative/over-range dims throw instead of wrapping.
namespace {
inline uint32_t checked_u32(int64_t v, const char* what) {
    if (v < 0 || v > (int64_t)UINT32_MAX)
        throw std::runtime_error(std::string("GPUComputeCuda: ") + what +
                                 " out of uint32 range");
    return (uint32_t)v;
}
} // namespace

typedef CUresult (*PFN_cuInit)(unsigned int);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*PFN_cuDeviceTotalMem)(size_t*, CUdevice);
typedef CUresult (*PFN_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*PFN_cuCtxDestroy)(CUcontext);
typedef CUresult (*PFN_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*PFN_cuModuleUnload)(CUmodule);
typedef CUresult (*PFN_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*PFN_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (*PFN_cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*PFN_cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*PFN_cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*PFN_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (*PFN_cuStreamDestroy)(CUstream);
typedef CUresult (*PFN_cuStreamSynchronize)(CUstream);
typedef CUresult (*PFN_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
typedef CUresult (*PFN_cuMemGetInfo)(size_t*, size_t*);
typedef CUresult (*PFN_cuMemHostRegister)(void*, size_t, unsigned int);
typedef CUresult (*PFN_cuMemHostUnregister)(void*);

static const char* CUDA_PTX_CODE = R"(
.version 6.5
.target sm_30
.address_size 64

.visible .entry gemm_kernel(
	.param .f32 alpha, .param .u64 A, .param .u64 B, .param .f32 beta, .param .u64 C,
	.param .u32 M, .param .u32 N, .param .u32 K) {
    .reg .u32 %tx, %ty, %bx, %by, %rCol, %rRow, %rM, %rN, %rK, %k, %aIdx, %bIdx, %cIdx;
    .reg .f32 %fA, %fB, %fSum, %fAlpha, %fBeta, %fC;
    .reg .pred %pExit, %pK;
    .reg .u64 %ptr, %ptrA, %ptrB, %ptrC, %offset;

    mov.u32 %tx, %tid.x; mov.u32 %ty, %tid.y;
    mov.u32 %bx, %ctaid.x; mov.u32 %by, %ctaid.y;
    mad.lo.u32 %rCol, %bx, 16, %tx;
    mad.lo.u32 %rRow, %by, 16, %ty;
    ld.param.u32 %rM, [M]; ld.param.u32 %rN, [N]; ld.param.u32 %rK, [K];
    setp.ge.u32 %pExit, %rCol, %rN; @%pExit bra L_EXIT;
    setp.ge.u32 %pExit, %rRow, %rM; @%pExit bra L_EXIT;

    mov.f32 %fSum, 0.0; mov.u32 %k, 0;
    ld.param.u64 %ptrA, [A]; ld.param.u64 %ptrB, [B];

L_LOOP:
    setp.ge.u32 %pK, %k, %rK; @%pK bra L_END_LOOP;
    mad.lo.u32 %aIdx, %rRow, %rK, %k;
    cvt.u64.u32 %offset, %aIdx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrA, %offset;
    ld.global.f32 %fA, [%ptr];
    mad.lo.u32 %bIdx, %k, %rN, %rCol;
    cvt.u64.u32 %offset, %bIdx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrB, %offset;
    ld.global.f32 %fB, [%ptr];
    fma.rn.f32 %fSum, %fA, %fB, %fSum;
    add.u32 %k, %k, 1;
    bra L_LOOP;
L_END_LOOP:
    ld.param.f32 %fAlpha, [alpha]; ld.param.f32 %fBeta, [beta]; ld.param.u64 %ptrC, [C];
    mul.f32 %fSum, %fSum, %fAlpha;
    mad.lo.u32 %cIdx, %rRow, %rN, %rCol;
    cvt.u64.u32 %offset, %cIdx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrC, %offset;
    setp.eq.f32 %pExit, %fBeta, 0.0; @%pExit bra L_STORE;
    ld.global.f32 %fC, [%ptr];
    fma.rn.f32 %fSum, %fC, %fBeta, %fSum;
L_STORE:
    st.global.f32 [%ptr], %fSum;
L_EXIT:
    ret;
}

.visible .entry relu_kernel(.param .u64 X, .param .u64 Y, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fX, %fZ; .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fX, [%ptr];
    mov.f32 %fZ, 0.0; max.f32 %fX, %fX, %fZ;
    ld.param.u64 %ptrY, [Y]; add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fX;
L_EXIT: ret;
}

.visible .entry silu_kernel(.param .u64 X, .param .u64 Y, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fX, %fZ; .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fX, [%ptr];
    mul.f32 %fZ, %fX, -1.44269504; ex2.approx.f32 %fZ, %fZ; add.f32 %fZ, %fZ, 1.0; rcp.approx.f32 %fZ, %fZ; mul.f32 %fX, %fX, %fZ;
    ld.param.u64 %ptrY, [Y]; add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fX;
L_EXIT: ret;
}

.visible .entry gelu_kernel(.param .u64 X, .param .u64 Y, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fX, %fZ; .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fX, [%ptr];
    mul.f32 %fZ, %fX, -2.4554649; ex2.approx.f32 %fZ, %fZ; add.f32 %fZ, %fZ, 1.0; rcp.approx.f32 %fZ, %fZ; mul.f32 %fX, %fX, %fZ;
    ld.param.u64 %ptrY, [Y]; add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fX;
L_EXIT: ret;
}

.visible .entry add_kernel(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fA, %fB; .reg .u64 %ptrA, %ptrB, %ptrC, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    ld.param.u64 %ptrA, [A]; add.u64 %ptr, %ptrA, %offset; ld.global.f32 %fA, [%ptr];
    ld.param.u64 %ptrB, [B]; add.u64 %ptr, %ptrB, %offset; ld.global.f32 %fB, [%ptr];
    add.f32 %fA, %fA, %fB;
    ld.param.u64 %ptrC, [C]; add.u64 %ptr, %ptrC, %offset; st.global.f32 [%ptr], %fA;
L_EXIT: ret;
}

.visible .entry mul_kernel(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fA, %fB; .reg .u64 %ptrA, %ptrB, %ptrC, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    ld.param.u64 %ptrA, [A]; add.u64 %ptr, %ptrA, %offset; ld.global.f32 %fA, [%ptr];
    ld.param.u64 %ptrB, [B]; add.u64 %ptr, %ptrB, %offset; ld.global.f32 %fB, [%ptr];
    mul.f32 %fA, %fA, %fB;
    ld.param.u64 %ptrC, [C]; add.u64 %ptr, %ptrC, %offset; st.global.f32 [%ptr], %fA;
L_EXIT: ret;
}

.visible .entry scale_kernel(.param .f32 S, .param .u64 X, .param .u64 Y, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fX, %fS; .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fX, [%ptr];
    ld.param.f32 %fS, [S]; mul.f32 %fX, %fX, %fS;
    ld.param.u64 %ptrY, [Y]; add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fX;
L_EXIT: ret;
}

.visible .entry fill_kernel(.param .f32 V, .param .u64 X, .param .u32 N) {
    .reg .u32 %idx, %rN; .reg .pred %p; .reg .f32 %fV; .reg .u64 %ptrX, %offset, %ptr;
    mov.u32 %idx, %ctaid.x; mov.u32 %rN, %ntid.x; mad.lo.u32 %idx, %idx, %rN, %tid.x;
    ld.param.u32 %rN, [N]; setp.ge.u32 %p, %idx, %rN; @%p bra L_EXIT;
    ld.param.u64 %ptrX, [X]; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.param.f32 %fV, [V]; st.global.f32 [%ptr], %fV;
L_EXIT: ret;
}

.visible .entry softmax_kernel(.param .u64 X, .param .u64 Y, .param .u32 rows, .param .u32 cols) {
    .reg .u32 %row, %col, %rRows, %rCols, %idx; .reg .pred %p; .reg .f32 %fMax, %fVal, %fSum, %fInv;
    .reg .u64 %ptrX, %ptrY, %offset, %ptr;
    mov.u32 %row, %ctaid.x; ld.param.u32 %rRows, [rows]; setp.ge.u32 %p, %row, %rRows; @%p bra L_EXIT;
    ld.param.u32 %rCols, [cols]; ld.param.u64 %ptrX, [X]; ld.param.u64 %ptrY, [Y];
    mov.u32 %col, 0; mov.f32 %fMax, -1000000.0;
L_MAX_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_MAX_END;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr]; max.f32 %fMax, %fMax, %fVal;
    add.u32 %col, %col, 1; bra L_MAX_LOOP;
L_MAX_END:
    mov.u32 %col, 0; mov.f32 %fSum, 0.0;
L_EXP_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_EXP_END;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr]; sub.f32 %fVal, %fVal, %fMax;
    mul.f32 %fVal, %fVal, 1.44269504; ex2.approx.f32 %fVal, %fVal;
    add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fVal; add.f32 %fSum, %fSum, %fVal;
    add.u32 %col, %col, 1; bra L_EXP_LOOP;
L_EXP_END:
    rcp.approx.f32 %fInv, %fSum; mov.u32 %col, 0;
L_NORM_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_EXIT;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrY, %offset; ld.global.f32 %fVal, [%ptr]; mul.f32 %fVal, %fVal, %fInv;
    st.global.f32 [%ptr], %fVal; add.u32 %col, %col, 1; bra L_NORM_LOOP;
L_EXIT: ret;
}

.visible .entry rmsnorm_kernel(.param .u64 X, .param .u64 W, .param .u64 Y, .param .u32 rows, .param .u32 cols, .param .f32 eps) {
    .reg .u32 %row, %col, %rRows, %rCols, %idx; .reg .pred %p; .reg .f32 %fSum, %fVal, %fW, %fEps, %fScale;
    .reg .u64 %ptrX, %ptrW, %ptrY, %offset, %ptr;
    mov.u32 %row, %ctaid.x; ld.param.u32 %rRows, [rows]; setp.ge.u32 %p, %row, %rRows; @%p bra L_EXIT;
    ld.param.u32 %rCols, [cols]; ld.param.u64 %ptrX, [X]; ld.param.u64 %ptrW, [W]; ld.param.u64 %ptrY, [Y];
    mov.u32 %col, 0; mov.f32 %fSum, 0.0;
L_SUM_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_SUM_END;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr]; fma.rn.f32 %fSum, %fVal, %fVal, %fSum;
    add.u32 %col, %col, 1; bra L_SUM_LOOP;
L_SUM_END:
    cvt.rn.f32.u32 %fScale, %rCols; rcp.approx.f32 %fScale, %fScale; mul.f32 %fSum, %fSum, %fScale;
    ld.param.f32 %fEps, [eps]; add.f32 %fSum, %fSum, %fEps; rsqrt.approx.f32 %fSum, %fSum;
    mov.u32 %col, 0;
L_NORM_LOOP:
    setp.ge.u32 %p, %col, %rCols; @%p bra L_EXIT;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr];
    cvt.u64.u32 %offset, %col; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrW, %offset; ld.global.f32 %fW, [%ptr];
    mul.f32 %fVal, %fVal, %fSum; mul.f32 %fVal, %fVal, %fW;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fVal; add.u32 %col, %col, 1; bra L_NORM_LOOP;
L_EXIT: ret;
}

.visible .entry rope_kernel(.param .u64 X, .param .u64 Q, .param .u32 seq_len, .param .u32 head_dim) {
    .reg .u32 %p_idx, %d_idx, %slen, %hdim, %idx1, %idx2, %x_idx;
    .reg .pred %p_exit, %p_loop;
    .reg .f32 %q1, %q2, %cos_val, %sin_val, %new_q1, %new_q2;
    .reg .u64 %ptrX, %ptrQ, %ptr1, %ptr2, %ptrC, %ptrS, %offset;

    mov.u32 %p_idx, %ctaid.x;
    ld.param.u32 %slen, [seq_len];
    setp.ge.u32 %p_exit, %p_idx, %slen;
    @%p_exit bra L_EXIT;

    ld.param.u32 %hdim, [head_dim];
    ld.param.u64 %ptrX, [X];
    ld.param.u64 %ptrQ, [Q];

    mov.u32 %d_idx, 0;
L_LOOP:
    setp.ge.u32 %p_loop, %d_idx, %hdim;
    @%p_loop bra L_EXIT;

    mad.lo.u32 %idx1, %p_idx, %hdim, %d_idx;
    cvt.u64.u32 %offset, %idx1; shl.b64 %offset, %offset, 2; add.u64 %ptr1, %ptrQ, %offset;
    ld.global.f32 %q1, [%ptr1];

    add.u32 %idx2, %idx1, 1;
    cvt.u64.u32 %offset, %idx2; shl.b64 %offset, %offset, 2; add.u64 %ptr2, %ptrQ, %offset;
    ld.global.f32 %q2, [%ptr2];

    cvt.u64.u32 %offset, %idx1; shl.b64 %offset, %offset, 2; add.u64 %ptrC, %ptrX, %offset;
    ld.global.f32 %cos_val, [%ptrC];
    add.u64 %ptrS, %ptrC, 4;
    ld.global.f32 %sin_val, [%ptrS];

    mul.f32 %new_q1, %q1, %cos_val;
    neg.f32 %sin_val, %sin_val;
    fma.rn.f32 %new_q1, %q2, %sin_val, %new_q1;
    neg.f32 %sin_val, %sin_val;

    mul.f32 %new_q2, %q1, %sin_val;
    fma.rn.f32 %new_q2, %q2, %cos_val, %new_q2;

    st.global.f32 [%ptr1], %new_q1;
    st.global.f32 [%ptr2], %new_q2;

    add.u32 %d_idx, %d_idx, 2;
    bra L_LOOP;

L_EXIT:
    ret;
}

.visible .entry attention_kernel(.param .u64 Q, .param .u64 K, .param .u64 V, .param .u64 O, .param .u32 seq_len, .param .u32 head_dim) {
    .reg .u32 %q_idx, %k_idx, %d_idx, %slen, %hdim, %idx;
    .reg .pred %p_exit, %p_k, %p_d;
    .reg .f32 %fQ, %fK, %fV, %score, %max_score, %sum_exp, %exp_score, %out_val, %scale;
    .reg .u64 %ptrQ, %ptrK, %ptrV, %ptrO, %ptr, %offset;

    mov.u32 %q_idx, %ctaid.x;
    ld.param.u32 %slen, [seq_len];
    setp.ge.u32 %p_exit, %q_idx, %slen;
    @%p_exit bra L_EXIT;

    ld.param.u32 %hdim, [head_dim];
    ld.param.u64 %ptrQ, [Q];
    ld.param.u64 %ptrK, [K];
    ld.param.u64 %ptrV, [V];
    ld.param.u64 %ptrO, [O];

    cvt.rn.f32.u32 %scale, %hdim;
    rsqrt.approx.f32 %scale, %scale;

    mov.f32 %max_score, -1000000.0;
    mov.u32 %k_idx, 0;
L_K1:
    setp.ge.u32 %p_k, %k_idx, %slen;
    @%p_k bra L_K1_END;
    
    mov.f32 %score, 0.0;
    mov.u32 %d_idx, 0;
L_D1:
    setp.ge.u32 %p_d, %d_idx, %hdim;
    @%p_d bra L_D1_END;
    
    mad.lo.u32 %idx, %q_idx, %hdim, %d_idx; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrQ, %offset;
    ld.global.f32 %fQ, [%ptr];
    
    mad.lo.u32 %idx, %k_idx, %hdim, %d_idx; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrK, %offset;
    ld.global.f32 %fK, [%ptr];
    
    fma.rn.f32 %score, %fQ, %fK, %score;
    add.u32 %d_idx, %d_idx, 1; bra L_D1;
L_D1_END:
    mul.f32 %score, %score, %scale;
    max.f32 %max_score, %max_score, %score;
    add.u32 %k_idx, %k_idx, 1; bra L_K1;
L_K1_END:

    mov.f32 %sum_exp, 0.0;
    mov.u32 %k_idx, 0;
L_K2:
    setp.ge.u32 %p_k, %k_idx, %slen;
    @%p_k bra L_K2_END;
    
    mov.f32 %score, 0.0;
    mov.u32 %d_idx, 0;
L_D2:
    setp.ge.u32 %p_d, %d_idx, %hdim;
    @%p_d bra L_D2_END;
    
    mad.lo.u32 %idx, %q_idx, %hdim, %d_idx; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrQ, %offset;
    ld.global.f32 %fQ, [%ptr];
    
    mad.lo.u32 %idx, %k_idx, %hdim, %d_idx; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrK, %offset;
    ld.global.f32 %fK, [%ptr];
    
    fma.rn.f32 %score, %fQ, %fK, %score;
    add.u32 %d_idx, %d_idx, 1; bra L_D2;
L_D2_END:
    mul.f32 %score, %score, %scale;
    sub.f32 %score, %score, %max_score;
    mul.f32 %score, %score, 1.44269504; ex2.approx.f32 %exp_score, %score;
    add.f32 %sum_exp, %sum_exp, %exp_score;
    add.u32 %k_idx, %k_idx, 1; bra L_K2;
L_K2_END:

    mov.u32 %d_idx, 0;
L_D3:
    setp.ge.u32 %p_d, %d_idx, %hdim;
    @%p_d bra L_D3_END;
    
    mov.f32 %out_val, 0.0;
    mov.u32 %k_idx, 0;
L_K3:
    setp.ge.u32 %p_k, %k_idx, %slen;
    @%p_k bra L_K3_END;
    
    mov.f32 %score, 0.0;
    .reg .u32 %d2;
    mov.u32 %d2, 0;
L_D4:
    setp.ge.u32 %p_exit, %d2, %hdim;
    @%p_exit bra L_D4_END;
    
    mad.lo.u32 %idx, %q_idx, %hdim, %d2; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrQ, %offset;
    ld.global.f32 %fQ, [%ptr];
    
    mad.lo.u32 %idx, %k_idx, %hdim, %d2; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrK, %offset;
    ld.global.f32 %fK, [%ptr];
    
    fma.rn.f32 %score, %fQ, %fK, %score;
    add.u32 %d2, %d2, 1; bra L_D4;
L_D4_END:
    mul.f32 %score, %score, %scale;
    sub.f32 %score, %score, %max_score;
    mul.f32 %score, %score, 1.44269504; ex2.approx.f32 %exp_score, %score;
    div.approx.f32 %exp_score, %exp_score, %sum_exp;
    
    mad.lo.u32 %idx, %k_idx, %hdim, %d_idx; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrV, %offset;
    ld.global.f32 %fV, [%ptr];
    
    fma.rn.f32 %out_val, %exp_score, %fV, %out_val;
    add.u32 %k_idx, %k_idx, 1; bra L_K3;
L_K3_END:
    
    mad.lo.u32 %idx, %q_idx, %hdim, %d_idx; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrO, %offset;
    st.global.f32 [%ptr], %out_val;
    
    add.u32 %d_idx, %d_idx, 1; bra L_D3;
L_D3_END:

L_EXIT:
    ret;
}

.visible .entry reduce_sum_kernel(.param .u64 X, .param .u64 result, .param .u32 N) {
    .reg .u32 %tid, %rN, %stride;
    .reg .pred %p;
    .reg .f32 %val, %val2;
    .reg .u64 %ptrX, %ptrRes, %offset, %ptr;
    .shared .align 4 .b8 smem[4096];

    mov.u32 %tid, %tid.x;
    ld.param.u32 %rN, [N];
    
    mov.f32 %val, 0.0;
    setp.ge.u32 %p, %tid, %rN;
    @%p bra L_STORE_SMEM;
    
    ld.param.u64 %ptrX, [X];
    cvt.u64.u32 %offset, %tid; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset;
    ld.global.f32 %val, [%ptr];

L_STORE_SMEM:
    cvt.u64.u32 %offset, %tid; shl.b64 %offset, %offset, 2;
    mov.u64 %ptr, smem; add.u64 %ptr, %ptr, %offset;
    st.shared.f32 [%ptr], %val;
    bar.sync 0;

    mov.u32 %stride, %ntid.x;
    shr.u32 %stride, %stride, 1;
L_REDUCE:
    setp.eq.u32 %p, %stride, 0;
    @%p bra L_DONE;
    setp.ge.u32 %p, %tid, %stride;
    @%p bra L_SYNC;

    mov.u64 %ptr, smem; cvt.u64.u32 %offset, %tid; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptr, %offset;
    ld.shared.f32 %val, [%ptr];
    add.u32 %stride, %tid, %stride;
    cvt.u64.u32 %offset, %stride; shl.b64 %offset, %offset, 2; mov.u64 %ptr, smem; add.u64 %ptr, %ptr, %offset;
    ld.shared.f32 %val2, [%ptr];
    add.f32 %val, %val, %val2;
    
    cvt.u64.u32 %offset, %tid; shl.b64 %offset, %offset, 2; mov.u64 %ptr, smem; add.u64 %ptr, %ptr, %offset;
    st.shared.f32 [%ptr], %val;

    sub.u32 %stride, %stride, %tid;
L_SYNC:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra L_REDUCE;

L_DONE:
    setp.ne.u32 %p, %tid, 0;
    @%p bra L_EXIT;
    ld.param.u64 %ptrRes, [result];
    mov.u64 %ptr, smem; ld.shared.f32 %val, [%ptr];
    st.global.f32 [%ptrRes], %val;

L_EXIT:
    ret;
}

.visible .entry reduce_max_kernel(.param .u64 X, .param .u64 result, .param .u32 N) {
    .reg .u32 %tid, %rN, %stride;
    .reg .pred %p;
    .reg .f32 %val, %val2;
    .reg .u64 %ptrX, %ptrRes, %offset, %ptr;
    .shared .align 4 .b8 smem[4096];

    mov.u32 %tid, %tid.x;
    ld.param.u32 %rN, [N];
    
    mov.f32 %val, -1000000.0;
    setp.ge.u32 %p, %tid, %rN;
    @%p bra L_STORE_SMEM;
    
    ld.param.u64 %ptrX, [X];
    cvt.u64.u32 %offset, %tid; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset;
    ld.global.f32 %val, [%ptr];

L_STORE_SMEM:
    cvt.u64.u32 %offset, %tid; shl.b64 %offset, %offset, 2;
    mov.u64 %ptr, smem; add.u64 %ptr, %ptr, %offset;
    st.shared.f32 [%ptr], %val;
    bar.sync 0;

    mov.u32 %stride, %ntid.x;
    shr.u32 %stride, %stride, 1;
L_REDUCE:
    setp.eq.u32 %p, %stride, 0;
    @%p bra L_DONE;
    setp.ge.u32 %p, %tid, %stride;
    @%p bra L_SYNC;

    mov.u64 %ptr, smem; cvt.u64.u32 %offset, %tid; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptr, %offset;
    ld.shared.f32 %val, [%ptr];
    add.u32 %stride, %tid, %stride;
    cvt.u64.u32 %offset, %stride; shl.b64 %offset, %offset, 2; mov.u64 %ptr, smem; add.u64 %ptr, %ptr, %offset;
    ld.shared.f32 %val2, [%ptr];
    max.f32 %val, %val, %val2;
    
    cvt.u64.u32 %offset, %tid; shl.b64 %offset, %offset, 2; mov.u64 %ptr, smem; add.u64 %ptr, %ptr, %offset;
    st.shared.f32 [%ptr], %val;

    sub.u32 %stride, %stride, %tid;
L_SYNC:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra L_REDUCE;

L_DONE:
    setp.ne.u32 %p, %tid, 0;
    @%p bra L_EXIT;
    ld.param.u64 %ptrRes, [result];
    mov.u64 %ptr, smem; ld.shared.f32 %val, [%ptr];
    st.global.f32 [%ptrRes], %val;

L_EXIT:
    ret;
}

.visible .entry moe_gather_kernel(.param .u64 expert_outputs, .param .u64 routing_indices, .param .u64 output, .param .u32 num_tokens, .param .u32 expert_dim) {
    .reg .u32 %t, %d, %ntok, %edim, %route, %idx_in, %idx_out;
    .reg .pred %p;
    .reg .f32 %val;
    .reg .u64 %ptrEO, %ptrRI, %ptrOut, %ptr, %offset;
    
    mov.u32 %t, %ctaid.x;
    ld.param.u32 %ntok, [num_tokens];
    setp.ge.u32 %p, %t, %ntok;
    @%p bra L_EXIT;

    ld.param.u64 %ptrRI, [routing_indices];
    cvt.u64.u32 %offset, %t; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrRI, %offset;
    ld.global.u32 %route, [%ptr];

    ld.param.u32 %edim, [expert_dim];
    ld.param.u64 %ptrEO, [expert_outputs];
    ld.param.u64 %ptrOut, [output];

    mov.u32 %d, %tid.x;
L_LOOP:
    setp.ge.u32 %p, %d, %edim;
    @%p bra L_EXIT;

    mad.lo.u32 %idx_in, %route, %edim, %d;
    cvt.u64.u32 %offset, %idx_in; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrEO, %offset;
    ld.global.f32 %val, [%ptr];

    mad.lo.u32 %idx_out, %t, %edim, %d;
    cvt.u64.u32 %offset, %idx_out; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrOut, %offset;
    st.global.f32 [%ptr], %val;

    add.u32 %d, %d, %ntid.x;
    bra L_LOOP;

L_EXIT:
    ret;
}

.visible .entry moe_scatter_kernel(.param .u64 input, .param .u64 routing_indices, .param .u64 expert_inputs, .param .u32 num_tokens, .param .u32 token_dim) {
    .reg .u32 %t, %d, %ntok, %tdim, %route, %idx_in, %idx_out;
    .reg .pred %p;
    .reg .f32 %val;
    .reg .u64 %ptrIn, %ptrRI, %ptrEI, %ptr, %offset;
    
    mov.u32 %t, %ctaid.x;
    ld.param.u32 %ntok, [num_tokens];
    setp.ge.u32 %p, %t, %ntok;
    @%p bra L_EXIT;

    ld.param.u64 %ptrRI, [routing_indices];
    cvt.u64.u32 %offset, %t; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrRI, %offset;
    ld.global.u32 %route, [%ptr];

    ld.param.u32 %tdim, [token_dim];
    ld.param.u64 %ptrIn, [input];
    ld.param.u64 %ptrEI, [expert_inputs];

    mov.u32 %d, %tid.x;
L_LOOP:
    setp.ge.u32 %p, %d, %tdim;
    @%p bra L_EXIT;

    mad.lo.u32 %idx_in, %t, %tdim, %d;
    cvt.u64.u32 %offset, %idx_in; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrIn, %offset;
    ld.global.f32 %val, [%ptr];

    mad.lo.u32 %idx_out, %route, %tdim, %d;
    cvt.u64.u32 %offset, %idx_out; shl.b64 %offset, %offset, 2; add.u64 %ptr, %ptrEI, %offset;
    st.global.f32 [%ptr], %val;

    add.u32 %d, %d, %ntid.x;
    bra L_LOOP;

L_EXIT:
    ret;
}

.visible .entry layernorm_kernel(.param .u64 X, .param .u64 Gamma, .param .u64 Beta, .param .u64 Y, .param .u32 rows, .param .u32 cols, .param .f32 eps) {
    .reg .u32 %row, %col, %rRows, %rCols, %idx;
    .reg .pred %p;
    .reg .f32 %fSum, %fMean, %fVar, %fVal, %fG, %fB, %fEps, %fScale;
    .reg .u64 %ptrX, %ptrG, %ptrB, %ptrY, %offset, %ptr;

    mov.u32 %row, %ctaid.x;
    ld.param.u32 %rRows, [rows];
    setp.ge.u32 %p, %row, %rRows;
    @%p bra L_EXIT;

    ld.param.u32 %rCols, [cols];
    ld.param.u64 %ptrX, [X];
    ld.param.u64 %ptrG, [Gamma];
    ld.param.u64 %ptrB, [Beta];
    ld.param.u64 %ptrY, [Y];

    mov.u32 %col, 0; mov.f32 %fSum, 0.0;
L_MEAN_LOOP:
    setp.ge.u32 %p, %col, %rCols;
    @%p bra L_MEAN_END;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr]; add.f32 %fSum, %fSum, %fVal;
    add.u32 %col, %col, 1; bra L_MEAN_LOOP;
L_MEAN_END:
    cvt.rn.f32.u32 %fScale, %rCols; rcp.approx.f32 %fScale, %fScale;
    mul.f32 %fMean, %fSum, %fScale;

    mov.u32 %col, 0; mov.f32 %fSum, 0.0;
L_VAR_LOOP:
    setp.ge.u32 %p, %col, %rCols;
    @%p bra L_VAR_END;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr];
    sub.f32 %fVal, %fVal, %fMean;
    fma.rn.f32 %fSum, %fVal, %fVal, %fSum;
    add.u32 %col, %col, 1; bra L_VAR_LOOP;
L_VAR_END:
    mul.f32 %fVar, %fSum, %fScale;
    ld.param.f32 %fEps, [eps]; add.f32 %fVar, %fVar, %fEps; rsqrt.approx.f32 %fVar, %fVar;

    mov.u32 %col, 0;
L_NORM_LOOP:
    setp.ge.u32 %p, %col, %rCols;
    @%p bra L_EXIT;
    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrX, %offset; ld.global.f32 %fVal, [%ptr];
    sub.f32 %fVal, %fVal, %fMean;
    mul.f32 %fVal, %fVal, %fVar;

    cvt.u64.u32 %offset, %col; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrG, %offset; ld.global.f32 %fG, [%ptr];
    add.u64 %ptr, %ptrB, %offset; ld.global.f32 %fB, [%ptr];

    fma.rn.f32 %fVal, %fVal, %fG, %fB;

    mad.lo.u32 %idx, %row, %rCols, %col; cvt.u64.u32 %offset, %idx; shl.b64 %offset, %offset, 2;
    add.u64 %ptr, %ptrY, %offset; st.global.f32 [%ptr], %fVal;
    
    add.u32 %col, %col, 1; bra L_NORM_LOOP;
L_EXIT:
    ret;
}
)";

struct GPUComputeCuda::Impl {
    bool ok = false;
    void* lib = nullptr;

    PFN_cuInit cuInit = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDeviceTotalMem cuDeviceTotalMem = nullptr;
    PFN_cuCtxCreate cuCtxCreate = nullptr;
    PFN_cuCtxDestroy cuCtxDestroy = nullptr;
    PFN_cuModuleLoadData cuModuleLoadData = nullptr;
    PFN_cuModuleUnload cuModuleUnload = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuMemAlloc cuMemAlloc = nullptr;
    PFN_cuMemFree cuMemFree = nullptr;
    PFN_cuMemcpyHtoDAsync cuMemcpyHtoDAsync = nullptr;
    PFN_cuMemcpyDtoHAsync cuMemcpyDtoHAsync = nullptr;
    PFN_cuMemcpyDtoDAsync cuMemcpyDtoDAsync = nullptr;
    PFN_cuStreamCreate cuStreamCreate = nullptr;
    PFN_cuStreamDestroy cuStreamDestroy = nullptr;
    PFN_cuStreamSynchronize cuStreamSynchronize = nullptr;
    PFN_cuLaunchKernel cuLaunchKernel = nullptr;

    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUstream compute_stream;
    CUstream copy_stream;

    CUfunction f_gemm, f_relu, f_silu, f_gelu, f_add, f_mul, f_scale, f_fill, f_softmax, f_rmsnorm;
    CUfunction f_layernorm, f_rope, f_attention, f_reduce_sum, f_reduce_max, f_moe_gather, f_moe_scatter;
    PFN_cuMemGetInfo cuMemGetInfo = nullptr;
    PFN_cuMemHostRegister cuMemHostRegister = nullptr;
    PFN_cuMemHostUnregister cuMemHostUnregister = nullptr;
    
    std::vector<CUdeviceptr> bufs;
    std::mutex mtx;

    void* load_func(const char* name) {
#if defined(_WIN32)
        return (void*)GetProcAddress((HMODULE)lib, name);
#else
        return dlsym(lib, name);
#endif
    }

    bool load_lib() {
#if defined(_WIN32)
        lib = LoadLibraryA("nvcuda.dll");
#else
        lib = dlopen("libcuda.so", RTLD_NOW);
        if (!lib) lib = dlopen("libcuda.so.1", RTLD_NOW);
#endif
        return lib != nullptr;
    }

    bool init(int64_t device_id) {
        if (!load_lib()) return false;
        
        cuInit = (PFN_cuInit)load_func("cuInit");
        cuDeviceGet = (PFN_cuDeviceGet)load_func("cuDeviceGet");
        cuDeviceTotalMem = (PFN_cuDeviceTotalMem)load_func("cuDeviceTotalMem");
        cuCtxCreate = (PFN_cuCtxCreate)load_func("cuCtxCreate");
        cuCtxDestroy = (PFN_cuCtxDestroy)load_func("cuCtxDestroy");
        cuModuleLoadData = (PFN_cuModuleLoadData)load_func("cuModuleLoadData");
        cuModuleUnload = (PFN_cuModuleUnload)load_func("cuModuleUnload");
        cuModuleGetFunction = (PFN_cuModuleGetFunction)load_func("cuModuleGetFunction");
        cuMemAlloc = (PFN_cuMemAlloc)load_func("cuMemAlloc");
        cuMemFree = (PFN_cuMemFree)load_func("cuMemFree");
        cuMemcpyHtoDAsync = (PFN_cuMemcpyHtoDAsync)load_func("cuMemcpyHtoDAsync");
        cuMemcpyDtoHAsync = (PFN_cuMemcpyDtoHAsync)load_func("cuMemcpyDtoHAsync");
        cuMemcpyDtoDAsync = (PFN_cuMemcpyDtoDAsync)load_func("cuMemcpyDtoDAsync");
        cuStreamCreate = (PFN_cuStreamCreate)load_func("cuStreamCreate");
        cuStreamDestroy = (PFN_cuStreamDestroy)load_func("cuStreamDestroy");
        cuStreamSynchronize = (PFN_cuStreamSynchronize)load_func("cuStreamSynchronize");
        cuLaunchKernel = (PFN_cuLaunchKernel)load_func("cuLaunchKernel");
        cuMemGetInfo = (PFN_cuMemGetInfo)load_func("cuMemGetInfo");
        cuMemHostRegister = (PFN_cuMemHostRegister)load_func("cuMemHostRegister");
        cuMemHostUnregister = (PFN_cuMemHostUnregister)load_func("cuMemHostUnregister");

        if (!cuInit || !cuLaunchKernel) return false;

        if (cuInit(0) != CUDA_SUCCESS) return false;
        if (cuDeviceGet(&device, (int)device_id) != CUDA_SUCCESS) return false;
        if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) return false;
        
        if (cuStreamCreate(&compute_stream, 0) != CUDA_SUCCESS) return false;
        if (cuStreamCreate(&copy_stream, 0) != CUDA_SUCCESS) return false;

        if (cuModuleLoadData(&module, CUDA_PTX_CODE) != CUDA_SUCCESS) return false;
        
        cuModuleGetFunction(&f_gemm, module, "gemm_kernel");
        cuModuleGetFunction(&f_relu, module, "relu_kernel");
        cuModuleGetFunction(&f_silu, module, "silu_kernel");
        cuModuleGetFunction(&f_gelu, module, "gelu_kernel");
        cuModuleGetFunction(&f_add, module, "add_kernel");
        cuModuleGetFunction(&f_mul, module, "mul_kernel");
        cuModuleGetFunction(&f_scale, module, "scale_kernel");
        cuModuleGetFunction(&f_fill, module, "fill_kernel");
        cuModuleGetFunction(&f_softmax, module, "softmax_kernel");
        cuModuleGetFunction(&f_rmsnorm, module, "rmsnorm_kernel");
        cuModuleGetFunction(&f_layernorm, module, "layernorm_kernel");
        cuModuleGetFunction(&f_rope, module, "rope_kernel");
        cuModuleGetFunction(&f_attention, module, "attention_kernel");
        cuModuleGetFunction(&f_reduce_sum, module, "reduce_sum_kernel");
        cuModuleGetFunction(&f_reduce_max, module, "reduce_max_kernel");
        cuModuleGetFunction(&f_moe_gather, module, "moe_gather_kernel");
        cuModuleGetFunction(&f_moe_scatter, module, "moe_scatter_kernel");

        ok = true;
        return true;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lk(mtx);
        if (!ok) return;
        for (auto b : bufs) cuMemFree(b);
        bufs.clear();
        if (compute_stream) cuStreamDestroy(compute_stream);
        if (copy_stream) cuStreamDestroy(copy_stream);
        if (module) cuModuleUnload(module);
        if (context) cuCtxDestroy(context);
#if defined(_WIN32)
        if (lib) FreeLibrary((HMODULE)lib);
#else
        if (lib) dlclose(lib);
#endif
        ok = false;
    }
};

GPUComputeCuda::GPUComputeCuda() : impl_(new Impl()) {}
GPUComputeCuda::~GPUComputeCuda() { delete impl_; }

bool GPUComputeCuda::init(int64_t device_id) {
    if (impl_->ok) return true;
    return impl_->init(device_id);
}
bool GPUComputeCuda::is_initialized() const { return impl_->ok; }
void GPUComputeCuda::shutdown() { impl_->shutdown(); }

void* GPUComputeCuda::alloc(int64_t bytes) {
    if (!impl_->ok || bytes <= 0) return nullptr;
    CUdeviceptr ptr = 0;
    if (impl_->cuMemAlloc(&ptr, bytes) == CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->bufs.push_back(ptr);
        return (void*)ptr;
    }
    return nullptr;
}

void GPUComputeCuda::free_buf(void* ptr) {
    if (!impl_->ok || !ptr) return;
    CUdeviceptr dptr = (CUdeviceptr)ptr;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->cuMemFree(dptr);
    for (auto it = impl_->bufs.begin(); it != impl_->bufs.end(); ++it) {
        if (*it == dptr) {
            impl_->bufs.erase(it);
            break;
        }
    }
}

void GPUComputeCuda::upload(const Tensor& src, void* dst) {
    if (!impl_->ok || !dst || src.numel() == 0) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->cuMemcpyHtoDAsync((CUdeviceptr)dst, src.data(), src.size_bytes(), impl_->copy_stream);
    impl_->cuStreamSynchronize(impl_->copy_stream);
}

void GPUComputeCuda::download(void* src, Tensor& dst) {
    if (!impl_->ok || !src || dst.numel() == 0) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->cuMemcpyDtoHAsync(dst.data(), (CUdeviceptr)src, dst.size_bytes(), impl_->copy_stream);
    impl_->cuStreamSynchronize(impl_->copy_stream);
}

void GPUComputeCuda::gemm(float alpha, const void* A, const void* B, float beta, void* C, int64_t M, int64_t N, int64_t K) {
    if (!impl_->ok || !impl_->f_gemm) return;
    CUdeviceptr dA = (CUdeviceptr)A, dB = (CUdeviceptr)B, dC = (CUdeviceptr)C;
    uint32_t uM = checked_u32(M, "M"), uN = checked_u32(N, "N"), uK = checked_u32(K, "K");
    void* args[] = { &alpha, &dA, &dB, &beta, &dC, &uM, &uN, &uK };
    unsigned int gx = (unsigned int)((N + 15) / 16);
    unsigned int gy = (unsigned int)((M + 15) / 16);
    impl_->cuLaunchKernel(impl_->f_gemm, gx, gy, 1, 16, 16, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::gemv(float alpha, const void* A, const void* x, float beta, void* y, int64_t M, int64_t N) {
    gemm(alpha, A, x, beta, y, M, 1, N);
}

void GPUComputeCuda::relu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !impl_->f_relu) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &dX, &dY, &uN };
    impl_->cuLaunchKernel(impl_->f_relu, (unsigned int)((n + 255) / 256), 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::gelu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !impl_->f_gelu) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &dX, &dY, &uN };
    impl_->cuLaunchKernel(impl_->f_gelu, (unsigned int)((n + 255) / 256), 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::silu(const void* x, void* y, int64_t n) {
    if (!impl_->ok || !impl_->f_silu) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &dX, &dY, &uN };
    impl_->cuLaunchKernel(impl_->f_silu, (unsigned int)((n + 255) / 256), 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::softmax(const void* x, void* y, int64_t rows, int64_t cols) {
    if (!impl_->ok || !impl_->f_softmax) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uRows = checked_u32(rows, "rows"), uCols = checked_u32(cols, "cols");
    void* args[] = { &dX, &dY, &uRows, &uCols };
    impl_->cuLaunchKernel(impl_->f_softmax, checked_u32(rows, "rows"), 1, 1, 1, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::rms_norm(const void* x, const void* weight, void* y, int64_t rows, int64_t cols, float eps) {
    if (!impl_->ok || !impl_->f_rmsnorm) return;
    CUdeviceptr dX = (CUdeviceptr)x, dW = (CUdeviceptr)weight, dY = (CUdeviceptr)y;
    uint32_t uRows = checked_u32(rows, "rows"), uCols = checked_u32(cols, "cols");
    void* args[] = { &dX, &dW, &dY, &uRows, &uCols, &eps };
    impl_->cuLaunchKernel(impl_->f_rmsnorm, checked_u32(rows, "rows"), 1, 1, 1, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::layer_norm(const void* x, const void* gamma, const void* beta, void* y, int64_t rows, int64_t cols, float eps) {
    if (!impl_->ok || !impl_->f_layernorm) return;
    CUdeviceptr dX = (CUdeviceptr)x, dG = (CUdeviceptr)gamma, dB = (CUdeviceptr)beta, dY = (CUdeviceptr)y;
    uint32_t uRows = checked_u32(rows, "rows"), uCols = checked_u32(cols, "cols");
    void* args[] = { &dX, &dG, &dB, &dY, &uRows, &uCols, &eps };
    impl_->cuLaunchKernel(impl_->f_layernorm, checked_u32(rows, "rows"), 1, 1, 1, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::add(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok || !impl_->f_add) return;
    CUdeviceptr dA = (CUdeviceptr)a, dB = (CUdeviceptr)b, dC = (CUdeviceptr)c;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &dA, &dB, &dC, &uN };
    impl_->cuLaunchKernel(impl_->f_add, (unsigned int)((n + 255) / 256), 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::mul(const void* a, const void* b, void* c, int64_t n) {
    if (!impl_->ok || !impl_->f_mul) return;
    CUdeviceptr dA = (CUdeviceptr)a, dB = (CUdeviceptr)b, dC = (CUdeviceptr)c;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &dA, &dB, &dC, &uN };
    impl_->cuLaunchKernel(impl_->f_mul, (unsigned int)((n + 255) / 256), 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::scale(float s, const void* x, void* y, int64_t n) {
    if (!impl_->ok || !impl_->f_scale) return;
    CUdeviceptr dX = (CUdeviceptr)x, dY = (CUdeviceptr)y;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &s, &dX, &dY, &uN };
    impl_->cuLaunchKernel(impl_->f_scale, (unsigned int)((n + 255) / 256), 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::fill(float val, void* x, int64_t n) {
    if (!impl_->ok || !impl_->f_fill) return;
    CUdeviceptr dX = (CUdeviceptr)x;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &val, &dX, &uN };
    impl_->cuLaunchKernel(impl_->f_fill, (unsigned int)((n + 255) / 256), 1, 1, 256, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::copy_buf(const void* src, void* dst, int64_t n) {
    if (!impl_->ok || !src || !dst) return;
    impl_->cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, n * sizeof(float), impl_->compute_stream);
}

void GPUComputeCuda::rope(const void* x, void* q, int64_t seq_len, int64_t head_dim) {
    if (!impl_->ok || !impl_->f_rope) return;
    CUdeviceptr dX = (CUdeviceptr)x, dQ = (CUdeviceptr)q;
    uint32_t uS = checked_u32(seq_len, "seq_len"), uH = checked_u32(head_dim, "head_dim");
    void* args[] = { &dX, &dQ, &uS, &uH };
    impl_->cuLaunchKernel(impl_->f_rope, checked_u32(seq_len, "seq_len"), 1, 1, 1, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::attention(const void* q, const void* k, const void* v, void* o, int64_t seq_len, int64_t head_dim) {
    if (!impl_->ok || !impl_->f_attention) return;
    CUdeviceptr dQ = (CUdeviceptr)q, dK = (CUdeviceptr)k, dV = (CUdeviceptr)v, dO = (CUdeviceptr)o;
    uint32_t uS = checked_u32(seq_len, "seq_len"), uH = checked_u32(head_dim, "head_dim");
    void* args[] = { &dQ, &dK, &dV, &dO, &uS, &uH };
    impl_->cuLaunchKernel(impl_->f_attention, checked_u32(seq_len, "seq_len"), 1, 1, 1, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::reduce_sum(const void* x, void* result, int64_t n) {
    if (!impl_->ok || !impl_->f_reduce_sum) return;
    CUdeviceptr dX = (CUdeviceptr)x, dR = (CUdeviceptr)result;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &dX, &dR, &uN };
    uint32_t threads = (uint32_t)(n > 1024 ? 1024 : n);
    impl_->cuLaunchKernel(impl_->f_reduce_sum, 1, 1, 1, threads, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::reduce_max(const void* x, void* result, int64_t n) {
    if (!impl_->ok || !impl_->f_reduce_max) return;
    CUdeviceptr dX = (CUdeviceptr)x, dR = (CUdeviceptr)result;
    uint32_t uN = checked_u32(n, "n");
    void* args[] = { &dX, &dR, &uN };
    uint32_t threads = (uint32_t)(n > 1024 ? 1024 : n);
    impl_->cuLaunchKernel(impl_->f_reduce_max, 1, 1, 1, threads, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::moe_gather(const void* expert_outputs, const void* routing_indices, void* output, int64_t num_tokens, int64_t expert_dim) {
    if (!impl_->ok || !impl_->f_moe_gather) return;
    CUdeviceptr dEO = (CUdeviceptr)expert_outputs, dRI = (CUdeviceptr)routing_indices, dO = (CUdeviceptr)output;
    uint32_t uN = checked_u32(num_tokens, "num_tokens"), uD = checked_u32(expert_dim, "expert_dim");
    void* args[] = { &dEO, &dRI, &dO, &uN, &uD };
    uint32_t threads = (uint32_t)(expert_dim > 1024 ? 1024 : expert_dim);
    impl_->cuLaunchKernel(impl_->f_moe_gather, checked_u32(num_tokens, "num_tokens"), 1, 1, threads, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::moe_scatter(const void* input, const void* routing_indices, void* expert_inputs, int64_t num_tokens, int64_t token_dim) {
    if (!impl_->ok || !impl_->f_moe_scatter) return;
    CUdeviceptr dI = (CUdeviceptr)input, dRI = (CUdeviceptr)routing_indices, dEI = (CUdeviceptr)expert_inputs;
    uint32_t uN = checked_u32(num_tokens, "num_tokens"), uD = checked_u32(token_dim, "token_dim");
    void* args[] = { &dI, &dRI, &dEI, &uN, &uD };
    uint32_t threads = (uint32_t)(token_dim > 1024 ? 1024 : token_dim);
    impl_->cuLaunchKernel(impl_->f_moe_scatter, checked_u32(num_tokens, "num_tokens"), 1, 1, threads, 1, 1, 0, impl_->compute_stream, args, nullptr);
}

void GPUComputeCuda::synchronize() {
    if (impl_->ok) {
        impl_->cuStreamSynchronize(impl_->compute_stream);
        impl_->cuStreamSynchronize(impl_->copy_stream);
    }
}

bool GPUComputeCuda::register_host_memory(void* ptr, size_t bytes) {
    if (!impl_->ok || !impl_->cuMemHostRegister || !ptr || bytes == 0) return false;
    return impl_->cuMemHostRegister(ptr, bytes, 0) == CUDA_SUCCESS;
}

bool GPUComputeCuda::unregister_host_memory(void* ptr) {
    if (!impl_->ok || !impl_->cuMemHostUnregister || !ptr) return false;
    return impl_->cuMemHostUnregister(ptr) == CUDA_SUCCESS;
}

void* GPUComputeCuda::create_stream() {
    if (!impl_->ok || !impl_->cuStreamCreate) return nullptr;
    CUstream stream = nullptr;
    if (impl_->cuStreamCreate(&stream, 0) == CUDA_SUCCESS) {
        return (void*)stream;
    }
    return nullptr;
}

void GPUComputeCuda::destroy_stream(void* stream) {
    if (!impl_->ok || !impl_->cuStreamDestroy || !stream) return;
    impl_->cuStreamDestroy((CUstream)stream);
}

void GPUComputeCuda::synchronize_stream(void* stream) {
    if (!impl_->ok || !impl_->cuStreamSynchronize || !stream) return;
    impl_->cuStreamSynchronize((CUstream)stream);
}

void GPUComputeCuda::async_upload(const void* src, void* dst, size_t bytes, void* stream) {
    if (!impl_->ok || !impl_->cuMemcpyHtoDAsync) return;
    CUstream s = stream ? (CUstream)stream : impl_->copy_stream;
    impl_->cuMemcpyHtoDAsync((CUdeviceptr)dst, src, bytes, s);
}

void GPUComputeCuda::async_download(const void* src, void* dst, size_t bytes, void* stream) {
    if (!impl_->ok || !impl_->cuMemcpyDtoHAsync) return;
    CUstream s = stream ? (CUstream)stream : impl_->copy_stream;
    impl_->cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, bytes, s);
}

int64_t GPUComputeCuda::memory_free() const {
    if (!impl_->ok || !impl_->cuMemGetInfo) return 0;
    size_t free = 0, total = 0;
    impl_->cuMemGetInfo(&free, &total);
    return free;
}

int64_t GPUComputeCuda::memory_total() const {
    if (!impl_->ok) return 0;
    size_t mem = 0;
    impl_->cuDeviceTotalMem(&mem, impl_->device);
    return mem;
}

static GPUComputeCuda g_cuda_compute;
GPUComputeCuda& get_cuda_compute() { return g_cuda_compute; }

} // namespace gpu
} // namespace quant
