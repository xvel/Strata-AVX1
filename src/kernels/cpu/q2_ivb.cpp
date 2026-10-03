// src/kernels/cpu/q2_ivb.cpp - Ivy Bridge tier Q2_0 kernels. Read q2_ivb.hpp first.
//
// COMPILE FLAGS ARE THE POINT: this TU must build with `-mssse3` (GCC/Clang) and NO
// /arch flag (MSVC) - i.e. nothing newer than SSSE3 may appear. `grep -c '_mm256_('`
// on this file must print 0 (no 256-bit intrinsics at all). The pooled dispatch (expert_layout.cpp) only calls it when the CPU
// passes cpu_ivb_ok() or STRATA_FORCE_IVB=1 is set for tests.
//
// The integer dot follows ggml's Q2_0 idiom (MIT, (c) 2023-2026 the ggml authors,
// ggml/src/ggml-cpu/arch/x86/quants.c + quants.c ggml_vec_dot_q2_0_q8_0), as adapted in
// expert.cpp and q2_avx2.cpp: `_mm_maddubs_epi16` (codes 0..3 UNSIGNED against the int8
// activation) then `_mm_madd_epi16` against ones. Bounds: 16 codes x 3 x 127 = 6096,
// no lane can overflow int16 (32767) or int32.
#include "strata/kernels/cpu/q2_ivb.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

// fp16 -> fp32, exact on all bit patterns including subnormals (same transcription as
// expert.cpp's h2f; the F16C intrinsic is deliberately NOT used so this TU needs no
// feature flag and matches the oracle on tiny scales).
inline float h2f_scalar(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// 16 code bytes -> 64 codes in value order, SSE2 only. The shift-and-mask is q2_avx2's
// unpack64 without its AVX2 recombination: c0..c3 hold code k of every byte; interleaving
// the byte pairs puts codes 4j..4j+3 of input bytes 4i..4i+3 into output group i.
inline void unpack64_sse(const uint8_t* in16, __m128i& g0, __m128i& g1, __m128i& g2, __m128i& g3) {
    const __m128i b = _mm_loadu_si128((const __m128i*) in16);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i l01 = _mm_unpacklo_epi8(c0, c1);   // codes 0,1 of bytes 0..7
    const __m128i h01 = _mm_unpackhi_epi8(c0, c1);   // codes 0,1 of bytes 8..15
    const __m128i l23 = _mm_unpacklo_epi8(c2, c3);   // codes 2,3 of bytes 0..7
    const __m128i h23 = _mm_unpackhi_epi8(c2, c3);   // codes 2,3 of bytes 8..15
    g0 = _mm_unpacklo_epi16(l01, l23);               // codes of bytes 0..3, in order
    g1 = _mm_unpackhi_epi16(l01, l23);               // codes of bytes 4..7, in order
    g2 = _mm_unpacklo_epi16(h01, h23);               // codes of bytes 8..11, in order
    g3 = _mm_unpackhi_epi16(h01, h23);               // codes of bytes 12..15, in order
}

// sum(codes[i] * q[i]) for 16 pairs. SSSE3 + SSE2 only; see the file header for bounds.
inline int32_t dot16(const __m128i& codes, const int8_t* q) {
    const __m128i qs = _mm_loadu_si128((const __m128i*) q);
    const __m128i s = _mm_madd_epi16(_mm_maddubs_epi16(codes, qs), _mm_set1_epi16(1));
    __m128i h = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0xB1));
    return _mm_cvtsi128_si32(h);
}

// One 64-weight canonical block pair: two 32-chunks against two activation chunks.
// Returns d*(s0*xs0 + s1*xs1) - d*(hx0+hx1); no FMA anywhere in this TU.
inline float block_pair(const uint8_t* cb16, float d, const ActQ& a, int chunk2) {
    __m128i g0, g1, g2, g3;
    unpack64_sse(cb16, g0, g1, g2, g3);
    const int8_t* q0 = a.q + (size_t) chunk2 * QKA;
    const int32_t s0 = dot16(g0, q0) + dot16(g1, q0 + 16);
    const int32_t s1 = dot16(g2, q0 + 32) + dot16(g3, q0 + 48);
    const float acc = d * a.scale[chunk2] * (float) s0 + d * a.scale[chunk2 + 1] * (float) s1;
    const float corr = d * (a.hx[chunk2] + a.hx[chunk2 + 1]);
    return acc - corr;
}

// Canonical blob row: `codes` holds nblocks*16 code bytes, `scales` nblocks fp16 scales.
// Two accumulators break the float dependency chain (out-of-order cores retire both).
inline float row_dot_canon(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    float acc0 = 0.f, acc1 = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const float v = block_pair(codes + (size_t) b * 16, h2f_scalar(scales + (size_t) b * 2), a, 2 * b);
        if (b & 1) acc1 += v;
        else acc0 += v;
    }
    return (acc0 + acc1);
}

// GGUF row: nblocks 18-byte blocks (fp16 scale + 16 code bytes).
inline float row_dot_gguf(const uint8_t* row, const ActQ& a, int nblocks) {
    float acc0 = 0.f, acc1 = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float v = block_pair(blk + 2, h2f_scalar(blk), a, 2 * b);
        if (b & 1) acc1 += v;
        else acc0 += v;
    }
    return (acc0 + acc1);
}

inline float silu_mul(float g, float u) { return (g / (1.f + std::exp(-g))) * u; }

}  // namespace

void act_quant_q8_1_ivb(const float* x, int n, ActQ& a) {
    // The legacy scalar contract from expert.cpp, moved not rewritten: amax, inv scale,
    // t + copysign(0.5,t) (== lround half-away, branchless), clamp. Bit-exact with it.
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + (size_t) k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) {
            const float v = xb[j] < 0.f ? -xb[j] : xb[j];
            if (v > amax) amax = v;
        }
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + (size_t) k * QKA;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

void q2_0_gguf_rows_multi_ivb(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                              float* const* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        for (int t = 0; t < nt; ++t) out[t][r] = row_dot_gguf(row, *a[t], nblocks);
    }
}

void s2_expert_gu_rows_ivb(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float g = row_dot_canon(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                                      blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot_canon(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                                      blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU);
        ff[r] = silu_mul(g, u);
    }
}

void s2_expert_down_rows_ivb(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        out[r] = row_dot_canon(blob + O_D_CODES + (size_t) r * ROW_D,
                               blob + O_D_SCALES + (size_t) r * SC_D * 2, a2, SC_D);
}

void s2_expert_vnni_q_ivb(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    s2_expert_gu_rows_ivb(blob, a1, ws.ff, 0, FF);
    act_quant_q8_1_ivb(ws.ff, FF, ws.a2);
    s2_expert_down_rows_ivb(blob, ws.a2, out, 0, H);
}

void s2_expert_gu_rows_multi_ivb(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff,
                                  int r0, int r1) {
    for (int t = 0; t < n_tokens; ++t) s2_expert_gu_rows_ivb(blob, *a1[t], ff[t], r0, r1);
}

void s2_expert_down_rows_multi_ivb(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out,
                                    int r0, int r1) {
    for (int t = 0; t < n_tokens; ++t) s2_expert_down_rows_ivb(blob, *a2[t], out[t], r0, r1);
}

}  // namespace strata::kernels::cpu
