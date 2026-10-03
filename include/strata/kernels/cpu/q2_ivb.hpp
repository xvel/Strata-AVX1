// include/strata/kernels/cpu/q2_ivb.hpp - Ivy Bridge tier Q2_0 CPU kernels.
//
// The third dispatch tier below AVX-512 and AVX2 (see expert_layout.hpp): for CPUs with
// SSE4.2 + AVX + F16C but WITHOUT AVX2/FMA (Intel Ivy Bridge, e.g. Xeon E5-2697 v2) - and,
// as a bonus, for anything with SSSE3, since this TU uses nothing newer.
//
// WHAT IT COVERS, and why only this:
//   * Q2_0 expert rows in both layouts the engine serves from the CPU: the canonical
//     Strata blob (separate codes/scales, tools/strata_pack.py) and the GGUF block layout
//     (18 B blocks, native Q2_0 packs with gu_type/d_type 42).
//   * The Q8 activation quantizer in the legacy scalar contract, bit-exact with the scalar
//     loop in expert.cpp (round-half-away, clamp [-127, 127]).
//   * i-quant experts need NO new kernel: native_expert.cpp already falls back to ggml-cpu's
//     vec_dot, which carries its own SSE paths. Only its AVX2 multi-token shortcuts must be
//     guarded (they fault here) - see native_expert.cpp.
//
// ISA BUDGET (everything else is a bug): scalar C++, SSE2 integer/float, SSSE3
// `_mm_maddubs_epi16`. No `_mm256*` of any kind (even float: YMM state is not assumed),
// no F16C intrinsic (scales convert through the exact scalar h2f), no FMA (mul+add are
// separate, so results agree with the AVX2 path to ~1 ulp, not bit-exact).
//
// The dot identity is the AVX2 kernel's (src/kernels/cpu/q2_avx2.cpp): per 32-value chunk
// sum(c*xhat) in integers, times weight-scale times chunk-scale, minus weight-scale times
// the chunk's hx (the -1 code offset). ggml's quants.c (MIT, (c) 2023-2026 ggml authors)
// is the reference for the maddubs+madd idiom, as in expert.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/kernels/cpu/expert.hpp"

namespace strata::kernels::cpu {

/// Legacy-contract Q8 activation quantizer (`n` multiple of QKA, at most H).
void act_quant_q8_1_ivb(const float* x, int n, ActQ& a);

/// Q2_0 rows in the GGUF block layout: `w` holds rows of `nblocks` 18-byte blocks
/// (fp16 scale + 16 code bytes); `a[t]` are ivb-quantized activations.
void q2_0_gguf_rows_multi_ivb(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                              float* const* out, int r0, int r1);

/// One canonical-blob expert: `out(H) = down(silu(gate(x)) * up(x))`, all Q2_0.
void s2_expert_vnni_q_ivb(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws);

/// Canonical-blob row ranges (pool split phases, bitwise the single expert above).
void s2_expert_gu_rows_ivb(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1);
void s2_expert_down_rows_ivb(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1);

/// Multi-token row ranges: each token bitwise its single-token rows (loops tokens;
/// the 1.38 MB blob stays L3-resident, so no DRAM is re-read).
void s2_expert_gu_rows_multi_ivb(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff,
                                  int r0, int r1);
void s2_expert_down_rows_multi_ivb(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out,
                                    int r0, int r1);

}  // namespace strata::kernels::cpu
