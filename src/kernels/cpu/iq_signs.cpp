// src/kernels/cpu/iq_signs.cpp - the IQ2_XS sign-expansion table on baseline codegen.
//
// WHY THIS FILE EXISTS (Xeon E5-2697 v2, 0xc000001d at process start): the table used to be
// built by a file-scope `static const EvenSigns` constructor inside iq_avx2.cpp, which compiles
// with /arch:AVX2 (MSVC) / -mavx2 (GCC). That constructor is scalar C++, but the flag lets the
// compiler use AVX2 anywhere in the TU - including CRT startup before main() - and on a CPU
// without AVX2 that is an instant illegal-instruction death with no log output. Building the
// same 128 u64 here, in a TU with NO special flags, removes the whole hazard class: baseline
// x86-64 codegen cannot emit AVX2.
//
// The 128 source bytes are keven_signs_q2xs (ggml's arch/x86/quants.c, also vendored in
// third_party/ggml/ggml-common.h as ksigns_iq2xs), copied rather than included: that header's
// tables are TU-local `static const`, so including it here would also duplicate megabytes of
// grids. Drift is caught by signs_test (WSL harness, validated 01.10.2026: 128/128 match).
//
// The table is built on FIRST USE (magic static, thread-safe), not at load: even a baseline
// TU should not pay for what a Q2_0-only run never touches.
#include "strata/kernels/cpu/iq_avx2.hpp"

namespace strata::kernels::cpu {

namespace {
// clang-format off
const uint8_t kSigns[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
// clang-format on

struct EvenSigns {
    uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= (uint64_t) (((kSigns[i] >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
}  // namespace

const uint64_t* iq2xs_even_signs() {
    static const EvenSigns t;
    return t.v;
}

}  // namespace strata::kernels::cpu
