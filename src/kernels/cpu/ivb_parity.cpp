// src/kernels/cpu/ivb_parity.cpp - parity + bench for the Ivy Bridge tier kernels.
//
// Runs on any x86-64 with SSSE3 (gated below): no AVX2/AVX-512 kernel is referenced,
// so this is the test that passes ON a Xeon E5-2697 v2, where expert_parity and
// pool_test cannot even start (their reference kernels are AVX-512 machine code).
// What it proves, on synthetic data with realistic-small fp16 scales:
//   T1  act_quant_q8_1_ivb is bit-exact vs the double-precision round-half-away rule
//       (incl. the all-zero edge).
//   T2  GGUF Q2_0 rows match a long-double dot to float-rounding (worst ~2e-5 rel,
//       cancellation-limited on near-zero sums).
//   T3  a full canonical expert matches long double to 2e-3 abs (acc-vs-corr
//       cancellation; the AVX-512 kernel shows the same profile - see below).
//   T4  multi-token / partial-range rows are bitwise the single-token rows.
//   T5  bench: ms/expert and GB/s, single thread.
// Kernel-vs-kernel numbers (measured on AVX-512 hardware, not part of this binary):
// act ivb == avx2 bitwise; gguf rows avx2-vs-ivb 7e-5 rel; expert avx512-vs-ivb
// 7e-4 abs worst, 1e-4 mean, 0/19200 ff-quant flips.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/q2_ivb.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

namespace strata::kernels::cpu {
namespace {

int failures = 0;
#define CHECK(cond, ...)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            ++failures;                                    \
            std::printf("FAIL %d: ", __LINE__);             \
            std::printf(__VA_ARGS__);                      \
            std::printf("\n");                             \
        }                                                  \
    } while (0)

float frand() { return (float) rand() / (float) RAND_MAX * 2.f - 1.f; }

int round_away(double t) {
    if (t >= 0) return (int) std::floor(t + 0.5);
    return (int) std::ceil(t - 0.5);
}

// realistic Q2_0 scale: finite fp16, |d| in [2^-5, 1]
uint16_t rscale() {
    const uint16_t sign = (uint16_t) ((rand() & 1) << 15);
    const uint16_t exp = (uint16_t) ((10 + rand() % 5) << 10);
    const uint16_t man = (uint16_t) (rand() & 0x3FF);
    return (uint16_t) (sign | exp | man);
}

void rand_finite_row(uint8_t* row, int nblocks) {
    for (int b = 0; b < nblocks; ++b) {
        uint8_t* blk = row + (size_t) b * 18;
        const uint16_t h = rscale();
        std::memcpy(blk, &h, 2);
        for (int j = 0; j < 16; ++j) blk[2 + j] = (uint8_t) (rand() & 0xFF);
    }
}

bool feq(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return a == b;
}

bool host_ssse3() {
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, 1);
    return ((unsigned) x[2] >> 9) & 1u;
#else
    unsigned a, b, c, d;
    __cpuid_count(1, 0, a, b, c, d);
    (void) a;
    (void) b;
    (void) d;
    return (c >> 9) & 1u;
#endif
}

}  // namespace
}  // namespace strata::kernels::cpu

int main() {
    using namespace strata::kernels::cpu;
    if (!host_ssse3()) {
        std::printf("ivb_parity: SKIP (this CPU has no SSSE3)\n");
        return 0;
    }
    srand(42);

    // T1: act_quant bit-exact vs the double rule.
    for (int trial = 0; trial < 200; ++trial) {
        const int n = (trial % 3 == 0) ? FF : H;
        std::vector<float> x((size_t) n);
        for (int i = 0; i < n; ++i) {
            x[(size_t) i] = frand() * (trial % 5 == 0 ? 100.f : 1.f);
            if (trial % 7 == 0 && i < 4) x[(size_t) i] = 0.f;
        }
        if (trial % 11 == 0) std::fill(x.begin(), x.end(), 0.f);
        ActQ a;
        act_quant_q8_1_ivb(x.data(), n, a);
        CHECK(a.nchunks == n / QKA, "nchunks %d", a.nchunks);
        for (int k = 0; k < a.nchunks; ++k) {
            double amax = 0;
            for (int j = 0; j < QKA; ++j) {
                const double v = std::fabs((double) x[(size_t) k * QKA + j]);
                if (v > amax) amax = v;
            }
            const double s = amax > 0 ? amax / 127.0 : 0.0;
            const double inv = s > 0 ? 1.0 / s : 0.0;
            CHECK(a.scale[k] == (float) s, "trial %d chunk %d scale", trial, k);
            long sum = 0;
            for (int j = 0; j < QKA; ++j) {
                int v = round_away((double) x[(size_t) k * QKA + j] * inv);
                if (v < -127) v = -127;
                if (v > 127) v = 127;
                sum += v;
                CHECK(a.q[(size_t) k * QKA + j] == (int8_t) v, "trial %d q[%d]", trial, j);
            }
            CHECK(a.sum[k] == (int32_t) sum, "trial %d sum", trial);
            CHECK(a.hx[k] == a.scale[k] * (float) a.sum[k], "trial %d hx", trial);
        }
    }
    std::printf("T1 act_quant: done\n");

    // T2: GGUF rows vs long double.
    {
        double worst = 0;
        for (int trial = 0; trial < 100; ++trial) {
            const int nblocks = 40;
            std::vector<uint8_t> row((size_t) nblocks * 18);
            rand_finite_row(row.data(), nblocks);
            std::vector<float> xf(H);
            for (int i = 0; i < H; ++i) xf[(size_t) i] = frand();
            ActQ a;
            act_quant_q8_1_ivb(xf.data(), H, a);
            float out = -1.f;
            const ActQ* ap = &a;
            float* op = &out;
            q2_0_gguf_rows_multi_ivb(row.data(), 18, nblocks, &ap, 1, &op, 0, 1);
            long double ref = 0;
            for (int b = 0; b < nblocks; ++b) {
                const uint8_t* blk = row.data() + (size_t) b * 18;
                uint16_t h;
                std::memcpy(&h, blk, 2);
                const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
                long double d;
                if (exp == 0) d = man == 0 ? 0.0L : std::ldexp((long double) man, -24);
                else if (exp == 31) d = man == 0 ? HUGE_VALL : NAN;
                else d = std::ldexp((long double) (man | 0x400u), (int) exp - 25);
                if (sign) d = -d;
                for (int j = 0; j < 64; ++j) {
                    const int c = (blk[2 + (j >> 2)] >> (2 * (j & 3))) & 3;
                    const long double xhat = (long double) a.q[b * 64 + j] * (long double) a.scale[(2 * b) + (j >= 32)];
                    ref += ((long double) c - 1.0L) * d * xhat;
                }
            }
            const double rel = std::fabs((double) ref) > 1e-6 ? std::fabs(((double) out - (double) ref) / (double) ref)
                                                              : std::fabs((double) out - (double) ref);
            const double absd = std::fabs((double) out - (double) ref);
            if (rel > worst && absd > 1e-9) worst = rel;
            CHECK(rel < 1e-5 || absd < 5e-4, "trial %d rel %g abs %g", trial, rel, absd);
        }
        std::printf("T2 gguf row: worst rel %.3g\n", worst);
    }

    // T3: a full canonical expert vs long double (abs tolerance: acc-vs-corr
    // cancellation makes some rows ill-conditioned; kernel-vs-kernel agrees far tighter).
    {
        std::vector<uint8_t> blob(BLOB);
        for (int r = 0; r < 2 * FF; ++r) {
            uint8_t* c = blob.data() + O_GU_CODES + (size_t) r * ROW_GU;
            uint8_t* s = blob.data() + O_GU_SCALES + (size_t) r * SC_GU * 2;
            for (int j = 0; j < ROW_GU; ++j) c[j] = (uint8_t) (rand() & 0xFF);
            for (int b = 0; b < SC_GU; ++b) {
                const uint16_t h = rscale();
                std::memcpy(s + (size_t) b * 2, &h, 2);
            }
        }
        for (int r = 0; r < H; ++r) {
            uint8_t* c = blob.data() + O_D_CODES + (size_t) r * ROW_D;
            uint8_t* s = blob.data() + O_D_SCALES + (size_t) r * SC_D * 2;
            for (int j = 0; j < ROW_D; ++j) c[j] = (uint8_t) (rand() & 0xFF);
            for (int b = 0; b < SC_D; ++b) {
                const uint16_t h = rscale();
                std::memcpy(s + (size_t) b * 2, &h, 2);
            }
        }
        auto h2d = [](const uint8_t* p) -> long double {
            uint16_t h;
            std::memcpy(&h, p, 2);
            const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
            long double d;
            if (exp == 0) d = man == 0 ? 0.0L : std::ldexp((long double) man, -24);
            else if (exp == 31) d = man == 0 ? HUGE_VALL : NAN;
            else d = std::ldexp((long double) (man | 0x400u), (int) exp - 25);
            return sign ? -d : d;
        };
        double worst_abs = 0;
        for (int trial = 0; trial < 3; ++trial) {
            std::vector<float> xf(H);
            for (int i = 0; i < H; ++i) xf[(size_t) i] = frand() * 0.5f;
            ActQ a1;
            act_quant_q8_1_ivb(xf.data(), H, a1);
            std::vector<float> out(H, 0);
            ExpertScratch ws;
            s2_expert_vnni_q_ivb(blob.data(), a1, out.data(), ws);
            std::vector<long double> ff(FF);
            for (int r = 0; r < FF; ++r) {
                long double g = 0, u = 0;
                for (int b = 0; b < SC_GU; ++b) {
                    const long double dg = h2d(blob.data() + O_GU_SCALES + ((size_t) (2 * r) * SC_GU + b) * 2);
                    const long double du = h2d(blob.data() + O_GU_SCALES + ((size_t) (2 * r + 1) * SC_GU + b) * 2);
                    const uint8_t* gc = blob.data() + O_GU_CODES + (size_t) (2 * r) * ROW_GU + (size_t) b * 16;
                    const uint8_t* uc = blob.data() + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU + (size_t) b * 16;
                    for (int j = 0; j < 64; ++j) {
                        const int cg = (gc[j >> 2] >> (2 * (j & 3))) & 3;
                        const int cu = (uc[j >> 2] >> (2 * (j & 3))) & 3;
                        const int ch = 2 * b + (j >= 32);
                        g += ((long double) cg - 1) * dg * (long double) a1.q[b * 64 + j] * (long double) a1.scale[ch];
                        u += ((long double) cu - 1) * du * (long double) a1.q[b * 64 + j] * (long double) a1.scale[ch];
                    }
                }
                ff[(size_t) r] = (g / (1.0L + std::exp(-(double) g))) * u;
            }
            std::vector<int8_t> fq(FF);
            std::vector<double> fs((size_t) FF / QKA);
            for (int k = 0; k < FF / QKA; ++k) {
                double amax = 0;
                for (int j = 0; j < QKA; ++j) {
                    const double v = std::fabs((double) ff[(size_t) k * QKA + j]);
                    if (v > amax) amax = v;
                }
                const double s = amax > 0 ? amax / 127 : 0, inv = s > 0 ? 1 / s : 0;
                fs[(size_t) k] = s;
                for (int j = 0; j < QKA; ++j) {
                    int v = round_away((double) ff[(size_t) k * QKA + j] * inv);
                    if (v < -127) v = -127;
                    if (v > 127) v = 127;
                    fq[(size_t) k * QKA + j] = (int8_t) v;
                }
            }
            for (int r = 0; r < H; ++r) {
                long double acc = 0;
                for (int b = 0; b < SC_D; ++b) {
                    const long double d = h2d(blob.data() + O_D_SCALES + ((size_t) r * SC_D + b) * 2);
                    const uint8_t* dc = blob.data() + O_D_CODES + (size_t) r * ROW_D + (size_t) b * 16;
                    for (int j = 0; j < 64; ++j) {
                        const int c = (dc[j >> 2] >> (2 * (j & 3))) & 3;
                        const int ch = 2 * b + (j >= 32);
                        acc += ((long double) c - 1) * d * (long double) fq[b * 64 + j] * fs[(size_t) ch];
                    }
                }
                const double a = std::fabs(out[(size_t) r] - (double) acc);
                if (a > worst_abs) worst_abs = a;
            }
        }
        std::printf("T3 expert: worst abs %.3g\n", worst_abs);
        CHECK(worst_abs < 2e-3, "expert abs %g", worst_abs);
    }

    // T4: multi-token / partial-range rows bitwise the single-token rows.
    {
        std::vector<uint8_t> blob(BLOB);
        for (size_t i = 0; i < blob.size(); ++i) blob[i] = (uint8_t) (rand() & 0xFF);
        std::vector<float> xf(H);
        for (int i = 0; i < H; ++i) xf[(size_t) i] = frand();
        ActQ a1;
        act_quant_q8_1_ivb(xf.data(), H, a1);
        std::vector<float> ff1(FF, 0), ff2(FF, 0);
        s2_expert_gu_rows_ivb(blob.data(), a1, ff1.data(), 0, FF);
        const ActQ* ap = &a1;
        float* fp = ff2.data();
        s2_expert_gu_rows_multi_ivb(blob.data(), &ap, 1, &fp, 0, FF);
        for (int r = 0; r < FF; ++r) CHECK(feq(ff1[(size_t) r], ff2[(size_t) r]), "gu multi row %d", r);
        ActQ a2;
        act_quant_q8_1_ivb(ff1.data(), FF, a2);
        std::vector<float> o1(H, 0), o2(H, 0);
        s2_expert_down_rows_ivb(blob.data(), a2, o1.data(), 0, H);
        const ActQ* bp = &a2;
        float* op = o2.data();
        s2_expert_down_rows_multi_ivb(blob.data(), &bp, 1, &op, 0, H);
        for (int r = 0; r < H; ++r) CHECK(feq(o1[(size_t) r], o2[(size_t) r]), "down multi row %d", r);
        std::vector<float> ff3(FF, 0);
        s2_expert_gu_rows_ivb(blob.data(), a1, ff3.data(), 100, 500);
        for (int r = 100; r < 500; ++r) CHECK(feq(ff3[(size_t) r], ff1[(size_t) r]), "gu range row %d", r);
        std::printf("T4 rows/multi: done\n");
    }

    // T5: bench, single thread.
    {
        std::vector<uint8_t> blob(BLOB);
        for (size_t i = 0; i < blob.size(); ++i) blob[i] = (uint8_t) (rand() & 0xFF);
        std::vector<float> xf(H);
        for (int i = 0; i < H; ++i) xf[(size_t) i] = frand();
        ActQ a1;
        act_quant_q8_1_ivb(xf.data(), H, a1);
        std::vector<float> out(H);
        ExpertScratch ws;
        const int it = 20;
        const clock_t t0 = clock();
        for (int i = 0; i < it; ++i) s2_expert_vnni_q_ivb(blob.data(), a1, out.data(), ws);
        const clock_t t1 = clock();
        const double ms = 1000.0 * (t1 - t0) / CLOCKS_PER_SEC / it;
        std::printf("T5 bench: %.2f ms/expert single-thread (%.2f GB/s)\n", ms, (BLOB / 1e9) / (ms / 1e3));
    }

    if (failures == 0) std::printf("ivb_parity: ALL OK\n");
    else std::printf("ivb_parity: %d FAILURES\n", failures);
    return failures ? 1 : 0;
}
