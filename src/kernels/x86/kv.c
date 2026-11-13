/*
 * kv.c — per-row quantize/dequant kernels for the KV cache.
 *
 * The engine stores each KV row (one token, `kv_dim` elements) in the format
 * chosen at load time (`--kv-type`).  The quantized row layouts are the
 * project's WEIGHT layouts, so the weight-side kernels stay the reference:
 *
 *   f16  : 2 B/elem.  f32 -> f16 is round-to-nearest-even, the same
 *          algorithm as `k32_f32_to_f16bits` in kernels32.c (what the
 *          weight-side f16 path stores), and f16 -> f32 is exact
 *          (the mirror of `k32_f16_to_f32`).
 *
 *   q8_0 : 34 B per 32-element block: f16 scale d @0, 32 x int8 q @2;
 *          y[k] = d * q[k].  Quantize: d32 = amax/127 (1 if amax==0),
 *          d = f16(d32), q[k] = clamp(round(x[k]/d32), -128, 127) —
 *          EXACTLY `quantize_q8_0` in kernels32.c (reused, not copied) and
 *          the dequant is `quant.dequantizeQ8_0` (it uses the f16-rounded
 *          d, not d32).
 *
 *   q4_0 : 18 B per 32-element block: f16 scale d @0, 16 packed-nibble
 *          bytes @2, byte j holds element j (low) and element 16+j (high);
 *          y = (q-8)*d.  Quantize: d32 = amax/8 (1 if amax==0),
 *          q[k] = clamp(round(x[k]/d32)+8, 0, 15).  Same layout as the
 *          weight q4_0 (`quant.dequantQ4_0`, quant_dequant.zig L78: 8 x 18 B
 *          per 256-elem row), so a KV row dequantizes with the weight-side
 *          kernels unchanged.
 *
 * Public API (exactly these — NOT static, no prefix):
 *   void q35_kv_quant_f16(int n, const float* x, uint8_t* out);
 *   void q35_kv_quant_q4_0(int n, const float* x, uint8_t* out);
 *   void q35_kv_dequant_f16(int n, const uint8_t* data, float* out);
 *   void q35_kv_dequant_q8_0(int n, const uint8_t* data, float* out);
 *   void q35_kv_dequant_q4_0(int n, const uint8_t* data, float* out);
 *
 * `n` is a multiple of 32 for the block formats, anything for f16.  The
 * q8_0 quantize is `quantize_q8_0` (kernels32.c); there is no q8_0 copy
 * here.  Every internal symbol is static and prefixed kv_ (sibling C files
 * in this directory are concatenated; collisions break the build).
 */
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* f16 <-> f32 bit conversions (exact; copied from kernels32.c's      */
/* k32_f16_to_f32 / k32_f32_to_f16bits, which this file's layouts     */
/* must match).                                                        */
/* ------------------------------------------------------------------ */

static inline float kv_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)(h & 0x3FFu);
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        /* f16 subnormal: value = mant * 2^-24; normalize mant into [512,1024) */
        uint32_t m = mant;
        uint32_t lz = 0;
        while (m < 0x200u) { m <<= 1; lz++; }
        /* value = (1 + (m-512)/512) * 2^(-15-lz) */
        out = sign | ((112u - lz) << 23) | ((m & 0x1FFu) << 14);
    } else if (exp == 31) {
        /* inf / nan */
        out = sign | 0x7F800000u | ((mant << 14) & 0x7FFFFFu);
    } else {
        /* normal: f32 exponent = f16 exponent + 112 */
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return *(const float *)&out;
}

static inline uint16_t kv_f32_to_f16bits(float x) {
    const uint32_t u = *(const uint32_t *)&x;
    const uint32_t sign = (uint32_t)((u & 0x80000000u) >> 16);
    const uint32_t e = (u >> 23) & 0xFFu;
    const uint32_t m = u & 0x7FFFFFu;
    if (e == 255u) return (uint16_t)(0x7C00u | sign);     /* inf / nan -> inf */
    if (e >= 143u) return (uint16_t)(0x7C00u | sign);     /* > 65504 -> inf   */
    if (e >= 113u) {
        /* normal f16: round 23-bit mantissa to 10 bits, round-to-nearest-even */
        uint32_t keep = m >> 13;
        const uint32_t drop = m & 0x1FFFu;
        const uint32_t up = (drop > 0x1000u) || (drop == 0x1000u && (keep & 1u));
        keep += up;
        uint32_t e16 = e - 112u;
        if (keep >= 0x400u) { keep = 0; e16 += 1; }
        if (e16 > 30u) return (uint16_t)(0x7C00u | sign); /* overflow -> inf */
        return (uint16_t)(sign | (e16 << 10) | keep);
    }
    /* e in [0,112]: f16 subnormal or zero.  k = RN((2^23+m) * 2^(e-126)) */
    {
        const uint32_t t = m | 0x800000u;
        const int s = (int)(126u - e);
        if (s >= 25) return (uint16_t)sign;               /* rounds to 0 */
        uint32_t keep = t >> s;
        const uint32_t drop = t & ((1u << s) - 1u);
        const uint32_t mid = 1u << (s - 1);
        const uint32_t up = (drop > mid) || (drop == mid && (keep & 1u));
        keep += up;
        if (keep >= 0x400u) return (uint16_t)(0x0380u | sign); /* rounds to 2^-14 */
        return (uint16_t)(sign | keep);
    }
}

/* ------------------------------------------------------------------ */

void q35_kv_quant_f16(int n, const float* x, uint8_t* out) {
    for (int i = 0; i < n; i++) {
        const uint16_t h = kv_f32_to_f16bits(x[i]);
        out[2 * i] = (uint8_t)(h & 0xFF);
        out[2 * i + 1] = (uint8_t)(h >> 8);
    }
}

void q35_kv_quant_q4_0(int n, const float* x, uint8_t* out) {
    const int nb = n / 32;
    for (int ib = 0; ib < nb; ib++) {
        const float* xb = x + ib * 32;
        uint8_t* yb = out + ib * 18;
        float amax = 0.0f;
        for (int k = 0; k < 32; k++) {
            const float a = xb[k] < 0.0f ? -xb[k] : xb[k];
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 8.0f : 1.0f;
        const float inv = amax > 0.0f ? 1.0f / scale : 0.0f;
        const uint16_t sbits = kv_f32_to_f16bits(scale);
        yb[0] = (uint8_t)(sbits & 0xFF);
        yb[1] = (uint8_t)(sbits >> 8);
        for (int k = 0; k < 16; k++) {
            /* byte k: element k in the low nibble, element 16+k in the high
             * nibble — the weight q4_0 packing (dequantQ4_0 reads it as
             * `qs[j]&0x0F` -> elem j, `qs[j]>>4` -> elem 16+j). */
            float v0 = xb[k] * inv;
            float v1 = xb[16 + k] * inv;
            int q0 = (int)(v0 + (v0 >= 0.0f ? 0.5f : -0.5f)) + 8;
            int q1 = (int)(v1 + (v1 >= 0.0f ? 0.5f : -0.5f)) + 8;
            if (q0 > 15) q0 = 15;
            if (q0 < 0) q0 = 0;
            if (q1 > 15) q1 = 15;
            if (q1 < 0) q1 = 0;
            yb[2 + k] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
}

void q35_kv_dequant_f16(int n, const uint8_t* data, float* out) {
    for (int i = 0; i < n; i++) {
        const uint16_t h = (uint16_t)(data[2 * i] | ((uint32_t)data[2 * i + 1] << 8));
        out[i] = kv_f16_to_f32(h);
    }
}

void q35_kv_dequant_q8_0(int n, const uint8_t* data, float* out) {
    const int nb = n / 32;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* yb = data + ib * 34;
        const uint16_t h = (uint16_t)(yb[0] | ((uint32_t)yb[1] << 8));
        const float d = kv_f16_to_f32(h);
        for (int k = 0; k < 32; k++) {
            out[ib * 32 + k] = d * (float)(int8_t)yb[2 + k];
        }
    }
}

void q35_kv_dequant_q4_0(int n, const uint8_t* data, float* out) {
    const int nb = n / 32;
    for (int ib = 0; ib < nb; ib++) {
        const uint8_t* yb = data + ib * 18;
        const uint16_t h = (uint16_t)(yb[0] | ((uint32_t)yb[1] << 8));
        const float d = kv_f16_to_f32(h);
        for (int k = 0; k < 16; k++) {
            const uint8_t b = yb[2 + k];
            out[ib * 32 + k] = ((float)((int)(b & 0x0F)) - 8.0f) * d;
            out[ib * 32 + 16 + k] = ((float)((int)(b >> 4)) - 8.0f) * d;
        }
    }
}
