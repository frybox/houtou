/* DSPINE code bridge (see dspine.c).  Declarations only -- the data lives in
 * the C unit, and the Zig oracle in `src/spec_ref.zig` is what these are
 * checked against. */
#ifndef Q35_DSPINE_H
#define Q35_DSPINE_H

#include <stdint.h>

uint16_t q35_f32_to_bf16_bits(float x);
float q35_f32_to_bf16(float x);

void q35_dspine_prepare_read(const float *x, const float *residual, float *raw,
                             float *normed, float *rms_out, long rows, long h,
                             float eps, int add);

void q35_dspine_normalize_codes(const float *projected, const float *codebook,
                                const int32_t *anchors, float *codes, long rows,
                                long c, long t, float eps, long row0);

void q35_dspine_feature_message(const float *packed, const float *packed_prev,
                                const float *bias, float *message, long rows,
                                long r, long t, long row0);

void q35_dspine_feature_finish(const float *raw, const float *delta,
                               const float *rms, const uint8_t *valid,
                               const float *weight, float *output,
                               float *normed, long rows, long h, long t,
                               float norm_eps, int write, int norm, long row0);

/* The tree reduction the device backend (`src/kernels/cuda/dspine.cu`
 * `block_reduce`) also uses.  Exposed so the two backends can be A/B'd on the
 * SAME order; see the long comment above `q35_dspine_sum_tree` for why this is
 * our own order rather than Triton's. */
float q35_dspine_sum_tree(const float *v, long n);
float q35_dspine_sum_sq(const float *v, long n);

#endif
