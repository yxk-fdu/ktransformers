#pragma once
#if defined(__aarch64__)
/* Independent prototype: native consecutive FP4 bytes + separate E8M0 scales.
 * No persistent repack, no allocation, no changes to installed kt-kernel. */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static const int8_t kt_native_lut[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static inline float half_scale(uint8_t e) {
    uint32_t bits=e<2 ? (uint32_t)0x00200000<<e : (uint32_t)(e-1)<<23;
    float f; memcpy(&f,&bits,4); return f;
}
static inline float activation_scale(const uint8_t *q) {
    __fp16 h; memcpy(&h,q,2); return (float)h;
}
static inline int32x4_t block_dot(const uint8_t *w,const uint8_t *q) {
    uint8x16_t b=vld1q_u8(w);
    int8x16_t table=vld1q_s8(kt_native_lut);
    int8x16_t lo=vqtbl1q_s8(table,vandq_u8(b,vdupq_n_u8(15)));
    int8x16_t hi=vqtbl1q_s8(table,vshrq_n_u8(b,4));
    /* Original K order. Matches the reference's two half-block dot groups. */
    int8x16_t a=vzip1q_s8(lo,hi),c=vzip2q_s8(lo,hi);
    return vdotq_s32(vdotq_s32(vdupq_n_s32(0),a,vld1q_s8((const int8_t*)q+2)),
                     c,vld1q_s8((const int8_t*)q+18));
}
static inline int kt_native_mxfp4_dot(int k,const uint8_t *w,const uint8_t *scales,const uint8_t *q,float *out) {
    if(k<=0 || k%32 || !w || !scales || !q || !out) return -1;
    int nb=k/32;
    /* 255 is reserved for NaN in E8M0. Reject explicitly. */
    for(int b=0;b<nb;b++) if(scales[b]==255) return -2;
    float32x4_t s0=vdupq_n_f32(0),s1=vdupq_n_f32(0);
    int b=0;
    for(;b+1<nb;b+=2) {
        s0=vfmaq_n_f32(s0,vcvtq_f32_s32(block_dot(w+16*b,q+34*b)),half_scale(scales[b])*activation_scale(q+34*b));
        s1=vfmaq_n_f32(s1,vcvtq_f32_s32(block_dot(w+16*(b+1),q+34*(b+1))),half_scale(scales[b+1])*activation_scale(q+34*(b+1)));
    }
    float result=vaddvq_f32(vaddq_f32(s0,s1));
    if(b<nb) {
        int sum=0;
        const int8_t *a=(const int8_t*)(q+34*b+2);
        for(int j=0;j<16;j++) sum+=kt_native_lut[w[16*b+j]&15]*a[2*j]+kt_native_lut[w[16*b+j]>>4]*a[2*j+1];
        result += (half_scale(scales[b])*activation_scale(q+34*b))*sum;
    }
    *out=result; return 0;
}

#elif defined(__x86_64__) || defined(_M_X64)
#include <stdint.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Read the original adjacent FP4 nibbles and separate E8M0 scales directly.
// No persistent repack or second weight allocation is needed on x86.
static const int8_t kt_native_lut[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static inline float half_scale(uint8_t e) {
    uint32_t bits = e < 2 ? (uint32_t)0x00200000 << e : (uint32_t)(e-1) << 23;
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}
static inline float activation_scale(const uint8_t* q) {
    uint16_t h; memcpy(&h, q, sizeof(h));
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 31, fraction = h & 1023;
    uint32_t bits;
    if (!exponent) {
        if (!fraction) bits = sign;
        else {
            int shift = 0;
            while (!(fraction & 1024)) { fraction <<= 1; ++shift; }
            bits = sign | uint32_t(113-shift) << 23 | (fraction & 1023) << 13;
        }
    } else if (exponent == 31) bits = sign | 0x7f800000 | fraction << 13;
    else bits = sign | (exponent+112) << 23 | fraction << 13;
    float f; memcpy(&f, &bits, sizeof(f)); return f;
#endif
}
static inline int32_t kt_native_block_dot(const uint8_t* w, const int8_t* q) {
#if defined(__AVX2__)
    const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kt_native_lut));
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w));
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i lo = _mm_shuffle_epi8(table, _mm_and_si128(packed, mask));
    const __m128i hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    const __m256i w0 = _mm256_cvtepi8_epi16(_mm_unpacklo_epi8(lo, hi));
    const __m256i w1 = _mm256_cvtepi8_epi16(_mm_unpackhi_epi8(lo, hi));
    const __m256i q0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q)));
    const __m256i q1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q+16)));
    const __m256i sum = _mm256_add_epi32(_mm256_madd_epi16(w0, q0), _mm256_madd_epi16(w1, q1));
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
#else
    int32_t sum = 0;
    for (int j=0; j<16; ++j)
        sum += kt_native_lut[w[j]&15]*q[2*j] + kt_native_lut[w[j]>>4]*q[2*j+1];
    return sum;
#endif
}
static inline int kt_native_mxfp4_dot(int k, const uint8_t* w, const uint8_t* scales,
                                     const uint8_t* q, float* out) {
    if (k<=0 || k%32 || !w || !scales || !q || !out) return -1;
    const int blocks = k/32;
    for (int b=0; b<blocks; ++b) if (scales[b]==255) return -2;
    float result = 0;
    for (int b=0; b<blocks; ++b) {
        const int32_t dot = kt_native_block_dot(w+16*b, reinterpret_cast<const int8_t*>(q+34*b+2));
        result += (half_scale(scales[b])*activation_scale(q+34*b))*dot;
    }
    *out=result; return 0;
}
#else
static inline int kt_native_mxfp4_dot(int, const unsigned char*, const unsigned char*,
                                    const unsigned char*, float*) { return -3; }
#endif
