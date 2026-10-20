#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
namespace kt_match_npu {
inline bool enabled() { static const bool value = std::getenv("KT_MOE_MATCH_NPU_MXFP8") != nullptr; return value; }
inline float bf16(float x) {
 uint32_t u; std::memcpy(&u,&x,4); u += 0x7fff + ((u >> 16) & 1); u &= 0xffff0000u;
 std::memcpy(&x,&u,4); return x;
}
inline void quant_dequant(float* values, int count) {
 if (!enabled()) return;
 for (int base=0; base<count; base+=32) {
  int end=std::min(base+32,count); float maximum=0;
  for(int i=base;i<end;++i) maximum=std::max(maximum,std::fabs(values[i]));
  if(maximum==0) continue;
  int e; std::frexp(maximum,&e); int scale_exp=std::max(-127,std::min(127,e-9));
  float scale=std::ldexp(1.0f,scale_exp);
  for(int i=base;i<end;++i) {
   float x=std::min(448.0f,std::fabs(values[i])/scale);
   int xe; std::frexp(x,&xe); float unit=std::ldexp(1.0f,std::max(-9,xe-4));
   float q=std::nearbyint(x/unit)*unit;
   values[i]=std::copysign(q*scale,values[i]);
  }
 }
}
inline float dot_gguf_mxfp4_mxfp8(const uint8_t* weights, const float* input, int blocks) {
 static const int8_t table_values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
#if defined(__AVX512BF16__) && defined(__AVX512F__)
 const __m128i table=_mm_loadu_si128(reinterpret_cast<const __m128i*>(table_values));
 const __m128i mask=_mm_set1_epi8(15);
 __m512 sum=_mm512_setzero_ps();
 for(int b=0;b<blocks;++b) {
  const uint8_t* w=weights+b*17;
  __m128i packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+1));
  __m128i lo=_mm_shuffle_epi8(table,_mm_and_si128(packed,mask));
  __m128i hi=_mm_shuffle_epi8(table,_mm_and_si128(_mm_srli_epi16(packed,4),mask));
  __m512 scale=_mm512_set1_ps(std::ldexp(1.0f,int(w[0])-128));
  __m512 wl=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)),scale);
  __m512 wh=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)),scale);
  __m512bh wb=_mm512_cvtne2ps_pbh(wh,wl);
  __m512bh xb=_mm512_cvtne2ps_pbh(_mm512_loadu_ps(input+b*32+16),_mm512_loadu_ps(input+b*32));
  sum=_mm512_dpbf16_ps(sum,wb,xb);
 }
 return _mm512_reduce_add_ps(sum);
#else
 float sum=0;
 for(int b=0;b<blocks;++b) {
  const uint8_t* w=weights+b*17;float scale=std::ldexp(1.0f,int(w[0])-128);
  for(int i=0;i<16;++i) {
   sum += float(table_values[w[1+i]&15])*scale*input[b*32+i];
   sum += float(table_values[w[1+i]>>4])*scale*input[b*32+16+i];
  }
 }
 return sum;
#endif
}
}
