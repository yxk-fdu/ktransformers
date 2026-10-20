#include "../cpu_backend/cpu_compute_trace.h"
#include <algorithm>
#include <vector>
#include <cstdio>
int main() {
  int64_t ids[] = {2, 7, -1, 7, 3};
  { kt::cpu_trace::Span warm(0, 0, 1, 5, ids, [](int64_t id){ return id < 0; }); }
  std::vector<double> samples;
  for (int i = 0; i < 1000; ++i) {
    auto a = kt::cpu_trace::now(CLOCK_MONOTONIC);
    { kt::cpu_trace::Span span(0, 0, 1, 5, ids, [](int64_t id){ return id < 0; }); }
    auto b = kt::cpu_trace::now(CLOCK_MONOTONIC);
    samples.push_back(double(b-a)/1000);
    usleep(1000);
  }
  std::sort(samples.begin(), samples.end());
  double sum = 0; for (auto v : samples) sum += v;
  printf("producer trace overhead: mean=%.3fus p50=%.3fus p90=%.3fus\n", sum/samples.size(), samples[500], samples[900]);
}
