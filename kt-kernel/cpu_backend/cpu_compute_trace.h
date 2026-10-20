#include <sys/resource.h>
#pragma once
// Actual CPU forward spans. Producers never perform file I/O or wait on the writer.
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>

namespace kt::cpu_trace {
inline uint64_t now(clockid_t id) {
  timespec ts{};
  clock_gettime(id, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}
struct Record {
  uint64_t wall_ns = 0, duration_ns = 0, coordinator_cpu_ns = 0;
  int transfer_count = 0;
  int layer = 0, part = 0, tokens = 0, count = 0, tid = 0;
  uint64_t task_id = 0, mono_ns = 0, raw_ns = 0;
  uint64_t thread_cpu_ns=0, trace_mutex_wait_ns=0;
  long nvcsw=0,nivcsw=0,minflt=0,majflt=0;
  int cpu = -1;
  const char* phase = nullptr;
  uint64_t expert_route_count = 0;
  bool truncated = false;
  std::array<int64_t, 256> experts{};
};
class Writer {
  static constexpr size_t CAPACITY = 16384;
  std::array<Record, CAPACITY> records_{};
  size_t head_ = 0, tail_ = 0, count_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopping_ = false;
  FILE* file_ = nullptr;
  std::thread thread_;
  std::atomic<uint64_t> dropped_{0};
  uint64_t written_ = 0;
  void run() {
    // std::thread inherits the producer's one-core affinity. Restore the process
    // leader's allowed mask so the writer does not compete on a CPU worker core.
    cpu_set_t allowed;
    if (sched_getaffinity(getpid(), sizeof(allowed), &allowed) == 0)
      sched_setaffinity(0, sizeof(allowed), &allowed);
    pthread_setname_np(pthread_self(), "kt_cpu_trace");
    std::array<Record, 16> batch{};
    for (;;) {
      size_t n = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stopping_ || count_; });
        if (!count_ && stopping_) break;
        while (count_ && n < batch.size()) {
          batch[n++] = records_[head_]; head_ = (head_ + 1) % CAPACITY; --count_;
        }
      }
      for (size_t i = 0; i < n; ++i) {
        const auto& r = batch[i];
        if (r.phase) {
          fprintf(file_, "{\"event\":\"cpu_moe.lifecycle\",\"phase\":\"%s\",\"task_id\":%llu,\"layer\":%d,\"part\":%d,\"pid\":%d,\"tid\":%d,\"cpu\":%d,\"wall_ns\":%llu,\"mono_ns\":%llu}\n", r.phase, (unsigned long long)r.task_id, r.layer,r.part,int(getpid()),r.tid,r.cpu,(unsigned long long)r.wall_ns,(unsigned long long)r.mono_ns);
          fprintf(file_, "{\"event\":\"cpu_moe.resource\",\"task_id\":%llu,\"phase\":\"%s\",\"tid\":%d,\"mono_ns\":%llu,\"thread_cpu_ns\":%llu,\"trace_mutex_wait_ns\":%llu,\"nvcsw\":%ld,\"nivcsw\":%ld,\"minflt\":%ld,\"majflt\":%ld}\n", (unsigned long long)r.task_id,r.phase,r.tid,(unsigned long long)r.mono_ns,(unsigned long long)r.thread_cpu_ns,(unsigned long long)r.trace_mutex_wait_ns,r.nvcsw,r.nivcsw,r.minflt,r.majflt);
          ++written_; continue;
        }
        fprintf(file_, "");
        fprintf(file_, "{\"event\":\"cpu_moe.compute\",\"pid\":%d,\"tid\":%d,\"layer\":%d,\"part\":%d,\"tokens\":%d,\"wall_ns\":%llu,\"duration_ns\":%llu,\"mono_raw_ns\":%llu,\"coordinator_cpu_ns\":%llu,\"task_id\":%llu,\"transfer_count\":%d,\"expert_count\":%d,\"expert_route_count\":%llu,\"expert_ids_truncated\":%s,\"expert_ids\":[",
            int(getpid()), r.tid, r.layer, r.part, r.tokens,
            (unsigned long long)r.wall_ns, (unsigned long long)r.duration_ns, (unsigned long long)r.raw_ns,
            (unsigned long long)r.coordinator_cpu_ns, (unsigned long long)r.task_id, r.transfer_count, r.count, (unsigned long long)r.expert_route_count, r.truncated ? "true" : "false");
        for (int j = 0; j < r.count; ++j) fprintf(file_, "%s%lld", j ? "," : "", (long long)r.experts[j]);
        fprintf(file_, "]}\n"); ++written_;
      }
      fprintf(file_, "{\"event\":\"cpu_moe.trace_flush\",\"pid\":%d,\"written\":%llu,\"dropped\":%llu}\n",
          int(getpid()), (unsigned long long)written_, (unsigned long long)dropped_.load());
      fflush(file_);
    }
    fflush(file_);
  }
 public:
  Writer() {
    const char* value = std::getenv("KT_MOE_CPU_TRACE");
    if (!value || !*value) return;
    std::string path(value), pid = std::to_string(getpid());
    for (size_t pos = 0; (pos = path.find("%p", pos)) != std::string::npos; pos += pid.size()) path.replace(pos, 2, pid);
    file_ = fopen(path.c_str(), "a");
    if (!file_) { perror("KT_MOE_CPU_TRACE open failed"); return; }
    setvbuf(file_, nullptr, _IOFBF, 64 * 1024);
    thread_ = std::thread([this] { run(); });
    fprintf(stderr, "[KT_CPU_TRACE] local_file=%s pid=%d\n", path.c_str(), int(getpid()));
  }
  ~Writer() {
    if (!file_) return;
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    cv_.notify_one(); thread_.join(); fclose(file_);
  }
  bool enabled() const { return file_ != nullptr; }
  void push(const Record& record) noexcept {
    const uint64_t trace_lock_begin=now(CLOCK_MONOTONIC);
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    // Sparse essential events take only the bounded in-memory buffer mutex;
    // file I/O always remains outside it. Detailed workers remain best-effort.
    if(record.phase && (record.phase[0]=='w'||record.phase[0]=='p'||record.phase[0]=='n')) lock.try_lock();
    else lock.lock();
    if (!lock.owns_lock() || count_ == CAPACITY) { ++dropped_; return; }
    records_[tail_] = record; records_[tail_].trace_mutex_wait_ns=now(CLOCK_MONOTONIC)-trace_lock_begin; tail_ = (tail_ + 1) % CAPACITY; ++count_;
    lock.unlock(); cv_.notify_one();
  }
};
inline Writer& writer() { static Writer instance; return instance; }
struct Context { uint64_t id=0; int layer=-1; };
inline thread_local Context context;
inline bool lifecycle_enabled() { static bool value = std::getenv("KT_MOE_LIFECYCLE_TRACE") != nullptr; return value; }
class ContextScope { Context old_; public: ContextScope(Context c): old_(context) {context=c;} ~ContextScope(){context=old_;} };
inline void mark(const char* phase, int part=0) {
  if (!lifecycle_enabled() || !context.id) return;
  // Detailed pool/worker events are opt-in: several hundred records per layer
  // can otherwise perturb the callback chain being measured.
  static bool detail=std::getenv("KT_MOE_LIFECYCLE_WORKERS") != nullptr;
  if (!detail && (phase[0]=='w' || phase[0]=='p' || phase[0]=='n')) return;
  auto& w=writer(); if(!w.enabled())return;
  Record r; r.phase=phase; r.task_id=context.id;r.layer=context.layer;r.part=part;
  r.mono_ns=now(CLOCK_MONOTONIC);r.wall_ns=now(CLOCK_REALTIME);
  r.tid=int(syscall(SYS_gettid));r.cpu=sched_getcpu();
  r.thread_cpu_ns=now(CLOCK_THREAD_CPUTIME_ID); struct rusage usage{};
  if(getrusage(RUSAGE_THREAD,&usage)==0){r.nvcsw=usage.ru_nvcsw;r.nivcsw=usage.ru_nivcsw;r.minflt=usage.ru_minflt;r.majflt=usage.ru_majflt;}
  w.push(r);
}
inline uint64_t begin(int layer) { static std::atomic<uint64_t> seq{0}; if(!lifecycle_enabled())return 0; context={++seq,layer};mark("callback_enter");return context.id; }
class Span {
  Writer* writer_ = nullptr;
  Record record_{};
  uint64_t start_ = 0, cpu_ = 0;
 public:
  template <typename Skip>
  Span(int layer, int part, int tokens, int k, const int64_t* ids, Skip skip) {
    auto& w = writer();
    if (!w.enabled() || tokens <= 0) return;
    writer_ = &w; record_.task_id=context.id; mark("compute_begin"); record_.layer = layer; record_.part = part; record_.tokens = tokens;
    record_.tid = int(syscall(SYS_gettid));
    for (int64_t i = 0; i < int64_t(tokens) * k; ++i) {
      // NPU encodes the layer-wide admission count as -1-T on skipped routes.
      // It may be repeated on many routes; decode the maximum, never sum it.
      if (ids[i] < -1) {
        const int count = static_cast<int>(-1 - ids[i]);
        if (count > record_.transfer_count) record_.transfer_count = count;
      }
      if (skip(ids[i])) continue;
      ++record_.expert_route_count;
      bool seen = false;
      for (int j = 0; j < record_.count; ++j) if (record_.experts[j] == ids[i]) { seen = true; break; }
      if (seen) continue;
      if (record_.count == int(record_.experts.size())) { record_.truncated = true; continue; }
      record_.experts[record_.count++] = ids[i];
    }
    record_.raw_ns = now(CLOCK_MONOTONIC_RAW);
    record_.wall_ns = now(CLOCK_REALTIME);
    cpu_ = now(CLOCK_THREAD_CPUTIME_ID);
    start_ = now(CLOCK_MONOTONIC);
  }
  ~Span() {
    if (!writer_) return;
    record_.duration_ns = now(CLOCK_MONOTONIC) - start_;
    record_.coordinator_cpu_ns = now(CLOCK_THREAD_CPUTIME_ID) - cpu_;
    writer_->push(record_); mark("compute_end");
  }
};
}  // namespace kt::cpu_trace
