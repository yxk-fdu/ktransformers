/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"
#include "cpu_compute_trace.h"

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace {
// Codex: active-wait for 50 ms before falling back to condition-variable sleep.
constexpr auto kActiveWaitBudget = std::chrono::milliseconds(50);

inline void cpu_relax() noexcept {
#if defined(__aarch64__) || defined(__arm__)
  // Codex: CPU instruction hint, not a yield to the OS scheduler.
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

template <typename Ready>
void active_wait(Ready ready) {
  if (ready()) return;
  const auto deadline = std::chrono::steady_clock::now() + kActiveWaitBudget;
  unsigned int spins = 0;
  while (!ready()) {
    cpu_relax();
    // Codex: amortize clock reads; the budget is elapsed time, not spin count.
    if ((++spins & 63u) == 0 &&
        std::chrono::steady_clock::now() >= deadline) {
      return;
    }
  }
}
}  // namespace

TaskQueue::TaskQueue() : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  // 新建队列后台线程，在本对象上运行 worker()；MoE 内部再调度 WorkerPool。
  workerThread = std::thread(&TaskQueue::worker, this);
}

TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  // 原子加一：未完成任务包括正在执行的和排队中的任务。
  pending.fetch_add(1, std::memory_order_acq_rel);
  auto ctx=kt::cpu_trace::context;
  Node* node = new Node([task,ctx](){kt::cpu_trace::ContextScope scope(ctx);kt::cpu_trace::mark("queue_take");task();kt::cpu_trace::mark("task_return");});
  node->trace_id=ctx.id; node->trace_layer=ctx.layer;
  kt::cpu_trace::mark("queue_publish");
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  // 唤醒队列线程后返回，不等待计算完成。
  cv.notify_one();
}

void TaskQueue::sync(size_t allow_n_pending) {
  kt::cpu_trace::mark("queue_sync_enter");
  kt::cpu_trace::mark("sync_spin_enter");
  // Codex: actively observe completion for 50 ms before sleeping.
  active_wait([&] {
    return pending.load(std::memory_order_acquire) <= allow_n_pending
        || done.load(std::memory_order_acquire);
  });
  kt::cpu_trace::mark("sync_spin_exit");
  kt::cpu_trace::mark("sync_mutex_enter");
  std::unique_lock<std::mutex> lock(mtx);
  kt::cpu_trace::mark("sync_mutex_acquired");
  if(pending.load(std::memory_order_acquire)>allow_n_pending && !done.load(std::memory_order_acquire)){
    kt::cpu_trace::mark("sync_cv_sleep_enter");
    cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire);
    });
    kt::cpu_trace::mark("sync_cv_sleep_exit");
  }
  kt::cpu_trace::mark("completion_observed");
}

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      if (next->task) {
        // 后台线程真正执行任务；MoE forward 会调用计算线程池。
        next->task();
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        // 整个任务返回后才算完成，将未完成任务数原子减一。
        kt::cpu_trace::ContextScope scope({curr->trace_id,curr->trace_layer});
        kt::cpu_trace::mark("completion_publish");
        pending.fetch_sub(1, std::memory_order_acq_rel);
        kt::cpu_trace::mark("queue_pending_decremented");
      }
      // 唤醒 sync() 等待者，重新检查 pending 是否满足条件。
      cv.notify_all();
      {kt::cpu_trace::ContextScope scope({curr->trace_id,curr->trace_layer});kt::cpu_trace::mark("queue_completion_notify_done");}
    } else {
      // Codex: keep the empty queue active for 50 ms after the previous task.
      active_wait([&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
      // Codex: the locked predicate closes the spin-to-sleep lost-wakeup window.
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
  }
}