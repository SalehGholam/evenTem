/* Copyright (C) 2025 Thomas Friedrich, Chu-Ping Yu, Arno Annys
 * University of Antwerp - All Rights Reserved. 
 * You may use, distribute and modify
 * this code under the terms of the GPL3 license.
 * You should have received a copy of the GPL3 license with
 * this file. If not, please visit: 
 * https://www.gnu.org/licenses/gpl-3.0.en.html
 * 
 * Authors: 
 *   Thomas Friedrich <>
 *   Chu-Ping Yu <>
 *   Arno Annys <arno.annys@uantwerpen.be>
 */

#ifndef BOUNDED_THREAD_POOL_H
#define BOUNDED_THREAD_POOL_H

#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>

class BoundedThreadPool
{
private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::atomic<bool> b_running;
    // A single mutex protects `tasks` (and `busy_workers` below) for both producers
    // (push_task) and consumers (worker/execute_task). The previous version used two
    // *separate* mutexes -- one for "queue full" waits, one for "queue empty" waits
    // -- guarding the same std::queue from both push and pop/front sides. That's a
    // genuine data race: a push (under mtx_queue_full) and a pop/front-access (under
    // mtx_queue_empty) could run concurrently on the same underlying queue. It went
    // unnoticed because nothing in this codebase actually ran the pool with more than
    // one worker thread before; enabling real multi-threaded declustering (n_threads
    // > 1) triggered it immediately (observed as a stray "bad function call" --
    // std::bad_function_call from a worker dequeuing a corrupted/empty task, and
    // duplicated startup log lines). One mutex for the whole queue is the standard,
    // correct pattern for a bounded blocking queue with multiple producers/consumers.
    std::mutex mtx;
    std::condition_variable cnd_not_full;
    std::condition_variable cnd_not_empty;
    // Counts workers currently executing a task (i.e. dequeued but not yet finished).
    // wait_for_completion() needs this: an empty queue does NOT mean all work is
    // done if a worker just dequeued its last task and hasn't returned from it yet.
    int busy_workers = 0;

    void create_threads()
    {
        for (int i = 0; i < n_threads; i++)
        {
            threads.push_back(std::thread(&BoundedThreadPool::worker, this));
        }
        std::cout << "Created " << n_threads << " threads." << std::endl;
    }

    void worker()
    {
        while (true)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cnd_not_empty.wait(lock, [this]
                                   { return !tasks.empty() || !b_running; });
                if (tasks.empty())
                {
                    // Only reachable via !b_running (shutdown) with nothing left to do.
                    return;
                }
                task = std::move(tasks.front());
                tasks.pop();
                ++busy_workers;
            }
            cnd_not_full.notify_one();
            try
            {
                task();
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << std::endl;
            }
            {
                std::lock_guard<std::mutex> lock(mtx);
                --busy_workers;
            }
            cnd_not_full.notify_one();
        }
    }

public:
    int n_threads;
    int limit;

    template <typename T>
    inline void push_task(const T &task)
    {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cnd_not_full.wait(lock, [this]
                             { return ((int)tasks.size() < limit); });
            tasks.push(std::function<void()>(task));
        }
        cnd_not_empty.notify_one();
    }

    template <typename T, typename... A>
    inline void push_task(const T &task, const A &...args)
    {
        push_task([task, args...]
                  { task(args...); });
    }

    void wait_for_completion()
    {
        std::unique_lock<std::mutex> lock(mtx);
        cnd_not_full.wait(lock, [this]
                         { return tasks.empty() && busy_workers == 0; });
    }

    void join_threads()
    {
        for (int i = 0; i < n_threads; i++)
        {
            threads[i].join();
        }
    }

    void init(int n_threads, int limit)
    {
        int n_threads_max = std::thread::hardware_concurrency();
        if (n_threads > n_threads_max || n_threads < 1)
        {
            this->n_threads = n_threads_max;
        }
        else
        {
            this->n_threads = n_threads;
        }
        this->limit = limit;
        b_running = true;
        create_threads();
    }

    explicit BoundedThreadPool(int n_threads) : limit(8)
    {
        init(n_threads, limit);
    }

    explicit BoundedThreadPool(int n_threads, int limit)
    {
        init(n_threads, limit);
    }

    BoundedThreadPool() : b_running(false), n_threads(0), limit(0) {}

    ~BoundedThreadPool()
    {
        wait_for_completion();
        {
            std::lock_guard<std::mutex> lock(mtx);
            b_running = false;
        }
        cnd_not_empty.notify_all();
        join_threads();
    }
};

#endif