/*
 * PATCH9: how long every step of texrecon takes and how much memory it then holds, on the console, so that a run on a
 * large mesh shows where the time and the memory go:
 *     [PATCH9] Step: <name> took 12.3 s; memory in use 80.1 GiB, the most so far 95.4 GiB
 * The memory figures are read from /proc/self (Linux); elsewhere only the time is printed.
 */

#ifndef TEX_STEP_TIMER_HEADER
#define TEX_STEP_TIMER_HEADER

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "defines.h"

TEX_NAMESPACE_BEGIN

/* the resident memory now and the most so far in GiB (VmRSS, VmHWM), or -1 where /proc/self/status cannot be read */
inline void
memory_in_use_gib(double * now, double * most) {
    *now = -1.0;
    *most = -1.0;
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        double * target = nullptr;
        if (line.compare(0, 6, "VmRSS:") == 0) target = now;
        else if (line.compare(0, 6, "VmHWM:") == 0) target = most;
        if (target == nullptr) continue;
        std::istringstream fields(line.substr(6));
        double kib = 0.0;
        if (fields >> kib) *target = kib / (1024.0 * 1024.0);
    }
}

class StepTimer {
    private:
        std::chrono::steady_clock::time_point last;

    public:
        StepTimer() : last(std::chrono::steady_clock::now()) {}

        /* prints the time since the timer was made or since the last step, and starts the next step */
        void step(std::string const & name) {
            std::chrono::steady_clock::time_point const now = std::chrono::steady_clock::now();
            double const seconds = std::chrono::duration<double>(now - last).count();
            double in_use, most;
            memory_in_use_gib(&in_use, &most);
            char text[160];
            if (in_use >= 0.0 && most >= 0.0)
                std::snprintf(text, sizeof(text), " took %.1f s; memory in use %.1f GiB, the most so far %.1f GiB",
                    seconds, in_use, most);
            else
                std::snprintf(text, sizeof(text), " took %.1f s", seconds);
            std::cout << "\t[PATCH9] Step: " << name << text << std::endl;
            last = std::chrono::steady_clock::now();
        }
};

TEX_NAMESPACE_END

#endif /* TEX_STEP_TIMER_HEADER */
