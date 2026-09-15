#include "walk_handoff.h"
#include <atomic>
#include <iostream>
#include <limits>
#include <thread>

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
static bool close(float a, float b) { return std::fabs(a - b) < 2e-5f; }

int main()
{
    try {
        g1::ControlHistory history;
        for (int k = 0; k <= 120; ++k) {
            g1::ControlFrame f;
            f.time = k * 0.001;
            f.q.fill(k);
            f.previous_target.fill(k - 1);
            history.push(f);
        }
        const auto frames = history.window(5, 0.02);
        for (int k = 0; k < 5; ++k) {
            require(close(frames[k].q[0], 40 + 20 * k), "history must sample 20 ms apart, oldest first");
            require(close(frames[k].previous_target[28], 39 + 20 * k), "history must retain previously sent target");
        }
        g1::ControlHistory short_history;
        short_history.push(frames[0]);
        require(short_history.window(5, .02).front().q == frames[0].q, "startup history padding");

        g1::WalkHandoff handoff;
        g1::JointVector seed{}, candidate{};
        seed.fill(.4f);
        candidate.fill(-.6f);
        handoff.begin(seed, 10);
        for (int ms = 0; ms < 80; ++ms) {
            require(handoff.sample(10 + ms * .001) == seed, "delayed inference must preserve last sent target");
            require(handoff.command_gain(10 + ms * .001) == 0, "no command before first inference");
        }
        handoff.publish(candidate, 10.08);
        require(handoff.sample(10.08) == seed, "first valid inference must not produce a target jump");
        float max_delta = 0;
        auto previous = seed;
        for (int ms = 1; ms <= 900; ++ms) {
            const double time = 10.08 + ms * .001;
            if (ms % 20 == 0) handoff.publish(candidate, time);
            const auto target = handoff.sample(time);
            max_delta = std::max(max_delta, std::fabs(target[0] - previous[0]));
            if (ms == 150) require(close(target[0], -.1f), "blend midpoint must be half way");
            if (ms >= 300) require(close(target[0], candidate[0]), "blend must reach the live policy target");
            if (ms <= 400) require(close(handoff.command_gain(time), 0), "zero command window");
            if (ms == 650) require(close(handoff.command_gain(time), .5), "command ramp midpoint");
            if (ms == 900) require(close(handoff.command_gain(time), 1), "command fully restored");
            previous = target;
        }
        require(max_delta < .0063f, "one-radian entry change must be spread across the 300 ms blend");
        std::cout << "PASS delayed first inference, smooth target/command entry; max 1 ms delta for 1 rad change = " << max_delta << " rad\n";

        // Reuse the state after a previous walk episode; candidate and all entry
        // timers must be fresh even though the same C++ object is retained.
        seed.fill(1.2f);
        handoff.begin(seed, 20);
        require(handoff.sample(20.01) == seed, "re-entry must not publish the previous walk candidate");
        require(handoff.command_gain(20.01) == 0, "re-entry must restart command hold");
        require(handoff.fault(20.201), "first inference timeout");
        handoff.begin(seed, 30);
        handoff.publish(candidate, 30.01);
        handoff.sample(30.01);
        require(handoff.fault(30.211), "running inference timeout");
        auto invalid = candidate;
        invalid[8] = std::numeric_limits<float>::quiet_NaN();
        bool rejected = false;
        try { handoff.publish(invalid, 30.02); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "NaN target must be rejected");
        handoff.fail();
        require(handoff.fault(30.02), "inference exception must signal a fault");
        std::cout << "PASS repeated entry, timeout, non-finite result, exception handling\n";

        // Complete vectors must survive simultaneous producer/consumer access.
        g1::WalkHandoff concurrent;
        seed.fill(0);
        concurrent.begin(seed, 40);
        concurrent.publish(seed, 40);
        concurrent.sample(40);
        concurrent.publish(seed, 40.4);
        std::atomic<bool> go{false}, failed{false};
        std::thread producer([&] {
            while (!go.load()) std::this_thread::yield();
            for (int i = 0; i < 30000; ++i) {
                g1::JointVector target;
                target.fill(i % 2 ? .7f : -.4f);
                concurrent.publish(target, 40.4);
            }
        });
        go = true;
        for (int i = 0; i < 30000; ++i) {
            const auto target = concurrent.sample(40.4);
            for (float v : target) if (v != target[0]) failed = true;
        }
        producer.join();
        require(!failed, "torn target snapshot across threads");

        g1::ControlHistory concurrent_history;
        std::thread recorder([&] {
            for (int i = 0; i < 5000; ++i) {
                g1::ControlFrame f;
                f.time = i * .001;
                f.q.fill(i);
                f.previous_target.fill(i - 1);
                concurrent_history.push(f);
            }
        });
        for (int i = 0; i < 5000; ++i) {
            for (const auto& frame : concurrent_history.window(5, .02)) {
                require(frame.q[0] == frame.q[28] && frame.previous_target[28] == frame.q[0] - 1,
                        "torn sensor/command history snapshot");
            }
        }
        recorder.join();
        std::cout << "PASS concurrent target and sensor/command snapshot stress tests\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
