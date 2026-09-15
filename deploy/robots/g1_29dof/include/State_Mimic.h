#pragma once

#include "FSM/State_RLBase.h"
#include <atomic>
#include <condition_variable>

class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();

    void run();
    
    void exit()
    {
        {
            std::lock_guard<std::mutex> lock(wake_mutex_);
            policy_thread_running = false;
        }
        env->alg->cancel_inference();
        wake_.notify_all();
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
    std::atomic<bool> action_ready_{false};
    std::atomic<bool> policy_fault_{false};
    std::atomic<bool> bad_orientation_{false};
    std::atomic<float> elapsed_{0};
    std::mutex wake_mutex_;
    std::condition_variable wake_;
};

REGISTER_FSM(State_Mimic)
