#pragma once

// Test-only replacement for include/dds_height_map_source.h: no DDS. Tests push
// grids straight into the instance that State_Walk created.
#include "height_map_gate.h"
#include <mutex>
#include <string>

namespace g1
{
class DdsHeightMapSource
{
public:
    explicit DdsHeightMapSource(const std::string& topic) : topic_(topic) { instance = this; }
    ~DdsHeightMapSource()
    {
        if (instance == this) instance = nullptr;
    }

    const std::string& topic() const { return topic_; }

    bool latest(HeightMapGrid& out) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_) return false;
        out = grid_;
        return true;
    }

    // Test hook: what the DDS callback would have stored.
    void publish(const HeightMapGrid& grid)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        grid_ = grid;
        has_ = true;
    }

    inline static DdsHeightMapSource* instance = nullptr;

private:
    std::string topic_;
    mutable std::mutex mutex_;
    HeightMapGrid grid_;
    bool has_ = false;
};
}  // namespace g1
