#pragma once

// 真机 / sim2sim 的高程图来源：订阅建图节点通过 Unitree DDS 发布的 HeightMap_ 消息。
// 复用 unitree_sdk2 自带的 unitree_go::msg::dds_::HeightMap_（Go2 的 utlidar 高程图 IDL），
// 不需要新增消息类型；Python 侧用 unitree_sdk2py.idl.unitree_go.msg.dds_.HeightMap_ 发布即可。
//
// 消息字段的约定见 height_map_gate.h 顶部：width/height/resolution/origin 必须和策略网格一致，
// data[iy * width + ix] = 地面高度 − 躯干高度，未知格 NaN，frame_id 不检查。
// stamp 字段不用：建图节点和控制器的时钟不一定同步，新鲜度按控制器收到消息的时刻算。
//
// 离线测试用 test/fakes/dds_height_map_source.h 里的同名假类替换本文件。
#include "height_map_gate.h"
#include "unitree/dds_wrapper/common/Subscription.h"
#include <unitree/idl/go2/HeightMap_.hpp>
#include <memory>
#include <mutex>
#include <string>

namespace g1
{
class DdsHeightMapSource
{
public:
    explicit DdsHeightMapSource(const std::string& topic) : sub_(std::make_shared<Subscription>(topic)) {}

    const std::string& topic() const { return sub_->topic; }

    // Copies the newest message. Returns false until the first message arrives.
    bool latest(HeightMapGrid& out) const
    {
        std::lock_guard<std::mutex> lock(sub_->mutex_);
        if (!sub_->received) return false;
        const auto& msg = sub_->msg_;
        out.stamp = sub_->received_at;
        out.width = msg.width();
        out.height = msg.height();
        out.resolution = msg.resolution();
        out.origin = msg.origin();
        out.data = msg.data();
        return true;
    }

private:
    struct Subscription : public unitree::robot::SubscriptionBase<unitree_go::msg::dds_::HeightMap_>
    {
        explicit Subscription(const std::string& topic_)
            : unitree::robot::SubscriptionBase<unitree_go::msg::dds_::HeightMap_>(topic_), topic(topic_) {}
        std::string topic;
        bool received = false;
        double received_at = 0;

    protected:
        // Runs inside the DDS callback while mutex_ is held, so latest() sees stamp and data together.
        void post_communication() override
        {
            received = true;
            received_at = steady_seconds_now();
        }
    };

    std::shared_ptr<Subscription> sub_;
};
}  // namespace g1
