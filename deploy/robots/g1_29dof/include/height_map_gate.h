#pragma once

// 高程图在控制器里的表示和"门"：把建图节点发来的网格变成策略可用的向量。
// 纯逻辑，不依赖 DDS / 机器人 I/O，离线测试直接覆盖。
//
// 约定（与训练侧 perception_cfg.py 的 HEIGHT_SCANNER_CFG 一一对应）：
//   * 网格在"躯干 yaw 对齐系"里：原点在 torso_link 投影，x 朝前、y 朝左，只跟随 yaw 不跟随 roll/pitch。
//   * width = x 方向格数 17（−0.8 .. +0.8 m），height = y 方向格数 11（−0.5 .. +0.5 m），分辨率 0.1 m。
//   * 排布 = Isaac Lab GridPattern "xy" 序：x 变化最快，data[iy * width + ix]，
//     cell (ix, iy) 的中心 = origin + (ix, iy) * resolution，origin = (−0.8, −0.5)。
//   * 每格数值 = 地面高度 − 躯干高度 [m]，平地站立约 −0.78；未知格填 NaN。
//   * 观测 = −value − offset（offset 在 deploy.yaml 的 height_scan.params 里，训练用 0.5），
//     由 observations.h 的 height_scan 项计算。
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace g1
{
inline double steady_seconds_now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct HeightMapGrid
{
    double stamp = 0;  // steady-clock seconds when the controller received it
    std::uint32_t width = 0, height = 0;  // cells along x (fastest) and y
    float resolution = 0;
    std::array<float, 2> origin{};  // (x, y) of cell (0, 0) in the torso-yaw frame
    std::vector<float> data;  // z_ground - z_torso, data[iy * width + ix]; NaN = unknown
};

struct HeightMapGateConfig
{
    std::uint32_t width = 17, height = 11;
    float resolution = 0.1f;
    std::array<float, 2> origin{-0.8f, -0.5f};
    double timeout = 0.3;  // s; older maps are replaced by the flat fallback
    float nominal_torso_height = 0.78f;  // flat fallback cell value = -nominal_torso_height
    float max_abs_height = 3.0f;  // cells beyond this are treated as unknown

    std::size_t size() const { return static_cast<std::size_t>(width) * height; }

    void validate() const
    {
        if (width == 0 || height == 0) throw std::invalid_argument("height_map grid must have non-zero width and height");
        if (!(resolution > 0) || !std::isfinite(resolution)) throw std::invalid_argument("height_map resolution must be positive");
        if (!(timeout > 0) || !std::isfinite(timeout)) throw std::invalid_argument("height_map timeout_s must be positive");
        if (!(nominal_torso_height > 0) || !std::isfinite(nominal_torso_height)) {
            throw std::invalid_argument("height_map nominal_torso_height must be positive");
        }
        if (!(max_abs_height > 0) || !std::isfinite(max_abs_height)) throw std::invalid_argument("height_map max_abs_height must be positive");
        if (!std::isfinite(origin[0]) || !std::isfinite(origin[1])) throw std::invalid_argument("height_map origin must be finite");
    }
};

class HeightMapGate
{
public:
    // none: nothing received yet; live: fresh and well-formed; stale: older than timeout;
    // invalid: dimensions / resolution / origin differ from the policy grid.
    enum class Source { none, live, stale, invalid };

    static const char* name(Source source)
    {
        switch (source) {
            case Source::none: return "none";
            case Source::live: return "live";
            case Source::stale: return "stale";
            case Source::invalid: return "invalid";
        }
        return "?";
    }

    explicit HeightMapGate(HeightMapGateConfig cfg) : cfg_(cfg) { cfg_.validate(); }

    const HeightMapGateConfig& config() const { return cfg_; }

    bool matches(const HeightMapGrid& grid, std::string* why = nullptr) const
    {
        const auto fail = [&](const std::string& reason) {
            if (why) *why = reason;
            return false;
        };
        if (grid.width != cfg_.width || grid.height != cfg_.height) {
            return fail("grid " + std::to_string(grid.width) + "x" + std::to_string(grid.height) + " != policy "
                        + std::to_string(cfg_.width) + "x" + std::to_string(cfg_.height));
        }
        if (grid.data.size() != cfg_.size()) return fail("data size " + std::to_string(grid.data.size()));
        if (std::fabs(grid.resolution - cfg_.resolution) > 1e-4f) return fail("resolution " + std::to_string(grid.resolution));
        for (int j = 0; j < 2; ++j) {
            if (std::fabs(grid.origin[j] - cfg_.origin[j]) > 1e-3f) {
                return fail("origin (" + std::to_string(grid.origin[0]) + ", " + std::to_string(grid.origin[1]) + ")");
            }
        }
        return true;
    }

    std::vector<float> fallback() const { return std::vector<float>(cfg_.size(), -cfg_.nominal_torso_height); }

    // Policy-layout cells for one control frame: never empty, never NaN, never out of range.
    // latest == nullptr means nothing has been received yet.
    std::vector<float> resolve(const HeightMapGrid* latest, double now, Source& source) const
    {
        if (!latest) {
            source = Source::none;
            return fallback();
        }
        if (!matches(*latest)) {
            source = Source::invalid;
            return fallback();
        }
        const double age = now - latest->stamp;
        if (!(age <= cfg_.timeout)) {  // also catches NaN stamps
            source = Source::stale;
            return fallback();
        }
        source = Source::live;
        std::vector<float> cells(latest->data);
        for (float& value : cells) {
            if (!std::isfinite(value) || std::fabs(value) > cfg_.max_abs_height) value = -cfg_.nominal_torso_height;
        }
        return cells;
    }

private:
    HeightMapGateConfig cfg_;
};
}  // namespace g1
