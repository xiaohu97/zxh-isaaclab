// HeightMapGate: the only place where a real-world height map becomes policy input.
// Pure logic; no DDS, no ONNX.
#include "height_map_gate.h"
#include <cmath>
#include <iostream>

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

int main()
{
    try {
        using Source = g1::HeightMapGate::Source;
        g1::HeightMapGateConfig cfg;  // defaults are the training grid
        require(cfg.width == 17 && cfg.height == 11 && cfg.size() == 187, "default grid must be the 17x11 training scan");
        g1::HeightMapGate gate(cfg);
        Source source;

        // Nothing received: flat ground at the nominal torso height, never empty.
        auto cells = gate.resolve(nullptr, 100.0, source);
        require(source == Source::none && cells.size() == 187, "no-message fallback");
        for (float v : cells) require(v == -cfg.nominal_torso_height, "fallback cell value");

        g1::HeightMapGrid grid;
        grid.stamp = 100.0;
        grid.width = 17;
        grid.height = 11;
        grid.resolution = 0.1f;
        grid.origin = {-0.8f, -0.5f};
        grid.data.assign(187, -0.78f);
        // cell (ix=5, iy=3) -> index iy*width+ix; a 15 cm step up in front-left
        const std::size_t step_index = 3 * 17 + 5;
        grid.data[step_index] = -0.63f;
        grid.data[7] = NAN;      // unknown cell from the mapper
        grid.data[9] = 12.0f;    // garbage
        grid.data[11] = -4.0f;   // beyond max_abs_height
        cells = gate.resolve(&grid, 100.2, source);
        require(source == Source::live, "fresh grid is live");
        require(cells[step_index] == -0.63f, "live cell passes through");
        require(cells[7] == -0.78f && cells[9] == -0.78f && cells[11] == -0.78f, "unknown/garbage cells become flat");
        for (float v : cells) require(std::isfinite(v), "live output must be finite");

        // Older than timeout: flat fallback, not the last map (the robot has moved on).
        cells = gate.resolve(&grid, 100.0 + cfg.timeout + 0.01, source);
        require(source == Source::stale && cells[step_index] == -0.78f, "stale grid falls back to flat");
        cells = gate.resolve(&grid, 100.0 + cfg.timeout - 0.01, source);
        require(source == Source::live, "grid inside the timeout is live");

        // Any layout mismatch is rejected: the policy would otherwise read cells in the wrong place.
        std::string why;
        g1::HeightMapGrid bad = grid;
        bad.width = 11;
        bad.height = 17;
        require(!gate.matches(bad, &why) && why.find("grid 11x17") != std::string::npos, "transposed grid rejected");
        cells = gate.resolve(&bad, 100.2, source);
        require(source == Source::invalid && cells[step_index] == -0.78f, "invalid grid falls back to flat");
        bad = grid;
        bad.origin = {0.0f, 0.0f};
        require(!gate.matches(bad), "origin mismatch rejected");
        bad = grid;
        bad.resolution = 0.05f;
        require(!gate.matches(bad), "resolution mismatch rejected");
        bad = grid;
        bad.data.resize(186);
        require(!gate.matches(bad), "short data rejected");

        bool rejected = false;
        try {
            g1::HeightMapGateConfig invalid;
            invalid.timeout = 0;
            g1::HeightMapGate g(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "zero timeout must be rejected");

        std::cout << "PASS height map gate: fallback/live/stale/invalid, cell sanitising, layout checks\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
