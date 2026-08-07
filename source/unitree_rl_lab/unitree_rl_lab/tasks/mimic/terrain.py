"""Terrain importers shared by mimic environments."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter


GRID_GROUND_USD_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "ground" / "default_environment.usd"
)


class LocalGridPlaneTerrainImporter(TerrainImporter):
    """Load Isaac Sim's original grid plane from the project-local asset copy."""

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)) -> None:
        prim_path = f"{self.cfg.prim_path}/{name}"
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. "
                f"Existing terrains: {', '.join(self.terrain_names)}."
            )
        if not GRID_GROUND_USD_PATH.is_file():
            raise FileNotFoundError(f"Local grid ground asset not found: {GRID_GROUND_USD_PATH}")

        self.terrain_prim_paths.append(prim_path)

        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            if "diffuse_color" in material:
                color = material["diffuse_color"]

        ground_plane_cfg = sim_utils.GroundPlaneCfg(
            usd_path=str(GRID_GROUND_USD_PATH),
            physics_material=self.cfg.physics_material,
            size=size,
            color=color,
        )
        ground_plane_cfg.func(prim_path, ground_plane_cfg)
