# Copyright 2022 GlobalFoundries PDK Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# ---------------- Pcells Utilities for Klayout of GF180MCU ----------------
# ============================================================================

import gdsfactory as gf
import pya
import os


def gf_to_pya(layout, c: gf.Component, device_name: str):
    c.write_gds(str(device_name) + "_temp.gds")
    layout.read(str(device_name) + "_temp.gds")
    os.remove(str(device_name) + "_temp.gds")

    return layout.cell(c.name)


def snap_to_grid(component: gf.Component, dbu: float = 0.005) -> gf.Component:
    """Returns a new Component with all polygons snapped to the nearest DBU grid (e.g. 5nm)."""
    # Step 1: flatten the component
    flat = component.copy()
    flat.flatten()

    # Step 2: create the cleaned component
    c_clean = gf.Component(name=f"{component.name}_snapped")

    # Step 3: snap polygons
    for layer, polygons in flat.get_polygons(by_spec=True).items():
        for points in polygons:
            if len(points) == 0:
                continue
            snapped = [(round(x / dbu) * dbu, round(y / dbu) * dbu) for x, y in points]
            c_clean.add_polygon(snapped, layer=layer)

    return c_clean
