"""Generate a collision-enabled Open Duck Mini V2 model for recovery training.

The upstream MuJoCo model only gives the two foot soles collision geometry
(verified: 3 of 47 geoms have contype/conaffinity set).  A robot lying on the
floor therefore has nothing to rest on and simply sinks through the ground, so
"fallen" is not a physically representable state and stand-up cannot be learned.

This script derives two files next to the originals:

  xmls/open_duck_mini_v2_getup.xml        robot + trunk/head collision boxes
  xmls/scene_flat_terrain_getup.xml       flat-terrain scene using that robot

The original files are never modified, so every deployed policy keeps training
and validating against exactly the physics it was tuned for.  The boxes are
sized from the visual mesh bounding boxes reported by ``geom_aabb.py``; the leg
links carry no collision geometry either, so no leg self-collision is
introduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

XMLS = Path(__file__).resolve().parent / "xmls"
ROBOT_SRC = XMLS / "open_duck_mini_v2.xml"
SCENE_SRC = XMLS / "scene_flat_terrain.xml"
ROBOT_DST = XMLS / "open_duck_mini_v2_getup.xml"
SCENE_DST = XMLS / "scene_flat_terrain_getup.xml"

# Bounding boxes measured from the visual meshes (see geom_aabb.py output).
#   trunk shell: x[-0.1549,+0.0461] y[-0.0789,+0.0760] z[-0.0788,+0.1410]
#   head shell : x[-0.0243,+0.0418] y[-0.1072,+0.0920] z[-0.1029,+0.1220]
TRUNK_BOX = dict(pos="-0.0545 -0.0015 0.0310", size="0.0980 0.0750 0.1060")
HEAD_BOX = dict(pos="0.0088 -0.0076 0.0096", size="0.0330 0.0980 0.1100")

TRUNK_ANCHOR = "<!-- Frame trunk -->"
HEAD_ANCHOR = '<site group="0" name="head"'


def insert_before(text: str, anchor: str, block: str, what: str) -> str:
    if block.strip().splitlines()[0] in text:
        print(f"  {what}: already present, skipping")
        return text
    index = text.find(anchor)
    if index < 0:
        raise SystemExit(f"ERROR: anchor not found for {what}: {anchor!r}")
    return text[:index] + block + text[index:]


def build_robot() -> None:
    src = ROBOT_SRC.read_text(encoding="utf-8")
    print(f"Reading {ROBOT_SRC.name} ({len(src)} bytes)")

    trunk_geom = (
        f'        <geom name="trunk_collision" class="collision" type="box"\n'
        f'          pos="{TRUNK_BOX["pos"]}" size="{TRUNK_BOX["size"]}"/>\n'
    )
    head_geom = (
        f'                <geom name="head_collision" class="collision" type="box"\n'
        f'                  pos="{HEAD_BOX["pos"]}" size="{HEAD_BOX["size"]}"/>\n'
    )

    src = insert_before(src, TRUNK_ANCHOR, trunk_geom, "trunk collision box")
    src = insert_before(src, HEAD_ANCHOR, head_geom, "head collision box")

    ROBOT_DST.write_text(src, encoding="utf-8")
    print(f"Wrote {ROBOT_DST.name} ({ROBOT_DST.stat().st_size} bytes)")


def build_scene() -> None:
    src = SCENE_SRC.read_text(encoding="utf-8")
    if 'include file="open_duck_mini_v2.xml"' not in src:
        raise SystemExit("ERROR: scene does not include the expected robot file")
    src = src.replace(
        'include file="open_duck_mini_v2.xml"',
        'include file="open_duck_mini_v2_getup.xml"',
    )
    SCENE_DST.write_text(src, encoding="utf-8")
    print(f"Wrote {SCENE_DST.name} ({SCENE_DST.stat().st_size} bytes)")


def main() -> int:
    build_robot()
    build_scene()
    return 0


if __name__ == "__main__":
    sys.exit(main())
