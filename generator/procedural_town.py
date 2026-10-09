"""Generate a deterministic, offline street-and-building plan for a town."""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union

from .osm import OSMFeature


ROAD_DESIGN_RULES = {
    "primary": {"speed_kph": 65, "intersection_spacing_m": 600,
                "curve_scale": 0.4},
    "secondary": {"speed_kph": 45, "intersection_spacing_m": 540,
                  "curve_scale": 0.65},
    "tertiary": {"speed_kph": 35, "intersection_spacing_m": 210,
                 "curve_scale": 0.85},
    "residential": {"speed_kph": 25, "intersection_spacing_m": 100,
                    "curve_scale": 1.15},
    "service": {"speed_kph": 20, "intersection_spacing_m": 260,
                "curve_scale": 0.45},
}


TOWN_PROFILES = {
    "custom": {},
    "small_town": {},
    "compact_city": {
        "block_size_m": 120, "building_density": 0.9, "park_chance": 0.05,
        "commercial_chance": 0.3, "civic_chance": 0.1,
        "industrial_chance": 0.02, "river_chance": 0.15,
        "street_pattern": "grid", "road_irregularity": 0.0,
    },
    "suburb": {
        "block_size_m": 230, "building_density": 0.56, "park_chance": 0.12,
        "commercial_chance": 0.08, "civic_chance": 0.035,
        "industrial_chance": 0.01, "river_chance": 0.12,
        "street_pattern": "organic", "road_irregularity": 0.12,
    },
    "rural_village": {
        "block_size_m": 300, "building_density": 0.42, "park_chance": 0.08,
        "commercial_chance": 0.06, "civic_chance": 0.03,
        "industrial_chance": 0.025, "river_chance": 0.32,
        "street_pattern": "organic", "road_irregularity": 0.3,
    },
    "industrial": {
        "block_size_m": 240, "building_density": 0.56, "park_chance": 0.06,
        "commercial_chance": 0.1, "civic_chance": 0.04,
        "industrial_chance": 0.22, "river_chance": 0.1,
        "street_pattern": "grid", "road_irregularity": 0.05,
    },
    "riverside": {
        "block_size_m": 180, "building_density": 0.62, "park_chance": 0.11,
        "commercial_chance": 0.14, "civic_chance": 0.06,
        "industrial_chance": 0.02, "water_chance": 0.08,
        "river_chance": 1.0, "street_pattern": "organic",
        "road_irregularity": 0.2,
    },
}


@dataclass(frozen=True)
class TownParameters:
    block_size_m: float = 180.0
    building_density: float = 0.68
    park_chance: float = 0.10
    commercial_chance: float = 0.16
    civic_chance: float = 0.06
    industrial_chance: float = 0.04
    water_chance: float = 0.04
    river_chance: float = 0.2
    town_type: str = "small_town"
    street_pattern: str = "organic"
    road_irregularity: float = 0.18
    max_building_side: int = 40
    seed: int = 1

    @classmethod
    def from_dict(cls, data: dict | None, default_seed: int = 1) -> "TownParameters":
        if not isinstance(data, dict):
            raise ValueError("Procedural town parameters must be an object.")
        town_type = data.get("town_type", "small_town")
        if not isinstance(town_type, str) or town_type not in TOWN_PROFILES:
            raise ValueError("town_type must be a supported town character.")
        limits = {
            "block_size_m": (60.0, 500.0),
            "building_density": (0.15, 1.0),
            "park_chance": (0.0, 0.6),
            "commercial_chance": (0.0, 0.8),
            "civic_chance": (0.0, 0.3),
            "industrial_chance": (0.0, 0.3),
            "water_chance": (0.0, 0.4),
            "river_chance": (0.0, 1.0),
            "road_irregularity": (0.0, 0.35),
            "max_building_side": (12, 200),
            "seed": (0, 2 ** 31 - 1),
        }
        values = {**TOWN_PROFILES[town_type], "seed": default_seed,
                  "town_type": town_type}
        for name, (low, high) in limits.items():
            if name not in data:
                continue
            raw = data[name]
            integer = name in {"seed", "max_building_side"}
            try:
                value = int(raw) if integer else float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a number.") from exc
            if integer and isinstance(raw, float) and not raw.is_integer():
                raise ValueError(f"{name} must be a whole number.")
            if (not integer and not math.isfinite(value)) \
                    or not low <= value <= high:
                raise ValueError(f"{name} must be between {low:g} and {high:g}.")
            values[name] = value
        street_pattern = data.get(
            "street_pattern", TOWN_PROFILES[town_type].get("street_pattern", "organic"))
        if not isinstance(street_pattern, str) \
                or street_pattern not in {"organic", "grid"}:
            raise ValueError("street_pattern must be 'organic' or 'grid'.")
        values["street_pattern"] = street_pattern
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


def _district_road_positions(extent: int, center: float, spacing: int,
                             rng: random.Random,
                             variation: float,
                             service_center: float | None = None,
                             industrial_chance: float = 0.0) -> list[float]:
    """Lay denser downtown streets, neighborhood collectors, and sparse fringe roads."""
    center = min(max(center, 1.0), extent - 2.0)
    positions = [center]
    for direction in (-1, 1):
        current = center
        while True:
            distance = abs(current - center)
            service_distance = (abs(current - service_center)
                               if service_center is not None else None)
            scale = _district_spacing_scale(
                distance, extent, spacing, service_distance,
                spacing * (0.65 + industrial_chance * 2.5))
            step = max(16.0, spacing * scale * rng.uniform(
                1.0 - variation, 1.0 + variation))
            next_position = current + direction * step
            if next_position <= 0:
                positions.append(0.0)
                break
            if next_position >= extent - 1:
                positions.append(float(extent - 1))
                break
            positions.append(next_position)
            current = next_position
    return sorted(set(round(position, 4) for position in positions))


def _smoothstep(value: float, start: float, end: float) -> float:
    if end <= start:
        return float(value >= end)
    t = min(1.0, max(0.0, (value - start) / (end - start)))
    return t * t * (3.0 - 2.0 * t)


def _proximity_weight(distance: float, full: float, none: float) -> float:
    return 1.0 - _smoothstep(distance, full, none)


def _district_spacing_scale(distance: float, extent: int, spacing: int,
                            service_distance: float | None = None,
                            service_radius: float = 0.0) -> float:
    fringe = _smoothstep(distance, spacing * 0.75, extent * 0.4)
    local_speed = ROAD_DESIGN_RULES["residential"]["speed_kph"]
    arterial_speed = ROAD_DESIGN_RULES["primary"]["speed_kph"]
    design_speed = local_speed + (arterial_speed - local_speed) * fringe
    scale = 0.68 + 0.87 * (design_speed - local_speed) / (
        arterial_speed - local_speed)
    if service_distance is not None and service_radius > 0:
        service = _proximity_weight(service_distance, spacing * 0.2,
                                    service_radius * 1.25)
        scale = scale * (1.0 - service) + 1.25 * service
    return scale


def _street_curve_weight(distance: float, spacing: int) -> float:
    return 0.25 + 0.75 * _smoothstep(
        distance, spacing * 0.65, spacing * 2.5)


def _collector_stride(spacing: int) -> int:
    target = ROAD_DESIGN_RULES["secondary"]["intersection_spacing_m"]
    return max(2, min(6, round(target / max(1, spacing))))


def _road_curve_scale(highway: str, distance: float, spacing: int) -> float:
    return (ROAD_DESIGN_RULES[highway]["curve_scale"]
            * _street_curve_weight(distance, spacing))

def _road_shape_offset(position: float, extent: int, spacing: int,
                       highway: str, route_index: int,
                       pattern: str, irregularity: float) -> float:
    """A seeded but structured lateral alignment for one road class."""
    t = min(1.0, max(0.0, position / max(1.0, extent - 1)))
    phase = math.pi if route_index % 2 else 0.0
    edge_fade = math.sin(math.pi * t)
    if highway == "primary":
        shape = edge_fade
    elif highway == "secondary":
        shape = edge_fade * math.sin(math.tau * t + phase)
    elif highway == "tertiary":
        sign = -1.0 if route_index % 2 else 1.0
        shape = (sign * max(0.0, 1.0 - abs(t - 0.5) / 0.16)
                 if route_index % 3 == 1 else 0.12 * edge_fade)
    elif highway == "service":
        shape = 0.2 * edge_fade
    else:
        shape = edge_fade * math.sin(2.5 * math.pi * t + phase)
    variation = max(0.25, min(1.0, irregularity * 2.0))
    pattern_scale = 0.55 if pattern == "grid" else 1.0
    return (spacing * 0.08 * ROAD_DESIGN_RULES[highway]["curve_scale"]
            * variation * pattern_scale * shape)

def _axis_road_role(index: int, position: float, center: float,
                    primary_index: int, spacing: int, industrial_center: float,
                    industrial_chance: float) -> str:
    if index == primary_index:
        return "primary"
    if (industrial_chance > 0
            and abs(position - industrial_center)
            < spacing * (0.65 + industrial_chance * 2.5)):
        return "service"
    if index % _collector_stride(spacing) == 0:
        return "secondary"
    if abs(position - center) < spacing * 1.2:
        return "tertiary"
    return "residential"


def _road_grid(width: int, height: int, town_center: tuple[float, float],
               spacing: int, rng: random.Random, pattern: str,
               irregularity: float,
               industrial_center: tuple[float, float] | None = None,
               industrial_chance: float = 0.0
               ) -> list[list[tuple[float, float]]]:
    """Create shared, connected intersections; local curves are node offsets."""
    variation = max(0.025, min(0.18, irregularity * 0.45))
    xs = _district_road_positions(width, town_center[0], spacing,
                                  rng, variation,
                                  industrial_center[0] if industrial_center else None,
                                  industrial_chance)
    ys = _district_road_positions(height, town_center[1], spacing,
                                  rng, variation,
                                  industrial_center[1] if industrial_center else None,
                                  industrial_chance)
    curve = max(0.025, min(0.16, irregularity * (0.48 if pattern == "organic"
                                                  else 0.16)))
    primary_row = min(range(len(ys)),
                      key=lambda row: abs(ys[row] - town_center[1]))
    primary_column = min(range(len(xs)),
                         key=lambda col: abs(xs[col] - town_center[0]))
    nodes = []
    for row, y in enumerate(ys):
        horizontal_role = _axis_road_role(
            row, y, town_center[1], primary_row, spacing,
            industrial_center[1] if industrial_center else town_center[1],
            industrial_chance)
        node_row = []
        for col, x in enumerate(xs):
            vertical_role = _axis_road_role(
                col, x, town_center[0], primary_column, spacing,
                industrial_center[0] if industrial_center else town_center[0],
                industrial_chance)
            x_offset = (0.0 if col in (0, len(xs) - 1) else
                        _road_shape_offset(y, height, spacing, vertical_role,
                                           col, pattern, irregularity))
            y_offset = (0.0 if row in (0, len(ys) - 1) else
                        _road_shape_offset(x, width, spacing, horizontal_role,
                                           row, pattern, irregularity))
            node_row.append((x + x_offset, y + y_offset))
        nodes.append(node_row)
    return nodes


def _district_boundaries(width: int, height: int,
                         centers: list[tuple[float, float]],
                         industrial_center: tuple[float, float], spacing: int,
                         commercial_chance: float, civic_chance: float,
                         industrial_chance: float) -> dict[str, Polygon]:
    """Plan non-overlapping land-use districts before adding local roads."""
    bounds = box(0, 0, width - 1, height - 1)
    downtown = unary_union([
        Point(*center).buffer(spacing * 1.25) for center in centers
    ]).intersection(bounds)
    industrial = (
        Point(*industrial_center).buffer(
            spacing * (0.65 + industrial_chance * 2.5)).intersection(bounds)
        .difference(downtown)
        if industrial_chance > 0 else Polygon())
    civic_center = (centers[0][0] - spacing * 0.55,
                    centers[0][1] + spacing * 0.3)
    civic = (
        Point(*civic_center).buffer(spacing * 0.55).intersection(bounds)
        .difference(unary_union([downtown, industrial]))
        if civic_chance > 0 else Polygon())
    arterials = unary_union([
        LineString([(0, centers[0][1]), (width - 1, centers[0][1])]),
        LineString([(centers[0][0], 0), (centers[0][0], height - 1)]),
    ])
    commercial = (
        arterials.buffer(spacing * 0.58).intersection(bounds)
        .difference(unary_union([downtown, industrial, civic]))
        if commercial_chance > 0 else Polygon())
    residential = bounds.difference(unary_union([
        downtown, industrial, civic, commercial]))
    return {"downtown": downtown, "commercial": commercial,
            "civic": civic, "industrial": industrial,
            "residential": residential}


def _blocks_from_roads(roads: list[LineString], width: int,
                       height: int) -> list[Polygon]:
    """Node a connected road network and derive its enclosed blocks."""
    boundary = LineString([(0, 0), (width - 1, 0),
                           (width - 1, height - 1), (0, height - 1), (0, 0)])
    network = unary_union([boundary, *roads])
    lines = list(network.geoms) if hasattr(network, "geoms") else [network]
    graph: dict[tuple[float, float], set[tuple[float, float]]] = {}
    for line in lines:
        if line.geom_type != "LineString":
            continue
        coords = list(line.coords)
        for start, end in zip(coords, coords[1:]):
            a = (round(start[0], 5), round(start[1], 5))
            b = (round(end[0], 5), round(end[1], 5))
            if a == b:
                continue
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    dangling = [node for node, neighbors in graph.items()
                if len(neighbors) < 2]
    if not graph or dangling:
        raise ValueError("Procedural road network contains an unconnected endpoint: "
                         f"{dangling[:6]}")
    unseen = set(graph)
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in graph[current] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
        if components > 1:
            raise ValueError("Procedural road network is disconnected.")
    map_boundary = Polygon([(0, 0), (width - 1, 0),
                            (width - 1, height - 1), (0, height - 1)])
    blocks = [poly for poly in polygonize(network)
              if poly.area >= 16
              and map_boundary.covers(poly.representative_point())]
    return sorted(blocks, key=lambda poly: (
        round(poly.centroid.y, 4), round(poly.centroid.x, 4)))


def _coordinate(proj, x: float, y: float) -> tuple[float, float]:
    lat, lon = proj.to_latlon(x, y)
    return float(lat), float(lon)


def _way(proj, feature_id: int, tags: dict,
         points: list[tuple[float, float]]) -> OSMFeature:
    geometry = [_coordinate(proj, x, y) for x, y in points]
    return OSMFeature(feature_id, "way", tags, geometry)


def _segment_intersects(a, b, c, d) -> bool:
    def orientation(p, q, r):
        return ((q[1] - p[1]) * (r[0] - q[0])
                - (q[0] - p[0]) * (r[1] - q[1]))

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return ((o1 == 0 and on_segment(a, c, b))
            or (o2 == 0 and on_segment(a, d, b))
            or (o3 == 0 and on_segment(c, a, d))
            or (o4 == 0 and on_segment(c, b, d)))


def _crosses_water(points, water_line) -> bool:
    return any(_segment_intersects(a, b, c, d)
               for a, b in zip(points, points[1:])
               for c, d in zip(water_line, water_line[1:]))


def _line_near_rect(line, left: float, top: float,
                    right: float, bottom: float, margin: float) -> bool:
    left, top = left - margin, top - margin
    right, bottom = right + margin, bottom + margin
    corners = ((left, top), (right, top), (right, bottom),
               (left, bottom), (left, top))
    for start, end in zip(line, line[1:]):
        if (left <= start[0] <= right and top <= start[1] <= bottom
                or left <= end[0] <= right and top <= end[1] <= bottom):
            return True
        if any(_segment_intersects(start, end, a, b)
               for a, b in zip(corners, corners[1:])):
            return True
    return False


def _riverbank(line, width: float) -> list[tuple[float, float]]:
    left_bank, right_bank = [], []
    for index, point in enumerate(line):
        before = line[max(0, index - 1)]
        after = line[min(len(line) - 1, index + 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = max(1.0, math.hypot(dx, dy))
        nx, ny = -dy / length * width, dx / length * width
        left_bank.append((point[0] + nx, point[1] + ny))
        right_bank.append((point[0] - nx, point[1] - ny))
    return left_bank + list(reversed(right_bank)) + [left_bank[0]]


def _river_path(width: int, height: int, horizontal: bool,
                rng: random.Random, irregularity: float) -> list[tuple[float, float]]:
    extent = width if horizontal else height
    cross_extent = height if horizontal else width
    low, high = cross_extent * 0.18, cross_extent * 0.82
    points = [(0.0, rng.uniform(low, high)),
              (float(extent - 1), rng.uniform(low, high))]
    for _ in range(5):
        refined = [points[0]]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            midpoint = (x0 + x1) / 2
            displacement = rng.uniform(-1, 1) * cross_extent * (
                0.10 + irregularity * 0.28)
            across = max(low, min(high, (y0 + y1) / 2 + displacement))
            refined.append((midpoint, across))
            refined.append((x1, y1))
        points = refined
    return points if horizontal else [(y, x) for x, y in points]


def _river_side_road(river: list[tuple[float, float]], width: int,
                     height: int, metres_per_tile: float
                     ) -> list[tuple[float, float]]:
    """Follow a river bank with a long rural road and an edge connection."""
    offset = max(4.0, 16.0 / metres_per_tile)
    horizontal = abs(river[-1][0] - river[0][0]) >= \
        abs(river[-1][1] - river[0][1])
    river_line = LineString(river)
    bounds = box(0, 0, width - 1, height - 1)
    candidates = []
    for side in ("left", "right"):
        shifted = river_line.parallel_offset(offset, side, join_style="round")
        parts = list(shifted.geoms) if hasattr(shifted, "geoms") else [shifted]
        for part in parts:
            if part.geom_type != "LineString":
                continue
            clipped = part.intersection(bounds)
            clipped_parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
            candidates.extend(line for line in clipped_parts
                              if line.geom_type == "LineString"
                              and line.length > offset * 2)
    if not candidates:
        return []
    road = list(max(candidates, key=lambda line: line.length).coords)
    if ((horizontal and road[0][0] > road[-1][0])
            or (not horizontal and road[0][1] > road[-1][1])):
        road.reverse()
    if horizontal:
        road = [(0.0, road[0][1]), *road,
                (float(width - 1), road[-1][1])]
    else:
        road = [(road[0][0], 0.0), *road,
                (road[-1][0], float(height - 1))]
    if horizontal:
        road[0] = (0.0, min(height - 2.0, max(1.0, road[0][1])))
        road[-1] = (float(width - 1),
                    min(height - 2.0, max(1.0, road[-1][1])))
    else:
        road[0] = (min(width - 2.0, max(1.0, road[0][0])), 0.0)
        road[-1] = (min(width - 2.0, max(1.0, road[-1][0])),
                    float(height - 1))
    return road


def _route_names(lines: list[LineString], classes: list[str]) -> list[str]:
    """Keep names continuous through junctions along the straightest route."""
    local_names = ("Oak Street", "Maple Avenue", "Pine Lane", "Cedar Street",
                   "School Road", "River Road", "Meadow Lane", "Elm Street",
                   "Parkway", "Church Street", "Mill Road", "Market Street")
    arterial_names = ("Main Street", "Market Avenue", "Center Street",
                      "Broadway", "Mill Road", "Parkway", "State Street")
    ends: dict[tuple[float, float], list[tuple[int, float, float]]] = {}
    for index, line in enumerate(lines):
        coords = list(line.coords)
        for start, next_point in ((coords[0], coords[1]),
                                  (coords[-1], coords[-2])):
            dx, dy = next_point[0] - start[0], next_point[1] - start[1]
            length = max(1e-9, math.hypot(dx, dy))
            key = (round(start[0], 5), round(start[1], 5))
            ends.setdefault(key, []).append((index, dx / length, dy / length))

    continuations = [set() for _ in lines]
    for incident in ends.values():
        pairs = []
        for left in range(len(incident)):
            for right in range(left + 1, len(incident)):
                a, ax, ay = incident[left]
                b, bx, by = incident[right]
                pairs.append((ax * bx + ay * by, a, b))
        used = set()
        for alignment, a, b in sorted(pairs):
            if alignment > -0.35 or a in used or b in used:
                continue
            used.update((a, b))
            continuations[a].add(b)
            continuations[b].add(a)

    components = []
    unseen = set(range(len(lines)))
    while unseen:
        first = min(unseen)
        stack, component = [first], []
        unseen.remove(first)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in continuations[current] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(component)
    components.sort(key=lambda group: (
        round(min(lines[i].centroid.y for i in group), 3),
        round(min(lines[i].centroid.x for i in group), 3)))

    names = [""] * len(lines)
    counters = {"primary": 0, "secondary": 0, "local": 0}
    rank = {"primary": 3, "secondary": 2, "tertiary": 1,
            "residential": 0, "service": 0}
    for component in components:
        hierarchy = max((classes[i] for i in component), key=rank.get)
        if hierarchy == "primary":
            number = counters["primary"] + 1
            name = f"County Road {number}"
            counters["primary"] += 1
        elif hierarchy == "secondary":
            name = arterial_names[counters["secondary"] % len(arterial_names)]
            counters["secondary"] += 1
        else:
            name = local_names[counters["local"] % len(local_names)]
            counters["local"] += 1
        for index in component:
            names[index] = name
    return names


def _frontage_lots(block: Polygon, rng: random.Random,
                   min_frontage: float, max_frontage: float,
                   depth: float, road_setback: float):
    """Place non-overlapping parcels along the block's street-facing edges."""
    frontage_geometry = block.simplify(
        max(1.0, min_frontage * 0.25), preserve_topology=True)
    if frontage_geometry.geom_type != "Polygon":
        frontage_geometry = block
    exterior = list(frontage_geometry.exterior.coords)
    for start, end in zip(exterior, exterior[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        edge_length = math.hypot(dx, dy)
        if edge_length < min_frontage + 8:
            continue
        tx, ty = dx / edge_length, dy / edge_length
        nx, ny = -ty, tx
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        if not block.buffer(-1).covers(
                Point(midpoint[0] + nx * 2, midpoint[1] + ny * 2)):
            nx, ny = -nx, -ny
        cursor = rng.uniform(3.0, 7.0)
        stop = edge_length - rng.uniform(3.0, 7.0)
        while cursor + min_frontage <= stop:
            frontage = min(rng.uniform(min_frontage, max_frontage),
                           stop - cursor)
            if frontage < min_frontage:
                break
            along = cursor + frontage / 2
            center_x = start[0] + tx * along + nx * (
                road_setback + depth / 2)
            center_y = start[1] + ty * along + ny * (
                road_setback + depth / 2)
            half_front, half_depth = frontage / 2, depth / 2
            polygon = Polygon([
                (center_x - tx * half_front - nx * half_depth,
                 center_y - ty * half_front - ny * half_depth),
                (center_x + tx * half_front - nx * half_depth,
                 center_y + ty * half_front - ny * half_depth),
                (center_x + tx * half_front + nx * half_depth,
                 center_y + ty * half_front + ny * half_depth),
                (center_x - tx * half_front + nx * half_depth,
                 center_y - ty * half_front + ny * half_depth),
            ])
            if block.buffer(-road_setback).covers(polygon):
                yield polygon, (tx, ty, nx, ny)
            cursor += frontage + rng.uniform(2.0, 7.0)


def _culdesac(block: Polygon, rng: random.Random,
              metres_per_tile: float):
    """Return a residential spur and its connected turnaround bulb, if viable."""
    vertices = list(block.exterior.coords)[:-1]
    rng.shuffle(vertices)
    for start in vertices:
        dx, dy = block.centroid.x - start[0], block.centroid.y - start[1]
        depth = math.hypot(dx, dy)
        if depth < 12.0 / metres_per_tile:
            continue
        inward = (dx / depth, dy / depth)
        for scale in (1.0, 0.75, 0.5, 0.35):
            radius = min(max(1.5, 5.0 / metres_per_tile) * scale,
                         depth * 0.2)
            center_distance = min(depth * 0.48, radius * 3.0)
            center = (start[0] + inward[0] * center_distance,
                      start[1] + inward[1] * center_distance)
            throat = (center[0] - inward[0] * radius,
                      center[1] - inward[1] * radius)
            rear_angle = math.atan2(-inward[1], -inward[0])
            bulb = [(center[0] + radius * math.cos(
                        rear_angle + math.tau * i / 20),
                     center[1] + radius * math.sin(
                        rear_angle + math.tau * i / 20))
                    for i in range(20)]
            bulb[0] = throat
            bulb.append(throat)
            if not block.covers(Polygon(bulb)):
                continue
            spur = [start,
                    (start[0] + (throat[0] - start[0]) * 0.33,
                     start[1] + (throat[1] - start[1]) * 0.33),
                    (start[0] + (throat[0] - start[0]) * 0.67,
                     start[1] + (throat[1] - start[1]) * 0.67),
                    throat]
            return spur, bulb
    return None


def _neighborhood_loop(block: Polygon):
    """Return a small residential loop with two connectors to its bounding streets."""
    left, top, right, bottom = block.bounds
    width, height = right - left, bottom - top
    radius_x, radius_y = width * 0.2, height * 0.2
    if radius_x < 4 or radius_y < 4:
        return None
    center_x, center_y = block.centroid.x, block.centroid.y
    loop = [(center_x + radius_x * math.cos(math.tau * i / 24),
             center_y + radius_y * math.sin(math.tau * i / 24))
            for i in range(24)]
    loop.append(loop[0])
    if not block.covers(Polygon(loop)):
        return None
    left_target = Point(center_x - radius_x * 1.5, center_y)
    right_target = Point(center_x + radius_x * 1.5, center_y)
    vertices = list(block.exterior.coords)[:-1]
    left_start = min(vertices, key=lambda point: Point(*point).distance(left_target))
    right_candidates = [point for point in vertices if point != left_start]
    right_start = min(
        right_candidates,
        key=lambda point: Point(*point).distance(right_target))
    connectors = [
        [left_start, (center_x - radius_x, center_y)],
        [right_start, (center_x + radius_x, center_y)],
    ]
    loop[0] = (center_x + radius_x, center_y)
    loop[12] = (center_x - radius_x, center_y)
    loop[-1] = loop[0]
    if not all(block.covers(LineString(connector)) for connector in connectors):
        return None
    return connectors, loop


def _pond(left: int, top: int, right: int, bottom: int,
          rng: random.Random) -> list[tuple[float, float]]:
    cx = rng.uniform(left + (right - left) * 0.4,
                     right - (right - left) * 0.4)
    cy = rng.uniform(top + (bottom - top) * 0.4,
                     bottom - (bottom - top) * 0.4)
    rx = (right - left) * rng.uniform(0.15, 0.25)
    ry = (bottom - top) * rng.uniform(0.15, 0.25)
    phase = rng.uniform(0, math.tau)
    points = []
    for index in range(13):
        angle = math.tau * index / 12
        shore = rng.uniform(0.82, 1.12)
        points.append((cx + math.cos(angle + phase) * rx * shore,
                       cy + math.sin(angle + phase) * ry * shore))
    points[-1] = points[0]
    return points


def _projected_shape(proj, shape_data: dict | None):
    """Project a checked GeoJSON outline into the town's tile coordinates."""
    if not shape_data:
        return None
    polygons = shape_data["coordinates"]
    if shape_data["type"] == "Polygon":
        polygons = [polygons]
    projected = []
    for polygon in polygons:
        rings = [[proj.to_px(lat, lon) for lon, lat in ring]
                 for ring in polygon]
        if rings and len(rings[0]) >= 4:
            projected.append(Polygon(rings[0], rings[1:]))
    return unary_union(projected) if projected else None


def generate(proj, parameters: TownParameters, templates=None,
             max_features: int = 100_000, should_stop=None,
             max_building_side: int | None = None,
             selection_shape: dict | None = None) -> list[OSMFeature]:
    """Build a connected street network, green/water features and town lots."""
    import knoxstop

    rng = random.Random(parameters.seed)
    metres_per_tile = proj.meters_per_tile
    build_area = _projected_shape(proj, selection_shape)
    spacing = max(16, round(parameters.block_size_m / metres_per_tile))
    building_side_limit = min(
        parameters.max_building_side,
        (max_building_side if max_building_side is not None
         else parameters.max_building_side))
    town_center = (proj.width * rng.uniform(0.44, 0.56),
                   proj.height * rng.uniform(0.44, 0.56))
    industrial_side = rng.choice(("east", "west"))
    industrial_center = (
        (proj.width * (0.88 if industrial_side == "east" else 0.12),
         town_center[1]))
    centers = [town_center]
    if parameters.town_type in {"compact_city", "suburb"}:
        spread = min(proj.width, proj.height) * (
            0.22 if parameters.town_type == "compact_city" else 0.3)
        centers.extend([
            (town_center[0] - spread, town_center[1] - spread * 0.52),
            (town_center[0] + spread, town_center[1] + spread * 0.48),
        ])
        if parameters.town_type == "compact_city":
            centers.append((town_center[0] - spread * 0.12,
                            town_center[1] + spread))
    districts = _district_boundaries(
        proj.width, proj.height, centers, industrial_center, spacing,
        parameters.commercial_chance, parameters.civic_chance,
        parameters.industrial_chance)
    organic = parameters.street_pattern == "organic"
    bend = parameters.road_irregularity
    nodes = _road_grid(proj.width, proj.height, town_center, spacing,
                       rng, parameters.street_pattern, bend,
                       industrial_center, parameters.industrial_chance)

    features: list[OSMFeature] = []
    feature_id = -1

    def add(tags: dict, points: list[tuple[float, float]]) -> None:
        nonlocal feature_id
        if build_area is not None and "building" in tags:
            footprint = Polygon(points)
            if not build_area.covers(footprint):
                return
        features.append(_way(proj, feature_id, tags, points))
        feature_id -= 1
        if len(features) > max_features:
            raise ValueError(
                f"This selection would create more than {max_features:,} "
                "procedural features. Draw a smaller area or increase block size.")

    river = None
    if rng.random() < parameters.river_chance:
        river = _river_path(proj.width, proj.height, rng.choice((True, False)),
                            rng, bend)

    railway_points = None
    if parameters.industrial_chance > 0:
        rail_y = min(max(proj.height * 0.82, 2.0), proj.height - 3.0)
        edge_x = float(proj.width - 1 if industrial_side == "east" else 0)
        railway_points = [
            (edge_x, rail_y),
            (industrial_center[0], rail_y),
            industrial_center,
        ]
        railway_tags = {"railway": "rail", "service": "spur",
                        "usage": "freight", "name": "Industrial Freight Spur"}
        if river and _crosses_water(railway_points, river):
            railway_tags.update({"bridge": "yes", "layer": "1"})
        add(railway_tags, railway_points)
    railway_line = (LineString(railway_points) if railway_points else None)

    def road_tags(index: int, name: str, points,
                  highway: str | None = None) -> dict:
        if highway is None:
            if index % 6 == 0:
                highway = "primary"
            elif index % 3 == 0:
                highway = "secondary"
            elif index % 2 == 0:
                highway = "tertiary"
            else:
                highway = "residential"
        width, lanes = {
            "primary": ("12", "2"), "secondary": ("8", "2"),
            "tertiary": ("6", "2"), "residential": ("5", "1"),
            "service": ("4", "1"),
        }[highway]
        speed = {"primary": 50, "secondary": 40, "tertiary": 30,
                 "residential": 20, "service": 15}[highway]
        design = ROAD_DESIGN_RULES[highway]
        tags = {"highway": highway, "name": name, "width": f"{width} m",
                "lanes": lanes, "maxspeed": f"{speed} km/h",
                "knoxmap:design_speed_kph": design["speed_kph"],
                "knoxmap:intersection_spacing_m": design["intersection_spacing_m"],
                "surface": "asphalt"}
        midpoint = LineString(points).interpolate(0.5, normalized=True)
        tags["knoxmap:district"] = next(
            (district for district in
             ("downtown", "commercial", "industrial", "civic")
             if districts[district].covers(midpoint)), "residential")
        if river and _crosses_water(points, river):
            tags.update({"bridge": "yes", "layer": "1"})
        if railway_line and LineString(points).intersects(railway_line):
            tags["railway_crossing"] = "yes"
        return tags

    road_specs: list[tuple[int, int, LineString, str]] = []
    row_count, column_count = len(nodes), len(nodes[0])
    collector_stride = _collector_stride(spacing)
    middle_row = row_count // 2
    middle_column = column_count // 2
    primary_row = min(
        range(row_count),
        key=lambda row: abs(nodes[row][middle_column][1] - town_center[1]))
    primary_column = min(
        range(column_count),
        key=lambda col: abs(nodes[middle_row][col][0] - town_center[0]))
    town_center_point = Point(*town_center)
    industrial_center_point = Point(*industrial_center)
    major_roads = []
    for row in range(row_count):
        for col in range(column_count - 1):
            line = LineString((nodes[row][col], nodes[row][col + 1]))
            center_distance = line.distance(town_center_point)
            service_distance = line.distance(industrial_center_point)
            highway = (
                "primary" if row == primary_row else
                "service" if (parameters.industrial_chance > 0
                              and service_distance < spacing * 0.8) else
                "secondary" if row % collector_stride == 0 else
                "tertiary" if center_distance < spacing * 0.85 else
                "residential")
            road_specs.append((row, col, line, highway))
    for col in range(column_count):
        for row in range(row_count - 1):
            line = LineString((nodes[row][col], nodes[row + 1][col]))
            center_distance = line.distance(town_center_point)
            service_distance = line.distance(industrial_center_point)
            highway = (
                "primary" if col == primary_column else
                "service" if (parameters.industrial_chance > 0
                              and service_distance < spacing * 0.8) else
                "secondary" if col % collector_stride == 0 else
                "tertiary" if center_distance < spacing * 0.85 else
                "residential")
            road_specs.append((row, col, line, highway))

    route_lines = [line for _row, _col, line, _highway in road_specs]
    route_classes = [highway for _row, _col, _line, highway in road_specs]
    route_names = _route_names(route_lines, route_classes)
    major_roads = [line for line, highway in zip(route_lines, route_classes)
                   if highway in {"primary", "secondary"}]
    arterial_geometries = [line for line, highway in zip(route_lines, route_classes)
                           if highway == "primary"]
    road_geometries = list(route_lines)
    horizontal_count = row_count * (column_count - 1)
    primary_segments = {
        *(primary_row * (column_count - 1) + col
          for col in range(column_count - 1)),
        *(horizontal_count + primary_column * (row_count - 1) + row
          for row in range(row_count - 1)),
    }
    add(road_tags(0, "Main Street", list(nodes[primary_row]), "primary"),
        list(nodes[primary_row]))
    main_vertical = [nodes[row][primary_column] for row in range(row_count)]
    add(road_tags(1, "County Road 1", main_vertical, "primary"), main_vertical)
    for index, (row, col, line, highway) in enumerate(road_specs):
        if index in primary_segments:
            continue
        points = list(line.coords)
        add(road_tags(index, route_names[index], points, highway), points)

    if parameters.town_type == "rural_village" and river:
        river_road = _river_side_road(
            river, proj.width, proj.height, metres_per_tile)
        if river_road:
            add(road_tags(2, "River Road", river_road, "tertiary"), river_road)
            road_geometries.append(LineString(river_road))

    diagonal_added = False
    for row in range(row_count - 1):
        for col in range(column_count - 1):
            cell_center = Polygon([
                nodes[row][col], nodes[row][col + 1],
                nodes[row + 1][col + 1], nodes[row + 1][col],
            ]).centroid
            if districts["commercial"].covers(cell_center):
                diagonal = [nodes[row][col], nodes[row + 1][col + 1]]
                add(road_tags(3, "Market Diagonal", diagonal, "tertiary"),
                    diagonal)
                road_geometries.append(LineString(diagonal))
                diagonal_added = True
                break
        if diagonal_added:
            break

    if river:
        bank_width = max(3.0, 12.0 / metres_per_tile)
        add({"leisure": "park", "name": "Riverbank Greenway"},
            _riverbank(river, bank_width))
        add({"waterway": "river", "name": "Town River"}, river)

    culdesac_added = False
    loop_added = False
    yard_turnaround_added = False
    for row in range(row_count - 1):
        for col in range(column_count - 1):
            cell = Polygon([nodes[row][col], nodes[row][col + 1],
                            nodes[row + 1][col + 1], nodes[row + 1][col]])
            if not cell.is_valid or cell.area < spacing * spacing * 0.12:
                continue
            center = cell.centroid
            if (not yard_turnaround_added
                    and districts["industrial"].covers(center)):
                yard = _neighborhood_loop(cell)
                if yard:
                    yard_tags = road_tags(
                        4, "Freight Yard Turnaround", yard[1], "service")
                    yard_tags.update({"service": "yard", "width": "8 m"})
                    for connector in yard[0]:
                        add(yard_tags, connector)
                        road_geometries.append(LineString(connector))
                    add(yard_tags, yard[1])
                    road_geometries.append(LineString(yard[1]))
                    yard_turnaround_added = True
            if not districts["residential"].covers(center):
                continue
            if not culdesac_added:
                court = _culdesac(cell, rng, metres_per_tile)
                if court:
                    tags = {"highway": "residential",
                            "name": "Residential Court", "width": "5 m",
                            "lanes": "1", "surface": "asphalt",
                            "maxspeed": "20 km/h",
                            "knoxmap:design_speed_kph":
                                ROAD_DESIGN_RULES["residential"]["speed_kph"],
                            "knoxmap:intersection_spacing_m":
                                ROAD_DESIGN_RULES["residential"]["intersection_spacing_m"]}
                    add(tags, court[0])
                    add(tags, court[1])
                    road_geometries.extend(
                        (LineString(court[0]), LineString(court[1])))
                    culdesac_added = True
                    continue
            if not loop_added:
                loop = _neighborhood_loop(cell)
                if loop:
                    tags = {"highway": "residential",
                            "name": "Neighborhood Loop", "width": "5 m",
                            "lanes": "1", "surface": "asphalt",
                            "maxspeed": "20 km/h",
                            "knoxmap:design_speed_kph":
                                ROAD_DESIGN_RULES["residential"]["speed_kph"],
                            "knoxmap:intersection_spacing_m":
                                ROAD_DESIGN_RULES["residential"]["intersection_spacing_m"]}
                    for connector in loop[0]:
                        add(tags, connector)
                        road_geometries.append(LineString(connector))
                    add(tags, loop[1])
                    road_geometries.append(LineString(loop[1]))
                    loop_added = True
                    continue
            if rng.random() < 0.04:
                court = _culdesac(cell, rng, metres_per_tile)
                if court:
                    tags = {"highway": "residential",
                            "name": "Residential Court", "width": "5 m",
                            "lanes": "1", "surface": "asphalt",
                            "maxspeed": "20 km/h",
                            "knoxmap:design_speed_kph":
                                ROAD_DESIGN_RULES["residential"]["speed_kph"],
                            "knoxmap:intersection_spacing_m":
                                ROAD_DESIGN_RULES["residential"]["intersection_spacing_m"]}
                    add(tags, court[0])
                    add(tags, court[1])
                    road_geometries.extend(
                        (LineString(court[0]), LineString(court[1])))

    blocks = _blocks_from_roads(road_geometries, proj.width, proj.height)
    block_plans = [(block, index, index)
                   for index, block in enumerate(blocks)]

    center_radius = max(spacing, math.hypot(proj.width, proj.height) * 0.52)
    major_geometries = [LineString(road.coords) for road in major_roads]
    town_center_point = Point(*town_center)
    industrial_center_point = Point(*industrial_center)
    civic_point = Point(town_center[0] - spacing * 0.55,
                        town_center[1] + spacing * 0.3)
    if block_plans:
        commercial_anchor = min(
            range(len(block_plans)),
            key=lambda i: block_plans[i][0].centroid.distance(town_center_point))
        industrial_anchor = min(
            range(len(block_plans)),
            key=lambda i: block_plans[i][0].centroid.distance(
                industrial_center_point))
        civic_candidates = [i for i in range(len(block_plans))
                            if i != commercial_anchor]
        civic_anchor = min(
            civic_candidates or range(len(block_plans)),
            key=lambda i: block_plans[i][0].centroid.distance(civic_point))
        corridor_candidates = [
            i for i, (candidate, _row, _col) in enumerate(block_plans)
            if i not in {commercial_anchor, industrial_anchor, civic_anchor}
            and candidate.centroid.distance(town_center_point) > spacing * 1.25
            and min((line.distance(candidate.centroid)
                     for line in major_geometries), default=float("inf"))
            <= spacing * 0.8]
        commercial_corridor_anchor = min(
            corridor_candidates,
            key=lambda i: block_plans[i][0].centroid.distance(town_center_point),
            default=-1)
        motel_anchor = -1
        motel_candidates = (corridor_candidates if len(corridor_candidates) > 1
                            else [i for i, (candidate, _row, _col)
                                  in enumerate(block_plans)
                                  if i not in {commercial_anchor,
                                               industrial_anchor, civic_anchor}
                                  and candidate.centroid.distance(town_center_point)
                                  > spacing * 1.25
                                  and min((line.distance(candidate.centroid)
                                           for line in major_geometries),
                                          default=float("inf"))
                                  <= spacing * 1.6])
        if parameters.commercial_chance >= 0.3:
            motel_anchor = min(
                (i for i in motel_candidates
                 if i != commercial_corridor_anchor),
                key=lambda i: block_plans[i][0].centroid.distance(town_center_point),
                default=-1)
    else:
        commercial_anchor = industrial_anchor = civic_anchor = -1
        commercial_corridor_anchor = -1
        motel_anchor = -1
    fuel_created = False
    motel_created = False
    for block_index, (block, row, col) in enumerate(block_plans):
        if block_index % 8 == 0:
            knoxstop.check(should_stop, "the procedural town")
        setback = max(3.0, min(spacing * 0.08, 7.0 / metres_per_tile))
        inner = block.buffer(-setback)
        if inner.is_empty or inner.geom_type != "Polygon":
            continue
        left, top, right, bottom = inner.bounds
        left, top = math.ceil(left), math.ceil(top)
        right, bottom = math.floor(right), math.floor(bottom)
        center_distance = min(block.centroid.distance(Point(*center))
                              for center in centers)
        centrality = max(0.0, 1.0 - center_distance / center_radius)
        downtown_weight = _proximity_weight(
            center_distance, spacing * 0.65, spacing * 2.5)
        block_center = Point(block.centroid.x, block.centroid.y)
        corridor_distance = min(
            (line.distance(block_center) for line in major_geometries),
            default=float("inf"))
        arterial_distance = min(
            (line.distance(block_center) for line in arterial_geometries),
            default=float("inf"))
        near_main_street = corridor_distance <= spacing * 0.58
        corridor_weight = _proximity_weight(
            arterial_distance, spacing * 0.2, spacing * 1.5)
        mixed_use_weight = max(
            downtown_weight * (1.0 - downtown_weight),
            corridor_weight * (1.0 - downtown_weight) * 0.7)
        if right - left < 12 or bottom - top < 12:
            continue
        service_distance = block_center.distance(industrial_center_point)
        industrial_core = districts["industrial"].covers(block_center)
        industrial_block = parameters.industrial_chance > 0 and (
            block_index == industrial_anchor or
            industrial_core or
            (service_distance <= spacing * 2.0
             and rng.random() < parameters.industrial_chance * 0.8))
        civic_block = (not industrial_block
                       and block_index != commercial_anchor
                       and parameters.civic_chance > 0
                       and (block_index == civic_anchor or
                            (districts["civic"].covers(block_center)
                             and rng.random() < min(
                                                                 0.7, parameters.civic_chance * 2.5))))
        downtown = downtown_weight >= 0.65
        commercial_corridor = (districts["commercial"].covers(block_center)
                               or corridor_weight > 0.2) and near_main_street
        commercial_probability = min(
            0.9,
            downtown_weight * (0.36 + parameters.commercial_chance * 0.55)
            + corridor_weight * (0.12 + parameters.commercial_chance * 0.88))
        commercial_block = (
            not industrial_block and not civic_block
            and parameters.commercial_chance > 0
            and (block_index in {commercial_anchor, commercial_corridor_anchor,
                                 motel_anchor}
                 or (commercial_corridor
                     and rng.random() < commercial_probability)))
        district_anchor = (
            block_index == commercial_anchor and parameters.commercial_chance > 0
            or block_index == motel_anchor and parameters.commercial_chance >= 0.3
            or block_index == civic_anchor and parameters.civic_chance > 0
            or block_index == industrial_anchor and parameters.industrial_chance > 0)
        river_in_block = bool(river and _line_near_rect(
            river, left, top, right, bottom, 12.0 / metres_per_tile))
        if not district_anchor and rng.random() < parameters.park_chance:
            add({"leisure": "park", "name": "Town Park"},
                list(inner.exterior.coords))
            path_y = (top + bottom) / 2
            path = LineString([(left + 3, path_y), (right - 3, path_y)])
            if inner.covers(path):
                add({"highway": "footway", "surface": "gravel",
                     "name": "Park Path"}, list(path.coords))
            continue
        if (not district_anchor and not river_in_block
            and rng.random() < parameters.water_chance):
            pond = _pond(left, top, right, bottom, rng)
            if inner.covers(Polygon(pond)):
                add({"natural": "water", "water": "pond",
                     "name": "Town Pond"}, pond)
                continue
        if commercial_block:
            add({"landuse": "commercial",
                 "name": "Town Centre" if downtown else "Commercial Corridor"},
                list(inner.exterior.coords))
        elif industrial_block:
            add({"landuse": "industrial", "name": "Industrial Estate"},
                list(inner.exterior.coords))
        elif not civic_block:
            add({"landuse": "residential", "name": "Residential District"},
                list(inner.exterior.coords))

        if industrial_block:
            family, kind = "industrial", "industrial"
            min_frontage, max_frontage = 44, 76
            occupancy = min(0.78, 0.34 + parameters.building_density * 0.3)
        elif commercial_block:
            family, kind = "commercial", "shop"
            if downtown:
                min_frontage, max_frontage = 12, 20
                occupancy = min(0.98, 0.84 + parameters.building_density * 0.14)
            else:
                junction_weight = _proximity_weight(
                    center_distance, spacing * 0.6, spacing * 2.2)
                min_frontage = round(24 + 10 * junction_weight)
                max_frontage = round(42 + 18 * junction_weight)
                occupancy = min(0.82, 0.56 + parameters.building_density * 0.25)
        elif civic_block:
            family, kind = "civic", "civic"
            min_frontage, max_frontage = 28, 48
            occupancy = min(0.9, 0.5 + parameters.building_density * 0.3)
        else:
            family, kind = "residential", "house"
            frontage_by_type = {
                "compact_city": (12, 20),
                "suburb": (18, 30),
                "rural_village": (25, 42),
                "industrial": (24, 38),
                "riverside": (20, 34),
            }
            min_frontage, max_frontage = frontage_by_type.get(
                parameters.town_type, (17, 29))
            density_floor = {
                "compact_city": 0.88, "suburb": 0.68,
                "rural_village": 0.58, "small_town": 0.98,
            }.get(parameters.town_type, 0.72)
            occupancy = min(
                0.98, 0.68 + parameters.building_density * 0.38)
            occupancy *= density_floor + (1.0 - density_floor) * centrality

        min_frontage /= metres_per_tile
        max_frontage /= metres_per_tile
        lot_depth = min(
            max(30.0 / metres_per_tile, (bottom - top) * rng.uniform(0.27, 0.36)),
            (bottom - top) * 0.44)
        road_setback = ((1.0 if commercial_block else
                         rng.uniform(1.0, 3.0)) / metres_per_tile)
        occupied: list[Polygon] = []
        house_number = rng.randrange(100, 900, 2)
        for lot, orientation in _frontage_lots(
                inner, rng, min_frontage, max_frontage,
                lot_depth, road_setback):
            if rng.random() > occupancy or civic_block and occupied:
                continue
            tx, ty, nx, ny = orientation
            kind_for_lot = kind
            family_for_lot = family
            if civic_block:
                kind_for_lot = rng.choices(
                    ("school", "church", "medical", "civic"),
                    weights=(0.38, 0.3, 0.16, 0.16))[0]
            elif commercial_block:
                if not downtown and block_index == commercial_corridor_anchor:
                    kind_for_lot = "fuel"
                    fuel_created = True
                elif not downtown and block_index == motel_anchor:
                    kind_for_lot = "motel"
                    motel_created = True
                elif not downtown and not fuel_created:
                    kind_for_lot = "fuel"
                    fuel_created = True
                elif not downtown and not motel_created:
                    kind_for_lot = "motel"
                    motel_created = True
                elif not downtown and rng.random() < 0.1:
                    kind_for_lot = "fuel"
                elif not downtown and rng.random() < 0.12:
                    kind_for_lot = "motel"
                else:
                    kind_for_lot = rng.choice(("shop", "restaurant"))
            elif (centrality > 0.48 and rng.random() <
                  (0.22 if parameters.town_type == "compact_city"
                   else 0.12 if parameters.town_type == "small_town"
                   else 0.07)):
                kind_for_lot = "apartment"
                family_for_lot = "residential"
            elif (not commercial_block and mixed_use_weight > 0.1
                and rng.random() < mixed_use_weight * 0.32):
                kind_for_lot = "mixed_shop"

            frontage = lot.exterior.length / 2 - lot_depth
            max_width = min(building_side_limit, frontage - 4.0 / metres_per_tile)
            max_depth = min(building_side_limit, lot_depth - 5.0 / metres_per_tile)
            width, depth = _building_size(
                family_for_lot, int(max_width), int(max_depth), rng, templates,
                max_side=building_side_limit, metres_per_tile=metres_per_tile)
            if width < 3 or depth < 3:
                continue
            if commercial_block and downtown:
                front_setback = rng.uniform(0.3, 1.2) / metres_per_tile
            elif commercial_block:
                front_setback = rng.uniform(1.5, 3.5) / metres_per_tile
            elif parameters.town_type == "compact_city":
                front_setback = rng.uniform(3.5, 7.0) / metres_per_tile
            elif parameters.town_type == "rural_village":
                front_setback = rng.uniform(6.0, 11.0) / metres_per_tile
            else:
                front_setback = rng.uniform(5.0, 9.0) / metres_per_tile
            side_offset = rng.uniform(-0.14, 0.14) * max(
                0.0, frontage - width)
            center_x = lot.centroid.x + tx * side_offset - nx * (
                lot_depth / 2 - front_setback - depth / 2)
            center_y = lot.centroid.y + ty * side_offset - ny * (
                lot_depth / 2 - front_setback - depth / 2)
            footprint = Polygon([
                (center_x - tx * width / 2 - nx * depth / 2,
                 center_y - ty * width / 2 - ny * depth / 2),
                (center_x + tx * width / 2 - nx * depth / 2,
                 center_y + ty * width / 2 - ny * depth / 2),
                (center_x + tx * width / 2 + nx * depth / 2,
                 center_y + ty * width / 2 + ny * depth / 2),
                (center_x - tx * width / 2 + nx * depth / 2,
                 center_y - ty * width / 2 + ny * depth / 2),
            ])
            if (not lot.covers(footprint)
                    or any(footprint.intersects(other) for other in occupied)
                    or (river and _line_near_rect(
                        river, *footprint.bounds, 18.0 / metres_per_tile))):
                continue
            occupied.append(footprint)
            if kind_for_lot == "shop":
                tags = {
                    "building": "retail", "shop": "convenience",
                    "building:levels": str(rng.choice((1, 1, 2))),
                }
            elif kind_for_lot == "restaurant":
                tags = {"building": "restaurant", "amenity": "restaurant",
                        "building:levels": str(rng.choice((1, 1, 2)))}
            elif kind_for_lot == "fuel":
                tags = {"building": "service", "amenity": "fuel",
                        "building:levels": "1"}
            elif kind_for_lot == "motel":
                tags = {"building": "hotel", "tourism": "motel",
                        "building:levels": "1"}
            elif kind_for_lot == "mixed_shop":
                tags = {"building": "apartments", "shop": "convenience",
                        "building:levels": "2"}
            elif kind_for_lot == "industrial":
                tags = {"building": "industrial",
                        "industrial": rng.choice(("factory", "warehouse", "workshop")),
                        "building:levels": "1"}
            elif kind_for_lot in ("school", "church", "medical", "civic"):
                amenity = {"school": "school", "church": "place_of_worship",
                           "medical": "clinic", "civic": "fire_station"}[kind_for_lot]
                tags = {"building": "public", "amenity": amenity,
                        "building:levels": str(rng.choice((1, 1, 2)))}
            elif kind_for_lot == "apartment":
                tags = {"building": "apartments",
                        "building:levels": str(rng.choice((3, 4, 5)))}
            else:
                tags = {"building": "house",
                        "building:levels": str(rng.choice((1, 1, 2))),
                        "building:material": rng.choice(
                            ("brick", "wood", "concrete")),
                        "addr:housenumber": str(house_number)}
                house_number += rng.choice((2, 4, 6))
            add(tags, list(footprint.exterior.coords))
    return features


def _building_size(family: str, available_width: int, available_height: int,
                   rng: random.Random, templates=None,
                   max_side: int = 40,
                   metres_per_tile: float = 1.0) -> tuple[int, int]:
    if templates is not None:
        sizes = [(width, height)
                 for width, height in templates.dimensions(
                     family, min(max_side, available_width - 2),
                     min(max_side, available_height - 2))
                 if width >= 3 and height >= 3]
        if sizes:
            return rng.choice(sizes)
    if family == "commercial":
        sizes = ((12, 12), (14, 16), (16, 14), (18, 18), (20, 16))
    elif family == "industrial":
        sizes = ((22, 30), (28, 36), (34, 24), (40, 32))
    elif family == "civic":
        sizes = ((18, 22), (24, 28), (30, 24), (32, 36))
    else:
        sizes = ((9, 11), (10, 13), (12, 14), (14, 16), (16, 13), (17, 18))
    sizes = tuple((max(3, round(width / metres_per_tile)),
                   max(3, round(height / metres_per_tile)))
                  for width, height in sizes)
    fitting = [(w, h) for w, h in sizes
               if w <= available_width - 2 and h <= available_height - 2]
    return rng.choice(fitting) if fitting else (0, 0)
