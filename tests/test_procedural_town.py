import random
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from generator import osm
from generator.procedural_town import (ROAD_DESIGN_RULES, TownParameters,
                                       _building_size, _collector_stride,
                                       _blocks_from_roads, _district_boundaries,
                                       _district_spacing_scale, _line_near_rect,
                                       _road_curve_scale, _road_shape_offset,
                                       _street_curve_weight,
                                       _road_grid, generate)
from knoxbuild.footprint import place
import numpy as np


class GridProjection:
    width = 720
    height = 720
    meters_per_tile = 1.0

    @staticmethod
    def to_latlon(x, y):
        return y, x

    @staticmethod
    def to_px(lat, lon):
        return lon, lat


class ProceduralTownTests(unittest.TestCase):
    def test_spacing_and_alignment_fade_smoothly_across_districts(self):
        distances = (0, 80, 140, 220, 350, 500)
        scales = [_district_spacing_scale(distance, 720, 100)
                  for distance in distances]

        self.assertEqual(scales[0], 0.68)
        self.assertTrue(all(left <= right
                            for left, right in zip(scales, scales[1:])))
        self.assertTrue(0.68 < scales[1] < scales[2] < 1.0)
        self.assertLess(_street_curve_weight(0, 100),
                        _street_curve_weight(300, 100))
        self.assertGreater(
            _district_spacing_scale(0, 720, 100, 0, 130), scales[0])

    def test_road_shapes_follow_class_specific_design_rules(self):
        positions = range(0, 721, 20)
        offsets = {
            road: max(abs(_road_shape_offset(
                position, 720, 120, road, 1, "organic", 0.2))
                for position in positions)
            for road in ("primary", "secondary", "residential", "service")
        }

        self.assertLess(offsets["primary"], offsets["secondary"])
        self.assertLess(offsets["service"], offsets["residential"])

    def test_district_boundaries_partition_the_map(self):
        zones = _district_boundaries(
            600, 500, [(300, 250)], (520, 250), 100, 0.5, 0.15, 0.2)
        shapes = list(zones.values())

        self.assertTrue(all(shapes[i].intersection(shapes[j]).area < 1e-6
                            for i in range(len(shapes))
                            for j in range(i + 1, len(shapes))))
        self.assertAlmostEqual(unary_union(shapes).area, box(0, 0, 599, 499).area,
                               delta=1e-4)

    def test_blocks_are_polygonized_from_the_connected_road_network(self):
        nodes = _road_grid(300, 300, (150, 150), 70, random.Random(13),
                           "organic", 0.2)
        roads = [LineString((nodes[row][col], nodes[row][col + 1]))
                 for row in range(len(nodes))
                 for col in range(len(nodes[row]) - 1)]
        roads.extend(LineString((nodes[row][col], nodes[row + 1][col]))
                     for row in range(len(nodes) - 1)
                     for col in range(len(nodes[row])))

        blocks = _blocks_from_roads(roads, 300, 300)

        self.assertGreater(len(blocks), 8)
        self.assertAlmostEqual(sum(block.area for block in blocks),
                               299 * 299, delta=1.0)

    def test_road_graph_rejects_dangling_internal_endpoints(self):
        roads = [LineString([(5, 0), (5, 10)]),
                 LineString([(0, 5), (10, 5)]),
                 LineString([(5, 5), (7, 7)])]

        with self.assertRaisesRegex(ValueError, "unconnected endpoint"):
            _blocks_from_roads(roads, 11, 11)

    def test_seed_makes_the_same_features(self):
        params = TownParameters.from_dict({"seed": 42})
        first = generate(GridProjection(), params)
        second = generate(GridProjection(), params)
        self.assertEqual(
            [(feature.tags, feature.geometry) for feature in first],
            [(feature.tags, feature.geometry) for feature in second])

    def test_town_types_select_distinct_realistic_layouts(self):
        city = TownParameters.from_dict({"town_type": "compact_city"})
        village = TownParameters.from_dict({"town_type": "rural_village"})
        self.assertEqual(city.street_pattern, "grid")
        self.assertGreater(city.building_density, village.building_density)
        self.assertGreater(village.block_size_m, city.block_size_m)
        self.assertEqual(
            TownParameters.from_dict(city.to_dict()), city)

    def test_features_include_roads_buildings_and_a_park(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 12, "park_chance": 0.6,
                "commercial_chance": 0.8, "building_density": 1.0,
            }))
        categories = [osm.classify(f.tags, f.geometry[0] == f.geometry[-1])
                      for f in features]
        self.assertIn("road_major", categories)
        self.assertIn("road_minor", categories)
        self.assertIn("building", categories)
        self.assertIn("park", categories)

    def test_network_has_road_hierarchy_and_crossing_bridges(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 23, "street_pattern": "grid", "river_chance": 1,
                "water_chance": 0, "park_chance": 0,
            }))
        highways = {feature.tags.get("highway") for feature in features}
        self.assertTrue({"primary", "secondary", "tertiary", "residential"}
                        <= highways)
        rivers = [feature for feature in features
                  if feature.tags.get("waterway") == "river"]
        self.assertEqual(len(rivers), 1)
        self.assertTrue(any(feature.tags.get("bridge") == "yes"
                            for feature in features))
        self.assertEqual(osm.classify(rivers[0].tags), "water")

    def test_grid_roads_form_a_wide_main_cross_and_narrow_local_streets(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 32, "town_type": "compact_city",
                "river_chance": 0, "park_chance": 0, "water_chance": 0,
            }))
        roads = [feature for feature in features if feature.tags.get("highway")]
        widths = {road.tags["highway"]: int(road.tags["width"].split()[0])
                  for road in roads}
        self.assertGreater(widths["primary"], widths["secondary"])
        self.assertGreater(widths["secondary"], widths["tertiary"])
        self.assertGreater(widths["tertiary"], widths["residential"])

        primary = [road for road in roads if road.tags["highway"] == "primary"]
        primary_lines = [LineString([(lon, lat) for lat, lon in road.geometry])
                 for road in primary]
        self.assertTrue(any(left.intersects(right)
                    for i, left in enumerate(primary_lines)
                    for right in primary_lines[i + 1:]))

    def test_design_speed_is_separate_and_controls_network_rules(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 23, "street_pattern": "grid", "river_chance": 0,
                "park_chance": 0, "water_chance": 0,
                "industrial_chance": 0.2,
            }))
        roads = [feature for feature in features if feature.tags.get("highway")]

        for road in roads:
            rules = ROAD_DESIGN_RULES[road.tags["highway"]]
            self.assertEqual(road.tags["knoxmap:design_speed_kph"],
                             rules["speed_kph"])
            self.assertEqual(road.tags["knoxmap:intersection_spacing_m"],
                             rules["intersection_spacing_m"])
            self.assertNotEqual(road.tags["maxspeed"],
                                f"{rules['speed_kph']} km/h")
        self.assertGreater(_collector_stride(120), _collector_stride(180))
        self.assertLess(_road_curve_scale("primary", 300, 120),
                        _road_curve_scale("residential", 300, 120))

    def test_commercial_core_corridors_and_industrial_edge_are_coherent(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 82, "street_pattern": "grid", "river_chance": 0,
                "park_chance": 0, "water_chance": 0,
                "commercial_chance": 0.5, "civic_chance": 0.15,
                "industrial_chance": 0.2,
            }))
        landuses = [feature for feature in features if feature.tags.get("landuse")]
        names = {feature.tags.get("name") for feature in landuses}
        self.assertIn("Town Centre", names)
        self.assertIn("Commercial Corridor", names)
        industrial = [Polygon([(lon, lat) for lat, lon in feature.geometry])
                      for feature in landuses
                      if feature.tags.get("name") == "Industrial Estate"]
        self.assertGreaterEqual(len(industrial), 2)
        self.assertTrue(any(left.distance(right) <= 16
                            for i, left in enumerate(industrial)
                            for right in industrial[i + 1:]))
        self.assertTrue(any(feature.tags.get("amenity") == "fuel"
                            for feature in features))
        self.assertTrue(any(feature.tags.get("tourism") == "motel"
                            for feature in features))
        self.assertTrue(any(feature.tags.get("name") == "Market Diagonal"
                    for feature in features))
        self.assertTrue(any(feature.tags.get("name") == "Freight Yard Turnaround"
                    for feature in features))
        railway = next(feature for feature in features
                   if feature.tags.get("railway") == "rail")
        rail_line = LineString([(lon, lat) for lat, lon in railway.geometry])
        crossings = [feature for feature in features
                 if feature.tags.get("railway_crossing") == "yes"]
        self.assertTrue(crossings)
        self.assertTrue(all(LineString([(lon, lat) for lat, lon in feature.geometry])
                    .intersects(rail_line) for feature in crossings))

    def test_transition_streets_include_mixed_use_buildings(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 42, "town_type": "compact_city",
                "commercial_chance": 0.5, "civic_chance": 0.1,
                "industrial_chance": 0.05, "river_chance": 0,
                "park_chance": 0, "water_chance": 0,
            }))

        mixed_use = [feature for feature in features
                     if feature.tags.get("building") == "apartments"
                     and feature.tags.get("shop")]
        self.assertTrue(mixed_use)

    def test_organic_streets_form_irregular_blocks_not_a_jittered_grid(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 2026, "river_chance": 0, "park_chance": 0,
                "water_chance": 0,
            }))
        roads = [f for f in features if f.tags.get("highway") in {
            "primary", "secondary", "tertiary", "residential"}]
        angled = sum(
            abs(b[1] - a[1]) > 1 and abs(b[0] - a[0]) > 1
            for road in roads for a, b in zip(road.geometry, road.geometry[1:]))
        self.assertGreater(angled, len(roads) // 3)
        classes = {road.tags["highway"] for road in roads}
        self.assertGreaterEqual(len(classes), 3)
        self.assertTrue(any(len(road.geometry) > 3 for road in roads))

    def test_generated_roads_share_intersections_and_zones_have_distinct_uses(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 82, "river_chance": 0, "park_chance": 0,
                "water_chance": 0, "commercial_chance": 0.5,
                "civic_chance": 0.15, "industrial_chance": 0.2,
            }))
        roads = [feature for feature in features
                 if feature.tags.get("highway") in {
                     "primary", "secondary", "tertiary", "residential"}]
        endpoints = {}
        for road in roads:
            for point in (road.geometry[0], road.geometry[-1]):
                key = (round(point[0], 7), round(point[1], 7))
                endpoints[key] = endpoints.get(key, 0) + 1
        self.assertGreater(sum(count >= 2 for count in endpoints.values()),
                           len(roads) // 4)
        landuses = {feature.tags.get("landuse") for feature in features}
        self.assertTrue({"residential", "commercial", "industrial"} <= landuses)

    def test_named_streets_continue_through_intersections(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 47, "town_type": "compact_city",
                "street_pattern": "grid", "river_chance": 0,
                "park_chance": 0, "water_chance": 0,
            }))
        roads = [feature for feature in features
                 if feature.tags.get("highway") in {
                     "primary", "secondary", "tertiary", "residential"}]
        named = {}
        for road in roads:
            named.setdefault(road.tags["name"], []).append(road)
        routes = [ways for ways in named.values() if len(ways) > 1]
        self.assertTrue(routes)
        self.assertTrue(any(
            len({(round(point[0], 7), round(point[1], 7))
                 for way in ways for point in (way.geometry[0], way.geometry[-1])})
            < 2 * len(ways)
            for ways in routes))

    def test_generated_road_names_survive_the_feature_cache_for_map_building(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 74, "river_chance": 0, "park_chance": 0,
                "water_chance": 0,
            }))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "procedural_osm.json.gz"
            osm.save_cache(str(cache), (0, 0, 720, 720), features)
            cached = osm.load_cache(str(cache), (0, 0, 720, 720))

        self.assertIsNotNone(cached)
        named_roads = [f for f in cached
                       if f.tags.get("highway") in {
                           "primary", "secondary", "tertiary", "residential"}
                       and f.tags.get("name")]
        self.assertTrue(named_roads)
        self.assertEqual(
            [(f.tags, f.geometry) for f in features],
            [(f.tags, f.geometry) for f in cached])

    def test_grid_blocks_have_spacing_variation_instead_of_one_repeated_pitch(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 48, "town_type": "compact_city",
                "street_pattern": "grid", "river_chance": 0,
                "park_chance": 0, "water_chance": 0,
            }))
        roads = [feature for feature in features
                 if feature.tags.get("highway") in {
                     "primary", "secondary", "tertiary", "residential"}]
        x_positions = sorted({round(lon, 4) for road in roads
                              for lat, lon in road.geometry
                              if max(p[0] for p in road.geometry)
                              - min(p[0] for p in road.geometry) < 1e-5})
        gaps = {round(b - a, 1) for a, b in zip(x_positions, x_positions[1:])}
        self.assertGreaterEqual(len(gaps), 3)

    def test_compact_city_has_multiple_centers_and_more_apartments(self):
        params = {
            "seed": 93, "river_chance": 0, "park_chance": 0,
            "water_chance": 0, "commercial_chance": 0,
            "civic_chance": 0,
        }
        compact = generate(
            GridProjection(),
            TownParameters.from_dict({**params, "town_type": "compact_city"}))
        village = generate(
            GridProjection(),
            TownParameters.from_dict({**params, "town_type": "rural_village"}))
        compact_flats = sum(f.tags.get("building") == "apartments"
                            for f in compact)
        village_flats = sum(f.tags.get("building") == "apartments"
                            for f in village)
        self.assertGreater(compact_flats, village_flats)

    def test_irregular_selection_does_not_cut_through_generated_buildings(self):
        outline = {
            "type": "Polygon",
            "coordinates": [[(100, 100), (620, 130), (540, 620),
                            (250, 560), (100, 100)]],
        }
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 54, "river_chance": 0, "park_chance": 0,
                "water_chance": 0,
            }),
            selection_shape=outline)
        boundary = Polygon(outline["coordinates"][0])
        buildings = [feature for feature in features
                     if "building" in feature.tags]
        self.assertTrue(buildings)
        for building in buildings:
            footprint = Polygon([(lon, lat) for lat, lon in building.geometry])
            self.assertTrue(boundary.covers(footprint))

    def test_river_has_a_greenway_and_buildings_respect_its_setback(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 31, "river_chance": 1, "water_chance": 0,
                "park_chance": 0, "building_density": 1,
            }))
        river = next(f for f in features if f.tags.get("waterway") == "river")
        river_xy = [(lon, lat) for lat, lon in river.geometry]
        self.assertTrue(any(f.tags.get("name") == "Riverbank Greenway"
                            for f in features))
        buildings = [f for f in features if "building" in f.tags]
        self.assertTrue(buildings)
        for building in buildings:
            points = [(lon, lat) for lat, lon in building.geometry]
            self.assertFalse(_line_near_rect(
                river_xy, min(x for x, _ in points), min(y for _, y in points),
                max(x for x, _ in points), max(y for _, y in points), 18))

    def test_rural_village_adds_a_road_along_the_riverbank(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 314, "town_type": "rural_village",
                "river_chance": 1, "park_chance": 0, "water_chance": 0,
            }))
        river = next(feature for feature in features
                     if feature.tags.get("waterway") == "river")
        river_road = next(feature for feature in features
                          if feature.tags.get("name") == "River Road")
        river_line = LineString([(lon, lat) for lat, lon in river.geometry])
        road_line = LineString([(lon, lat) for lat, lon in river_road.geometry])

        self.assertGreater(road_line.distance(river_line), 4)
        self.assertTrue(any(
            min(x, y, GridProjection.width - 1 - x,
                GridProjection.height - 1 - y) < 1e-5
            for lat, lon in (river_road.geometry[0], river_road.geometry[-1])
            for x, y in [(lon, lat)]))

    def test_homes_are_spread_along_frontages_with_yards_and_no_overlap(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 25, "river_chance": 0, "park_chance": 0,
                "water_chance": 0,
            }))
        homes = [Polygon([(lon, lat) for lat, lon in feature.geometry])
                 for feature in features
                 if feature.tags.get("building") == "house"]
        self.assertGreaterEqual(len(homes), 40)
        for index, home in enumerate(homes):
            for other in homes[index + 1:]:
                self.assertLessEqual(home.intersection(other).area, 0.01)
        self.assertGreater(sum(home.area for home in homes), 1_000)
        self.assertLess(sum(home.area for home in homes), 80_000)
        nearest = sorted(
            min(home.distance(other) for j, other in enumerate(homes)
                if index != j)
            for index, home in enumerate(homes))
        self.assertGreater(nearest[len(nearest) // 2], 6)

    def test_residential_culdesacs_have_a_connected_turnaround(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 91, "town_type": "compact_city",
                "street_pattern": "grid", "river_chance": 0,
                "park_chance": 0, "water_chance": 0,
                "commercial_chance": 0, "civic_chance": 0,
                "industrial_chance": 0,
            }))
        courts = [feature for feature in features
                  if feature.tags.get("name") == "Residential Court"]
        loops = [feature for feature in courts
                 if feature.geometry[0] == feature.geometry[-1]]
        spurs = [feature for feature in courts
                 if feature.geometry[0] != feature.geometry[-1]]
        self.assertTrue(loops)
        self.assertTrue(spurs)
        loop_lines = [LineString([(lon, lat) for lat, lon in feature.geometry])
                      for feature in loops]
        spur_lines = [LineString([(lon, lat) for lat, lon in feature.geometry])
                      for feature in spurs]
        self.assertTrue(any(
            min(loop.distance(Point(endpoint))
                for endpoint in (spur.coords[0], spur.coords[-1])) < 1e-5
            for loop in loop_lines for spur in spur_lines))

    def test_house_footprints_keep_real_world_size_at_coarser_scale(self):
        width, depth = _building_size(
            "residential", 10, 12, random.Random(4),
            metres_per_tile=2)
        self.assertGreaterEqual(width * 2, 9)
        self.assertLessEqual(width * 2, 17)
        self.assertGreaterEqual(depth * 2, 11)
        self.assertLessEqual(depth * 2, 18)

    def test_ponds_and_organic_roads_follow_parameters(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 24, "street_pattern": "organic",
                "road_irregularity": 0.3, "river_chance": 0,
                "water_chance": 0.4, "park_chance": 0,
            }))
        self.assertTrue(any(feature.tags.get("natural") == "water"
                            for feature in features))
        self.assertTrue(any(feature.tags.get("highway") == "primary"
                            and len(feature.geometry) > 2
                            for feature in features))

    def test_rectangular_building_lot_keeps_its_declared_dimensions(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({"seed": 4, "park_chance": 0.0}))
        building = next(f for f in features if "building" in f.tags)
        points = [(lon, lat) for lat, lon in building.geometry]
        placed, reason = place(points, np.zeros((720, 720), dtype=bool),
                               alignment="real")
        self.assertIsNotNone(placed, reason)
        area = Polygon(points).area
        self.assertGreater(placed.tiles, area * 0.8)
        self.assertLess(placed.tiles, area * 1.2)

    def test_invalid_parameters_are_rejected(self):
        for data in ({"block_size_m": 0}, {"park_chance": 0.7},
                     {"building_density": float("nan")}, {"seed": -1},
                     {"street_pattern": "radial"},
                     {"town_type": "unknown"},
                     {"road_irregularity": 0.5},
                     {"max_building_side": 201},
                     {"max_building_side": 12.5}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                TownParameters.from_dict(data)


if __name__ == "__main__":
    unittest.main()
