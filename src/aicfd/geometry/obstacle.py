"""Closed polyline obstacles for the first two-dimensional prototypes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import acos, atan, cos, degrees, isfinite, pi, radians, sin, sqrt

from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.validation import explain_validity

Point2D = tuple[float, float]


@dataclass(frozen=True, slots=True)
class BoundaryVertex2D:
    """A polyline vertex and simple estimates of its local shape."""

    point: Point2D
    turning_angle: float
    curvature: float
    local_length: float

    @property
    def turning_angle_degrees(self) -> float:
        """Change in tangent direction at the vertex, in degrees."""

        return degrees(self.turning_angle)


@dataclass(frozen=True, slots=True)
class BoundarySample2D:
    """Closest point and local boundary data for one query point."""

    point: Point2D
    signed_distance: float
    normal: Point2D
    curvature: float
    segment_index: int


class Obstacle2D:
    """A solid obstacle bounded by a valid, counter-clockwise polyline.

    The polyline is the geometry authority. This mirrors an STL-based workflow:
    clipping and snapping are exact with respect to the supplied segments, while
    the fidelity to an underlying CAD curve depends on how finely it was sampled.
    """

    def __init__(self, vertices: Sequence[Sequence[float]], *, name: str) -> None:
        clean_vertices = self._clean_vertices(vertices)
        polygon = orient(Polygon(clean_vertices), sign=1.0)
        if not polygon.is_valid:
            reason = explain_validity(polygon)
            raise ValueError(f"obstacle polygon is invalid: {reason}")
        if polygon.is_empty or polygon.area <= 0.0:
            raise ValueError("obstacle must enclose a positive area")

        self.name = str(name)
        self._polygon = polygon
        self._vertices = tuple(
            (float(x), float(y)) for x, y in polygon.exterior.coords[:-1]
        )
        self._vertex_features = self._measure_vertex_features(self._vertices)

    @staticmethod
    def _clean_vertices(vertices: Sequence[Sequence[float]]) -> tuple[Point2D, ...]:
        if isinstance(vertices, (str, bytes)):
            raise TypeError("vertices must be a sequence of coordinate pairs")

        clean: list[Point2D] = []
        for raw_point in vertices:
            if len(raw_point) != 2:
                raise ValueError("every obstacle vertex must have two coordinates")
            point = (float(raw_point[0]), float(raw_point[1]))
            if not all(isfinite(value) for value in point):
                raise ValueError("obstacle coordinates must be finite")
            if not clean or point != clean[-1]:
                clean.append(point)

        if len(clean) >= 2 and clean[0] == clean[-1]:
            clean.pop()
        if len(clean) < 3:
            raise ValueError("an obstacle needs at least three distinct vertices")
        return tuple(clean)

    @staticmethod
    def _measure_vertex_features(
        vertices: tuple[Point2D, ...],
    ) -> tuple[BoundaryVertex2D, ...]:
        features: list[BoundaryVertex2D] = []
        count = len(vertices)
        for index, vertex in enumerate(vertices):
            previous = vertices[(index - 1) % count]
            following = vertices[(index + 1) % count]
            incoming = (vertex[0] - previous[0], vertex[1] - previous[1])
            outgoing = (following[0] - vertex[0], following[1] - vertex[1])
            incoming_length = sqrt(incoming[0] ** 2 + incoming[1] ** 2)
            outgoing_length = sqrt(outgoing[0] ** 2 + outgoing[1] ** 2)
            if incoming_length == 0.0 or outgoing_length == 0.0:
                raise ValueError("obstacle contains a zero-length segment")

            cosine = (
                incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
            ) / (incoming_length * outgoing_length)
            turning_angle = acos(max(-1.0, min(1.0, cosine)))
            local_length = 0.5 * (incoming_length + outgoing_length)
            features.append(
                BoundaryVertex2D(
                    point=vertex,
                    turning_angle=turning_angle,
                    curvature=turning_angle / local_length,
                    local_length=local_length,
                )
            )
        return tuple(features)

    @property
    def polygon(self) -> Polygon:
        """Shapely polygon used for robust predicates and clipping."""

        return self._polygon

    @property
    def boundary(self) -> BaseGeometry:
        """Closed polyline forming the obstacle boundary."""

        return self._polygon.boundary

    @property
    def vertices(self) -> tuple[Point2D, ...]:
        """Counter-clockwise vertices without a repeated closing point."""

        return self._vertices

    @property
    def vertex_features(self) -> tuple[BoundaryVertex2D, ...]:
        """Turning angle and discrete curvature at every boundary vertex."""

        return self._vertex_features

    @classmethod
    def circle(
        cls,
        center: Point2D,
        radius: float,
        *,
        segments: int = 128,
        name: str = "circle",
    ) -> Obstacle2D:
        """Create a regular-polygon approximation to a circular obstacle."""

        if radius <= 0.0 or not isfinite(radius):
            raise ValueError("circle radius must be finite and positive")
        if not isinstance(segments, int) or isinstance(segments, bool):
            raise TypeError("segments must be an integer")
        if segments < 12:
            raise ValueError("a circle needs at least 12 segments")
        center_x, center_y = (float(value) for value in center)
        vertices = tuple(
            (
                center_x + radius * cos(2.0 * pi * index / segments),
                center_y + radius * sin(2.0 * pi * index / segments),
            )
            for index in range(segments)
        )
        return cls(vertices, name=name)

    @classmethod
    def naca4(
        cls,
        code: str = "0012",
        *,
        chord: float = 1.0,
        leading_edge: Point2D = (0.0, 0.0),
        angle_degrees: float = 0.0,
        points_per_surface: int = 81,
        sharp_trailing_edge: bool = True,
    ) -> Obstacle2D:
        """Create a cosine-sampled NACA four-digit airfoil polygon.

        ``leading_edge`` locates the unrotated leading edge and ``angle_degrees``
        rotates the section counter-clockwise about that point.
        """

        if len(code) != 4 or not code.isdigit():
            raise ValueError("a NACA four-digit code must contain exactly four digits")
        if chord <= 0.0 or not isfinite(chord):
            raise ValueError("airfoil chord must be finite and positive")
        if not isinstance(points_per_surface, int) or isinstance(
            points_per_surface, bool
        ):
            raise TypeError("points_per_surface must be an integer")
        if points_per_surface < 8:
            raise ValueError("points_per_surface must be at least 8")

        maximum_camber = int(code[0]) / 100.0
        camber_location = int(code[1]) / 10.0
        thickness = int(code[2:]) / 100.0
        if thickness == 0.0:
            raise ValueError("a zero-thickness airfoil does not enclose an obstacle")
        if maximum_camber > 0.0 and camber_location == 0.0:
            raise ValueError("a cambered NACA code needs a non-zero camber location")

        normalized_x = tuple(
            0.5 * (1.0 - cos(pi * index / (points_per_surface - 1)))
            for index in range(points_per_surface)
        )
        upper: list[Point2D] = []
        lower: list[Point2D] = []
        final_coefficient = 0.1036 if sharp_trailing_edge else 0.1015

        for x_over_chord in normalized_x:
            thickness_offset = 5.0 * thickness * (
                0.2969 * sqrt(x_over_chord)
                - 0.1260 * x_over_chord
                - 0.3516 * x_over_chord**2
                + 0.2843 * x_over_chord**3
                - final_coefficient * x_over_chord**4
            )
            if sharp_trailing_edge and x_over_chord == 1.0:
                thickness_offset = 0.0

            if maximum_camber == 0.0:
                camber = 0.0
                slope = 0.0
            elif x_over_chord < camber_location:
                camber = maximum_camber / camber_location**2 * (
                    2.0 * camber_location * x_over_chord - x_over_chord**2
                )
                slope = 2.0 * maximum_camber / camber_location**2 * (
                    camber_location - x_over_chord
                )
            else:
                camber = maximum_camber / (1.0 - camber_location) ** 2 * (
                    1.0
                    - 2.0 * camber_location
                    + 2.0 * camber_location * x_over_chord
                    - x_over_chord**2
                )
                slope = 2.0 * maximum_camber / (1.0 - camber_location) ** 2 * (
                    camber_location - x_over_chord
                )

            surface_angle = atan(slope)
            upper.append(
                (
                    x_over_chord - thickness_offset * sin(surface_angle),
                    camber + thickness_offset * cos(surface_angle),
                )
            )
            lower.append(
                (
                    x_over_chord + thickness_offset * sin(surface_angle),
                    camber - thickness_offset * cos(surface_angle),
                )
            )

        normalized_vertices = list(reversed(upper)) + lower[1:]
        if sharp_trailing_edge:
            normalized_vertices.pop()

        rotation = radians(float(angle_degrees))
        cosine = cos(rotation)
        sine = sin(rotation)
        leading_x, leading_y = (float(value) for value in leading_edge)
        vertices = tuple(
            (
                leading_x + chord * (x * cosine - y * sine),
                leading_y + chord * (x * sine + y * cosine),
            )
            for x, y in normalized_vertices
        )
        return cls(vertices, name=f"NACA {code}")

    def nearest_boundary(self, point: Sequence[float]) -> BoundarySample2D:
        """Project a point onto the polyline and return local geometry data."""

        if len(point) != 2:
            raise ValueError("a 2D query point must have two coordinates")
        query = (float(point[0]), float(point[1]))
        best_distance_squared = float("inf")
        best_point: Point2D = self._vertices[0]
        best_normal: Point2D = (0.0, 0.0)
        best_curvature = 0.0
        best_segment = 0
        count = len(self._vertices)

        for index, start in enumerate(self._vertices):
            end = self._vertices[(index + 1) % count]
            delta = (end[0] - start[0], end[1] - start[1])
            length_squared = delta[0] ** 2 + delta[1] ** 2
            parameter = (
                (query[0] - start[0]) * delta[0]
                + (query[1] - start[1]) * delta[1]
            ) / length_squared
            parameter = max(0.0, min(1.0, parameter))
            projected = (
                start[0] + parameter * delta[0],
                start[1] + parameter * delta[1],
            )
            distance_squared = (
                (query[0] - projected[0]) ** 2 + (query[1] - projected[1]) ** 2
            )
            if distance_squared >= best_distance_squared:
                continue

            length = sqrt(length_squared)
            next_index = (index + 1) % count
            best_distance_squared = distance_squared
            best_point = projected
            best_normal = (delta[1] / length, -delta[0] / length)
            best_curvature = (
                (1.0 - parameter) * self._vertex_features[index].curvature
                + parameter * self._vertex_features[next_index].curvature
            )
            best_segment = index

        distance = sqrt(best_distance_squared)
        if distance <= 1.0e-14:
            signed_distance = 0.0
        elif self._polygon.contains(Point(query)):
            signed_distance = -distance
        else:
            signed_distance = distance
        return BoundarySample2D(
            point=best_point,
            signed_distance=signed_distance,
            normal=best_normal,
            curvature=best_curvature,
            segment_index=best_segment,
        )

    def features_near(
        self, region: BaseGeometry, max_distance: float
    ) -> tuple[BoundaryVertex2D, ...]:
        """Return polyline vertices lying in or close to a geometric region."""

        if max_distance < 0.0:
            raise ValueError("max_distance must be non-negative")
        return tuple(
            feature
            for feature in self._vertex_features
            if region.distance(Point(feature.point)) <= max_distance
        )
