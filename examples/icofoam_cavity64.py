#!/usr/bin/env python3
"""A readable, one-file finite-volume/PISO solver for ``cavity64``.

This program is an educational translation of the *solution procedure* used by
OpenFOAM Foundation v7 ``icoFoam``.  It solves the same incompressible, laminar
lid-driven cavity problem as our reference trajectory:

    domain:       0.1 m x 0.1 m
    cells:        64 x 64
    lid velocity: (1, 0) m/s
    viscosity:    0.01 m^2/s
    time step:    0.00125 s
    end time:     0.5 s (400 steps)
    PISO:         two pressure correctors

The important OpenFOAM-to-Python correspondence is:

    OpenFOAM icoFoam                     this file
    -----------------------------------  -----------------------------------
    fvm::ddt(U)                          implicit Euler diagonal
    fvm::div(phi, U)                     centered implicit face convection
    -fvm::laplacian(nu, U)               centered implicit diffusion
    solve(UEqn == -grad(p))              solve_momentum_component()
    rAU = 1/UEqn.A()                     coefficients.r_au
    HbyA = rAU*UEqn.H()                  calculate_h_by_a()
    phiHbyA                              provisional_face_velocity()
    laplacian(rAU,p) == div(phiHbyA)     solve_pressure()
    phi = phiHbyA - pEqn.flux()          correct_face_velocity()
    U = HbyA - rAU*grad(p)               cell-velocity correction

It is intentionally not a bit-for-bit rewrite of OpenFOAM.  OpenFOAM handles
general polyhedral meshes, patch fields, dimensions, run-time-selectable
schemes, and production linear solvers.  Here we specialize those abstractions
to one uniform Cartesian mesh and use only NumPy.  Pressure is fixed by a
zero-mean gauge rather than OpenFOAM's ``pRefCell=0``; only pressure differences
have physical meaning.

The corresponding OpenFOAM Foundation v7 source is:

    https://cpp.openfoam.org/v7/icoFoam_8C_source.html

Run all 400 steps:

    uv run python examples/icofoam_cavity64.py

Run a quick two-step walkthrough with detailed PISO output:

    uv run python examples/icofoam_cavity64.py --steps 2 --verbose-piso

Compare the final state with our imported OpenFOAM trajectory:

    uv run python examples/icofoam_cavity64.py \
        --reference .data/processed/cavity64

The output is one compressed NumPy archive.  ``U`` and ``p`` are cell-centered;
``face_u`` and ``face_v`` are the final conservative face-normal velocities.
Array row ``j=0`` is the bottom of the cavity and column ``i=0`` is the left.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CavityCase:
    """The physical and numerical controls that OpenFOAM reads from dictionaries."""

    nx: int = 64
    ny: int = 64
    length_x: float = 0.1
    length_y: float = 0.1
    viscosity: float = 0.01
    lid_velocity: float = 1.0
    delta_t: float = 0.00125
    steps: int = 400
    piso_correctors: int = 2
    momentum_tolerance: float = 1.0e-8
    pressure_tolerance: float = 1.0e-10
    momentum_max_iterations: int = 400
    pressure_max_iterations: int = 400

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("the cavity needs at least two cells per direction")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.piso_correctors < 1:
            raise ValueError("at least one PISO corrector is required")
        positive = (
            self.length_x,
            self.length_y,
            self.viscosity,
            self.delta_t,
            self.momentum_tolerance,
            self.pressure_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "physical lengths, coefficients, and tolerances must be positive"
            )
        if self.momentum_max_iterations < 1 or self.pressure_max_iterations < 1:
            raise ValueError("linear-solver iteration limits must be positive")

    @property
    def dx(self) -> float:
        return self.length_x / self.nx

    @property
    def dy(self) -> float:
        return self.length_y / self.ny

    @property
    def end_time(self) -> float:
        return self.steps * self.delta_t

    @property
    def reynolds_number(self) -> float:
        return self.lid_velocity * self.length_x / self.viscosity


@dataclass(frozen=True, slots=True)
class MomentumCoefficients:
    """Five-point coefficients for one component of the momentum equation.

    The equation in every cell is

        diagonal*U_P + east*U_E + west*U_W
                     + north*U_N + south*U_S = right_hand_side.

    Both velocity components share these coefficients because the frozen face
    flux ``phi`` advects them in exactly the same way.
    """

    diagonal: FloatArray
    east: FloatArray
    west: FloatArray
    north: FloatArray
    south: FloatArray

    @property
    def r_au(self) -> FloatArray:
        """OpenFOAM's reciprocal momentum diagonal, ``1/UEqn.A()``."""

        return 1.0 / self.diagonal


@dataclass(frozen=True, slots=True)
class LinearSolve:
    """Small diagnostic returned by each iterative linear solve."""

    iterations: int
    initial_residual: float
    final_residual: float


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Diagnostics for one physical time step."""

    momentum_u: LinearSolve
    momentum_v: LinearSolve
    pressure: tuple[LinearSolve, ...]
    courant_max: float
    continuity_l1: float
    continuity_linf: float
    kinetic_energy: float


def l2_norm(values: FloatArray) -> float:
    """Root-mean-square norm; its magnitude does not grow with mesh size."""

    return float(np.sqrt(np.mean(np.square(values))))


def divergence(face_u: FloatArray, face_v: FloatArray, case: CavityCase) -> FloatArray:
    """Cell divergence from conservative face-normal velocities.

    ``face_u[j, i]`` is positive from left to right on a vertical face.
    ``face_v[j, i]`` is positive from bottom to top on a horizontal face.
    Consequently, each internal face enters two adjacent cells with opposite
    signs: this is the finite-volume conservation property in array form.
    """

    return (face_u[:, 1:] - face_u[:, :-1]) / case.dx + (
        face_v[1:, :] - face_v[:-1, :]
    ) / case.dy


def cell_gradient_neumann(
    field: FloatArray, case: CavityCase
) -> tuple[FloatArray, FloatArray]:
    """Gauss-linear cell gradient with zero normal gradient at every wall."""

    face_x = np.empty((case.ny, case.nx + 1), dtype=np.float64)
    face_y = np.empty((case.ny + 1, case.nx), dtype=np.float64)

    face_x[:, 1:-1] = 0.5 * (field[:, :-1] + field[:, 1:])
    face_x[:, 0] = field[:, 0]
    face_x[:, -1] = field[:, -1]

    face_y[1:-1, :] = 0.5 * (field[:-1, :] + field[1:, :])
    face_y[0, :] = field[0, :]
    face_y[-1, :] = field[-1, :]

    gradient_x = (face_x[:, 1:] - face_x[:, :-1]) / case.dx
    gradient_y = (face_y[1:, :] - face_y[:-1, :]) / case.dy
    return gradient_x, gradient_y


def interpolate_normal_velocity(
    u: FloatArray,
    v: FloatArray,
    case: CavityCase,
) -> tuple[FloatArray, FloatArray]:
    """Linearly interpolate cell velocity to faces.

    All cavity walls are impermeable.  The moving lid has tangential velocity,
    but its normal velocity is still zero, so every boundary face flux is zero.
    """

    face_u = np.zeros((case.ny, case.nx + 1), dtype=np.float64)
    face_v = np.zeros((case.ny + 1, case.nx), dtype=np.float64)
    face_u[:, 1:-1] = 0.5 * (u[:, :-1] + u[:, 1:])
    face_v[1:-1, :] = 0.5 * (v[:-1, :] + v[1:, :])
    return face_u, face_v


def assemble_momentum_coefficients(
    face_u: FloatArray,
    face_v: FloatArray,
    case: CavityCase,
) -> MomentumCoefficients:
    """Assemble implicit Euler + centered convection + diffusion coefficients.

    The face velocities are held fixed during this time step.  That makes the
    nonlinear convection term a linear transport operator for the new ``U``.
    This is what ``fvm::div(phi,U)`` means at a high level.
    """

    shape = (case.ny, case.nx)
    east = np.zeros(shape, dtype=np.float64)
    west = np.zeros(shape, dtype=np.float64)
    north = np.zeros(shape, dtype=np.float64)
    south = np.zeros(shape, dtype=np.float64)

    diffusion_x = case.viscosity / case.dx**2
    diffusion_y = case.viscosity / case.dy**2

    # Each centered convective face value is (U_P + U_N)/2.  Its coefficient
    # therefore contributes half to the current cell and half to its neighbor.
    east[:, :-1] = 0.5 * face_u[:, 1:-1] / case.dx - diffusion_x
    west[:, 1:] = -0.5 * face_u[:, 1:-1] / case.dx - diffusion_x
    north[:-1, :] = 0.5 * face_v[1:-1, :] / case.dy - diffusion_y
    south[1:, :] = -0.5 * face_v[1:-1, :] / case.dy - diffusion_y

    # An interior diffusive face adds nu/h^2 to the diagonal.  A Dirichlet wall
    # is only h/2 from the cell center, so its contribution is 2*nu/h^2.
    diagonal = np.full(
        shape,
        1.0 / case.delta_t + 2.0 * diffusion_x + 2.0 * diffusion_y,
        dtype=np.float64,
    )
    diagonal[:, 0] += diffusion_x
    diagonal[:, -1] += diffusion_x
    diagonal[0, :] += diffusion_y
    diagonal[-1, :] += diffusion_y

    # Half of div(phi) appears on the diagonal for centered convection.  Once
    # pressure correction has made phi conservative, this is nearly zero.
    diagonal += 0.5 * (
        (face_u[:, 1:] - face_u[:, :-1]) / case.dx
        + (face_v[1:, :] - face_v[:-1, :]) / case.dy
    )
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        raise RuntimeError("momentum diagonal lost positivity")

    return MomentumCoefficients(diagonal, east, west, north, south)


def momentum_source(
    old_value: FloatArray,
    case: CavityCase,
    *,
    component: str,
) -> FloatArray:
    """Euler old-time source plus fixed-value wall diffusion sources."""

    source = old_value / case.delta_t
    if component == "u":
        # Only the top wall is nonzero.  The distance from its face to the top
        # cell center is dy/2, hence the factor two.
        source = source.copy()
        source[-1, :] += 2.0 * case.viscosity * case.lid_velocity / case.dy**2
    elif component != "v":
        raise ValueError("momentum component must be 'u' or 'v'")
    return source


def neighbor_sum(coefficients: MomentumCoefficients, value: FloatArray) -> FloatArray:
    """Return all off-diagonal terms of the five-point momentum matrix."""

    result = np.zeros_like(value)
    result[:, :-1] += coefficients.east[:, :-1] * value[:, 1:]
    result[:, 1:] += coefficients.west[:, 1:] * value[:, :-1]
    result[:-1, :] += coefficients.north[:-1, :] * value[1:, :]
    result[1:, :] += coefficients.south[1:, :] * value[:-1, :]
    return result


def apply_momentum(coefficients: MomentumCoefficients, value: FloatArray) -> FloatArray:
    """Matrix-free momentum matrix-vector product."""

    return coefficients.diagonal * value + neighbor_sum(coefficients, value)


def solve_momentum_component(
    coefficients: MomentumCoefficients,
    right_hand_side: FloatArray,
    initial: FloatArray,
    case: CavityCase,
) -> tuple[FloatArray, LinearSolve]:
    """Solve one momentum component with red-black Gauss-Seidel.

    A Cartesian five-point stencil connects only opposite checkerboard colors.
    Updating red cells and then black cells is therefore a compact vectorized
    version of Gauss-Seidel.  OpenFOAM's tutorial uses symmetric Gauss-Seidel;
    the algorithms differ in details but solve the same assembled equation.
    """

    value = np.array(initial, dtype=np.float64, copy=True)
    rows, columns = np.indices(value.shape)
    red = (rows + columns) % 2 == 0
    black = ~red

    residual = right_hand_side - apply_momentum(coefficients, value)
    initial_residual = l2_norm(residual)
    if initial_residual == 0.0:
        return value, LinearSolve(0, 0.0, 0.0)

    target = max(1.0e-12, case.momentum_tolerance * initial_residual)
    final_residual = initial_residual
    for iteration in range(1, case.momentum_max_iterations + 1):
        for color in (red, black):
            estimate = (
                right_hand_side - neighbor_sum(coefficients, value)
            ) / coefficients.diagonal
            value[color] = estimate[color]

        # Checking every second sweep reduces Python overhead without changing
        # the converged equation.
        if iteration == 1 or iteration % 2 == 0:
            residual = right_hand_side - apply_momentum(coefficients, value)
            final_residual = l2_norm(residual)
            if final_residual <= target:
                return value, LinearSolve(
                    iteration,
                    initial_residual,
                    final_residual,
                )

    raise RuntimeError(
        "momentum solve did not converge: "
        f"initial={initial_residual:.3e}, final={final_residual:.3e}"
    )


def calculate_h_by_a(
    coefficients: MomentumCoefficients,
    source: FloatArray,
    current_value: FloatArray,
) -> FloatArray:
    """Return ``rAU*H``: all momentum terms except the current pressure gradient."""

    return (source - neighbor_sum(coefficients, current_value)) / coefficients.diagonal


def provisional_face_velocity(
    h_u: FloatArray,
    h_v: FloatArray,
    old_u: FloatArray,
    old_v: FloatArray,
    old_face_u: FloatArray,
    old_face_v: FloatArray,
    r_au: FloatArray,
    case: CavityCase,
) -> tuple[FloatArray, FloatArray]:
    """Construct ``phiHbyA`` including a simplified Euler ``ddtCorr``.

    A collocated grid stores velocity at cells but conservation is enforced on
    faces.  Interpolating the two independently can introduce an inconsistency.
    OpenFOAM's ``ddtCorr(U,phi)`` carries the old face/cell difference into the
    new provisional flux.  For this fixed Cartesian Euler case it reduces to the
    correction below.
    """

    face_u, face_v = interpolate_normal_velocity(h_u, h_v, case)
    old_interpolated_u, old_interpolated_v = interpolate_normal_velocity(
        old_u,
        old_v,
        case,
    )
    r_au_x = 0.5 * (r_au[:, :-1] + r_au[:, 1:])
    r_au_y = 0.5 * (r_au[:-1, :] + r_au[1:, :])
    face_u[:, 1:-1] += (
        r_au_x / case.delta_t * (old_face_u[:, 1:-1] - old_interpolated_u[:, 1:-1])
    )
    face_v[1:-1, :] += (
        r_au_y / case.delta_t * (old_face_v[1:-1, :] - old_interpolated_v[1:-1, :])
    )
    return face_u, face_v


def pressure_face_coefficients(
    r_au: FloatArray,
    case: CavityCase,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Coefficients of ``-div(rAU grad(p))`` with zero-gradient walls."""

    coefficient_x = 0.5 * (r_au[:, :-1] + r_au[:, 1:]) / case.dx**2
    coefficient_y = 0.5 * (r_au[:-1, :] + r_au[1:, :]) / case.dy**2
    diagonal = np.zeros_like(r_au)
    diagonal[:, :-1] += coefficient_x
    diagonal[:, 1:] += coefficient_x
    diagonal[:-1, :] += coefficient_y
    diagonal[1:, :] += coefficient_y
    return coefficient_x, coefficient_y, diagonal


def apply_pressure_operator(
    pressure: FloatArray,
    coefficient_x: FloatArray,
    coefficient_y: FloatArray,
) -> FloatArray:
    """Apply the positive semi-definite operator ``-div(rAU grad(p))``."""

    result = np.zeros_like(pressure)
    difference_x = pressure[:, :-1] - pressure[:, 1:]
    result[:, :-1] += coefficient_x * difference_x
    result[:, 1:] -= coefficient_x * difference_x
    difference_y = pressure[:-1, :] - pressure[1:, :]
    result[:-1, :] += coefficient_y * difference_y
    result[1:, :] -= coefficient_y * difference_y
    return result


def solve_pressure(
    provisional_u: FloatArray,
    provisional_v: FloatArray,
    r_au: FloatArray,
    initial_pressure: FloatArray,
    case: CavityCase,
) -> tuple[FloatArray, LinearSolve]:
    """Solve the pressure equation with projected, Jacobi-preconditioned CG.

    All pressure boundaries are zero-gradient, so adding a constant to pressure
    changes nothing.  The matrix therefore has a one-dimensional null space.  We
    remove the mean from every Krylov vector, which is the numerical equivalent
    of choosing one pressure gauge.
    """

    coefficient_x, coefficient_y, diagonal = pressure_face_coefficients(r_au, case)
    right_hand_side = -divergence(provisional_u, provisional_v, case)
    right_hand_side -= np.mean(right_hand_side)

    pressure = np.array(initial_pressure, dtype=np.float64, copy=True)
    pressure -= np.mean(pressure)
    residual = right_hand_side - apply_pressure_operator(
        pressure,
        coefficient_x,
        coefficient_y,
    )
    residual -= np.mean(residual)
    initial_residual = l2_norm(residual)
    if initial_residual == 0.0:
        return pressure, LinearSolve(0, 0.0, 0.0)

    target = max(1.0e-12, case.pressure_tolerance * initial_residual)
    preconditioned = residual / diagonal
    preconditioned -= np.mean(preconditioned)
    direction = preconditioned.copy()
    residual_dot_preconditioned = float(np.sum(residual * preconditioned))
    final_residual = initial_residual

    for iteration in range(1, case.pressure_max_iterations + 1):
        operator_direction = apply_pressure_operator(
            direction,
            coefficient_x,
            coefficient_y,
        )
        denominator = float(np.sum(direction * operator_direction))
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("pressure CG lost positive definiteness")
        alpha = residual_dot_preconditioned / denominator
        pressure += alpha * direction
        residual -= alpha * operator_direction
        residual -= np.mean(residual)
        final_residual = l2_norm(residual)
        if final_residual <= target:
            pressure -= np.mean(pressure)
            return pressure, LinearSolve(
                iteration,
                initial_residual,
                final_residual,
            )

        preconditioned = residual / diagonal
        preconditioned -= np.mean(preconditioned)
        next_dot = float(np.sum(residual * preconditioned))
        beta = next_dot / residual_dot_preconditioned
        direction = preconditioned + beta * direction
        direction -= np.mean(direction)
        residual_dot_preconditioned = next_dot

    raise RuntimeError(
        "pressure solve did not converge: "
        f"initial={initial_residual:.3e}, final={final_residual:.3e}"
    )


def correct_face_velocity(
    provisional_u: FloatArray,
    provisional_v: FloatArray,
    pressure: FloatArray,
    r_au: FloatArray,
    case: CavityCase,
) -> tuple[FloatArray, FloatArray]:
    """Subtract the pressure-equation flux from provisional face velocities."""

    corrected_u = np.array(provisional_u, copy=True)
    corrected_v = np.array(provisional_v, copy=True)
    r_au_x = 0.5 * (r_au[:, :-1] + r_au[:, 1:])
    r_au_y = 0.5 * (r_au[:-1, :] + r_au[1:, :])
    corrected_u[:, 1:-1] -= r_au_x * (pressure[:, 1:] - pressure[:, :-1]) / case.dx
    corrected_v[1:-1, :] -= r_au_y * (pressure[1:, :] - pressure[:-1, :]) / case.dy
    return corrected_u, corrected_v


def courant_number(face_u: FloatArray, face_v: FloatArray, case: CavityCase) -> float:
    """OpenFOAM-style cell Courant estimate for an incompressible mesh."""

    cell_courant = (
        0.5
        * case.delta_t
        * (
            (np.abs(face_u[:, 1:]) + np.abs(face_u[:, :-1])) / case.dx
            + (np.abs(face_v[1:, :]) + np.abs(face_v[:-1, :])) / case.dy
        )
    )
    return float(np.max(cell_courant))


class IcoFoamCavity:
    """State and one-step PISO evolution for the structured cavity."""

    def __init__(self, case: CavityCase) -> None:
        self.case = case
        shape = (case.ny, case.nx)
        self.u = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.pressure = np.zeros(shape, dtype=np.float64)
        self.face_u = np.zeros((case.ny, case.nx + 1), dtype=np.float64)
        self.face_v = np.zeros((case.ny + 1, case.nx), dtype=np.float64)

    def advance(self, *, verbose_piso: bool = False) -> StepDiagnostics:
        """Advance one implicit-Euler step followed by the PISO loop."""

        case = self.case
        old_u = self.u.copy()
        old_v = self.v.copy()
        old_face_u = self.face_u.copy()
        old_face_v = self.face_v.copy()

        # 1. Momentum predictor: assemble once using the old conservative phi.
        coefficients = assemble_momentum_coefficients(old_face_u, old_face_v, case)
        source_u = momentum_source(old_u, case, component="u")
        source_v = momentum_source(old_v, case, component="v")
        pressure_gradient_x, pressure_gradient_y = cell_gradient_neumann(
            self.pressure,
            case,
        )
        self.u, momentum_u = solve_momentum_component(
            coefficients,
            source_u - pressure_gradient_x,
            old_u,
            case,
        )
        self.v, momentum_v = solve_momentum_component(
            coefficients,
            source_v - pressure_gradient_y,
            old_v,
            case,
        )

        # 2. PISO: reconstruct H/A, solve pressure, then correct phi and U.
        r_au = coefficients.r_au
        pressure_solves: list[LinearSolve] = []
        for corrector in range(case.piso_correctors):
            h_u = calculate_h_by_a(coefficients, source_u, self.u)
            h_v = calculate_h_by_a(coefficients, source_v, self.v)
            provisional_u, provisional_v = provisional_face_velocity(
                h_u,
                h_v,
                old_u,
                old_v,
                old_face_u,
                old_face_v,
                r_au,
                case,
            )
            self.pressure, pressure_solve = solve_pressure(
                provisional_u,
                provisional_v,
                r_au,
                self.pressure,
                case,
            )
            pressure_solves.append(pressure_solve)
            self.face_u, self.face_v = correct_face_velocity(
                provisional_u,
                provisional_v,
                self.pressure,
                r_au,
                case,
            )
            pressure_gradient_x, pressure_gradient_y = cell_gradient_neumann(
                self.pressure,
                case,
            )
            self.u = h_u - r_au * pressure_gradient_x
            self.v = h_v - r_au * pressure_gradient_y

            if verbose_piso:
                corrected_divergence = divergence(self.face_u, self.face_v, case)
                print(
                    f"      PISO {corrector + 1}: "
                    f"pCG={pressure_solve.iterations:3d}, "
                    f"pRes={pressure_solve.final_residual:.3e}, "
                    f"max|div(phi)|={np.max(np.abs(corrected_divergence)):.3e}"
                )

        if not all(
            np.all(np.isfinite(field))
            for field in (self.u, self.v, self.pressure, self.face_u, self.face_v)
        ):
            raise RuntimeError("the solution contains a non-finite value")

        continuity = divergence(self.face_u, self.face_v, case)
        return StepDiagnostics(
            momentum_u=momentum_u,
            momentum_v=momentum_v,
            pressure=tuple(pressure_solves),
            courant_max=courant_number(self.face_u, self.face_v, case),
            continuity_l1=float(np.mean(np.abs(continuity))),
            continuity_linf=float(np.max(np.abs(continuity))),
            kinetic_energy=float(
                0.5 * np.sum(np.square(self.u) + np.square(self.v)) * case.dx * case.dy
            ),
        )


def run_case(
    case: CavityCase,
    *,
    report_every: int,
    verbose_piso: bool,
) -> tuple[dict[str, FloatArray], list[StepDiagnostics]]:
    """Run the requested trajectory and retain every physical state."""

    solver = IcoFoamCavity(case)
    times = np.arange(case.steps + 1, dtype=np.float64) * case.delta_t
    velocity = np.empty((case.steps + 1, case.ny, case.nx, 2), dtype=np.float64)
    pressure = np.empty((case.steps + 1, case.ny, case.nx), dtype=np.float64)
    velocity[0, :, :, 0] = solver.u
    velocity[0, :, :, 1] = solver.v
    pressure[0] = solver.pressure
    diagnostics: list[StepDiagnostics] = []

    print(
        f"cavity: {case.nx}x{case.ny}, Re={case.reynolds_number:g}, "
        f"dt={case.delta_t:g} s, steps={case.steps}, "
        f"PISO correctors={case.piso_correctors}"
    )
    started = time.perf_counter()
    for step in range(1, case.steps + 1):
        diagnostic = solver.advance(verbose_piso=verbose_piso)
        diagnostics.append(diagnostic)
        velocity[step, :, :, 0] = solver.u
        velocity[step, :, :, 1] = solver.v
        pressure[step] = solver.pressure

        should_report = (
            step == 1
            or step == case.steps
            or (report_every > 0 and step % report_every == 0)
        )
        if should_report:
            pressure_iterations = "/".join(
                str(item.iterations) for item in diagnostic.pressure
            )
            print(
                f"step={step:4d} t={times[step]:.5f} "
                f"Co={diagnostic.courant_max:.3f} "
                f"max|div(phi)|={diagnostic.continuity_linf:.3e} "
                f"Uiter={diagnostic.momentum_u.iterations}/"
                f"{diagnostic.momentum_v.iterations} "
                f"Piter={pressure_iterations}"
            )

    elapsed = time.perf_counter() - started
    print(f"completed in {elapsed:.3f} s")
    arrays = {
        "times": times,
        "steps": np.arange(case.steps + 1, dtype=np.int64),
        "U": velocity,
        "p": pressure,
        "face_u": solver.face_u,
        "face_v": solver.face_v,
    }
    return arrays, diagnostics


def diagnostic_arrays(
    diagnostics: list[StepDiagnostics], case: CavityCase
) -> dict[str, FloatArray]:
    """Convert Python diagnostic records into transparent numerical arrays."""

    count = len(diagnostics)
    momentum_iterations = np.empty((count, 2), dtype=np.int64)
    pressure_iterations = np.empty(
        (count, case.piso_correctors),
        dtype=np.int64,
    )
    courant = np.empty(count, dtype=np.float64)
    continuity_l1 = np.empty(count, dtype=np.float64)
    continuity_linf = np.empty(count, dtype=np.float64)
    kinetic_energy = np.empty(count, dtype=np.float64)
    for index, item in enumerate(diagnostics):
        momentum_iterations[index] = (
            item.momentum_u.iterations,
            item.momentum_v.iterations,
        )
        pressure_iterations[index] = [solve.iterations for solve in item.pressure]
        courant[index] = item.courant_max
        continuity_l1[index] = item.continuity_l1
        continuity_linf[index] = item.continuity_linf
        kinetic_energy[index] = item.kinetic_energy
    return {
        "momentum_iterations": momentum_iterations,
        "pressure_iterations": pressure_iterations,
        "courant_max": courant,
        "continuity_l1": continuity_l1,
        "continuity_linf": continuity_linf,
        "kinetic_energy": kinetic_energy,
    }


def case_metadata(case: CavityCase) -> str:
    """JSON metadata stored inside the NPZ without pickled Python objects."""

    return json.dumps(
        {
            "algorithm": "educational Cartesian finite-volume PISO",
            "openfoam_analogue": "OpenFOAM Foundation v7 icoFoam",
            "mesh_cells": [case.nx, case.ny],
            "extent_m": [case.length_x, case.length_y],
            "viscosity_m2_per_s": case.viscosity,
            "lid_velocity_m_per_s": case.lid_velocity,
            "delta_t_s": case.delta_t,
            "steps": case.steps,
            "end_time_s": case.end_time,
            "piso_correctors": case.piso_correctors,
            "pressure_gauge": "zero spatial mean",
            "array_axes": {"cell_fields": ["time", "y", "x"], "j0": "bottom"},
        },
        sort_keys=True,
    )


def write_output(
    path: Path,
    arrays: dict[str, FloatArray],
    diagnostics: list[StepDiagnostics],
    case: CavityCase,
    *,
    overwrite: bool,
) -> None:
    """Write all states and diagnostics to one compressed archive."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = dict(arrays)
    payload.update(diagnostic_arrays(diagnostics, case))
    payload["metadata_json"] = np.array(case_metadata(case))
    np.savez_compressed(path, **payload)
    print(f"wrote {path} ({path.stat().st_size / 1_000_000:.2f} MB)")


def compare_with_reference(
    arrays: dict[str, FloatArray],
    reference_path: Path,
    case: CavityCase,
) -> None:
    """Compare the final state with our imported fixed-grid OpenFOAM trajectory."""

    try:
        from aicfd import load_trajectory
    except ImportError as error:  # pragma: no cover - only relevant outside the repo
        raise RuntimeError(
            "reference comparison needs the ai-native-cfd package; run with 'uv run'"
        ) from error

    reference = load_trajectory(reference_path)
    final_time = float(arrays["times"][-1])
    frame = int(np.argmin(np.abs(reference.times - final_time)))
    if not math.isclose(
        float(reference.times[frame]),
        final_time,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"reference trajectory has no frame at t={final_time:g}")
    if reference.layout.dimension != 2 or len(reference.layout) != case.nx * case.ny:
        raise ValueError("reference topology does not match this Cartesian cavity")

    reference_u = np.empty((case.ny, case.nx, 2), dtype=np.float64)
    reference_p = np.empty((case.ny, case.nx), dtype=np.float64)
    source_u = reference["U"].values[frame]
    source_p = reference["p"].values[frame]
    for row, cell in enumerate(reference.layout.cells):
        i, j = cell.index
        reference_u[j, i] = source_u[row]
        reference_p[j, i] = source_p[row]

    calculated_u = arrays["U"][-1]
    calculated_p = arrays["p"][-1]
    calculated_p = calculated_p - np.mean(calculated_p)
    reference_p = reference_p - np.mean(reference_p)
    velocity_error = calculated_u - reference_u
    pressure_error = calculated_p - reference_p
    reference_velocity_rms = max(l2_norm(reference_u), np.finfo(float).tiny)
    reference_pressure_rms = max(l2_norm(reference_p), np.finfo(float).tiny)
    speed_correlation = float(
        np.corrcoef(
            np.linalg.norm(calculated_u, axis=-1).ravel(),
            np.linalg.norm(reference_u, axis=-1).ravel(),
        )[0, 1]
    )

    print("OpenFOAM final-state comparison (pressure means removed):")
    print(
        f"  velocity RMSE       = {l2_norm(velocity_error):.6e} m/s "
        f"({l2_norm(velocity_error) / reference_velocity_rms:.3%} of reference RMS)"
    )
    print(
        f"  pressure RMSE       = {l2_norm(pressure_error):.6e} m^2/s^2 "
        f"({l2_norm(pressure_error) / reference_pressure_rms:.3%} of reference RMS)"
    )
    print(f"  speed correlation   = {speed_correlation:.6f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-file educational icoFoam/PISO cavity64 solver.",
    )
    parser.add_argument(
        "--steps", type=int, default=400, help="physical steps (default: 400)"
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=64,
        help="cells in each direction; cavity64 uses 64 (default: 64)",
    )
    parser.add_argument(
        "--correctors",
        type=int,
        default=2,
        help="PISO pressure correctors (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".data/python/icofoam_cavity64.npz"),
        help="compressed output archive",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="optional imported OpenFOAM trajectory directory",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=20,
        help="progress interval; 0 prints only first and last steps",
    )
    parser.add_argument(
        "--verbose-piso",
        action="store_true",
        help="print every pressure-correction solve",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output archive",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.report_every < 0:
        raise ValueError("--report-every must be non-negative")
    case = CavityCase(
        nx=arguments.cells,
        ny=arguments.cells,
        steps=arguments.steps,
        piso_correctors=arguments.correctors,
    )
    arrays, diagnostics = run_case(
        case,
        report_every=arguments.report_every,
        verbose_piso=arguments.verbose_piso,
    )
    write_output(
        arguments.output,
        arrays,
        diagnostics,
        case,
        overwrite=arguments.overwrite,
    )
    if arguments.reference is not None:
        compare_with_reference(arrays, arguments.reference, case)


if __name__ == "__main__":
    main()
