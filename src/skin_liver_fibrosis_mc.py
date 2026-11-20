"""
Skin-to-liver Monte Carlo simulation for normal vs fibrotic liver optics.

Research direction implemented here:
1. Primary: optical differences between normal liver and fibrotic liver after
   photons pass through skin.
2. Backup: feasibility of photons crossing skin, visiting liver, and returning
   through skin after reflection/scattering.

The model is a compact 2-D photon-packet Monte Carlo prototype. It is meant for
idea validation and figure generation, not as a clinical optical-property table.
Replace the coefficient table with measured values when a specific wavelength,
species, device, or abdominal anatomy is fixed.

Boundary scope: ``TissueOptics`` contains mu_a, mu_s and g only.  Refractive
index n, Snell refraction and Fresnel reflection are not implemented, so n must
not be shown as an active model parameter in the manuscript or workflow figure.

Units:
    Length is cm. Optical coefficients are cm^-1.

Examples:
    python skin_liver_fibrosis_mc.py --out skin_liver_outputs
    python skin_liver_fibrosis_mc.py --photons 15000 --out skin_liver_outputs_hi
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


EPS = 1.0e-12
CM_TO_MM = 10.0


TISSUE_NAMES = {
    0: "outside",
    1: "epidermis",
    2: "dermis",
    3: "liver",
}


@dataclass(frozen=True)
class Geometry:
    nx: int = 170
    nz: int = 260
    x_min: float = -0.075
    x_max: float = 0.075
    z_min: float = 0.0
    z_max: float = 0.160
    epidermis_thickness: float = 0.008
    skin_thickness: float = 0.030

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / self.nz

    @property
    def x_centers(self) -> np.ndarray:
        return self.x_min + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def z_centers(self) -> np.ndarray:
        return self.z_min + (np.arange(self.nz) + 0.5) * self.dz


@dataclass(frozen=True)
class SourceDetector:
    source_x: float = 0.0
    detector_x: float = 0.0
    detector_radius: float = 0.010
    beam_sigma: float = 0.003
    numerical_aperture: float = 0.035


@dataclass(frozen=True)
class TissueOptics:
    mua: np.ndarray
    mus: np.ndarray
    g: np.ndarray
    meta: Dict[str, Union[float, str]]


def build_layered_tissue(geo: Geometry) -> np.ndarray:
    """Return a [nz, nx] tissue-id map: epidermis, dermis, liver."""
    z = geo.z_centers[:, None]
    tissue = np.full((geo.nz, geo.nx), 3, dtype=np.uint8)
    tissue[z[:, 0] < geo.skin_thickness, :] = 2
    tissue[z[:, 0] < geo.epidermis_thickness, :] = 1
    return tissue


def optical_properties(
    wavelength_nm: int,
    liver_state: str,
    fibrosis_severity: Optional[float] = None,
) -> TissueOptics:
    """Return illustrative optical properties constrained by literature trends.

    Notes:
    - Skin values are compact NIR-friendly coefficients used for simulation.
    - Liver normal/fibrotic values are designed to express the reported trend
      that liver cirrhosis/fibrosis changes both absorption and reduced
      scattering. Nishio et al. reported that liver optical properties measured
      by NIR time-resolved spectroscopy relate to cirrhosis severity, with
      reduced scattering decreasing in cirrhotic liver.
    - musp is converted to mus using mus = musp / (1 - g).
    """
    state = liver_state.lower()
    if state not in {"normal", "fibrosis"}:
        raise ValueError("liver_state must be 'normal' or 'fibrosis'")
    # ``liver_state`` retains the convenient endpoint API used by the reference
    # examples.  When a severity is supplied explicitly, however, it is the
    # source of truth for *both* classes.  This is essential for cohorts such
    # as F0--F1 (severity 0--0.3) and continuous staging (severity 0--1): the
    # previous implementation silently collapsed every ``normal`` sample to
    # severity zero.
    if fibrosis_severity is None:
        severity = 0.0 if state == "normal" else 1.0
    else:
        severity = float(np.clip(fibrosis_severity, 0.0, 1.0))

    # mua and musp in cm^-1. musp is reduced scattering.
    table = {
        660: {
            "epi": (4.20, 18.0),
            "dermis": (1.25, 16.0),
            "liver_normal": (3.00, 18.0),
            "liver_fibrotic": (3.85, 14.2),
        },
        730: {
            "epi": (2.15, 16.0),
            "dermis": (0.82, 14.0),
            "liver_normal": (2.35, 16.0),
            "liver_fibrotic": (2.20, 12.5),
        },
        808: {
            "epi": (1.20, 14.0),
            "dermis": (0.55, 12.2),
            "liver_normal": (1.90, 14.0),
            "liver_fibrotic": (1.48, 10.5),
        },
        940: {
            "epi": (1.55, 12.0),
            "dermis": (0.82, 10.5),
            "liver_normal": (1.55, 11.5),
            "liver_fibrotic": (1.18, 8.8),
        },
        1064: {
            "epi": (1.05, 10.8),
            "dermis": (0.66, 9.2),
            "liver_normal": (1.25, 10.0),
            "liver_fibrotic": (0.98, 7.6),
        },
    }
    if wavelength_nm not in table:
        raise ValueError(f"unsupported wavelength {wavelength_nm}; choose one of {sorted(table)}")

    row = table[wavelength_nm]
    g_val = 0.90
    mua = np.zeros(4, dtype=np.float64)
    mus = np.zeros(4, dtype=np.float64)
    g = np.zeros(4, dtype=np.float64)
    mua[1], musp_epi = row["epi"]
    mua[2], musp_dermis = row["dermis"]
    normal_mua, normal_musp = row["liver_normal"]
    fib_mua, fib_musp = row["liver_fibrotic"]
    mua[3] = (1.0 - severity) * normal_mua + severity * fib_mua
    musp_liver = (1.0 - severity) * normal_musp + severity * fib_musp
    mus[1] = musp_epi / (1.0 - g_val)
    mus[2] = musp_dermis / (1.0 - g_val)
    mus[3] = musp_liver / (1.0 - g_val)
    g[1:] = g_val
    return TissueOptics(
        mua=mua,
        mus=mus,
        g=g,
        meta={
            "wavelength_nm": float(wavelength_nm),
            "liver_state": state,
            "fibrosis_severity": severity,
            "liver_mua": float(mua[3]),
            "liver_musp": float(musp_liver),
        },
    )


def tissue_at(x: float, z: float, tissue: np.ndarray, geo: Geometry) -> int:
    if x < geo.x_min or x >= geo.x_max or z < geo.z_min or z >= geo.z_max:
        return 0
    ix = int((x - geo.x_min) / geo.dx)
    iz = int((z - geo.z_min) / geo.dz)
    return int(tissue[iz, ix])


def deposit(A: np.ndarray, x: float, z: float, value: float, geo: Geometry) -> None:
    if x < geo.x_min or x >= geo.x_max or z < geo.z_min or z >= geo.z_max:
        return
    ix = int((x - geo.x_min) / geo.dx)
    iz = int((z - geo.z_min) / geo.dz)
    A[iz, ix] += value


def sample_hg_cosine(rng: np.random.Generator, g: float) -> float:
    r = max(float(rng.random()), EPS)
    if abs(g) < 1.0e-9:
        return 2.0 * r - 1.0
    temp = (1.0 - g * g) / (1.0 - g + 2.0 * g * r)
    return float(np.clip((1.0 + g * g - temp * temp) / (2.0 * g), -1.0, 1.0))


def rotate_2d(ux: float, uz: float, cos_theta: float, rng: np.random.Generator) -> Tuple[float, float]:
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    if rng.random() < 0.5:
        sin_theta = -sin_theta
    new_ux = ux * cos_theta - uz * sin_theta
    new_uz = ux * sin_theta + uz * cos_theta
    norm = math.sqrt(new_ux * new_ux + new_uz * new_uz) + EPS
    return new_ux / norm, new_uz / norm


def launch(rng: np.random.Generator, sd: SourceDetector) -> Tuple[float, float, float, float]:
    x = sd.source_x + rng.normal(0.0, sd.beam_sigma)
    z = 1.0e-7
    theta = rng.normal(0.0, sd.numerical_aperture)
    ux = math.sin(theta)
    uz = math.cos(theta)
    return x, z, ux, uz


def simulate(
    geo: Geometry,
    tissue: np.ndarray,
    optics: TissueOptics,
    sd: SourceDetector,
    photons: int,
    seed: int,
    record_paths: int = 0,
    max_steps: int = 2800,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    A = np.zeros((geo.nz, geo.nx), dtype=np.float64)
    paths: List[np.ndarray] = []
    depth_samples: List[float] = []
    liver_depth_samples: List[float] = []
    scatter_samples: List[int] = []

    stats: Dict[str, float] = {
        "photons": float(photons),
        "exit_top": 0.0,
        "detected_top": 0.0,
        "detected_after_liver": 0.0,
        "exit_bottom": 0.0,
        "exit_side": 0.0,
        "terminated": 0.0,
        "reached_liver": 0.0,
        "returned_after_liver": 0.0,
        "skin_absorption": 0.0,
        "liver_absorption": 0.0,
        "path_length_cm": 0.0,
        "path_in_liver_cm": 0.0,
        "max_depth_cm": 0.0,
        "scatter_count": 0.0,
    }

    for ip in range(photons):
        x, z, ux, uz = launch(rng, sd)
        W = 1.0
        reached_liver = False
        returned_after_liver = False
        path_len = 0.0
        path_in_liver = 0.0
        max_depth = z
        scatters = 0
        photon_path: List[Tuple[float, float]] = [(x, z)]

        for _ in range(max_steps):
            tid = tissue_at(x, z, tissue, geo)
            if tid == 0:
                if z < geo.z_min:
                    stats["exit_top"] += 1.0
                    if reached_liver:
                        stats["returned_after_liver"] += 1.0
                    if abs(x - sd.detector_x) <= sd.detector_radius:
                        stats["detected_top"] += 1.0
                        if reached_liver:
                            stats["detected_after_liver"] += 1.0
                elif z >= geo.z_max:
                    stats["exit_bottom"] += 1.0
                else:
                    stats["exit_side"] += 1.0
                break

            mua = float(optics.mua[tid])
            mus = float(optics.mus[tid])
            mut = max(mua + mus, EPS)
            step = -math.log(max(float(rng.random()), EPS)) / mut
            next_x = x + ux * step
            next_z = z + uz * step
            mid_x = 0.5 * (x + next_x)
            mid_z = 0.5 * (z + next_z)
            mid_tid = tissue_at(mid_x, mid_z, tissue, geo)
            score_tid = tid if mid_tid == 0 else mid_tid
            absorb = W * (mua / mut)
            W -= absorb
            deposit(A, mid_x, mid_z, absorb, geo)

            if score_tid in (1, 2):
                stats["skin_absorption"] += absorb
            elif score_tid == 3:
                stats["liver_absorption"] += absorb
                reached_liver = True
                path_in_liver += step

            x, z = next_x, next_z
            path_len += step
            max_depth = max(max_depth, z)
            scatters += 1

            if reached_liver and z < geo.skin_thickness:
                returned_after_liver = True
            if ip < record_paths and (scatters % 4 == 0 or score_tid == 3):
                photon_path.append((x, z))

            if W < 0.004:
                if rng.random() < 0.10:
                    W /= 0.10
                else:
                    stats["terminated"] += 1.0
                    break

            ux, uz = rotate_2d(ux, uz, sample_hg_cosine(rng, float(optics.g[tid])), rng)
        else:
            stats["terminated"] += 1.0

        if reached_liver:
            stats["reached_liver"] += 1.0
            liver_depth_samples.append(max_depth)
        if returned_after_liver:
            returned_after_liver = True
        stats["path_length_cm"] += path_len
        stats["path_in_liver_cm"] += path_in_liver
        stats["max_depth_cm"] += max_depth
        stats["scatter_count"] += scatters
        depth_samples.append(max_depth)
        scatter_samples.append(scatters)
        if ip < record_paths:
            photon_path.append((x, z))
            paths.append(np.asarray(photon_path, dtype=np.float64))

    for key in [
        "exit_top",
        "detected_top",
        "detected_after_liver",
        "exit_bottom",
        "exit_side",
        "terminated",
        "reached_liver",
        "returned_after_liver",
    ]:
        stats[key + "_probability"] = stats[key] / photons
    for key in ["skin_absorption", "liver_absorption", "path_length_cm", "path_in_liver_cm", "max_depth_cm", "scatter_count"]:
        stats["mean_" + key] = stats[key] / photons
    if stats["reached_liver"] > 0:
        stats["conditional_detected_after_liver_probability"] = stats["detected_after_liver"] / stats["reached_liver"]
        stats["conditional_returned_after_liver_probability"] = stats["returned_after_liver"] / stats["reached_liver"]
    else:
        stats["conditional_detected_after_liver_probability"] = 0.0
        stats["conditional_returned_after_liver_probability"] = 0.0

    mua_map = optics.mua[tissue]
    fluence = np.divide(
        A,
        np.maximum(mua_map, EPS) * photons * geo.dx * geo.dz,
        out=np.zeros_like(A),
        where=mua_map > 0,
    )
    return {
        "A": A,
        "fluence": fluence,
        "stats": stats,
        "paths": paths,
        "depth_samples": np.asarray(depth_samples),
        "liver_depth_samples": np.asarray(liver_depth_samples),
        "scatter_samples": np.asarray(scatter_samples),
        "optics": optics,
    }


def log_image(data: np.ndarray, percentile_floor: float = 1.0) -> np.ndarray:
    positive = data[data > 0]
    floor = 1.0e-30 if positive.size == 0 else np.percentile(positive, percentile_floor)
    return np.log10(np.maximum(data, floor))


def extent_mm(geo: Geometry) -> List[float]:
    return [geo.x_min * CM_TO_MM, geo.x_max * CM_TO_MM, geo.z_max * CM_TO_MM, geo.z_min * CM_TO_MM]


def overlay_layers(ax: plt.Axes, geo: Geometry, color: str = "white") -> None:
    ax.axhline(geo.epidermis_thickness * CM_TO_MM, color=color, lw=0.8, alpha=0.8)
    ax.axhline(geo.skin_thickness * CM_TO_MM, color=color, lw=1.0, alpha=0.9)


def save_csv(path: Path, rows: List[Dict[str, Union[float, str]]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_geometry(tissue: np.ndarray, geo: Geometry, out: Path) -> None:
    cmap = ListedColormap(["#c9955d", "#e3b17c", "#8a2d2d"])
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=180)
    ax.imshow(tissue, origin="upper", extent=extent_mm(geo), aspect="equal", cmap=cmap, vmin=1, vmax=3)
    overlay_layers(ax, geo, "black")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("depth z (mm)")
    ax.set_title("Skin + liver layered phantom")
    handles = [plt.Line2D([0], [0], color=cmap(i - 1), lw=6) for i in (1, 2, 3)]
    ax.legend(handles, [TISSUE_NAMES[i] for i in (1, 2, 3)], loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig01_skin_liver_geometry.png")
    plt.close(fig)


def plot_normal_vs_fibrosis(
    normal: Dict[str, object],
    fibrosis: Dict[str, object],
    geo: Geometry,
    out: Path,
    wavelength: int,
) -> None:
    normal_log = log_image(normal["fluence"])
    fibrosis_log = log_image(fibrosis["fluence"])
    diff = fibrosis_log - normal_log
    vmin = min(float(np.percentile(normal_log, 1)), float(np.percentile(fibrosis_log, 1)))
    vmax = max(float(np.percentile(normal_log, 99)), float(np.percentile(fibrosis_log, 99)))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), dpi=180, constrained_layout=True)
    panels = [
        (normal_log, f"normal liver\nlog10 fluence {wavelength} nm", "magma", vmin, vmax),
        (fibrosis_log, f"fibrotic liver\nlog10 fluence {wavelength} nm", "magma", vmin, vmax),
        (diff, "fibrosis - normal\nlog10 fluence", "coolwarm", -0.8, 0.8),
    ]
    for ax, (data, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(data, origin="upper", extent=extent_mm(geo), aspect="equal", cmap=cmap, vmin=lo, vmax=hi)
        overlay_layers(ax, geo)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(out / "fig02_normal_vs_fibrotic_fluence.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), dpi=180, constrained_layout=True)
    z = geo.z_centers * CM_TO_MM
    normal_profile = np.mean(normal["fluence"], axis=1)
    fib_profile = np.mean(fibrosis["fluence"], axis=1)
    axes[0].semilogx(normal_profile + EPS, z, label="normal")
    axes[0].semilogx(fib_profile + EPS, z, label="fibrosis")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean fluence (a.u.)")
    axes[0].set_ylabel("depth z (mm)")
    axes[0].set_title("Depth fluence profile")
    axes[0].legend()

    normal_A = np.mean(normal["A"], axis=1)
    fib_A = np.mean(fibrosis["A"], axis=1)
    axes[1].semilogx(normal_A + EPS, z, label="normal")
    axes[1].semilogx(fib_A + EPS, z, label="fibrosis")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("absorbed weight profile")
    axes[1].set_ylabel("depth z (mm)")
    axes[1].set_title("Absorption profile")
    axes[1].legend()
    fig.savefig(out / "fig03_depth_profiles_normal_vs_fibrosis.png")
    plt.close(fig)


def plot_paths(result: Dict[str, object], tissue: np.ndarray, geo: Geometry, out: Path) -> None:
    cmap = ListedColormap(["#c9955d", "#e3b17c", "#8a2d2d"])
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=180)
    ax.imshow(tissue, origin="upper", extent=extent_mm(geo), aspect="equal", cmap=cmap, alpha=0.52, vmin=1, vmax=3)
    for path in result["paths"]:
        ax.plot(path[:, 0] * CM_TO_MM, path[:, 1] * CM_TO_MM, lw=0.7, alpha=0.62)
    overlay_layers(ax, geo, "black")
    ax.set_xlim(geo.x_min * CM_TO_MM, geo.x_max * CM_TO_MM)
    ax.set_ylim(geo.z_max * CM_TO_MM, geo.z_min * CM_TO_MM)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("depth z (mm)")
    ax.set_title("Representative paths for feasibility analysis")
    fig.tight_layout()
    fig.savefig(out / "fig04_representative_return_paths.png")
    plt.close(fig)


def run_spectral_difference(
    geo: Geometry,
    tissue: np.ndarray,
    wavelengths: Iterable[int],
    photons: int,
    sd: SourceDetector,
    seed: int,
    out: Path,
) -> List[Dict[str, Union[float, str]]]:
    rows: List[Dict[str, Union[float, str]]] = []
    for i, wl in enumerate(wavelengths):
        normal = simulate(geo, tissue, optical_properties(wl, "normal"), sd, photons, seed + 10 * i + 1)
        fib = simulate(geo, tissue, optical_properties(wl, "fibrosis"), sd, photons, seed + 10 * i + 2)
        ns = normal["stats"]
        fs = fib["stats"]
        rows.append(
            {
                "wavelength_nm": wl,
                "normal_detected": ns["detected_top_probability"],
                "fibrosis_detected": fs["detected_top_probability"],
                "delta_detected": fs["detected_top_probability"] - ns["detected_top_probability"],
                "normal_liver_reached": ns["reached_liver_probability"],
                "fibrosis_liver_reached": fs["reached_liver_probability"],
                "normal_detected_after_liver": ns["detected_after_liver_probability"],
                "fibrosis_detected_after_liver": fs["detected_after_liver_probability"],
                "normal_liver_absorption": ns["mean_liver_absorption"],
                "fibrosis_liver_absorption": fs["mean_liver_absorption"],
                "delta_liver_absorption": fs["mean_liver_absorption"] - ns["mean_liver_absorption"],
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), dpi=180, constrained_layout=True)
    w = [float(r["wavelength_nm"]) for r in rows]
    axes[0].plot(w, [float(r["normal_detected"]) for r in rows], marker="o", label="normal")
    axes[0].plot(w, [float(r["fibrosis_detected"]) for r in rows], marker="o", label="fibrosis")
    axes[0].set_ylabel("top detected probability")
    axes[0].set_title("Observable reflectance proxy")
    axes[1].plot(w, [float(r["normal_detected_after_liver"]) for r in rows], marker="o", label="normal")
    axes[1].plot(w, [float(r["fibrosis_detected_after_liver"]) for r in rows], marker="o", label="fibrosis")
    axes[1].set_ylabel("detected after liver visit")
    axes[1].set_title("Liver-origin return signal")
    axes[2].bar([x - 8 for x in w], [float(r["delta_detected"]) for r in rows], width=12, label="detected")
    axes[2].bar([x + 8 for x in w], [float(r["delta_liver_absorption"]) for r in rows], width=12, label="liver absorption")
    axes[2].set_ylabel("fibrosis - normal")
    axes[2].set_title("Differential optical contrast")
    for ax in axes:
        ax.set_xlabel("wavelength (nm)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out / "fig05_spectral_normal_fibrosis_difference.png")
    plt.close(fig)
    return rows


def run_separation_scan(
    geo: Geometry,
    tissue: np.ndarray,
    wavelengths: Iterable[int],
    separations_mm: Iterable[float],
    photons: int,
    seed: int,
    out: Path,
) -> List[Dict[str, Union[float, str]]]:
    rows: List[Dict[str, Union[float, str]]] = []
    for wl in wavelengths:
        for j, sep_mm in enumerate(separations_mm):
            sd = SourceDetector(source_x=0.0, detector_x=sep_mm / CM_TO_MM, detector_radius=0.006)
            normal = simulate(geo, tissue, optical_properties(wl, "normal"), sd, photons, seed + wl + j)
            fib = simulate(geo, tissue, optical_properties(wl, "fibrosis"), sd, photons, seed + wl + 100 + j)
            ns = normal["stats"]
            fs = fib["stats"]
            rows.append(
                {
                    "wavelength_nm": wl,
                    "separation_mm": sep_mm,
                    "normal_detected": ns["detected_top_probability"],
                    "fibrosis_detected": fs["detected_top_probability"],
                    "delta_detected": fs["detected_top_probability"] - ns["detected_top_probability"],
                    "normal_detected_after_liver": ns["detected_after_liver_probability"],
                    "fibrosis_detected_after_liver": fs["detected_after_liver_probability"],
                    "delta_detected_after_liver": fs["detected_after_liver_probability"] - ns["detected_after_liver_probability"],
                }
            )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=180, constrained_layout=True)
    for wl in wavelengths:
        subset = [r for r in rows if int(r["wavelength_nm"]) == wl]
        axes[0].plot([float(r["separation_mm"]) for r in subset], [float(r["delta_detected"]) for r in subset], marker="o", label=f"{wl} nm")
        axes[1].plot(
            [float(r["separation_mm"]) for r in subset],
            [float(r["delta_detected_after_liver"]) for r in subset],
            marker="o",
            label=f"{wl} nm",
        )
    axes[0].set_title("Normal/fibrosis reflectance contrast")
    axes[0].set_ylabel("delta detected probability")
    axes[1].set_title("Normal/fibrosis liver-return contrast")
    axes[1].set_ylabel("delta liver-return probability")
    for ax in axes:
        ax.set_xlabel("source-detector separation (mm)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out / "fig06_source_detector_separation_scan.png")
    plt.close(fig)
    return rows


def run_feasibility_grid(
    wavelengths: Iterable[int],
    skin_thickness_mm: Iterable[float],
    photons: int,
    seed: int,
    out: Path,
) -> List[Dict[str, Union[float, str]]]:
    rows: List[Dict[str, Union[float, str]]] = []
    for wl in wavelengths:
        for j, thick_mm in enumerate(skin_thickness_mm):
            geo = Geometry(skin_thickness=thick_mm / CM_TO_MM, epidermis_thickness=min(0.008, thick_mm / CM_TO_MM * 0.25))
            tissue = build_layered_tissue(geo)
            sd = SourceDetector(detector_radius=0.010)
            res = simulate(geo, tissue, optical_properties(wl, "fibrosis"), sd, photons, seed + wl * 3 + j)
            stats = res["stats"]
            rows.append(
                {
                    "wavelength_nm": wl,
                    "skin_thickness_mm": thick_mm,
                    "reached_liver_probability": stats["reached_liver_probability"],
                    "returned_after_liver_probability": stats["returned_after_liver_probability"],
                    "detected_after_liver_probability": stats["detected_after_liver_probability"],
                    "conditional_detected_after_liver_probability": stats["conditional_detected_after_liver_probability"],
                    "mean_max_depth_mm": stats["mean_max_depth_cm"] * CM_TO_MM,
                }
            )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), dpi=180, constrained_layout=True)
    for wl in wavelengths:
        subset = [r for r in rows if int(r["wavelength_nm"]) == wl]
        x = [float(r["skin_thickness_mm"]) for r in subset]
        axes[0].plot(x, [float(r["reached_liver_probability"]) for r in subset], marker="o", label=f"{wl} nm")
        axes[1].plot(x, [float(r["returned_after_liver_probability"]) for r in subset], marker="o", label=f"{wl} nm")
        axes[2].plot(x, [float(r["detected_after_liver_probability"]) for r in subset], marker="o", label=f"{wl} nm")
    axes[0].set_title("Photons reaching liver")
    axes[1].set_title("Photons returning after liver")
    axes[2].set_title("Detected after liver visit")
    for ax in axes:
        ax.set_xlabel("skin thickness (mm)")
        ax.set_ylabel("probability")
        ax.set_yscale("symlog", linthresh=1.0e-4)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out / "fig07_skin_thickness_feasibility.png")
    plt.close(fig)
    return rows


def run_fibrosis_severity_sweep(
    geo: Geometry,
    tissue: np.ndarray,
    wavelengths: Iterable[int],
    severities: Iterable[float],
    photons: int,
    seed: int,
    out: Path,
) -> List[Dict[str, Union[float, str]]]:
    rows: List[Dict[str, Union[float, str]]] = []
    sd = SourceDetector(detector_radius=0.010)
    for wl in wavelengths:
        ref = simulate(geo, tissue, optical_properties(wl, "normal"), sd, photons, seed + wl)
        ref_detected = ref["stats"]["detected_top_probability"]
        ref_liver_abs = ref["stats"]["mean_liver_absorption"]
        for k, sev in enumerate(severities):
            state = "normal" if sev <= 0 else "fibrosis"
            res = simulate(geo, tissue, optical_properties(wl, state, fibrosis_severity=sev), sd, photons, seed + wl * 5 + k)
            stats = res["stats"]
            rows.append(
                {
                    "wavelength_nm": wl,
                    "fibrosis_severity": sev,
                    "detected_probability": stats["detected_top_probability"],
                    "delta_detected_vs_normal": stats["detected_top_probability"] - ref_detected,
                    "mean_liver_absorption": stats["mean_liver_absorption"],
                    "delta_liver_absorption_vs_normal": stats["mean_liver_absorption"] - ref_liver_abs,
                    "reached_liver_probability": stats["reached_liver_probability"],
                }
            )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), dpi=180, constrained_layout=True)
    for wl in wavelengths:
        subset = [r for r in rows if int(r["wavelength_nm"]) == wl]
        axes[0].plot([float(r["fibrosis_severity"]) for r in subset], [float(r["delta_detected_vs_normal"]) for r in subset], marker="o", label=f"{wl} nm")
        axes[1].plot(
            [float(r["fibrosis_severity"]) for r in subset],
            [float(r["delta_liver_absorption_vs_normal"]) for r in subset],
            marker="o",
            label=f"{wl} nm",
        )
    axes[0].set_title("Detector contrast vs fibrosis severity")
    axes[0].set_xlabel("fibrosis severity")
    axes[0].set_ylabel("delta detected probability")
    axes[1].set_title("Liver absorption contrast vs severity")
    axes[1].set_xlabel("fibrosis severity")
    axes[1].set_ylabel("delta liver absorption")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out / "fig08_fibrosis_severity_sweep.png")
    plt.close(fig)
    return rows


def plot_feasibility_budget(normal: Dict[str, object], fibrosis: Dict[str, object], out: Path) -> None:
    labels = ["reached liver", "returned after liver", "detected after liver", "skin absorbed", "liver absorbed"]
    values_normal = [
        normal["stats"]["reached_liver_probability"],
        normal["stats"]["returned_after_liver_probability"],
        normal["stats"]["detected_after_liver_probability"],
        normal["stats"]["mean_skin_absorption"],
        normal["stats"]["mean_liver_absorption"],
    ]
    values_fib = [
        fibrosis["stats"]["reached_liver_probability"],
        fibrosis["stats"]["returned_after_liver_probability"],
        fibrosis["stats"]["detected_after_liver_probability"],
        fibrosis["stats"]["mean_skin_absorption"],
        fibrosis["stats"]["mean_liver_absorption"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=180)
    ax.bar(x - 0.18, values_normal, width=0.36, label="normal")
    ax.bar(x + 0.18, values_fib, width=0.36, label="fibrosis")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("probability or mean absorbed weight")
    ax.set_yscale("log")
    ax.set_title("Feasibility budget for skin-liver round trip")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig09_feasibility_budget.png")
    plt.close(fig)


def run_convergence(
    geo: Geometry,
    tissue: np.ndarray,
    wavelength: int,
    photon_counts: Iterable[int],
    seed: int,
    out: Path,
) -> List[Dict[str, Union[float, str]]]:
    rows: List[Dict[str, Union[float, str]]] = []
    sd = SourceDetector()
    for i, n in enumerate(photon_counts):
        normal = simulate(geo, tissue, optical_properties(wavelength, "normal"), sd, int(n), seed + 300 + i)
        fib = simulate(geo, tissue, optical_properties(wavelength, "fibrosis"), sd, int(n), seed + 330 + i)
        rows.append(
            {
                "photons": n,
                "normal_detected": normal["stats"]["detected_top_probability"],
                "fibrosis_detected": fib["stats"]["detected_top_probability"],
                "delta_detected": fib["stats"]["detected_top_probability"] - normal["stats"]["detected_top_probability"],
                "normal_detected_after_liver": normal["stats"]["detected_after_liver_probability"],
                "fibrosis_detected_after_liver": fib["stats"]["detected_after_liver_probability"],
                "delta_detected_after_liver": fib["stats"]["detected_after_liver_probability"]
                - normal["stats"]["detected_after_liver_probability"],
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=180, constrained_layout=True)
    axes[0].plot([float(r["photons"]) for r in rows], [float(r["delta_detected"]) for r in rows], marker="o")
    axes[1].plot([float(r["photons"]) for r in rows], [float(r["delta_detected_after_liver"]) for r in rows], marker="o", color="#2a9d8f")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("photon packets")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fibrosis-normal detected probability")
    axes[0].set_title("Observed contrast convergence")
    axes[1].set_ylabel("fibrosis-normal liver-return probability")
    axes[1].set_title("Liver-return contrast convergence")
    fig.savefig(out / "fig10_convergence.png")
    plt.close(fig)
    return rows


def write_notes(out: Path) -> None:
    text = """# 皮肤-肝脏光传输研究方案说明

## 当前主线

首选方向建议写成：**经皮肤近红外反射光谱下，正常肝脏与肝纤维化肝脏的路径感知蒙特卡洛差异建模**。

它比单纯复现 MCXYZ 更有研究味道的地方在于：传统 MC 图只看 fluence/absorption，而这里额外统计“是否到达肝脏”“到达肝脏后是否重新穿过皮肤返回”“返回光是否进入探测器”“肝脏贡献的反射信号占比”等路径级指标。这些指标直接回答备选问题中的可行性，也能支撑首选问题中的正常/纤维化差异分析。

## 建议创新点

1. **Skin-gated liver optical contrast**：把皮肤层作为真实遮挡层，不直接比较裸肝组织，而比较光经过皮肤后的可观测差异。
2. **Liver-return photon biomarker**：提出 detected-after-liver probability，用来估计探测信号中真正访问过肝脏的成分。
3. **Normal-fibrosis differential spectrum**：在 660/730/808/940/1064 nm 下比较正常与纤维化肝脏的反射差异、肝吸收差异和路径差异。
4. **Source-detector separation optimization**：扫描源探测间距，寻找更偏向肝脏贡献而非皮肤表层散射的探测几何。
5. **Skin-thickness feasibility map**：扫描皮肤厚度，判断“光穿过皮肤-进入肝脏-散射返回-再次穿过皮肤被探测”的概率是否足够。

## 重要假设

本脚本中的光学参数是文献趋势约束下的示范参数，不是临床定值。正式论文需要替换为特定动物/人体、波长、组织状态下测得或文献报道的 `μa`、`μs'`、`g`。
"""
    (out / "research_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Skin-liver normal/fibrosis Monte Carlo simulator")
    parser.add_argument("--out", default="skin_liver_outputs")
    parser.add_argument("--photons", type=int, default=5500, help="baseline photons per normal/fibrosis map")
    parser.add_argument("--sweep-photons", type=int, default=1500, help="photons per spectral/severity point")
    parser.add_argument("--sep-photons", type=int, default=900, help="photons per source-detector separation point")
    parser.add_argument("--feasibility-photons", type=int, default=1100, help="photons per feasibility-grid point")
    parser.add_argument("--wavelength", type=int, default=808, choices=[660, 730, 808, 940, 1064])
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geo = Geometry()
    tissue = build_layered_tissue(geo)
    sd = SourceDetector()
    wavelengths = [660, 730, 808, 940, 1064]

    print("1/9 plotting geometry")
    plot_geometry(tissue, geo, out)

    print("2/9 running baseline normal/fibrosis maps")
    normal = simulate(geo, tissue, optical_properties(args.wavelength, "normal"), sd, args.photons, args.seed, record_paths=34)
    fibrosis = simulate(
        geo,
        tissue,
        optical_properties(args.wavelength, "fibrosis"),
        sd,
        args.photons,
        args.seed + 1,
        record_paths=34,
    )
    plot_normal_vs_fibrosis(normal, fibrosis, geo, out, args.wavelength)
    plot_paths(fibrosis, tissue, geo, out)
    plot_feasibility_budget(normal, fibrosis, out)

    print("3/9 running spectral normal/fibrosis difference")
    spectral_rows = run_spectral_difference(geo, tissue, wavelengths, args.sweep_photons, sd, args.seed + 20, out)
    save_csv(out / "spectral_difference.csv", spectral_rows)

    print("4/9 running source-detector separation scan")
    sep_rows = run_separation_scan(geo, tissue, [730, 808, 940, 1064], [0.0, 0.6, 1.2, 1.8, 2.4], args.sep_photons, args.seed + 40, out)
    save_csv(out / "source_detector_separation.csv", sep_rows)

    print("5/9 running skin-thickness feasibility grid")
    feasibility_rows = run_feasibility_grid([730, 808, 940, 1064], [1.5, 2.0, 3.0, 4.0, 5.0], args.feasibility_photons, args.seed + 60, out)
    save_csv(out / "skin_thickness_feasibility.csv", feasibility_rows)

    print("6/9 running fibrosis severity sweep")
    severity_rows = run_fibrosis_severity_sweep(
        geo,
        tissue,
        [730, 808, 940, 1064],
        [0.0, 0.25, 0.50, 0.75, 1.0],
        args.sweep_photons,
        args.seed + 80,
        out,
    )
    save_csv(out / "fibrosis_severity_sweep.csv", severity_rows)

    print("7/9 running convergence check")
    conv_rows = run_convergence(geo, tissue, args.wavelength, [300, 700, 1500, 3000, 6000], args.seed + 100, out)
    save_csv(out / "convergence.csv", conv_rows)

    print("8/9 saving baseline statistics")
    normal_stats = {"state": "normal", "wavelength_nm": args.wavelength}
    normal_stats.update({k: v for k, v in normal["stats"].items() if isinstance(v, (int, float, np.floating))})
    fib_stats = {"state": "fibrosis", "wavelength_nm": args.wavelength}
    fib_stats.update({k: v for k, v in fibrosis["stats"].items() if isinstance(v, (int, float, np.floating))})
    save_csv(out / "baseline_stats.csv", [normal_stats, fib_stats])

    print("9/9 writing research notes")
    write_notes(out)
    print(f"done. outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
