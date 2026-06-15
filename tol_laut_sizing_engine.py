"""
================================================================================
TOL LAUT BATTERY-ELECTRIC PROPULSION SIZING ENGINE — UNIFIED BACKBONE (v2.0)
================================================================================
Single backbone calculation for the Tol Laut electrification study. This file
merges the two prior engines into one authoritative source:

  * tol_laut_sizing_engine_v1_1.py        -> route matrix (Table 2), per-route
                                             arrival intervals, and pier-side
                                             charging power P_port.
  * tol_laut_sizing_engine_consolidated.py -> ceiling solver, sensitivity sweep,
                                             trade-off curve, microgrid sizing.

Both prior engines shared an identical physics core (same constants, same
motor-efficiency treatment in which P_AUX is NOT divided by eta_motor) and both
returned a 544.2 nm ceiling with 5 viable routes. This merge therefore changes
no result; it removes duplication and a single drift risk.

Cleanups applied during the merge:
  * One constants block (no duplicated/competing definitions).
  * Core functions accept rho_grav / cargo_target as arguments, so the
    sensitivity sweep no longer mutates module globals.
  * The pier-side charger (P_port) and the microgrid sizing both use each
    route's real arrival interval rather than a fixed default.

Authors : Manunk (RAM engineer), Dr. Betty
================================================================================
"""

import numpy as np
import pandas as pd


# ==========================================================
# 1. PHYSICS CONSTANTS & BASELINE PARAMETERS  (paper Chapter 3, Table 1)
# ==========================================================

# Vessel / voyage
V_KNOTS      = 10.0                 # Service speed, knots (75% MCR cruise) [14]
V_KMPH       = V_KNOTS * 1.852      # Speed, km/h  (= 18.52 km/h)
DWT_DESIGN   = 2000.0              # Design deadweight of the 100 TEU hull, t [3]
DWT_TARGET   = 1000.0              # Minimum commercial cargo target, t [1]
M_SAVED      = 120.0              # Mass credit from removing the diesel plant, t [14]

# Propulsion / electrical
P_PROP       = 1500.0             # Shaft propulsion power at 75% MCR, kW [14]
P_AUX        = 150.0              # Auxiliary (hotel/reefer) electrical load, kW [14]
ETA_MOTOR    = 0.95              # Motor + VFD efficiency (applies to P_PROP only) [13]
ETA_BESS     = 0.90              # Battery round-trip DC-DC efficiency [12]
ETA_CHARGER  = 0.92              # Shore-side AC/DC rectifier efficiency [6]

# Battery
RHO_GRAV     = 0.14              # Gravimetric energy density, MWh/t [16]
DOD          = 0.80              # Max depth of discharge per voyage [15]
S_RES        = 0.20              # Operational reserve margin [15]

# Charging schedule
T_BUFFER     = 12.0             # Handling/inspection buffer subtracted from the
                                #   arrival interval before sizing the charger, h

NM_TO_KM = 1.852                # 1 nautical mile in kilometres


# ==========================================================
# 2. CORE SIZING PHYSICS
# ==========================================================
# Energy demand and installed capacity are independent of rho_grav; only the
# battery MASS (and therefore residual cargo and feasibility) depend on it.

def calculate_battery_requirement(d_nm):
    """Energy and installed-capacity requirement for one uninterrupted leg.

    Returns a dict of intermediate and final energy quantities (MWh).
    E_seg (usable at terminals) = P_PROP/eta_motor + P_AUX, integrated over the
    voyage; E_installed adds the DoD, pack-efficiency, and reserve scale-ups.
    """
    d_km   = d_nm * NM_TO_KM
    t_hour = d_km / V_KMPH

    e_prop           = t_hour * P_PROP / 1000.0          # shaft work, MWh
    e_prop_delivered = e_prop / ETA_MOTOR                # electrical input to motor
    e_aux            = t_hour * P_AUX / 1000.0           # hotel load (already electric)
    e_seg            = e_prop_delivered + e_aux          # usable energy at terminals

    e_installed = e_seg * (1 + S_RES) / (DOD * ETA_BESS)  # nameplate capacity

    return {
        "distance_nm": d_nm,
        "distance_km": d_km,
        "transit_hours": t_hour,
        "e_seg_mwh": e_seg,
        "e_installed_mwh": e_installed,
    }


def calculate_battery_mass(e_installed_mwh, rho_grav=RHO_GRAV):
    """Battery mass (t) = installed capacity (MWh) / gravimetric density (MWh/t)."""
    return e_installed_mwh / rho_grav


def calculate_residual_cargo(d_nm, rho_grav=RHO_GRAV, cargo_target=DWT_TARGET):
    """Residual cargo deadweight (t) and feasibility for a leg of length d_nm.

    cargo = DWT_DESIGN - (M_BESS - M_SAVED); feasible if cargo >= cargo_target.
    """
    req     = calculate_battery_requirement(d_nm)
    m_bess  = calculate_battery_mass(req["e_installed_mwh"], rho_grav)
    cargo   = DWT_DESIGN - (m_bess - M_SAVED)
    req.update({
        "rho_grav": rho_grav,
        "m_bess_t": m_bess,
        "residual_cargo_t": cargo,
        "cargo_feasible": cargo >= cargo_target,
    })
    return req


def calculate_port_charging_power(e_installed_mwh, arrival_days):
    """Continuous pier-side charging power P_port (kW).

    The usable energy refilled each cycle (E_installed x DoD) is spread across
    the arrival interval minus the handling buffer, then grossed up for charger
    losses:  P_port = E_installed*DoD*1000 / ((T_arrival - T_buffer)*eta_charger).
    """
    t_arrival_h = arrival_days * 24.0
    window_h    = t_arrival_h - T_BUFFER
    return (e_installed_mwh * DOD * 1000.0) / (window_h * ETA_CHARGER)


# ==========================================================
# 3. FEASIBILITY CEILING  (numeric solver + closed form)
# ==========================================================

def find_ceiling(cargo_target=DWT_TARGET, rho_grav=RHO_GRAV,
                 tolerance_nm=0.01, max_distance=2000.0):
    """Numeric (bisection) maximum leg distance that still meets cargo_target."""
    low, high = 0.0, max_distance
    while high - low > tolerance_nm:
        mid = 0.5 * (low + high)
        cargo = calculate_residual_cargo(mid, rho_grav, cargo_target)["residual_cargo_t"]
        if cargo > cargo_target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def analytical_ceiling(rho_grav=RHO_GRAV, cargo_target=DWT_TARGET):
    """Closed-form ceiling, walking the constraint back through the modules.

    M_BESS_max = DWT_DESIGN - cargo_target + M_SAVED
    -> E_inst_max -> E_seg_max -> d_km_max -> d_nm_max.
    """
    m_bess_max = DWT_DESIGN - cargo_target + M_SAVED
    e_inst_max = m_bess_max * rho_grav
    e_seg_max  = e_inst_max * DOD * ETA_BESS / (1 + S_RES)
    d_km_max   = e_seg_max * 1000.0 * V_KMPH / (P_PROP / ETA_MOTOR + P_AUX)
    return d_km_max / NM_TO_KM


# ==========================================================
# 4. ROUTE DATASET  (2026 Kemendag registry — paper Table 2)
# ==========================================================
# Columns: official trayek code; Case A (asymmetric, long return leg) vs
# Case B (symmetric loop); longest uninterrupted leg D_max (nm); and the
# operational arrival interval (days) used to size the pier charger.

ROUTE_DATA = {
    "Route_Code": [
        "H-1", "H-4", "S-4B", "T-2", "T-9", "T-12", "H-2", "H-3",
        "S-1", "S-3", "S-4A", "S-5", "T-5", "T-10", "T-11", "T-20",
    ],
    "Case_Category": [
        "Case A", "Case A", "Case A", "Case A", "Case A", "Case A",
        "Case B", "Case B", "Case B", "Case B", "Case B", "Case B",
        "Case B", "Case B", "Case B", "Case B",
    ],
    "D_max_nm": [
        562, 715, 745, 554, 710, 818, 766, 647,
        616, 358, 160, 487, 617, 415, 507, 1152,
    ],
    "Arrival_Interval_Days": [
        7, 10, 7, 14, 10, 12, 14, 14,
        5, 4, 2, 6, 10, 5, 5, 10,
    ],
}


def build_route_matrix(route_data=None, rho_grav=RHO_GRAV, cargo_target=DWT_TARGET,
                       sort_by_distance=True):
    """Full sizing matrix (paper Table 2) for the candidate routes.

    Produces one row per route with installed capacity, battery mass, residual
    cargo, feasibility flag, and pier-side charging power P_port.
    """
    data = ROUTE_DATA if route_data is None else route_data
    df = pd.DataFrame(data)

    recs = []
    for _, r in df.iterrows():
        calc = calculate_residual_cargo(r["D_max_nm"], rho_grav, cargo_target)
        p_port = calculate_port_charging_power(calc["e_installed_mwh"],
                                               r["Arrival_Interval_Days"])
        recs.append({
            "Route_Code": r["Route_Code"],
            "Case_Category": r["Case_Category"],
            "D_max_nm": r["D_max_nm"],
            "D_max_km": round(r["D_max_nm"] * NM_TO_KM, 1),
            "E_seg_MWh": round(calc["e_seg_mwh"], 2),
            "E_installed_MWh": round(calc["e_installed_mwh"], 2),
            "M_BESS_tons": round(calc["m_bess_t"], 1),
            "DWT_Cargo_Remaining": round(calc["residual_cargo_t"], 1),
            "Payload_Feasible": calc["cargo_feasible"],
            "Arrival_Interval_Days": r["Arrival_Interval_Days"],
            "P_port_kW": round(p_port, 1),
        })

    out = pd.DataFrame(recs)
    if sort_by_distance:
        out = out.sort_values("D_max_nm").reset_index(drop=True)
    return out


def trade_off_curve(d_min=0, d_max=1200, step=5, rho_grav=RHO_GRAV):
    """Continuous payload-distance curve for plotting (Figure 4)."""
    d = np.arange(d_min, d_max + step, step)
    cargo = [calculate_residual_cargo(x, rho_grav)["residual_cargo_t"] for x in d]
    return pd.DataFrame({"distance_nm": d, "residual_cargo_t": cargo})


# ==========================================================
# 5. SENSITIVITY SWEEP  (Figure 6) — no global mutation
# ==========================================================

def ceiling_sensitivity(rho_grav_values, cargo_target_values):
    """Ceiling distance for each (rho_grav, cargo_target) pair.

    Pure function: parameters are passed straight into find_ceiling, so module
    globals are never reassigned (the prior consolidated engine mutated them).
    """
    return {
        (rho, cargo): find_ceiling(cargo_target=cargo, rho_grav=rho)
        for rho in rho_grav_values
        for cargo in cargo_target_values
    }


def viable_route_grid(rho_grav_values, cargo_target_values, route_data=None):
    """Count of viable routes across a (rho_grav, cargo_target) grid (Figure 6 heatmap)."""
    data = ROUTE_DATA if route_data is None else route_data
    distances = np.array(data["D_max_nm"], dtype=float)
    grid = {}
    for rho in rho_grav_values:
        for cargo in cargo_target_values:
            ceil = find_ceiling(cargo_target=cargo, rho_grav=rho)
            grid[(rho, cargo)] = int(np.sum(distances <= ceil))
    return grid


# ==========================================================
# 6. PORT MICROGRID SIZING  (Figure 8 context)
# ==========================================================

def size_port_microgrid(route_code, d_nm, e_installed_mwh, arrival_days,
                        community_load_mwh_per_day=8.0,
                        solar_yield_kwh_per_kwp_day=4.0):
    """First-order solar + stationary-BESS sizing for a port serving one route.

    Uses the route's real arrival interval to spread the ship recharge energy.
    Returns physical sizing only (solar capacity, storage, land). Capital-cost
    estimation is intentionally omitted: no microgrid cost figure is reported
    anywhere in the paper, so it is not computed here.
    """
    gen_required_mwh = (e_installed_mwh * DOD) / ETA_CHARGER     # refill per cycle
    daily_gen_ship   = gen_required_mwh / arrival_days
    daily_gen_total  = daily_gen_ship + community_load_mwh_per_day

    solar_kwp = daily_gen_total * 1000.0 / solar_yield_kwh_per_kwp_day
    bess_mwh  = daily_gen_total * 2.0 * 1.3                      # 2-day buffer + 30% reserve
    land_ha   = (solar_kwp / 1000.0) * 1.25                     # 1.25 ha/MWp tropical

    return {
        "route_code": route_code,
        "distance_nm": d_nm,
        "arrival_days": arrival_days,
        "ship_battery_mwh": e_installed_mwh,
        "ship_daily_gen_mwh": daily_gen_ship,
        "total_daily_gen_mwh": daily_gen_total,
        "solar_capacity_mwp": solar_kwp / 1000.0,
        "stationary_bess_mwh": bess_mwh,
        "land_area_ha": land_ha,
    }


# ==========================================================
# 7. DEMO / REPRODUCIBILITY DRIVER
# ==========================================================

def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print("=" * 84)
    print("TOL LAUT BATTERY-ELECTRIC PROPULSION — UNIFIED SIZING BACKBONE (v2.0)")
    print("=" * 84)

    # --- Route matrix (Table 2) ---
    mat = build_route_matrix()
    print("\n[1] Route sizing matrix (Table 2):")
    print(mat.to_string(index=False))
    mat.to_csv("Tol_Laut_Sizing_Engine_Results.csv", index=False)

    # --- Ceiling (numeric + analytical) ---
    c_num = find_ceiling()
    c_ana = analytical_ceiling()
    n_viable = int(mat["Payload_Feasible"].sum())
    print(f"\n[2] Feasibility ceiling: numeric = {c_num:.1f} nm | "
          f"analytical = {c_ana:.1f} nm")
    print(f"    Viable routes (>= {DWT_TARGET:.0f} t cargo): {n_viable} of {len(mat)}")

    # --- Charging ranges ---
    viable = mat[mat["Payload_Feasible"]]
    print(f"\n[3] Pier-side charging demand P_port:")
    print(f"    Full 16-route range : {mat['P_port_kW'].min():.0f}"
          f"-{mat['P_port_kW'].max():.0f} kW")
    print(f"    Viable-subset range : {viable['P_port_kW'].min():.0f}"
          f"-{viable['P_port_kW'].max():.0f} kW")

    # --- Sensitivity ---
    print("\n[4] Ceiling sensitivity (rho_grav x cargo_target):")
    sens = ceiling_sensitivity([0.14, 0.15, 0.16], [840, 1000])
    for (rho, cargo), c in sorted(sens.items()):
        print(f"    rho={rho:.2f} MWh/t, cargo={cargo:.0f} t -> ceiling = {c:.1f} nm")

    # --- Microgrid example (T-11, its real 5-day interval) ---
    t11 = mat[mat["Route_Code"] == "T-11"].iloc[0]
    mg = size_port_microgrid("T-11", t11["D_max_nm"], t11["E_installed_MWh"],
                             arrival_days=int(t11["Arrival_Interval_Days"]))
    print("\n[5] Port microgrid sizing example (T-11):")
    print(f"    Solar capacity : {mg['solar_capacity_mwp']:.1f} MWp")
    print(f"    Stationary BESS: {mg['stationary_bess_mwh']:.0f} MWh")
    print(f"    Land footprint : {mg['land_area_ha']:.1f} ha")

    print("\n" + "=" * 84)
    print("Saved route matrix -> Tol_Laut_Sizing_Engine_Results.csv")
    print("=" * 84)
    return mat


if __name__ == "__main__":
    main()
