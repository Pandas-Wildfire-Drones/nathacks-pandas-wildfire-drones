"""Sizing math behind context.md section 6. Run: python tools/sizing.py
Every input marked [S] is SOURCED (see context.md section 14), [A] is an ASSUMPTION."""
import math

G, RHO = 9.81, 1.225

# ---------- 1. Calibrate fuel burn to ACC Thunder Wasp GT (turbine, mechanical drive) ----------
# [S] MTOW 800 kg, empty 375 kg; 248 kg fuel + 177 kg payload ~ "just under 4 h"; 400 kg payload + ~25 kg fuel ~ 20 min.
BURN_800_ACC = 80.0                      # [A] kg/h at 800 kg, fitted to the two [S] points below
def burn_kg_h(m, burn800):               # hover-ish burn scales ~ m^1.5 at constant disk loading
    return burn800 * (m / 800.0) ** 1.5

def endurance_h(m0, fuel, burn800, reserve_frac=0.0, dt=1.0):
    m, f, t = m0, fuel, 0.0
    f_stop = fuel * reserve_frac
    while f > f_stop:
        b = burn_kg_h(m, burn800) / 3600 * dt
        f -= b; m -= b; t += dt
    return t / 3600

print("== calibration vs ACC Thunder Wasp GT ==")
print(f"248 kg fuel, 177 kg payload: {endurance_h(800,248,BURN_800_ACC):.2f} h   (source: just under 4 h)")
print(f"25 kg fuel, 400 kg payload : {endurance_h(800,25,BURN_800_ACC)*60:.0f} min (source: ~20 min)")

# ---------- 2. Our hybrid extinguisher ----------
MTOW, EMPTY = 800.0, 420.0               # [S] 800 MTOW; empty = 375 [S] + 20 genset + 25 battery buffer [A]
HYBRID_BURN_FACTOR = 0.65                # [A] efficient heavy-fuel genset + electric drive vs small turbine (VERIFY)
burn800 = BURN_800_ACC * HYBRID_BURN_FACTOR
print("\n== hybrid extinguisher: tank vs endurance (MTOW 800, empty 420) ==")
print("water_L  fuel_kg  endurance_h  (to dry)  with 15% reserve")
for W in (120, 150, 170, 200, 250, 300):
    F = MTOW - EMPTY - W
    print(f"{W:6d} {F:8.0f} {endurance_h(MTOW,F,burn800):10.2f} {endurance_h(MTOW,F,burn800,0.15):14.2f}")

W, F = 170.0, MTOW - EMPTY - 170.0
print(f"\nchosen: water {W:.0f} L, fuel {F:.0f} kg (= {F/0.8:.0f} L at 0.8 kg/L), MTOW {MTOW:.0f}")
print(f"burn at full mass {burn_kg_h(MTOW,burn800):.0f} kg/h ; at empty+reserve mass {burn_kg_h(EMPTY+40,burn800):.0f} kg/h")
print(f"hover-equivalent shaft power @800 kg ~138 kW (4 x 2.6 m rotors, FM 0.7, momentum theory) [A]")
print(f"battery buffer 5 kWh at 138 kW: {5/138*60:.1f} min hover (emergency landing reserve)")

# ---------- 3. Stopping distance vs lidar ----------
print("\n== stopping distance v^2/2a ==")
for name, a, speeds in (("extinguisher a=2.0", 2.0, (5, 10, 20, 25)), ("seeker a=4.0", 4.0, (5, 10, 15, 20))):
    print(name, {v: round(v*v/(2*a), 1) for v in speeds})

# ---------- 4. Cycle throughput and fuel per litre ----------
V, FILL, DROP = 20.0, 120.0, 8.0
def cycle(d_m, queue=0.0):
    return 2*d_m/V + FILL + DROP + queue
def fuel_per_cycle(d_m, queue=0.0):
    """Integrate burn: loaded outbound leg (mass 800-fuel_used), empty return, hover during fill/drop (counted at mean mass)."""
    t_out = d_m/V; t_back = d_m/V; t_hov = FILL + DROP + queue
    m_loaded = EMPTY + 100 + W            # mean fuel ~100 kg mid-mission [A]
    m_empty = EMPTY + 100
    # engine idles during fill if on a ground pad; assume 30% of hover burn there [A]
    burn = (burn_kg_h(m_loaded, burn800)*t_out + burn_kg_h(m_empty, burn800)*t_back
            + burn_kg_h(m_loaded, burn800)*0.3*FILL + burn_kg_h(m_loaded, burn800)*DROP)/3600
    return burn
print("\n== refill point throughput (170 L, 20 m/s, fill 120 s, drop 8 s) ==")
print("point              cycle_min  L/min   L/h    fuel_kg/cycle  L_per_kg_fuel")
for label, d, q in (("station 15 km", 15000, 0), ("station 5 km", 5000, 0), ("station 5 km +3min q", 5000, 180),
                    ("basin 1 km", 1000, 0), ("basin 0.5 km", 500, 0)):
    c = cycle(d, q); fb = fuel_per_cycle(d, q)
    print(f"{label:22s} {c/60:8.2f} {W/(c/60):7.2f} {W/(c/3600):7.0f} {fb:12.2f} {W/fb:12.1f}")
print(f"\nratio basin1km / station15km throughput: {cycle(15000)/cycle(1000):.1f}x")
print(f"sorties per full tank of fuel at station 5 km: {F*0.85/fuel_per_cycle(5000):.0f} ; at basin 1 km: {F*0.85/fuel_per_cycle(1000):.0f}")
print(f"total water per fuel load (85% usable): station 5 km {W*F*0.85/fuel_per_cycle(5000):.0f} L ; basin 1 km {W*F*0.85/fuel_per_cycle(1000):.0f} L")

# ---------- 5. Electric seeker ----------
print("\n== electric seeker (multirotor, 10 kg MTOW) ==")
M = 10.0; A = 4*math.pi*0.33**2; FM, ETA = 0.65, 0.80
p_hover = (M*G)**1.5 / math.sqrt(2*RHO*A) / FM / ETA
p_cruise = 0.85 * p_hover + 48
batt_kg, wh_kg, usable = 4.0, 250.0, 0.80
e_wh = batt_kg*wh_kg*usable
print(f"rotor area {A:.2f} m2, hover electric {p_hover:.0f} W, cruise(15 m/s, x0.85)+avionics {p_cruise:.0f} W")
print(f"battery {batt_kg} kg x {wh_kg} Wh/kg x {usable} usable = {e_wh:.0f} Wh -> {e_wh/p_cruise*60:.0f} min cruise, {e_wh/(p_hover+48)*60:.0f} min hover")
print(f"mass budget: battery 4.0 + avionics 1.0 + airframe/motors/props 3.2 + margin 1.8 = {4.0+1.0+3.2+1.8:.1f} kg")
swath = 2*80*math.tan(math.radians(16))
print(f"search: swath {swath:.1f} m x 15 m/s = {swath*15*3600/1e6:.2f} km2/h ideal ; x 0.8 h = {swath*15*3600*0.8/1e6:.1f} km2 per sortie")
print("FireSwarm-style VTOL option: 12 kg, L/D~10 => ", end="")
p_fw = 12*G*20/10/0.7 + 48
print(f"{p_fw:.0f} W at 20 m/s -> {1000/p_fw:.1f} h on 1 kWh usable")

# ---------- 6. Cycle simulation with falling mass: how long does one fuel load last? ----------
print("\n== one fuel load, repeated cycles (mass falls as fuel burns; engine idle 8 kg/h on the pad [A]) ==")
def sim_fuel_load(d_m, queue=0.0, idle_kg_h=8.0):
    fuel, t, litres, cycles = F, 0.0, 0.0, 0
    reserve = F * 0.15
    def burn_leg(mass_fn, dur):
        nonlocal fuel
        used = burn_kg_h(mass_fn(), burn800) * dur / 3600
        fuel -= used
    leg = d_m / V
    while True:
        # predicted need for one more cycle + return-to-station margin of one leg
        m_full = EMPTY + fuel + W
        need = (burn_kg_h(m_full, burn800) * (leg + DROP) + burn_kg_h(EMPTY + fuel, burn800) * leg) / 3600 + idle_kg_h * (FILL + queue) / 3600
        if fuel - need < reserve: break
        burn_leg(lambda: EMPTY + fuel + W, leg + DROP)            # loaded outbound + hover drop
        burn_leg(lambda: EMPTY + fuel, leg)                       # empty return
        fuel -= idle_kg_h * (FILL + queue) / 3600                 # fill on the pad
        t += leg * 2 + DROP + FILL + queue; litres += W; cycles += 1
    return cycles, t / 3600, litres
for label, d, q in (("station 15 km", 15000, 0), ("station 5 km", 5000, 0), ("station 5 km +3min q", 5000, 180),
                    ("basin 1 km", 1000, 0), ("basin 0.5 km", 500, 0)):
    c, h, L = sim_fuel_load(d, q)
    print(f"{label:22s} cycles {c:4d}  hours {h:5.2f}  water {L/1000:6.1f} m3  L per kg fuel (usable {F*0.85:.0f}) {L/(F*0.85):6.1f}")
