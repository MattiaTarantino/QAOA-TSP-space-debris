# ==========================================
# SIMULATEUR DE RENDEZ-VOUS ORBITAL (LAMBERT)
# Cœur physique : lambert-gemini.py (Évolution Différentielle + Tsiolkovski Séquentiel)
# Visualisation  : lambert.py / lambert-mistral.py (Animation 3D complète)
# ==========================================

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import itertools
from scipy.optimize import root_scalar, differential_evolution
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. CONSTANTES PHYSIQUES (Standard ESA/NASA)
# ==========================================
MU       = 398600.4418        # km^3/s^2  — Paramètre gravitationnel standard (EGM-96)
R_EARTH  = 6378.137           # km        — Rayon équatorial WGS-84
G0       = 9.80665 / 1000.0   # km/s^2    — Accélération gravitationnelle standard
ISP      = 310.0              # s         — Impulsion spécifique (Green Bi-propellant)

M_INITIAL   = 500.0   # kg   — Masse initiale du Chaser
ALPHA_WEIGHT_TIME = 0 # Poids (0.0 à 1.0) : 0 = 100% Delta-V, 1 = 100% Temps
MAX_DV_KMS  = 5.0     # km/s — Valeur max empirique pour normalisation
MAX_TIME_DAYS = 115.0  # Jours — Valeur max globale pour normalisation uniforme du temps
TAILLE      = 50      # Unité de rendu visuel (normalisée)
SCALE_EARTH = 1 / 10  # Échelle Terre dans la scène

# ==========================================
# 2. MÉCANIQUE ORBITALE
# ==========================================

def get_state_vectors(r_km, inc, omega, anomaly, ecc=0.0, a=None):
    """
    Calcule position (r) et vitesse (v) en ECI.
    Utilise l'équation de vis-viva — valide pour toute orbite képlérienne.
    Pour les orbites circulaires (ecc=0), a = r_km.
    """
    if a is None:
        a = r_km
    v_mag = np.sqrt(MU * (2.0 / r_km - 1.0 / a))

    r_p = np.array([r_km * np.cos(anomaly), r_km * np.sin(anomaly), 0.0])
    v_p = np.array([-v_mag * np.sin(anomaly),  v_mag * np.cos(anomaly), 0.0])

    R_omega = np.array([
        [ np.cos(omega), -np.sin(omega), 0],
        [ np.sin(omega),  np.cos(omega), 0],
        [ 0,              0,             1]
    ])
    R_inc = np.array([
        [1, 0,            0           ],
        [0, np.cos(inc), -np.sin(inc) ],
        [0, np.sin(inc),  np.cos(inc) ]
    ])
    rot = R_omega @ R_inc
    return rot @ r_p, rot @ v_p


def tsiolkovsky_sequential(dv1_kms, dv2_kms, m_initial=M_INITIAL):
    """
    Tsiolkovski séquentiel : la 2ème poussée est calculée sur la masse
    résiduelle après la 1ère brûlure — formulation rigoureuse.
    """
    m_after_dv1 = m_initial * np.exp(-dv1_kms / (G0 * ISP))
    fuel_dv1    = m_initial - m_after_dv1
    m_final     = m_after_dv1 * np.exp(-dv2_kms / (G0 * ISP))
    fuel_dv2    = m_after_dv1 - m_final
    return fuel_dv1 + fuel_dv2, m_final


def get_synodic_period(a1, a2):
    """
    3ème loi de Kepler → période synodique.
    Définit la borne supérieure physique pour le temps d'attente :
    au-delà de T_syn, la géométrie se répète exactement.
    """
    T1 = 2 * np.pi * np.sqrt(a1**3 / MU)
    T2 = 2 * np.pi * np.sqrt(a2**3 / MU)
    if abs(T1 - T2) < 1e-5:
        return 86400 * 5   # 5 jours si orbites quasi-identiques
    return abs((T1 * T2) / (T1 - T2))


# ==========================================
# 3. FONCTIONS DE STUMPFF & SOLVEUR DE LAMBERT
# ==========================================

def stumpff_S(z):
    if z > 0:  return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z * np.sqrt(z))
    elif z < 0: return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (-z * np.sqrt(-z))
    else:       return 1.0 / 6.0

def stumpff_C(z):
    if z > 0:  return (1.0 - np.cos(np.sqrt(z))) / z
    elif z < 0: return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    else:       return 1.0 / 2.0


def solve_lambert(r1_vec, r2_vec, tof, short_way=True, max_iter=200, tol=1e-6):
    """
    Solveur universel de Lambert (variable universelle z de Battin/Stumpff).
    Robuste : tente plusieurs intervalles de bracketing si le premier échoue.
    """
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    cos_dnu = np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0)
    dnu = np.arccos(cos_dnu)
    if not short_way:
        dnu = 2 * np.pi - dnu
    A = np.sin(dnu) * np.sqrt(r1 * r2 / (1.0 - np.cos(dnu)))

    def tof_equation(z):
        S, C = stumpff_S(z), stumpff_C(z)
        y = r1 + r2 + A * (z * S - 1.0) / np.sqrt(C)
        if y <= 0: return 1e5
        x = np.sqrt(y / C)
        return (x**3 * S + A * np.sqrt(y)) / np.sqrt(MU) - tof

    try:
        for blo, bhi in [(-10.0, 30.0), (-50.0, 100.0), (-100.0, 200.0)]:
            try:
                sol = root_scalar(tof_equation, bracket=[blo, bhi],
                                  method='bisect', maxiter=max_iter, xtol=tol)
                z = sol.root
                break
            except ValueError:
                continue
        else:
            return None, None
    except Exception:
        return None, None

    S, C = stumpff_S(z), stumpff_C(z)
    y = r1 + r2 + A * (z * S - 1.0) / np.sqrt(C)
    if y <= 0: return None, None

    f      = 1.0 - y / r1
    g      = A * np.sqrt(y / MU)
    g_dot  = 1.0 - y / r2
    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (g_dot * r2_vec - r1_vec) / g
    return v1_vec, v2_vec


def equations_mouvement(t, state):
    r, v = state[:3], state[3:]
    a = -MU * r / np.linalg.norm(r)**3
    return np.concatenate((v, a))


# ==========================================
# 4. GÉNÉRATION DES DÉBRIS & MATRICE DE MISSION
# ==========================================

def generate_debris_cluster(num_debris=4):
    np.random.seed(42)
    cluster = []
    for i in range(num_debris):
        h = 800.0 + np.random.uniform(-5.0, 5.0)
        cluster.append({
            'id':   i,
            'h':    h,
            'i':    np.radians(98.0) + np.random.uniform(-0.01, 0.01),
            'o':    np.radians(45.0) + np.random.uniform(-0.05, 0.05),
            'anom': np.random.uniform(0, 2 * np.pi),
            'ecc':  0.0,          # Orbites LEO quasi-circulaires
            'a':    h + R_EARTH,  # Demi-grand axe
            'size_category': 'Medium'
        })
    return cluster


def build_mission_dataframe(cluster, alpha=ALPHA_WEIGHT_TIME):
    """
    Construit la matrice de mission par optimisation GLOBALE (Évolution Différentielle).
    Borne supérieure du temps d'attente = période synodique (physiquement motivé).
    Utilise Tsiolkovski séquentiel pour le carburant.
    Retourne un DataFrame (matrice d'adjacence) ET un dict des paramètres optimaux par paire.
    """
    n = len(cluster)
    debris_names = [f"D{deb['id']}" for deb in cluster]
    df_mission = pd.DataFrame(index=debris_names, columns=debris_names, dtype=object)
    for name in debris_names:
        df_mission.at[name, name] = "-"
        
    opt_params = {}   # (start, target) -> (t_w, t_t)

    for start, target in itertools.permutations(range(n), 2):
        deb_s, deb_t = cluster[start], cluster[target]
        r1_km, r2_km = R_EARTH + deb_s['h'], R_EARTH + deb_t['h']
        a1, a2       = deb_s['a'], deb_t['a']

        T_syn = get_synodic_period(a1, a2)

        def objective(x):
            t_w, t_t = x
            anom_dep  = deb_s['anom']  + np.sqrt(MU / a1**3) * t_w
            anom_arr  = deb_t['anom']  + np.sqrt(MU / a2**3) * (t_w + t_t)
            r_dep, v_s = get_state_vectors(r1_km, deb_s['i'], deb_s['o'], anom_dep, 0.0, a1)
            r_arr, v_t = get_state_vectors(r2_km, deb_t['i'], deb_t['o'], anom_arr, 0.0, a2)
            v1, v2 = solve_lambert(r_dep, r_arr, t_t)
            if v1 is None: return 1e5
            
            dv_total = np.linalg.norm(v1 - v_s) + np.linalg.norm(v_t - v2)
            
            # Normalisation (0.0 -> 1.0 environ)
            norm_dv = dv_total / MAX_DV_KMS
            norm_time = (t_w + t_t) / (MAX_TIME_DAYS * 86400.0)
            
            return (1.0 - alpha) * norm_dv + alpha * norm_time

        bounds = [(0, T_syn), (1800, 43200)]
        res = differential_evolution(
            objective, bounds,
            strategy='best1bin', popsize=15, tol=1e-3,
            mutation=(0.5, 1.0), recombination=0.7,
            seed=42
        )
        t_w_opt, t_t_opt = res.x

        if res.fun >= 1e5 or t_w_opt < 0 or t_t_opt < 1800 or t_t_opt > 43200:
            continue

        # Recalcul final propre
        anom_dep  = deb_s['anom'] + np.sqrt(MU / a1**3) * t_w_opt
        anom_arr  = deb_t['anom'] + np.sqrt(MU / a2**3) * (t_w_opt + t_t_opt)
        r_dep, v_s = get_state_vectors(r1_km, deb_s['i'], deb_s['o'], anom_dep, 0.0, a1)
        r_arr, v_t = get_state_vectors(r2_km, deb_t['i'], deb_t['o'], anom_arr, 0.0, a2)

        v_trans_dep, v_trans_arr = solve_lambert(r_dep, r_arr, t_t_opt)
        dv1 = np.linalg.norm(v_trans_dep - v_s)
        dv2 = np.linalg.norm(v_t - v_trans_arr)
        fuel, _ = tsiolkovsky_sequential(dv1, dv2)

        opt_params[(start, target)] = (t_w_opt, t_t_opt)
        
        info_tuple = (
            round(dv1 * 1000, 2),
            round(dv2 * 1000, 2),
            round(t_w_opt / 3600, 2),
            round(t_t_opt / 3600, 2),
            round(fuel, 3)
        )
        df_mission.at[f"D{deb_s['id']}", f"D{deb_t['id']}"] = info_tuple

    df_mission.fillna("X", inplace=True)

    return df_mission, opt_params


# ==========================================
# 5. VISUALISATION — Animation 3D complète
# ==========================================

def km_to_norm(r_km):
    return (r_km / R_EARTH) * TAILLE


def plot_mission(num_debris=4, target_debris_index=1):
    my_cluster = generate_debris_cluster(num_debris=num_debris)

    print("\nOptimisation de la matrice de mission (Évolution Différentielle)…")
    df_mission, opt_params = build_mission_dataframe(my_cluster)

    print("\n" + "=" * 100)
    print("  MATRICE DE MISSION (Adjacence)")
    print("  Cellule : (ΔV1 [m/s], ΔV2 [m/s], T_w [h], T_vol [h], Fuel [kg])")
    print("=" * 100)
    print(df_mission.to_string())
    print("=" * 100 + "\n")

    # ── Paramètres du transfert visualisé (Chaser → cible) ──────────────
    chaser = my_cluster[0]
    target = next(d for d in my_cluster if d['id'] == target_debris_index)
    key    = (chaser['id'], target['id'])

    if key not in opt_params:
        print("Erreur : le transfert spécifié n'a pas convergé.")
        return

    t_wait, t_trans = opt_params[key]

    n_chaser = np.sqrt(MU / chaser['a']**3)
    n_target = np.sqrt(MU / target['a']**3)

    # Points de départ et d'arrivée
    anom_dep = chaser['anom'] + n_chaser * t_wait
    anom_arr = target['anom'] + n_target * (t_wait + t_trans)
    r_dep, v_dep_pre = get_state_vectors(chaser['a'],  chaser['i'], chaser['o'], anom_dep,  0.0, chaser['a'])
    r_arr, v_arr_pre = get_state_vectors(target['a'],  target['i'], target['o'], anom_arr,  0.0, target['a'])

    v_trans_dep, v_trans_arr = solve_lambert(r_dep, r_arr, t_trans)
    dv1 = np.linalg.norm(v_trans_dep - v_dep_pre)
    dv2 = np.linalg.norm(v_arr_pre   - v_trans_arr)
    fuel_used, _ = tsiolkovsky_sequential(dv1, dv2)

    print(f"  ΔV1 (Départ)      : {dv1 * 1000:.2f} m/s")
    print(f"  ΔV2 (Arrivée)     : {dv2 * 1000:.2f} m/s")
    print(f"  ΔV total          : {(dv1 + dv2) * 1000:.2f} m/s")
    print(f"  Carburant utilisé : {fuel_used:.2f} kg  (Tsiolkovski séquentiel)")
    print(f"  T_wait            : {t_wait / 3600:.2f} h")
    print(f"  T_vol             : {t_trans / 3600:.2f} h\n")

    # Arc de transfert (RK45 haute précision)
    sol = solve_ivp(
        equations_mouvement, [0, t_trans],
        np.concatenate((r_dep, v_trans_dep)),
        t_eval=np.linspace(0, t_trans, 80),
        rtol=1e-9, atol=1e-9
    )
    arc_pts = np.array([km_to_norm(r) for r in sol.y[:3, :].T])

    # ── Construction de la figure ────────────────────────────────────────
    fig = go.Figure()

    # Terre
    re_disp = TAILLE * SCALE_EARTH
    theta_e, phi_e = np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 60)
    x_e = re_disp * np.outer(np.cos(theta_e), np.sin(phi_e))
    y_e = re_disp * np.outer(np.sin(theta_e), np.sin(phi_e))
    z_e = re_disp * np.outer(np.ones(len(theta_e)), np.cos(phi_e))
    fig.add_trace(go.Surface(
        x=x_e, y=y_e, z=z_e,
        colorscale=[[0, 'royalblue'], [1, 'steelblue']],
        showscale=False, name='Terre', hoverinfo='skip'
    ))

    # Orbites + collecte des données d'animation
    debris_data = []
    t_circle = np.linspace(0, 2 * np.pi, 150)

    for deb in my_cluster:
        r_norm = km_to_norm(deb['a'])
        U = np.array([np.cos(deb['o']), np.sin(deb['o']), 0.0])
        W = np.array([np.sin(deb['i']) * np.sin(deb['o']),
                      -np.sin(deb['i']) * np.cos(deb['o']),
                       np.cos(deb['i'])])
        V = np.cross(W, U)

        ox = r_norm * (np.cos(t_circle) * U[0] + np.sin(t_circle) * V[0])
        oy = r_norm * (np.cos(t_circle) * U[1] + np.sin(t_circle) * V[1])
        oz = r_norm * (np.cos(t_circle) * U[2] + np.sin(t_circle) * V[2])

        color      = 'red' if deb['id'] == 0 else f"hsl({(deb['id'] * 137) % 360}, 80%, 65%)"
        vis_size   = {'Small': 5, 'Medium': 8, 'Large': 12}.get(deb.get('size_category', 'Medium'), 8)

        if deb['id'] == 0:
            fig.add_trace(go.Scatter3d(
                x=ox, y=oy, z=oz, mode='lines',
                line=dict(color=color, width=3, dash='dash'),
                opacity=0.7, name='Chaser Orbit', hoverinfo='skip'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=ox, y=oy, z=oz, mode='lines',
                line=dict(color=color, width=1),
                opacity=0.7, showlegend=False, hoverinfo='none'
            ))
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0], mode='markers',
                marker=dict(size=vis_size, color=color, line=dict(width=1, color='white')),
                showlegend=False, name=f'Debris {deb["id"]}'
            ))

        debris_data.append({
            'id':     deb['id'],
            'r_norm': r_norm,
            'n':      np.sqrt(MU / deb['a']**3),
            'anom':   deb['anom'],
            'U': U, 'V': V,
            'color':  color
        })

    # Marqueur Chaser (sera animé)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=12, color='red', symbol='diamond',
                    line=dict(width=2, color='white')),
        name='Chaser'
    ))

    # Arc de transfert Lambert (visible à la demande)
    fig.add_trace(go.Scatter3d(
        x=arc_pts[:, 0], y=arc_pts[:, 1], z=arc_pts[:, 2],
        mode='lines', line=dict(color='yellow', width=5, dash='dot'),
        name='Lambert Transfer', visible='legendonly'
    ))
    fig.add_trace(go.Scatter3d(
        x=[arc_pts[0, 0]], y=[arc_pts[0, 1]], z=[arc_pts[0, 2]],
        mode='markers', marker=dict(size=6, color='#FF6A00', symbol='diamond'),
        name='Burn 1 (Départ)'
    ))
    fig.add_trace(go.Scatter3d(
        x=[arc_pts[-1, 0]], y=[arc_pts[-1, 1]], z=[arc_pts[-1, 2]],
        mode='markers', marker=dict(size=6, color='#00FFB2', symbol='diamond'),
        name='Burn 2 (Arrivée)'
    ))

    # ── Animation (phasing → transfert → rendez-vous) ───────────────────
    MAX_WAIT_SHOWN = 2 * 3600.0
    t_wait_sim = min(t_wait, MAX_WAIT_SHOWN)
    t_skip     = t_wait - t_wait_sim

    num_frames      = 200
    total_sim_time  = t_wait_sim + t_trans + 3600.0
    dt              = total_sim_time / num_frames

    # Placement initial (après skip)
    for tr in fig.data:
        if getattr(tr, 'name', '') == 'Chaser':
            a_c = chaser['anom'] + n_chaser * t_skip
            tr.x = (debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][0] + np.sin(a_c) * debris_data[0]['V'][0]),)
            tr.y = (debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][1] + np.sin(a_c) * debris_data[0]['V'][1]),)
            tr.z = (debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][2] + np.sin(a_c) * debris_data[0]['V'][2]),)

    for deb in debris_data[1:]:
        a_d = deb['anom'] + deb['n'] * t_skip
        for tr in fig.data:
            if getattr(tr, 'name', '') == f'Debris {deb["id"]}':
                tr.x = (deb['r_norm'] * (np.cos(a_d) * deb['U'][0] + np.sin(a_d) * deb['V'][0]),)
                tr.y = (deb['r_norm'] * (np.cos(a_d) * deb['U'][1] + np.sin(a_d) * deb['V'][1]),)
                tr.z = (deb['r_norm'] * (np.cos(a_d) * deb['U'][2] + np.sin(a_d) * deb['V'][2]),)

    idx_chaser = next((i for i, tr in enumerate(fig.data) if getattr(tr, 'name', '') == 'Chaser'), -1)
    idx_debris = [i for i, tr in enumerate(fig.data) if (getattr(tr, 'name', '') or '').startswith('Debris ')]
    anim_traces = ([idx_chaser] if idx_chaser >= 0 else []) + idx_debris

    frames = []
    for step in range(num_frames):
        t_sim = step * dt
        t_real = t_skip + t_sim
        fd = []

        # Position du Chaser selon la phase
        if t_sim <= t_wait_sim:
            a_c  = chaser['anom'] + n_chaser * t_real
            cx = debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][0] + np.sin(a_c) * debris_data[0]['V'][0])
            cy = debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][1] + np.sin(a_c) * debris_data[0]['V'][1])
            cz = debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][2] + np.sin(a_c) * debris_data[0]['V'][2])
        elif t_sim <= t_wait_sim + t_trans:
            idx = min(int(((t_sim - t_wait_sim) / t_trans) * len(arc_pts)), len(arc_pts) - 1)
            cx, cy, cz = arc_pts[idx]
        else:
            tsa = t_real - (t_wait + t_trans)
            td  = next(d for d in debris_data if d['id'] == target['id'])
            a_rv = anom_arr + td['n'] * tsa
            cx = td['r_norm'] * (np.cos(a_rv) * td['U'][0] + np.sin(a_rv) * td['V'][0])
            cy = td['r_norm'] * (np.cos(a_rv) * td['U'][1] + np.sin(a_rv) * td['V'][1])
            cz = td['r_norm'] * (np.cos(a_rv) * td['U'][2] + np.sin(a_rv) * td['V'][2])

        fd.append(go.Scatter3d(x=[cx], y=[cy], z=[cz]))

        # Débris mobiles
        for deb in debris_data[1:]:
            a_t = deb['anom'] + deb['n'] * t_real
            fd.append(go.Scatter3d(
                x=[deb['r_norm'] * (np.cos(a_t) * deb['U'][0] + np.sin(a_t) * deb['V'][0])],
                y=[deb['r_norm'] * (np.cos(a_t) * deb['U'][1] + np.sin(a_t) * deb['V'][1])],
                z=[deb['r_norm'] * (np.cos(a_t) * deb['U'][2] + np.sin(a_t) * deb['V'][2])],
            ))

        frames.append(go.Frame(data=fd, traces=anim_traces, name=str(step)))

    fig.frames = frames

    # ── Layout ───────────────────────────────────────────────────────────
    max_r = km_to_norm(R_EARTH + 2000.0) * 1.1
    vis_with_arc    = [True] * len(fig.data)
    vis_without_arc = [True] * len(fig.data)
    for i, tr in enumerate(fig.data):
        if (getattr(tr, 'name', '') or '').startswith('Lambert'):
            vis_without_arc[i] = False

    fig.update_layout(
        title=dict(
            text=(f"3D Rendez-Vous Lambert — "
                  f"T_wait: {t_wait/3600:.1f} h | T_vol: {t_trans/3600:.1f} h | "
                  f"Fuel: {fuel_used:.1f} kg"),
            font=dict(color='white', size=18), x=0.5, y=0.97
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        scene=dict(
            xaxis=dict(visible=False, range=[-max_r, max_r]),
            yaxis=dict(visible=False, range=[-max_r, max_r]),
            zaxis=dict(visible=False, range=[-max_r, max_r]),
            bgcolor='black', aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        updatemenus=[
            dict(
                type="buttons", showactive=True,
                bgcolor='white', font=dict(color='black'),
                y=0.10, x=0.10,
                buttons=[dict(
                    label="► Play / ❚❚ Pause",
                    method="animate",
                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)],
                    args2=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                )]
            ),
            dict(
                type="buttons", showactive=True,
                bgcolor='white', font=dict(color='black'),
                y=0.20, x=0.10,
                buttons=[dict(
                    label="Afficher l'arc",
                    method="restyle",
                    args=[{"visible": vis_with_arc}],
                    args2=[{"visible": vis_without_arc}]
                )]
            ),
        ]
    )

    out = "mission_lambert_final.html"
    fig.write_html(out, auto_open=True)
    print(f"→ Rendu 3D généré : {out}")


# ==========================================
# 6. ENTRÉE PRINCIPALE
# ==========================================
if __name__ == "__main__":
    plot_mission(num_debris=4, target_debris_index=1)
