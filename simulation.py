import numpy as np
import plotly.graph_objects as go
import pandas as pd
import itertools
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

# ==========================================
# 1. CONSTANTES PHYSIQUES (Standard ESA/NASA)
# ==========================================
MU = 398600.4418      # km^3/s^2  — Paramètre gravitationnel standard
R_EARTH = 6378.137    # km        — Rayon équatorial WGS-84
G0 = 9.80665 / 1000.0 # km/s^2   — Accélération gravitationnelle standard
ISP = 310.0           # s         — Impulsion spécifique (Green Bi-propellant)

M_INITIAL = 500.0     # kg  — Masse initiale du Chaser
TAILLE = 50           # Échelle de rendu visuel
SCALE_EARTH = 1 / 10

# ==========================================
# 2. MÉCANIQUE ORBITALE & SOLVEUR DE LAMBERT
# ==========================================

def get_state_vectors(r_km, inc, omega, anomaly):
    """Calcule les vecteurs position r et vitesse v (Inertiel ECI)."""
    r_p = np.array([r_km * np.cos(anomaly), r_km * np.sin(anomaly), 0.0])
    v_p = np.array([
        -np.sqrt(MU / r_km) * np.sin(anomaly),
         np.sqrt(MU / r_km) * np.cos(anomaly),
         0.0
    ])
    R_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega),  np.cos(omega), 0],
        [0,              0,             1]
    ])
    R_inc = np.array([
        [1, 0,             0           ],
        [0, np.cos(inc), -np.sin(inc) ],
        [0, np.sin(inc),  np.cos(inc) ]
    ])
    rot = R_omega @ R_inc
    return rot @ r_p, rot @ v_p

def solve_phasing_lambert(h1, i1, o1, anom1, h2, i2, o2, anom2, dnu_deg=120.0):
    """
    Calcule le temps d'attente pour un transfert ciblant un angle spécifique (ex: 120°).
    Évite l'immense transfert de Hohmann (180°).
    """
    r1, r2 = R_EARTH + h1, R_EARTH + h2
    n1, n2 = np.sqrt(MU / r1**3), np.sqrt(MU / r2**3)

    # Temps de vol estimé pour parcourir l'angle demandé
    transfer_angle = np.radians(dnu_deg)
    t_trans = transfer_angle / n1
    
    # Angle parcouru par la cible pendant ce temps
    target_travel = n2 * t_trans
    target_lead_angle = transfer_angle - target_travel
    current_phase = anom2 - anom1
    
    omega_rel = n1 - n2
    if abs(omega_rel) < 1e-10: omega_rel = 1e-10

    if omega_rel > 0:
        angle_to_catch_up = (current_phase - target_lead_angle) % (2 * np.pi)
        t_wait = angle_to_catch_up / omega_rel
    else:
        angle_to_catch_up = (target_lead_angle - current_phase) % (2 * np.pi)
        t_wait = angle_to_catch_up / abs(omega_rel)

    return t_wait, t_trans

def stumpff_S(z):
    if z > 0: return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z * np.sqrt(z))
    elif z < 0: return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (-z * np.sqrt(-z))
    else: return 1.0 / 6.0

def stumpff_C(z):
    if z > 0: return (1.0 - np.cos(np.sqrt(z))) / z
    elif z < 0: return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    else: return 1.0 / 2.0

def solve_lambert(r1_vec, r2_vec, tof, prograde=True):
    """Solveur universel de Lambert (Trouve les vitesses de transfert exactes)."""
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    
    cross_r = np.cross(r1_vec, r2_vec)
    cos_dnu = np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0)
    
    if prograde:
        dnu = np.arccos(cos_dnu) if cross_r[2] >= 0 else 2*np.pi - np.arccos(cos_dnu)
    else:
        dnu = np.arccos(cos_dnu) if cross_r[2] < 0 else 2*np.pi - np.arccos(cos_dnu)

    A = np.sin(dnu) * np.sqrt(r1 * r2 / (1.0 - np.cos(dnu)))

    def tof_equation(z):
        S, C = stumpff_S(z), stumpff_C(z)
        y = r1 + r2 + A * (z * S - 1.0) / np.sqrt(C)
        if y < 0: return -1e5 # Pénalité si mathématiquement invalide
        x = np.sqrt(y / C)
        return (x**3 * S + A * np.sqrt(y)) / np.sqrt(MU) - tof

    # Recherche de la racine (Variable universelle z)
    sol = root_scalar(tof_equation, bracket=[-10.0, 30.0], method='bisection')
    z = sol.root

    # Calcul des vecteurs finaux
    S, C = stumpff_S(z), stumpff_C(z)
    y = r1 + r2 + A * (z * S - 1.0) / np.sqrt(C)
    
    f = 1.0 - y / r1
    g = A * np.sqrt(y / MU)
    g_dot = 1.0 - y / r2
    
    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (g_dot * r2_vec - r1_vec) / g
    
    return v1_vec, v2_vec

def tsiolkovsky(total_dv, m_initial=M_INITIAL):
    m_final = m_initial * np.exp(-total_dv / (G0 * ISP))
    return m_final, m_initial - m_final

def equations_mouvement(t, state):
    """Modèle physique RK45 pour dessiner l'orbite de transfert."""
    r = state[:3]
    v = state[3:]
    a = -MU * r / np.linalg.norm(r)**3
    return np.concatenate((v, a))

# ==========================================
# 3. GÉNÉRATION DE L'AMAS ET GRAPHE
# ==========================================

def generate_debris_cluster(num_debris=5, mode='real'):
    np.random.seed(42)
    cluster = []
    
    if mode == 'random':
        cluster.append({'id': 0, 'h': 400.0, 'i': np.radians(98.0), 'o': np.radians(45.0), 'anom': 0.0, 'size_category': 'Medium'})
        cluster.append({'id': 1, 'h': 450.0, 'i': np.radians(98.5), 'o': np.radians(46.0), 'anom': np.radians(180.0), 'size_category': 'Large'})
        return cluster

    for i in range(num_debris):
        cluster.append({
            'id': i, 'h': 800.0 + np.random.uniform(-5.0, 5.0), 
            'i': np.radians(98.0) + np.random.uniform(-0.01, 0.01), 
            'o': np.radians(45.0) + np.random.uniform(-0.05, 0.05),
            'anom': np.random.uniform(0, 2 * np.pi), 
            'size_category': 'Medium'
        })
    return cluster

def build_fuel_distance_matrix(cluster):
    n = len(cluster)
    fuel_matrix = np.zeros((n, n))
    
    for start, target in itertools.permutations(range(n), 2):
        deb_start, deb_target = cluster[start], cluster[target]
        
        t_wait, t_trans = solve_phasing_lambert(
            deb_start['h'], deb_start['i'], deb_start['o'], deb_start['anom'],
            deb_target['h'], deb_target['i'], deb_target['o'], deb_target['anom']
        )
        
        r_dep_km, r_arr_km = R_EARTH + deb_start['h'], R_EARTH + deb_target['h']
        n_start, n_target = np.sqrt(MU / r_dep_km**3), np.sqrt(MU / r_arr_km**3)
        
        anom_dep = deb_start['anom'] + n_start * t_wait
        anom_arr = deb_target['anom'] + n_target * (t_wait + t_trans)
        
        r_dep, v_start_pre = get_state_vectors(r_dep_km, deb_start['i'], deb_start['o'], anom_dep)
        r_arr, v_target_pre = get_state_vectors(r_arr_km, deb_target['i'], deb_target['o'], anom_arr)
        
        try:
            v_trans_dep, v_trans_arr = solve_lambert(r_dep, r_arr, t_trans)
            total_dv = np.linalg.norm(v_trans_dep - v_start_pre) + np.linalg.norm(v_target_pre - v_trans_arr)
            _, fuel_used = tsiolkovsky(total_dv)
            fuel_matrix[start, target] = fuel_used
        except Exception:
            fuel_matrix[start, target] = np.inf # Trajectoire impossible
            
    return fuel_matrix

# ==========================================
# 4. VISUALISATION
# ==========================================

def km_to_norm(r_km): return (r_km / R_EARTH) * TAILLE

def plot_mission(num_debris=3, target_debris_index=1, mode='real'):
    my_cluster = generate_debris_cluster(num_debris=num_debris, mode=mode)
    fuel_mat = build_fuel_distance_matrix(my_cluster)
    
    df_fuel = pd.DataFrame(fuel_mat, columns=[f"V {i['id']}" for i in my_cluster], index=[f"De {i['id']}" for i in my_cluster])
    print("\n=== MATRICE DE CARBURANT (KG) ===")
    print(df_fuel.round(2))

    chaser = my_cluster[0]
    target = next(d for d in my_cluster if d['id'] == target_debris_index)

    r1_km = R_EARTH + chaser['h']
    
    fig = go.Figure()

    # Terre
    theta_e, phi_e = np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 60)
    re_disp = TAILLE * SCALE_EARTH
    x_e = re_disp * np.outer(np.cos(theta_e), np.sin(phi_e))
    y_e = re_disp * np.outer(np.sin(theta_e), np.sin(phi_e))
    z_e = re_disp * np.outer(np.ones(len(theta_e)), np.cos(phi_e))
    fig.add_trace(go.Surface(x=x_e, y=y_e, z=z_e, colorscale=[[0, 'royalblue'], [1, 'steelblue']], showscale=False, hoverinfo='skip'))

    # Orbites statiques et Débris (simplifié pour l'affichage)
    debris_data = []
    t_circle = np.linspace(0, 2 * np.pi, 150)
    
    for deb in my_cluster:
        r_km = R_EARTH + deb['h']
        U = np.array([np.cos(deb['o']), np.sin(deb['o']), 0.0])
        W = np.array([np.sin(deb['i']) * np.sin(deb['o']), -np.sin(deb['i']) * np.cos(deb['o']), np.cos(deb['i'])])
        V = np.cross(W, U)
        
        ox = km_to_norm(r_km) * (np.cos(t_circle) * U[0] + np.sin(t_circle) * V[0])
        oy = km_to_norm(r_km) * (np.cos(t_circle) * U[1] + np.sin(t_circle) * V[1])
        oz = km_to_norm(r_km) * (np.cos(t_circle) * U[2] + np.sin(t_circle) * V[2])
        
        color = 'red' if deb['id'] == 0 else f"hsl({(deb['id'] * 137) % 360}, 80%, 65%)"
        fig.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode='lines', line=dict(color=color, width=2, dash='dash' if deb['id']==0 else 'solid'), opacity=0.4))
        
        debris_data.append({'id': deb['id'], 'r_norm': km_to_norm(r_km), 'n': np.sqrt(MU / r_km**3), 'anom': deb['anom'], 'U': U, 'V': V, 'color': color})

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=12, color='red', symbol='diamond'), name='Chaser'))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=8, color=debris_data[1]['color']), name='Target'))

    # --- Physique du Transfert (Lambert + Intégration RK45) ---
    t_wait, t_trans = solve_phasing_lambert(
        chaser['h'], chaser['i'], chaser['o'], chaser['anom'],
        target['h'], target['i'], target['o'], target['anom']
    )

    n_chaser = debris_data[0]['n']
    n_target = debris_data[1]['n']
    
    anom_dep = chaser['anom'] + n_chaser * t_wait
    anom_arr = target['anom'] + n_target * (t_wait + t_trans)

    r_dep, v_chaser_pre = get_state_vectors(r1_km, chaser['i'], chaser['o'], anom_dep)
    r_arr, v_target_pre = get_state_vectors(R_EARTH + target['h'], target['i'], target['o'], anom_arr)

    # Solveur de Lambert
    v_trans_dep, v_trans_arr = solve_lambert(r_dep, r_arr, t_trans)
    
    dv1 = np.linalg.norm(v_trans_dep - v_chaser_pre)
    dv2 = np.linalg.norm(v_target_pre - v_trans_arr)
    _, fuel_used = tsiolkovsky(dv1 + dv2)

    # Simulation exacte de l'arc de transfert
    sol = solve_ivp(equations_mouvement, [0, t_trans], np.concatenate((r_dep, v_trans_dep)), t_eval=np.linspace(0, t_trans, 80))
    arc_pts = np.array([km_to_norm(r) for r in sol.y[:3, :].T])

    fig.add_trace(go.Scatter3d(x=arc_pts[:, 0], y=arc_pts[:, 1], z=arc_pts[:, 2], mode='lines', line=dict(color='yellow', width=5, dash='dot'), name='Lambert Transfer'))

    print(f"\n=== MISSION : INTERCEPT DEBRIS {target_debris_index} ===")
    print(f"  ΔV1 (Départ)     : {dv1 * 1000:.2f} m/s")
    print(f"  ΔV2 (Arrivée)    : {dv2 * 1000:.2f} m/s")
    print(f"  Carburant utilisé: {fuel_used:.2f} kg")
    print(f"========================================\n")

    # --- Animation (Simplifiée pour la fluidité) ---
    frames = []
    num_frames = 150
    dt = (t_wait + t_trans) / num_frames

    for step in range(num_frames):
        t_sim = step * dt
        frame_data = []

        # Chaser
        if t_sim <= t_wait:
            a_c = chaser['anom'] + n_chaser * t_sim
            cx = debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][0] + np.sin(a_c) * debris_data[0]['V'][0])
            cy = debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][1] + np.sin(a_c) * debris_data[0]['V'][1])
            cz = debris_data[0]['r_norm'] * (np.cos(a_c) * debris_data[0]['U'][2] + np.sin(a_c) * debris_data[0]['V'][2])
        else:
            idx = min(int(((t_sim - t_wait) / t_trans) * len(arc_pts)), len(arc_pts) - 1)
            cx, cy, cz = arc_pts[idx]
            
        frame_data.append(go.Scatter3d(x=[cx], y=[cy], z=[cz]))

        # Cible
        a_t = target['anom'] + n_target * t_sim
        tx = debris_data[1]['r_norm'] * (np.cos(a_t) * debris_data[1]['U'][0] + np.sin(a_t) * debris_data[1]['V'][0])
        ty = debris_data[1]['r_norm'] * (np.cos(a_t) * debris_data[1]['U'][1] + np.sin(a_t) * debris_data[1]['V'][1])
        tz = debris_data[1]['r_norm'] * (np.cos(a_t) * debris_data[1]['U'][2] + np.sin(a_t) * debris_data[1]['V'][2])
        frame_data.append(go.Scatter3d(x=[tx], y=[ty], z=[tz]))

        frames.append(go.Frame(data=frame_data, name=str(step)))

    fig.frames = frames

    fig.update_layout(
        title=dict(text="3D Rendez-Vous — Lambert Solver (NASA-Grade)", font=dict(color='white', size=22), x=0.5, y=0.95),
        paper_bgcolor='black',
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='black', aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=0),
        updatemenus=[dict(type="buttons", showactive=True, bgcolor='white', y=0.10, x=0.10, buttons=[dict(label="► Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])]
    )

    fig.write_html("mission_lambert.html", auto_open=True)
    print("Fichier interactif généré !")

if __name__ == "__main__":
    plot_mission(mode='random', target_debris_index=1)