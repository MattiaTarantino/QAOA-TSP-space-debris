import numpy as np
import plotly.graph_objects as go
import pandas as pd
import itertools

# ==========================================
# 1. CONSTANTES PHYSIQUES (Standard ESA/NASA)
# ==========================================
MU = 398600.4418      # km^3/s^2  — Paramètre gravitationnel standard
R_EARTH = 6378.137    # km        — Rayon équatorial WGS-84
G0 = 9.80665 / 1000.0 # km/s^2   — Accélération gravitationnelle standard
ISP = 310.0           # s         — Impulsion spécifique (Green Bi-propellant)

# Paramètres mission
M_INITIAL = 500.0     # kg  — Masse initiale du Chaser
M_KIT = 24.0          # kg  — Masse du kit de déorbitisation

# Paramètres visuels (échelle de rendu)
TAILLE = 50
SCALE_EARTH = 1 / 10

# ==========================================
# 2. FONCTIONS DE MÉCANIQUE ORBITALE
# ==========================================

def get_state_vectors(r_km, inc, omega, anomaly):
    """Calcule les vecteurs position r et vitesse v (Inertiel ECI) d'un satellite."""
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


def solve_phasing(h1, i1, o1, anom1, h2, i2, o2, anom2):
    """
    Calcule le temps d'attente optimal pour un rendez-vous de Hohmann.
    Retourne : (t_wait, t_trans, a_trans)
    """
    r1, r2 = R_EARTH + h1, R_EARTH + h2
    n1, n2 = np.sqrt(MU / r1**3), np.sqrt(MU / r2**3)

    a_trans = (r1 + r2) / 2.0
    t_trans = np.pi * np.sqrt(a_trans**3 / MU)

    alpha = n2 * t_trans
    target_lead_angle = np.pi - alpha
    current_phase = anom2 - anom1

    omega_rel = n1 - n2
    if abs(omega_rel) < 1e-10:
        omega_rel = 1e-10

    angle_to_catch_up = (target_lead_angle - current_phase) % (2 * np.pi)
    
    if omega_rel > 0:
        # Le Chaser va plus vite (orbite plus basse). Il comble son retard.
        # target_lead_angle doit être la phase finale (target - chaser).
        # Donc (anom2 - anom1)_final = (anom2 - anom1)_initial - omega_rel * t_wait = target_lead_angle
        # omega_rel * t_wait = current_phase - target_lead_angle
        angle_to_catch_up = (current_phase - target_lead_angle) % (2 * np.pi)
        t_wait = angle_to_catch_up / omega_rel
    else:
        # Le Débris va plus vite (Chaser sur orbite plus haute).
        # (anom2 - anom1)_final = current_phase + abs(omega_rel) * t_wait = target_lead_angle
        # abs(omega_rel) * t_wait = target_lead_angle - current_phase
        angle_to_catch_up = (target_lead_angle - current_phase) % (2 * np.pi)
        t_wait = angle_to_catch_up / abs(omega_rel)

    return t_wait, t_trans, a_trans


def compute_dv_vectors(r_dep, r_arr, v_chaser_pre, v_target_pre, a_trans):
    """
    Calcule les Delta-V vectoriels (ΔV1 et ΔV2) pour un transfert 3D.
    [CORRECTIF PHYSIQUE] : Si l'angle de transfert est de 180° (vecteurs opposés),
    le plan de transfert utilise le plan bissecteur pour optimiser la répartition
    du changement d'inclinaison.
    """
    h_chaser = np.cross(r_dep, v_chaser_pre)
    h_target = np.cross(r_arr, v_target_pre)
    h_vec = np.cross(r_dep, r_arr)
    norm_h = np.linalg.norm(h_vec)
    
    # Détection de la dégénérescence à 180° (vecteurs colinéaires)
    # NOTE: Pour r=7000km, norm_h est de ~1e7 même pour un écart de 0.1°. 
    # On utilise un seuil élevé pour garantir la stabilité du plan bissecteur.
    if norm_h < 1e7:
        # On crée le plan bissecteur exact (Moyenne des moments cinétiques)
        h_chaser_hat = h_chaser / np.linalg.norm(h_chaser)
        h_target_hat = h_target / np.linalg.norm(h_target)
        h_unit = h_chaser_hat + h_target_hat
        norm_h_unit = np.linalg.norm(h_unit)
        
        # Sécurité si les orbites sont parfaitement inversées (rétrogrades)
        if norm_h_unit < 1e-10:
            h_unit = h_chaser_hat
        else:
            h_unit = h_unit / norm_h_unit
    else:
        h_unit = h_vec / norm_h

    # Garantir que h_unit est dans le même sens que l'orbite initiale
    if np.dot(h_unit, h_chaser) < 0:
        h_unit = -h_unit

    # Vitesses sur l'orbite de transfert
    v_p_mag = np.sqrt(MU * (2.0 / np.linalg.norm(r_dep) - 1.0 / a_trans))
    v_p_vec = v_p_mag * np.cross(h_unit, r_dep / np.linalg.norm(r_dep))

    v_a_mag = np.sqrt(MU * (2.0 / np.linalg.norm(r_arr) - 1.0 / a_trans))
    v_a_vec = v_a_mag * np.cross(h_unit, r_arr / np.linalg.norm(r_arr))

    dv1 = np.linalg.norm(v_p_vec - v_chaser_pre)
    dv2 = np.linalg.norm(v_target_pre - v_a_vec)
    
    return dv1, dv2, dv1 + dv2, h_unit


def tsiolkovsky(total_dv, m_initial=M_INITIAL):
    """Applique l'équation de Tsiolkovsky."""
    m_final = m_initial * np.exp(-total_dv / (G0 * ISP))
    return m_final, m_initial - m_final

# ==========================================
# 3. GÉNÉRATION DE L'AMAS ET GRAPHE
# ==========================================

def generate_debris_cluster(num_debris=5, mode='real', base_h=800.0, base_i=np.radians(98.0), base_o=np.radians(45.0)):
    np.random.seed(42)
    cluster = []
    
    if mode == 'random':
        # Mode Démonstration Physique : 2 orbites très différentes
        # Chaser (index 0)
        cluster.append({
            'id': 0, 'h': 400.0, 'i': np.radians(98.0), 'o': np.radians(45.0),
            'anom': 0.0, 'size_category': 'Medium', 'size_m': 2.0
        })
        # Débris cible (index 1)
        cluster.append({
            'id': 1, 'h': 1200.0, 'i': np.radians(110.0), 'o': np.radians(60.0),
            'anom': np.radians(180.0), 'size_category': 'Large', 'size_m': 10.0
        })
        return cluster

    # Mode Réel : Amas SSO (groupé)
    for i in range(num_debris):
        h_deb = base_h + np.random.uniform(-5.0, 5.0)       
        i_deb = base_i + np.random.uniform(-0.01, 0.01)     
        o_deb = base_o + np.random.uniform(-0.05, 0.05)     
        anom_deb = np.random.uniform(0, 2 * np.pi)          
        
        r_size = np.random.random()
        if r_size < 0.6:
            cat, s_m = 'Small', np.random.uniform(0.1, 1.0)
        elif r_size < 0.9:
            cat, s_m = 'Medium', np.random.uniform(1.0, 5.0)
        else:
            cat, s_m = 'Large', np.random.uniform(5.0, 15.0)

        cluster.append({
            'id': i, 'h': h_deb, 'i': i_deb, 'o': o_deb,
            'anom': anom_deb, 'size_category': cat, 'size_m': s_m
        })
    return cluster

def build_fuel_distance_matrix(cluster):
    n = len(cluster)
    fuel_matrix = np.zeros((n, n))
    dv_matrix = np.zeros((n, n))
    
    for start, target in itertools.permutations(range(n), 2):
        deb_start, deb_target = cluster[start], cluster[target]
        
        t_wait, t_trans, a_trans = solve_phasing(
            deb_start['h'], deb_start['i'], deb_start['o'], deb_start['anom'],
            deb_target['h'], deb_target['i'], deb_target['o'], deb_target['anom']
        )
        
        r_dep_km, r_arr_km = R_EARTH + deb_start['h'], R_EARTH + deb_target['h']
        n_start, n_target = np.sqrt(MU / r_dep_km**3), np.sqrt(MU / r_arr_km**3)
        
        anom_dep = deb_start['anom'] + n_start * t_wait
        anom_arr = deb_target['anom'] + n_target * (t_wait + t_trans)
        
        r_dep, v_start_pre = get_state_vectors(r_dep_km, deb_start['i'], deb_start['o'], anom_dep)
        r_arr, v_target_pre = get_state_vectors(r_arr_km, deb_target['i'], deb_target['o'], anom_arr)
        
        dv1, dv2, total_dv, _ = compute_dv_vectors(r_dep, r_arr, v_start_pre, v_target_pre, a_trans)
        _, fuel_used = tsiolkovsky(total_dv, m_initial=M_INITIAL)
        
        fuel_matrix[start, target] = fuel_used
        dv_matrix[start, target] = total_dv
        
    return fuel_matrix, dv_matrix

# ==========================================
# 4. VISUALISATION
# ==========================================

def km_to_norm(r_km): return (r_km / R_EARTH) * TAILLE
def norm_to_km(r_norm): return (r_norm / TAILLE) * R_EARTH

def plot_mission(num_debris=3, target_debris_index=1, mode='real'):
    print(f"Mode de simulation : {mode.upper()}")
    print("Génération de l'amas de débris...")
    my_cluster = generate_debris_cluster(num_debris=num_debris, mode=mode)
    
    print("Calcul du graphe complet (Matrice de coûts)...")
    fuel_mat, dv_mat = build_fuel_distance_matrix(my_cluster)
    
    df_fuel = pd.DataFrame(fuel_mat, 
                           columns=[f"Vers {i['id']}" for i in my_cluster], 
                           index=[f"De {i['id']}" for i in my_cluster])
    
    print("\n=== MATRICE DES DISTANCES (CARBURANT EN KG) ===")
    print(df_fuel.round(2))

    # --- Paramètres Chaser ---
    h_chaser = my_cluster[0]['h']
    i_chaser = my_cluster[0]['i']
    o_chaser = my_cluster[0]['o']
    anom_chaser = my_cluster[0]['anom']
    r1_km = R_EARTH + h_chaser
    n_chaser = np.sqrt(MU / r1_km**3)
    chaser_r_norm = km_to_norm(r1_km)

    U_chaser = np.array([np.cos(o_chaser), np.sin(o_chaser), 0.0])
    W_chaser = np.array([np.sin(i_chaser) * np.sin(o_chaser), -np.sin(i_chaser) * np.cos(o_chaser), np.cos(i_chaser)])
    V_chaser = np.cross(W_chaser, U_chaser)

    fig = go.Figure()

    # Terre
    theta_e, phi_e = np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 60)
    re_disp = TAILLE * SCALE_EARTH
    x_e = re_disp * np.outer(np.cos(theta_e), np.sin(phi_e))
    y_e = re_disp * np.outer(np.sin(theta_e), np.sin(phi_e))
    z_e = re_disp * np.outer(np.ones(len(theta_e)), np.cos(phi_e))
    fig.add_trace(go.Surface(x=x_e, y=y_e, z=z_e, colorscale=[[0, 'royalblue'], [1, 'steelblue']], showscale=False, name='Earth', hoverinfo='skip'))

    # Orbite Chaser
    t_circle = np.linspace(0, 2 * np.pi, 150)
    c_ox = chaser_r_norm * np.cos(t_circle) * U_chaser[0] + chaser_r_norm * np.sin(t_circle) * V_chaser[0]
    c_oy = chaser_r_norm * np.cos(t_circle) * U_chaser[1] + chaser_r_norm * np.sin(t_circle) * V_chaser[1]
    c_oz = chaser_r_norm * np.cos(t_circle) * U_chaser[2] + chaser_r_norm * np.sin(t_circle) * V_chaser[2]
    fig.add_trace(go.Scatter3d(x=c_ox, y=c_oy, z=c_oz, mode='lines', line=dict(color='red', width=3, dash='dash'), opacity=0.5, name='Chaser Orbit', hoverinfo='skip'))

    # --- Débris ---
    debris_data = []
    for deb_info in my_cluster[1:]:
        r_km = R_EARTH + deb_info['h']
        r_norm = km_to_norm(r_km)
        n_deb = np.sqrt(MU / r_km**3)

        U = np.array([np.cos(deb_info['o']), np.sin(deb_info['o']), 0.0])
        W = np.array([np.sin(deb_info['i']) * np.sin(deb_info['o']), -np.sin(deb_info['i']) * np.cos(deb_info['o']), np.cos(deb_info['i'])])
        V = np.cross(W, U)

        orbit_x = r_norm * np.cos(t_circle) * U[0] + r_norm * np.sin(t_circle) * V[0]
        orbit_y = r_norm * np.cos(t_circle) * U[1] + r_norm * np.sin(t_circle) * V[1]
        orbit_z = r_norm * np.cos(t_circle) * U[2] + r_norm * np.sin(t_circle) * V[2]

        visual_size = {'Small': 5, 'Medium': 8, 'Large': 12}.get(deb_info['size_category'], 8)
        color = f"hsl({(deb_info['id'] * 137) % 360}, 80%, 65%)"

        debris_data.append({'id': deb_info['id'], 'r_norm': r_norm, 'r_km': r_km, 'anomaly': deb_info['anom'], 'n': n_deb, 'U': U, 'V': V, 'W': W, 'inc': deb_info['i'], 'omega': deb_info['o'], 'color': color})

        fig.add_trace(go.Scatter3d(x=orbit_x, y=orbit_y, z=orbit_z, mode='lines', line=dict(color=color, width=1), opacity=0.5, showlegend=False, hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=visual_size, color=color, line=dict(width=1, color='white')), showlegend=False, name=f'Debris {deb_info["id"]}'))

    # Position du Chaser (initialisée à 0, mise à jour plus bas)
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='white')), name='Chaser'))

    # --- Physique du Transfert ---
    target = next((d for d in debris_data if d['id'] == target_debris_index), debris_data[0])
    h_target = target['r_km'] - R_EARTH

    t_wait, t_trans, a_trans = solve_phasing(
        h_chaser, i_chaser, o_chaser, anom_chaser,
        h_target, target['inc'], target['omega'], target['anomaly']
    )

    n_chaser_val = np.sqrt(MU / r1_km**3)
    anom_dep = anom_chaser + n_chaser_val * t_wait
    anom_arr = target['anomaly'] + np.sqrt(MU / (R_EARTH + h_target)**3) * (t_wait + t_trans)

    r_dep, v_chaser_pre = get_state_vectors(r1_km, i_chaser, o_chaser, anom_dep)
    r_arr, v_target_pre = get_state_vectors(R_EARTH + h_target, target['inc'], target['omega'], anom_arr)

    # Note : h_unit donne maintenant le VRAI plan de transfert 3D (bissecteur)
    dv1, dv2, total_dv, h_unit = compute_dv_vectors(r_dep, r_arr, v_chaser_pre, v_target_pre, a_trans)
    m_final, fuel_used = tsiolkovsky(total_dv)

    # --- Construction de l'arc visuel dans le vrai plan 3D ---
    r_dep_norm = km_to_norm(np.linalg.norm(r_dep))
    r_arr_norm = km_to_norm(np.linalg.norm(r_arr))
    e_trans = abs(r_arr_norm - r_dep_norm) / (r_dep_norm + r_arr_norm)

    dep_pt = r_dep_norm * (r_dep / np.linalg.norm(r_dep))
    arr_pt = r_arr_norm * (r_arr / np.linalg.norm(r_arr))
    
    # [APPROCHE PHYSIQUE PURE] : Aucune triche sur l'anomalie du débris.
    # On calcule le plan géométrique UNIQUE qui connecte r_dep et r_arr.
    h_arc = np.cross(r_dep, r_arr)
    # GARANTIE : L'arc doit tourner dans le même sens que l'orbite (pas de marche arrière !)
    if np.dot(h_arc, h_unit) < 0:
        h_arc = -h_arc
        
    norm_h_arc = np.linalg.norm(h_arc)
    if norm_h_arc < 1e-10:
        h_arc = h_unit  # Retour au bissecteur si parfaitement alignés (coplanaires)
    else:
        h_arc = h_arc / norm_h_arc

    P_arc = dep_pt / np.linalg.norm(dep_pt)
    Q_arc = np.cross(h_arc, P_arc)
    Q_arc = Q_arc / np.linalg.norm(Q_arc)

    # Calcul de l'angle physique d'ouverture entre les deux points 
    # (très proche de 180°, ex: 179.5°)
    arr_cos = np.dot(arr_pt / r_arr_norm, P_arc)
    arr_sin = np.dot(arr_pt / r_arr_norm, Q_arc)
    arc_angle = np.arctan2(arr_sin, arr_cos)
    if arc_angle < 0:
        arc_angle += 2 * np.pi

    arc_pts = []
    nu_arc = np.linspace(0, arc_angle, 80)
    for nu in nu_arc:
        # On mappe l'ellipse de Hohmann classique (0 -> pi) sur 
        # l'angle de transfert visuel 3D exact (0 -> arc_angle)
        true_anom_trans = (nu / arc_angle) * np.pi
        r_nu_km = a_trans * (1 - e_trans**2) / (1 + e_trans * np.cos(true_anom_trans))
        r_nu_norm = km_to_norm(r_nu_km)
        pt = r_nu_norm * (np.cos(nu) * P_arc + np.sin(nu) * Q_arc)
        arc_pts.append(pt)

    arc_pts = np.array(arc_pts)
    burn2_pt = arc_pts[-1]

    fig.add_trace(go.Scatter3d(x=arc_pts[:, 0], y=arc_pts[:, 1], z=arc_pts[:, 2], mode='lines', line=dict(color='yellow', width=5, dash='dot'), name=f'Hohmann 3D Transfer', visible='legendonly'))
    fig.add_trace(go.Scatter3d(x=[dep_pt[0]], y=[dep_pt[1]], z=[dep_pt[2]], mode='markers', marker=dict(size=6, color='#FF6A00', symbol='diamond'), name='Burn 1'))
    fig.add_trace(go.Scatter3d(x=[burn2_pt[0]], y=[burn2_pt[1]], z=[burn2_pt[2]], mode='markers', marker=dict(size=6, color='#00FFB2', symbol='diamond'), name='Burn 2'))

    # --- Rapport ---
    print(f"\n==================================================")
    print(f" MISSION : INTERCEPT DEBRIS {target_debris_index}")
    print(f"==================================================")
    print(f"  ΔV1 (Départ)     : {dv1 * 1000:.2f} m/s")
    print(f"  ΔV2 (Rendez-vous): {dv2 * 1000:.2f} m/s")
    print(f"  ΔV total         : {total_dv * 1000:.2f} m/s")
    print(f"  Carburant utilisé: {fuel_used:.2f} kg")
    print(f"==================================================\n")

    # --- Animation ---
    MAX_WAIT_SHOWN = 2 * 3600.0
    t_wait_sim = min(t_wait, MAX_WAIT_SHOWN)
    t_skip = t_wait - t_wait_sim
    
    num_frames = 300
    total_sim_time = t_wait_sim + t_trans + 3600.0
    dt = total_sim_time / num_frames

    # Mise à jour des positions de départ pour le rendu (après le "skip" d'attente longue)
    anom_chaser_at_skip = anom_chaser + n_chaser_val * t_skip
    for tr in fig.data:
        if getattr(tr, 'name', '') == 'Chaser':
            tr.x = (chaser_r_norm * np.cos(anom_chaser_at_skip) * U_chaser[0] + chaser_r_norm * np.sin(anom_chaser_at_skip) * V_chaser[0],)
            tr.y = (chaser_r_norm * np.cos(anom_chaser_at_skip) * U_chaser[1] + chaser_r_norm * np.sin(anom_chaser_at_skip) * V_chaser[1],)
            tr.z = (chaser_r_norm * np.cos(anom_chaser_at_skip) * U_chaser[2] + chaser_r_norm * np.sin(anom_chaser_at_skip) * V_chaser[2],)
            
    for deb in debris_data:
        a_deb_skip = deb['anomaly'] + deb['n'] * t_skip
        for tr in fig.data:
            if getattr(tr, 'name', '') == f'Debris {deb["id"]}':
                tr.x = (deb['r_norm'] * np.cos(a_deb_skip) * deb['U'][0] + deb['r_norm'] * np.sin(a_deb_skip) * deb['V'][0],)
                tr.y = (deb['r_norm'] * np.cos(a_deb_skip) * deb['U'][1] + deb['r_norm'] * np.sin(a_deb_skip) * deb['V'][1],)
                tr.z = (deb['r_norm'] * np.cos(a_deb_skip) * deb['U'][2] + deb['r_norm'] * np.sin(a_deb_skip) * deb['V'][2],)

    idx_chaser_trace = next((i for i, tr in enumerate(fig.data) if getattr(tr, 'name', '') == 'Chaser'), -1)
    idx_debris_traces = [idx for idx, tr in enumerate(fig.data) if (tr.name or '').startswith('Debris ')]
    marker_traces = ([idx_chaser_trace] if idx_chaser_trace >= 0 else []) + idx_debris_traces

    frames = []
    for step in range(num_frames):
        t_sim_logical = step * dt
        t_real = t_skip + t_sim_logical
        frame_data = []

        if t_sim_logical <= t_wait_sim:
            new_anom = anom_chaser + n_chaser_val * t_real
            cx = chaser_r_norm * np.cos(new_anom) * U_chaser[0] + chaser_r_norm * np.sin(new_anom) * V_chaser[0]
            cy = chaser_r_norm * np.cos(new_anom) * U_chaser[1] + chaser_r_norm * np.sin(new_anom) * V_chaser[1]
            cz = chaser_r_norm * np.cos(new_anom) * U_chaser[2] + chaser_r_norm * np.sin(new_anom) * V_chaser[2]
        elif t_sim_logical <= t_wait_sim + t_trans:
            idx = min(int(((t_sim_logical - t_wait_sim) / t_trans) * len(arc_pts)), len(arc_pts) - 1)
            cx, cy, cz = arc_pts[idx]
        else:
            tsa = t_real - (t_wait + t_trans)
            # Phase 3 : Chaser traque le débris naturellement
            a_rv = anom_arr + target['n'] * tsa
            cx = target['r_norm'] * np.cos(a_rv) * target['U'][0] + target['r_norm'] * np.sin(a_rv) * target['V'][0]
            cy = target['r_norm'] * np.cos(a_rv) * target['U'][1] + target['r_norm'] * np.sin(a_rv) * target['V'][1]
            cz = target['r_norm'] * np.cos(a_rv) * target['U'][2] + target['r_norm'] * np.sin(a_rv) * target['V'][2]

        frame_data.append(go.Scatter3d(x=[cx], y=[cy], z=[cz]))

        for deb in debris_data:
            a_deb = deb['anomaly'] + deb['n'] * t_real
            dx = deb['r_norm'] * np.cos(a_deb) * deb['U'][0] + deb['r_norm'] * np.sin(a_deb) * deb['V'][0]
            dy = deb['r_norm'] * np.cos(a_deb) * deb['U'][1] + deb['r_norm'] * np.sin(a_deb) * deb['V'][1]
            dz = deb['r_norm'] * np.cos(a_deb) * deb['U'][2] + deb['r_norm'] * np.sin(a_deb) * deb['V'][2]
            frame_data.append(go.Scatter3d(x=[dx], y=[dy], z=[dz]))

        frames.append(go.Frame(data=frame_data, traces=marker_traces, name=str(step)))

    fig.frames = frames

    # --- Layout ---
    max_r = km_to_norm(R_EARTH + 2000.0) * 1.1
    vis_with_arc, vis_without_arc = [True] * len(fig.data), [True] * len(fig.data)
    for idx, tr in enumerate(fig.data):
        if (tr.name or '').startswith('Hohmann'): vis_without_arc[idx] = False

    fig.update_layout(
        title=dict(text="3D Rendez-Vous — Physique 3D Exacte", font=dict(color='white', size=22), x=0.5, y=0.95),
        paper_bgcolor='black',
        scene=dict(xaxis=dict(visible=False, range=[-max_r, max_r]), yaxis=dict(visible=False, range=[-max_r, max_r]), zaxis=dict(visible=False, range=[-max_r, max_r]), bgcolor='black', aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=0), font=dict(color='white'),
        updatemenus=[
            dict(type="buttons", showactive=True, bgcolor='white', font=dict(color='black'), y=0.10, x=0.10, buttons=[
                dict(label="► Play / ❚❚ Pause", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)], args2=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]),
            dict(type="buttons", showactive=True, bgcolor='white', font=dict(color='black'), y=0.20, x=0.10, buttons=[
                dict(label="Afficher l'arc", method="restyle", args=[{"visible": vis_with_arc}], args2=[{"visible": vis_without_arc}])
            ])
        ]
    )

    fig.write_html("mission_rendez_vous.html", auto_open=True)
    print("Fichier interactif mission_rendez_vous.html généré !")

if __name__ == "__main__":
    print("Choisissez le mode de simulation :")
    print("1. Mode RÉEL (Amas SSO groupé)")
    print("2. Mode RANDOM (Physique - 2 orbites éloignées)")
    
    choix = input("Votre choix (1 ou 2) : ")
    
    if choix == "2":
        plot_mission(mode='random', target_debris_index=1)
    else:
        plot_mission(num_debris=3, target_debris_index=1, mode='real')