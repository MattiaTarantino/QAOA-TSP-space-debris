# ==========================================
# MOTEUR DE TRAJECTOIRE POUR QAOA (LAMBERT 3D TENSOR)
# Discrétisation du temps pour un Time-Dependent TSP
# ==========================================

import numpy as np
import pandas as pd
import itertools
from scipy.optimize import root_scalar, minimize_scalar
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. CONSTANTES PHYSIQUES (Standard ESA/NASA)
# ==========================================
MU       = 398600.4418        # km^3/s^2  — Paramètre gravitationnel standard (EGM-96)
R_EARTH  = 6378.137           # km        — Rayon équatorial WGS-84
G0       = 9.80665 / 1000.0   # km/s^2    — Accélération gravitationnelle standard
ISP      = 310.0              # s         — Impulsion spécifique (Green Bi-propellant)
M_INITIAL = 500.0             # kg        — Masse initiale du Chaser

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

# ==========================================
# 4. GÉNÉRATION DES DÉBRIS
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
            'ecc':  0.0,
            'a':    h + R_EARTH
        })
    return cluster

# ==========================================
# 5. CONSTRUCTION DU DATASET LONG FORMAT (CSV)
# ==========================================

def generate_and_export_dataset(cluster, duration_h=4320.0, step_h=0.5, filename="lambert_dataset_6months.csv"):
    """
    Génère les coûts de mission sur une longue durée et exporte au format Tidy (plat).
    La sauvegarde se fait ligne par ligne pour éviter de saturer la mémoire RAM.
    """
    import os
    import time
    
    n = len(cluster)
    t_dep_hours = np.arange(0, duration_h + step_h, step_h)
    total_steps = len(t_dep_hours)
    
    # Création / Ouverture du fichier CSV
    print(f"Création du fichier {filename}...")
    with open(filename, 'w') as f:
        # Écriture de l'en-tête
        f.write("Time_Dep_h,Debris_Start,Debris_Target,Delta_V1_m_s,Delta_V2_m_s,Travel_Time_h\n")
        
        start_time_global = time.time()
        
        for idx, t_h in enumerate(t_dep_hours):
            t_dep_sec = t_h * 3600.0
            
            # Indication de progression pour le suivi du calcul en console
            if idx % 100 == 0 and idx > 0:
                elapsed = time.time() - start_time_global
                eta = (elapsed / idx) * (total_steps - idx)
                print(f"Progression: {idx}/{total_steps} tranches ({idx/total_steps*100:.1f}%) - ETA: {eta/60:.1f} min")
                
            for start, target in itertools.permutations(range(n), 2):
                deb_s, deb_t = cluster[start], cluster[target]
                r1_km, r2_km = R_EARTH + deb_s['h'], R_EARTH + deb_t['h']
                a1, a2       = deb_s['a'], deb_t['a']
                
                n_s = np.sqrt(MU / a1**3)
                n_t = np.sqrt(MU / a2**3)

                # --- POSITION EXACTE AU MOMENT DU DÉPART (t_dep_sec) ---
                anom_dep = deb_s['anom'] + n_s * t_dep_sec
                r_dep, v_s = get_state_vectors(r1_km, deb_s['i'], deb_s['o'], anom_dep, 0.0, a1)

                def objective(t_trans):
                    anom_arr = deb_t['anom'] + n_t * (t_dep_sec + t_trans)
                    r_arr, v_t = get_state_vectors(r2_km, deb_t['i'], deb_t['o'], anom_arr, 0.0, a2)
                    
                    v1, v2 = solve_lambert(r_dep, r_arr, t_trans)
                    if v1 is None: 
                        return 1e5
                    
                    dv1 = np.linalg.norm(v1 - v_s)
                    dv2 = np.linalg.norm(v_t - v2)
                    return dv1 + dv2

                # OPTIMISATION 1D (Temps de vol entre 30 min et 12h)
                res = minimize_scalar(objective, bounds=(1800, 43200), method='bounded')
                
                if not res.success or res.fun >= 1e5:
                    # En cas d'échec de Lambert (pas de solution viable trouvée)
                    f.write(f"{t_h},{deb_s['id']},{deb_t['id']},X,X,X\n")
                    continue

                t_t_opt = res.x
                
                # Recalcul final propre pour stocker les vraies valeurs distinctes dv1 / dv2
                anom_arr = deb_t['anom'] + n_t * (t_dep_sec + t_t_opt)
                r_arr, v_t = get_state_vectors(r2_km, deb_t['i'], deb_t['o'], anom_arr, 0.0, a2)
                v_trans_dep, v_trans_arr = solve_lambert(r_dep, r_arr, t_t_opt)
                
                dv1 = np.linalg.norm(v_trans_dep - v_s)
                dv2 = np.linalg.norm(v_t - v_trans_arr)

                # Formatage propre (m/s avec 2 décimales, temps de vol en heures avec 3 décimales)
                dv1_ms = round(dv1 * 1000, 2)
                dv2_ms = round(dv2 * 1000, 2)
                t_vol_h = round(t_t_opt / 3600, 3)
                
                # Écriture immédiate de la ligne dans le fichier
                f.write(f"{t_h},{deb_s['id']},{deb_t['id']},{dv1_ms},{dv2_ms},{t_vol_h}\n")
                
    print(f"\n✅ Terminé ! Le dataset de {total_steps} tranches temporelles a été sauvegardé dans : {filename}")


# ==========================================
# 6. ENTRÉE PRINCIPALE
# ==========================================
if __name__ == "__main__":
    print("Génération du cluster de 4 débris...")
    my_cluster = generate_debris_cluster(num_debris=10)
    
    # Paramètres de l'étude (6 MOIS SIMULÉS)
    DUREE_ETUDE_H = 4320.0  # 180 jours * 24 heures = 4320 heures
    PAS_TEMPS_H = 0.5       # Échantillonnage toutes les 30 minutes
    
    FICHIER_SORTIE = "lambert_dataset_6months.csv"
    
    print(f"Objectif : Tenseur 3D via fichier CSV plat (0 à {DUREE_ETUDE_H}h, pas {PAS_TEMPS_H}h).")
    print(f"Cela représente {int((DUREE_ETUDE_H/PAS_TEMPS_H)+1)} tranches horaires à analyser...")
    print("Lancement du calcul... (veuillez patienter, cela prendra plusieurs minutes)")
    
    generate_and_export_dataset(my_cluster, duration_h=DUREE_ETUDE_H, step_h=PAS_TEMPS_H, filename=FICHIER_SORTIE)