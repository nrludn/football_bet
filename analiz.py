import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.stats import poisson
import matplotlib.pyplot as plt

# --- Fonksiyonlar ---
def rho_correction(x, y, lambda_x, mu_y, rho):
    if x == 0 and y == 0:
        return max(1 - lambda_x * mu_y * rho, 1e-10)
    elif x == 0 and y == 1:
        return 1 + lambda_x * rho
    elif x == 1 and y == 0:
        return 1 + mu_y * rho
    elif x == 1 and y == 1:
        return max(1 - rho, 1e-10)
    else:
        return 1.0

def dc_log_like(x, y, alpha_x, beta_x, alpha_y, beta_y, rho, gamma):
    lambda_x = np.exp(alpha_x + beta_y + gamma)
    mu_y = np.exp(alpha_y + beta_x)
    log_lambda_x = np.log(max(poisson.pmf(x, lambda_x), 1e-10))
    log_mu_y = np.log(max(poisson.pmf(y, mu_y), 1e-10))
    return (
        np.log(max(rho_correction(x, y, lambda_x, mu_y, rho), 1e-10)) +
        log_lambda_x +
        log_mu_y
    )

def solve_parameters(dataset, init_vals=None, options={"disp": False, "maxiter": 100}):
    teams = np.sort(
        list(set(dataset["home_team"].unique()) | set(dataset["away_team"].unique()))
    )
    n_teams = len(teams)

    if init_vals is None:
        avg_attack = np.ones(n_teams) * 0.1
        avg_defence = np.zeros(n_teams)
        init_vals = np.concatenate([avg_attack, avg_defence, np.array([0.0, 0.1])])
    
    def estimate_parameters(params):
        attack_coeffs = dict(zip(teams, params[:n_teams]))
        defence_coeffs = dict(zip(teams, params[n_teams:2 * n_teams]))
        rho, gamma = params[-2:]
        log_likelihoods = []
        for row in dataset.itertuples():
            try:
                ll = dc_log_like(
                    int(row.home_score),
                    int(row.away_score),
                    attack_coeffs[row.home_team],
                    defence_coeffs[row.home_team],
                    attack_coeffs[row.away_team],
                    defence_coeffs[row.away_team],
                    rho, gamma
                )
                log_likelihoods.append(ll)
            except Exception as e:
                return np.inf
        return -np.sum(log_likelihoods)
    
    constraints = [{"type": "eq", "fun": lambda x: sum(x[:n_teams]) - n_teams}]
    bounds = Bounds(
        [-3.0] * n_teams + [-3.0] * n_teams + [-0.2, 0],
        [3.0] * n_teams + [3.0] * n_teams + [0.2, 1.0]
    )
    
    opt_output = minimize(
        estimate_parameters, init_vals, method='SLSQP',
        options=options, constraints=constraints, bounds=bounds
    )
    return dict(zip(
        ["attack_" + team for team in teams] +
        ["defence_" + team for team in teams] +
        ["rho", "home_adv"],
        opt_output.x
    ))

def dixon_coles_simulate_match(params_dict, home_team, away_team, max_goals=10):
    def calc_means(param_dict, home_team, away_team):
        return [
            np.exp(param_dict["attack_" + home_team] + param_dict["defence_" + away_team] + param_dict["home_adv"]),
            np.exp(param_dict["defence_" + home_team] + param_dict["attack_" + away_team])
        ]
    
    team_avgs = calc_means(params_dict, home_team, away_team)
    team_pred = [[poisson.pmf(i, team_avg) for i in range(max_goals + 1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([
        [rho_correction(h, a, team_avgs[0], team_avgs[1], params_dict["rho"]) for a in range(2)]
        for h in range(2)
    ])
    output_matrix[:2, :2] *= correction_matrix
    return output_matrix

# Takım istatistiklerini hesaplama fonksiyonu
def compute_team_stats(team, role, dataset):
    if role == "home":
        team_data = dataset[dataset['home_team'] == team]
        avg_goals = team_data['home_score'].mean() if not team_data.empty else 0
        wins = team_data[team_data['home_score'] > team_data['away_score']].shape[0]
        draws = team_data[team_data['home_score'] == team_data['away_score']].shape[0]
        losses = team_data[team_data['home_score'] < team_data['away_score']].shape[0]
    elif role == "away":
        team_data = dataset[dataset['away_team'] == team]
        avg_goals = team_data['away_score'].mean() if not team_data.empty else 0
        wins = team_data[team_data['away_score'] > team_data['home_score']].shape[0]
        draws = team_data[team_data['away_score'] == team_data['home_score']].shape[0]
        losses = team_data[team_data['away_score'] < team_data['home_score']].shape[0]
    return avg_goals, wins, draws, losses

# --- Veri Yükleme ---
@st.cache_data(show_spinner="Veri yükleniyor...")
def load_data(data_version):
    # new_data yükle
    json_file_path = "match_results.json"
    new_data = pd.read_json(json_file_path, orient="records")
    new_data['home_score'] = pd.to_numeric(new_data['home_score'], errors='coerce')
    new_data['away_score'] = pd.to_numeric(new_data['away_score'], errors='coerce')
    new_data = new_data.dropna(subset=['home_score', 'away_score'])
    
    if data_version == "2024-2025 verilerini kullan":
        # old_data yükle
        old_data = pd.read_json("match_results_old.json", orient="records")
        old_data['home_score'] = pd.to_numeric(old_data['home_score'], errors='coerce')
        old_data['away_score'] = pd.to_numeric(old_data['away_score'], errors='coerce')
        old_data = old_data.dropna(subset=['home_score', 'away_score'])
        data = pd.concat([old_data, new_data], ignore_index=True)
        return data
    else:
        return new_data

# --- Sidebar Seçimleri ---
st.sidebar.header("Veri Seti Seçimi")
data_version = st.radio(
    label="Hangi veri setini kullanmak istersiniz?",
    options=["2025 verilerini kullan", "2024-2025 verilerini kullan"],
    index=0
)

# Geçici veri yükleyip takım listesini alıyoruz
data_temp = load_data(data_version)
teams = sorted(list(set(data_temp['home_team'].unique()) | set(data_temp['away_team'].unique())))

st.sidebar.header("Takım Seçimi")
home_team = st.sidebar.selectbox("Ev sahibi takım:", teams)
away_team = st.sidebar.selectbox("Deplasman takımı:", teams)

if home_team == away_team:
    st.sidebar.error("Ev sahibi ve deplasman takımı aynı olamaz!")

# --- Analiz Butonu ---
if st.sidebar.button("Analizi Başlat"):
    with st.spinner("Analiz başlatıldı, lütfen bekleyin..."):
        # Seçime göre veri setini yükle
        data = load_data(data_version)
        params = solve_parameters(data)
        
        if f"attack_{home_team}" in params and f"attack_{away_team}" in params:
            sim_matrix = dixon_coles_simulate_match(params, home_team, away_team)
            
            # Takım istatistiklerini hesapla
            home_stats = compute_team_stats(home_team, "home", data)
            away_stats = compute_team_stats(away_team, "away", data)
            
            # İstatistikleri içeren küçük tablo oluşturma
            stats_data = {
                "Takım": [home_team, away_team],
                "Ortalama Gol": [f"{home_stats[0]:.2f}", f"{away_stats[0]:.2f}"],
                "Galibiyet": [home_stats[1], away_stats[1]],
                "Beraberlik": [home_stats[2], away_stats[2]],
                "Mağlubiyet": [home_stats[3], away_stats[3]]
            }
            stats_df = pd.DataFrame(stats_data)
            
            # İstatistik tablosunu ekranda göster
            st.write("### Takım İstatistikleri")
            st.table(stats_df)
            
            # Skor olasılıklarını DataFrame'e aktar
            score_probs = []
            for i in range(7):
                for j in range(7):
                    if i < sim_matrix.shape[0] and j < sim_matrix.shape[1]:
                        score_probs.append({
                            'home_goals': i,
                            'away_goals': j,
                            'probability': sim_matrix[i, j],
                            'score_label': f"{i} - {j}"
                        })
            score_df = pd.DataFrame(score_probs)
            score_df = score_df.sort_values('probability', ascending=False).reset_index(drop=True)
            top_20 = score_df.head(20).copy()
            top_20['probability_pct'] = top_20['probability'] * 100
            top_20 = top_20.sort_values('probability_pct', ascending=True)
            
            # --- Grafik Oluşturma ---
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.barh(top_20['score_label'], top_20['probability_pct'], 
                   color=plt.cm.Reds(np.linspace(0.3, 0.9, len(top_20))))
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_xlabel('Probability (%)')
            ax.set_ylabel('Goal Combinations')
            ax.set_title(f'{home_team} vs {away_team} - Skor Olasılıkları')
            ax.set_xlim(0, top_20['probability_pct'].max() * 1.1)
            
            st.pyplot(fig)
        else:
            st.error("Seçilen takımlardan biri veri setinde bulunamadı!")
