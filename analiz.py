import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.stats import poisson
import matplotlib.pyplot as plt

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
        old_data = pd.read_json("match_results_old.json", orient="records")
        old_data['home_score'] = pd.to_numeric(old_data['home_score'], errors='coerce')
        old_data['away_score'] = pd.to_numeric(old_data['away_score'], errors='coerce')
        old_data = old_data.dropna(subset=['home_score', 'away_score'])
        data = pd.concat([old_data, new_data], ignore_index=True)
        return data
    else:
        return new_data

# --- Ön İşleme: Takım indeksleri ve skor dizileri ---
def preprocess_dataset(dataset):
    # Takım isimlerini indekslere çeviriyoruz.
    teams = np.sort(list(set(dataset['home_team'].unique()) | set(dataset['away_team'].unique())))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    
    # Her maç için takım indekslerini ve skorları diziye aktaralım.
    home_idx = dataset['home_team'].map(team_to_idx).to_numpy(dtype=np.int32)
    away_idx = dataset['away_team'].map(team_to_idx).to_numpy(dtype=np.int32)
    home_scores = dataset['home_score'].to_numpy(dtype=np.int32)
    away_scores = dataset['away_score'].to_numpy(dtype=np.int32)
    
    return teams, team_to_idx, home_idx, away_idx, home_scores, away_scores

# --- Model Fonksiyonları ---
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


def vectorized_log_likelihood(params, teams, home_idx, away_idx, home_scores, away_scores):
    n_teams = len(teams)
    # Parametreleri ayıralım:
    attack = params[:n_teams]
    defence = params[n_teams:2*n_teams]
    rho, gamma = params[-2:]
    # Ev sahibi avantajı: gamma (home_adv) direkt gamma
    home_adv = gamma

    # Her maç için lambda ve mu hesapla:
    lam = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
    mu = np.exp(attack[away_idx] + defence[home_idx])
    
    # Poisson PMF hesaplamalarını vektörize ediyoruz:
    # np.maximum ile alt sınır uygulayarak log-likelihood hesaplamalarında 1e-10 koruması sağlıyoruz.
    log_pmf_home = np.log(np.maximum(poisson.pmf(home_scores, lam), 1e-10))
    log_pmf_away = np.log(np.maximum(poisson.pmf(away_scores, mu), 1e-10))
    
    # Rho düzeltmesini vektörize edelim:
    corr = rho_correction_vec(home_scores, away_scores, lam, mu, rho)
    log_corr = np.log(np.maximum(corr, 1e-10))
    
    log_likelihood = log_corr + log_pmf_home + log_pmf_away
    return -np.sum(log_likelihood)

def solve_parameters(dataset, init_vals=None, options={"disp": False, "maxiter": 100}):
    teams, team_to_idx, home_idx, away_idx, home_scores, away_scores = preprocess_dataset(dataset)
    n_teams = len(teams)
    
    if init_vals is None:
        avg_attack = np.ones(n_teams) * 0.1
        avg_defence = np.zeros(n_teams)
        # gamma (home_adv) başlangıç değeri 0.1, rho başlangıç 0.0
        init_vals = np.concatenate([avg_attack, avg_defence, np.array([0.0, 0.1])])
    
    def objective(params):
        return vectorized_log_likelihood(params, teams, home_idx, away_idx, home_scores, away_scores)
    
    # Toplam saldırı katsayıları için constraint: sum(attack) = n_teams
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:n_teams]) - n_teams}]
    bounds = Bounds(
        [-3.0] * n_teams + [-3.0] * n_teams + [-0.2, 0],
        [3.0] * n_teams + [3.0] * n_teams + [0.2, 1.0]
    )
    
    opt_output = minimize(objective, init_vals, method='SLSQP',
                          options=options, constraints=constraints, bounds=bounds)
    
    # Sonuçları sözlük şeklinde döndürüyoruz:
    param_dict = dict()
    for i, team in enumerate(teams):
        param_dict[f"attack_{team}"] = opt_output.x[i]
    for i, team in enumerate(teams):
        param_dict[f"defence_{team}"] = opt_output.x[n_teams + i]
    param_dict["rho"] = opt_output.x[-2]
    param_dict["home_adv"] = opt_output.x[-1]
    
    return param_dict

def dixon_coles_simulate_match(params_dict, home_team, away_team, max_goals=10):
    # Hesaplamaları basit tutuyoruz; vektörleştirme burada da benzer mantıkla yapılabilir.
    lam = np.exp(params_dict[f"attack_{home_team}"] + params_dict[f"defence_{away_team}"] + params_dict["home_adv"])
    mu = np.exp(params_dict[f"defence_{home_team}"] + params_dict[f"attack_{away_team}"])
    team_pred = [[poisson.pmf(i, lam) for i in range(max_goals + 1)],
                 [poisson.pmf(i, mu) for i in range(max_goals + 1)]]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    
    # Sadece 0-1 skorlar için düzeltme
    for i in range(2):
        for j in range(2):
            corr = rho_correction(i, j, lam, mu, params_dict["rho"])
            output_matrix[i, j] *= corr
    return output_matrix

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

# --- Sidebar Seçimleri ---
st.sidebar.header("Veri Seti Seçimi")
data_version = st.radio(
    label="Hangi veri setini kullanmak istersiniz?",
    options=["2025 verilerini kullan", "2024-2025 verilerini kullan"],
    index=0
)

data_temp = load_data(data_version)
teams_list = sorted(list(set(data_temp['home_team'].unique()) | set(data_temp['away_team'].unique())))

st.sidebar.header("Takım Seçimi")
home_team = st.sidebar.selectbox("Ev sahibi takım:", teams_list)
away_team = st.sidebar.selectbox("Deplasman takımı:", teams_list)
if home_team == away_team:
    st.sidebar.error("Ev sahibi ve deplasman takımı aynı olamaz!")

# --- Analiz Butonu ---
if st.sidebar.button("Analizi Başlat"):
    with st.spinner("Analiz başlatıldı, lütfen bekleyin..."):
        data = load_data(data_version)
        params = solve_parameters(data)
        
        if f"attack_{home_team}" in params and f"attack_{away_team}" in params:
            sim_matrix = dixon_coles_simulate_match(params, home_team, away_team)
            
            # Takım istatistiklerini hesapla
            home_stats = compute_team_stats(home_team, "home", data)
            away_stats = compute_team_stats(away_team, "away", data)
            
            stats_data = {
                "Takım": [home_team, away_team],
                "Ortalama Gol": [f"{home_stats[0]:.2f}", f"{away_stats[0]:.2f}"],
                "Galibiyet": [home_stats[1], away_stats[1]],
                "Beraberlik": [home_stats[2], away_stats[2]],
                "Mağlubiyet": [home_stats[3], away_stats[3]]
            }
            stats_df = pd.DataFrame(stats_data)
            st.write("### Takım İstatistikleri")
            st.table(stats_df)
            
            # Skor olasılıklarını hesapla
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
            
            # Grafik Oluşturma
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
