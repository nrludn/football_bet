import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
from datafc.sofascore import (
    match_data,
    match_odds_data,
    match_stats_data,
    momentum_data,
    lineups_data,
    coordinates_data,
    substitutions_data,
    goal_networks_data,
    shots_data,
    standings_data
)


# --- Veri Yükleme ---
@st.cache_data(show_spinner="Veri yükleniyor...")
def load_data(data_version, tournament=None):
    # tournament parametresi ekledik, böylece seçilen lige göre farklı dosyalar yükleyebiliriz
    
    # Eğer La Liga seçildiyse match_results_spa.json dosyasını kullan
    if tournament == "LaLiga":
        json_file_path = "match_results_spa.json"
        new_data = pd.read_json(json_file_path, orient="records")
        new_data['home_score'] = pd.to_numeric(new_data['home_score'], errors='coerce')
        new_data['away_score'] = pd.to_numeric(new_data['away_score'], errors='coerce')
        new_data = new_data.dropna(subset=['home_score', 'away_score'])
        
        if data_version == "2024-2025 verilerini kullan":
            # Eğer eski veriler de eklenecekse burada ekle (şu an yok)
            return new_data
        else:
            return new_data
    
    # Super Lig için mevcut kodu kullan    
    else:
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
    
    # Her maç için takım indekslerini ve skorları NumPy dizilerine aktaralım.
    home_idx = dataset['home_team'].map(team_to_idx).to_numpy(dtype=np.int32)
    away_idx = dataset['away_team'].map(team_to_idx).to_numpy(dtype=np.int32)
    home_scores = dataset['home_score'].to_numpy(dtype=np.int32)
    away_scores = dataset['away_score'].to_numpy(dtype=np.int32)
    
    return teams, team_to_idx, home_idx, away_idx, home_scores, away_scores

# --- Model Fonksiyonları ---

# Klasik rho düzeltme fonksiyonu (tek değerler için)
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

# Vektörleştirilmiş rho düzeltme fonksiyonu: 
def rho_correction_vec(hs, as_, lam, mu, rho):
    # hs ve as_ dizileri skorları içeriyor.
    # lam ve mu, vektör şeklinde lambda ve mu değerleridir.
    corr = np.ones_like(hs, dtype=np.float64)
    
    # 0-0 için:
    mask = (hs == 0) & (as_ == 0)
    corr[mask] = np.maximum(1 - lam[mask] * mu[mask] * rho, 1e-10)
    # 0-1 için:
    mask = (hs == 0) & (as_ == 1)
    corr[mask] = 1 + lam[mask] * rho
    # 1-0 için:
    mask = (hs == 1) & (as_ == 0)
    corr[mask] = 1 + mu[mask] * rho
    # 1-1 için:
    mask = (hs == 1) & (as_ == 1)
    corr[mask] = np.maximum(1 - rho, 1e-10)
    return corr

# Vektörleştirilmiş log-likelihood fonksiyonu
def vectorized_log_likelihood(params, teams, home_idx, away_idx, home_scores, away_scores):
    n_teams = len(teams)
    # Parametreleri ayıralım:
    attack = params[:n_teams]
    defence = params[n_teams:2*n_teams]
    rho, gamma = params[-2:]
    # Ev sahibi avantajı: gamma (home_adv)
    home_adv = gamma

    # Her maç için lambda ve mu hesapla:
    lam = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
    mu = np.exp(attack[away_idx] + defence[home_idx])
    
    # Poisson PMF hesaplamalarını vektörleştiriyoruz:
    log_pmf_home = np.log(np.maximum(poisson.pmf(home_scores, lam), 1e-10))
    log_pmf_away = np.log(np.maximum(poisson.pmf(away_scores, mu), 1e-10))
    
    # Rho düzeltmesini vektörleştir:
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
    param_dict = {}
    for i, team in enumerate(teams):
        param_dict[f"attack_{team}"] = opt_output.x[i]
    for i, team in enumerate(teams):
        param_dict[f"defence_{team}"] = opt_output.x[n_teams + i]
    param_dict["rho"] = opt_output.x[-2]
    param_dict["home_adv"] = opt_output.x[-1]
    
    return param_dict

def dixon_coles_simulate_match(params_dict, home_team, away_team, max_goals=10):
    # Hesaplamaları basit tutuyoruz.
    lam = np.exp(params_dict[f"attack_{home_team}"] + params_dict[f"defence_{away_team}"] + params_dict["home_adv"])
    mu = np.exp(params_dict[f"defence_{home_team}"] + params_dict[f"attack_{away_team}"])
    team_pred = [
        [poisson.pmf(i, lam) for i in range(max_goals + 1)],
        [poisson.pmf(i, mu) for i in range(max_goals + 1)]
    ]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    
    # Sadece 0-1 skorlar için düzeltme uygulayalım:
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


# --- Ana Uygulama Düzeni ---
# Sol tarafta veri seti seçimi ve takım seçimlerini koyalım ve daha fazla boşluk bırakalım
st.set_page_config(layout="wide")  # Geniş ekran düzeni kullanıyoruz


# Ana düzen için daha dengeli kolonlar oluşturuyoruz
left_col, right_col = st.columns([1, 3])  # Sol tarafa daha az, sağ tarafa daha fazla alan

with left_col:
    
    st.header("Veri Seti Seçimi")
    data_version = st.radio(
        label="Hangi veri setini kullanmak istersiniz?",
        options=["2025 verilerini kullan", "2024-2025 verilerini kullan"],
        index=0
    )

    # Lig seçimi ekleyelim - Burada hem "Super Lig" hem de "LaLiga" seçeneği olsun
    st.header("Lig Seçimi")
    
    # Session state'te lig seçimini saklayalım
    if 'selected_tournament' not in st.session_state:
        # İlk kez yüklendiğinde varsayılan olarak Super Lig seçilsin
        st.session_state.selected_tournament = "Super Lig"
    
    selected_tournament = st.selectbox(
        "Lig seçiniz:", 
        ["Super Lig", "LaLiga"],
        index=0 if st.session_state.selected_tournament == "Super Lig" else 1,
        key="tournament_select"
    )
    
    # Seçilen ligi session_state'e kaydedelim
    st.session_state.selected_tournament = selected_tournament
    
    # Seçilen lige göre veriyi yükleyelim
    data_temp = load_data(data_version, tournament=selected_tournament)
    
    # Takım listesini filtreleyelim
    teams_list = sorted(list(set(data_temp['home_team'].unique()) | set(data_temp['away_team'].unique())))

    # Takım seçimlerini session state'te saklayalım
    if 'home_team' not in st.session_state or st.session_state.home_team not in teams_list:
        # İlk kez yüklendiğinde veya seçili takım listede artık yoksa
        # Varsa, ilk takımı seçelim
        if teams_list:
            st.session_state.home_team = teams_list[0]
    
    if 'away_team' not in st.session_state or st.session_state.away_team not in teams_list:
        # İlk kez yüklendiğinde veya seçili takım listede artık yoksa
        # Varsa ve farklı ise ikinci takımı, eğer sadece bir takım varsa yine onu seçelim
        if len(teams_list) > 1:
            st.session_state.away_team = teams_list[1]
        elif teams_list:
            st.session_state.away_team = teams_list[0]

    st.header("Takım Seçimi")
    
    # Session state'ten değerleri alarak selectbox'ları oluşturalım
    home_team = st.selectbox(
        "Ev sahibi takım:", 
        teams_list, 
        index=teams_list.index(st.session_state.home_team) if st.session_state.home_team in teams_list else 0,
        key="home_team_select"
    )
    
    # Seçilen ev sahibi takımı session_state'e kaydedelim
    st.session_state.home_team = home_team
    
    # Deplasman takımı için, ev sahibi takımının seçilmemesi için kontrol yapalım
    away_options = [team for team in teams_list if team != home_team]
    
    # Eğer deplasman takımı ev sahibi ile aynıysa, başka bir takım seçelim
    if st.session_state.away_team == home_team and away_options:
        st.session_state.away_team = away_options[0]
    
    away_team = st.selectbox(
        "Deplasman takımı:", 
        teams_list,
        index=teams_list.index(st.session_state.away_team) if st.session_state.away_team in teams_list else (1 if len(teams_list) > 1 else 0),
        key="away_team_select"
    )
    
    # Seçilen deplasman takımını session_state'e kaydedelim
    st.session_state.away_team = away_team
    
    if home_team == away_team:
        st.error("Ev sahibi ve deplasman takımı aynı olamaz!")

    if st.button("Analizi Başlat"):
        if home_team == away_team:
            st.error("Ev sahibi ve deplasman takımı aynı olamaz!")
        else:
            with st.spinner("Analiz başlatıldı, lütfen bekleyin..."):
                # Lig bazında filtrelenmiş veriyi kullanalım
                params = solve_parameters(data_temp)
                
                # Analiz sonuçlarını right_col'de göstereceğiz
                if f"attack_{home_team}" in params and f"attack_{away_team}" in params:
                    # Bu değişkeni analysis_completed olarak ayarlayalım
                    st.session_state.analysis_completed = True
                    st.session_state.sim_matrix = dixon_coles_simulate_match(params, home_team, away_team)
                    st.session_state.home_team = home_team
                    st.session_state.away_team = away_team
                    st.session_state.data = data_temp
                    st.session_state.params = params

# Sağ taraf için analiz sonuçları
with right_col:
    if 'analysis_completed' in st.session_state and st.session_state.analysis_completed:
        sim_matrix = st.session_state.sim_matrix
        home_team = st.session_state.home_team
        away_team = st.session_state.away_team
        data = st.session_state.data
        params = st.session_state.params
        
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
        score_df['probability_pct'] = score_df['probability'] * 100
        
        # Toplam olasılık %80'e ulaşana kadar olan skorları seçelim
        score_df['cumulative_prob'] = score_df['probability'].cumsum()
        scores_80pct = score_df[score_df['cumulative_prob'] <= 0.8].copy()
        # Eğer hiçbir satır seçilmediyse (ilk satır bile %80'den büyükse), en azından ilk satırı alalım
        if len(scores_80pct) == 0:
            scores_80pct = score_df.head(1).copy()
        # Olasılıklara göre yüksekten düşüğe sıralayalım (grafikte düşükten yükseğe göstereceğiz)
        scores_80pct = scores_80pct.sort_values('probability_pct', ascending=True)
        
        # Tab'lar oluşturalım
        bar_tab, heatmap_tab, standing_tab = st.tabs(["Skor Olasılıkları", "Skor Dağılım Heatmap", "Lig Sıralaması"])
        
        # Bar chart - Bar tab'ında gösteriyoruz
        with bar_tab:
            # Bar chart - daha küçük boyutta gösteriyoruz
            fig, ax = plt.subplots(figsize=(10, 7))  # Boyutu küçülttük (eski boyut: 12,9)
            
            # Mavi tonlarında barlar oluşturalım
            bars = ax.barh(scores_80pct['score_label'], scores_80pct['probability_pct'], 
                   color=plt.cm.Blues(np.linspace(0.4, 0.9, len(scores_80pct))))
            
            # Bar içinde değerleri gösterelim
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width - width * 0.1  # Bar içinde sağ tarafa yakın
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                        f'%{scores_80pct["probability_pct"].iloc[i]:.1f}', 
                        va='center', ha='right', color='white', fontweight='bold', fontsize=9)
            
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_xlabel('Olasılık (%)', fontsize=12)  # Font boyutu artırıldı
            ax.set_ylabel('Gol Kombinasyonları', fontsize=12)  # Font boyutu artırıldı
            
            # Başlıkta toplam olasılığın yüzde kaçını gösterdiğimizi belirtelim
            total_prob_shown = scores_80pct['probability'].sum() * 100
            ax.set_title(f'{home_team} vs {away_team} - {selected_tournament} - Skor Olasılıkları (Toplam: %{total_prob_shown:.1f})', fontsize=14)
            
            ax.set_xlim(0, scores_80pct['probability_pct'].max() * 1.1)
            ax.tick_params(axis='both', which='major', labelsize=11)  # Eksen etiketleri boyutu artırıldı
            
            st.pyplot(fig)
            
            # --- Ek Tahmin Gruplamaları Hesaplama ---
            # Simülasyon matrisinden toplam gol olasılıklarını hesaplayalım:
            p_under = 0.0
            p_over = 0.0
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    if i + j <= 2:  # 2 veya daha az gol: 2.5 alt
                        p_under += sim_matrix[i, j]
                    else:           # 3 veya daha fazla gol: 2.5 üst
                        p_over += sim_matrix[i, j]

            if p_under > p_over:
                goal_group = "2.5 alt"
                goal_group_prob = p_under * 100  # yüzdeye çevirmek için
            else:
                goal_group = "2.5 üst"
                goal_group_prob = p_over * 100

            # Maç sonucu olasılıklarını hesaplayalım:
            p_home_win = 0.0
            p_draw = 0.0
            p_away_win = 0.0
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    if i > j:
                        p_home_win += sim_matrix[i, j]
                    elif i == j:
                        p_draw += sim_matrix[i, j]
                    else:
                        p_away_win += sim_matrix[i, j]

            # Ev sahibi kaybetme (deplasman kazanma) olasılığı daha yüksekse grup 1,
            # ev sahibi galibiyeti daha yüksekse grup 0,
            # aksi durumda (örneğin beraberlik baskınsa) grup 2 şeklinde gruplandıralım:
            if p_away_win > p_home_win and p_away_win > p_draw:
                result_group = 1
            elif p_home_win > p_away_win and p_home_win > p_draw:
                result_group = 0
            else:
                result_group = 2

            # Sonuçları yazdıralım:
            st.write("### Tahmin Gruplamaları ve Olasılıkları")

            # Daha güzel bir tablo formatında gösterelim
            col1, col2 = st.columns(2)

            # Toplam gol tahmini için güzel bir tablo
            with col1:
                # Toplam gol tablosu
                goal_data = {
                    "Tahmin Türü": ["2.5 Alt", "2.5 Üst"],
                    "Olasılık (%)": [f"{p_under*100:.1f}", f"{p_over*100:.1f}"]
                }
                goal_df = pd.DataFrame(goal_data)
                
                

            # İlerleme çubuğu ile görselleştirelim
            st.markdown("#### Olasılık Dağılımı")
            st.progress(float(p_home_win), text=f"Ev Sahibi Kazanır ({p_home_win*100:.1f}%)")
            st.progress(float(p_draw), text=f"Beraberlik ({p_draw*100:.1f}%)")
            st.progress(float(p_away_win), text=f"Deplasman Kazanır ({p_away_win*100:.1f}%)")
            st.progress(float(p_under), text=f"2.5 Alt ({p_under*100:.1f}%)")
            st.progress(float(p_over), text=f"2.5 Üst ({p_over*100:.1f}%)")
        
        # Heatmap - Heatmap tab'ında gösteriyoruz
        with heatmap_tab:
            # Heatmap için veriyi hazırlayalım (0-6 arası skorlar için)
            max_goals_heatmap = 7
            heatmap_data = np.zeros((max_goals_heatmap, max_goals_heatmap))
            
            for i in range(max_goals_heatmap):
                for j in range(max_goals_heatmap):
                    if i < sim_matrix.shape[0] and j < sim_matrix.shape[1]:
                        heatmap_data[i, j] = sim_matrix[i, j]
            
            # Heatmap oluşturalım - boyutu küçültüyoruz
            fig, ax = plt.subplots(figsize=(6, 4))  # Daha küçük boyut
            sns.heatmap(heatmap_data * 100, annot=True, fmt=".2f", cmap="YlOrRd",
                        xticklabels=range(max_goals_heatmap), 
                        yticklabels=range(max_goals_heatmap),
                        ax=ax, annot_kws={"size": 8})  # Annotation yazı boyutu da küçültüldü
            
            ax.set_title(f"{home_team} vs {away_team} - {selected_tournament} - Skor Olasılıkları (%)")
            ax.set_xlabel('Deplasman Golleri')
            ax.set_ylabel('Ev Sahibi Golleri')
            ax.tick_params(axis='both', which='major', labelsize=7)
            
            # Grafik düzenini iyileştir ve daha kompakt hale getir
            plt.tight_layout()
            
            st.pyplot(fig)
            
        with standing_tab:
            # Lig Tablosu
            st.header(f"{selected_tournament} - Lig Tablosu")
            
            @st.cache_data(ttl=3600)  # 1 saat boyunca cache'leyeceğiz
            def get_standings():
                # API'den canlı verileri çekelim
                # Seçilen lige göre tournament_id ve season_id değerlerini ayarlayalım
                tournament_params = {
                    "Super Lig": {"tournament_id": 52, "season_id": 63814},
                    "LaLiga": {"tournament_id": 8, "season_id": 61643}
                    # Daha fazla lig eklendiğinde buraya eklenebilir
                }
                
                # Seçilen lig için parametreleri alalım, varsayılan olarak Super Lig kullanılır
                default_params = tournament_params["Super Lig"]  # Varsayılan olarak Super Lig
                params = tournament_params.get(selected_tournament, default_params)
                
                standings_df = standings_data(
                    tournament_id=params["tournament_id"],
                    season_id=params["season_id"]
                )
                standings_df = standings_df[standings_df['category']=='Total'].reset_index(drop=True)
                # position zaten 1'den başlıyor, onu doğrudan kullanalım
                standings_df = standings_df[['position', 'team_name','matches','wins','draws','losses','scores_for','scores_against','points']]
                standings_df.columns = ['Sıra', 'Takım','O','G','B','M','A','Y','Puan']
                return standings_df
            
            standings_df = get_standings()
            st.dataframe(
                standings_df, 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Sıra": st.column_config.NumberColumn(format="%d"),
                    "Takım": st.column_config.TextColumn(width="medium"),
                    "O": st.column_config.NumberColumn(format="%d", help="Maç sayısı"),
                    "G": st.column_config.NumberColumn(format="%d", help="Galibiyet"),
                    "B": st.column_config.NumberColumn(format="%d", help="Beraberlik"),
                    "M": st.column_config.NumberColumn(format="%d", help="Mağlubiyet"),
                    "A": st.column_config.NumberColumn(format="%d", help="Attığı gol"),
                    "Y": st.column_config.NumberColumn(format="%d", help="Yediği gol"),
                    "Puan": st.column_config.NumberColumn(format="%d", help="Puan", width="small"),
                }
            )
