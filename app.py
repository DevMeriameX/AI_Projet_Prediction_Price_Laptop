import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Laptop Price Predictor Pro", 
    page_icon="💻", 
    layout="wide"
)

# --- STYLE CSS POUR LE LOOK FASCINANT ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { 
        width: 100%; 
        background-color: #1e3a8a; 
        color: white; 
        border-radius: 10px;
        height: 3.5em;
        font-weight: bold;
        font-size: 1.2em;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #3b82f6; border: none; }
    .price-box { 
        padding: 40px; 
        border-radius: 20px; 
        background-color: #ffffff; 
        border-top: 12px solid #1e3a8a;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CHARGEMENT DES DONNÉES ET MODÈLES
@st.cache_resource
def load_assets():
    df = pickle.load(open('df.pkl', 'rb'))
    models = {
        'XGBoost': pickle.load(open('xgb_model.pkl', 'rb')),
        'Random Forest': pickle.load(open('rf_model.pkl', 'rb')),
        'Linear Regression': pickle.load(open('linear_model.pkl', 'rb')),
        'Polynomial': pickle.load(open('poly_model.pkl', 'rb'))
    }
    return df, models

df, models_dict = load_assets()

# 3. BARRE LATÉRALE (SIDEBAR)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/428/428001.png", width=100)
st.sidebar.title("⚙️ Configuration IA")

selected_model_name = st.sidebar.selectbox(
    "🤖 Choisir l'algorithme", 
    ['XGBoost', 'Random Forest', 'Linear Regression', 'Polynomial']
)
model = models_dict[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Modèle actif :** {selected_model_name}
""")

# 4. INTERFACE PRINCIPALE
st.title("💻 Laptop Price Predictor Pro")
st.markdown("##### Anticipez le prix du marché grâce au Machine Learning")

# Utilisation de colonnes pour l'organisation
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🛠️ Composants Système")
    company = st.selectbox(' Marque', df['Company'].unique())
    type_name = st.selectbox(' Type de Laptop', df['TypeName'].unique())
    ram = st.select_slider(' Mémoire RAM (GB)', options=sorted(df['RAM (GB)'].unique()), value=8)
    weight = st.number_input(' Poids (kg)', value=2.0, step=0.1)
    cpu_freq = st.number_input(' Fréquence CPU (GHz)', value=2.5, step=0.1)
    os = st.selectbox(' Système d\'exploitation', df['os'].unique())

with col2:
    st.subheader("🖥️ Affichage & Stockage")
    cpu_brand = st.selectbox(' Gamme Processeur', df['Cpu brand'].unique())
    gpu_brand = st.selectbox(' Marque GPU', df['GPU_Company'].unique())
    
    # Zone Écran
    st.write("**Détails Écran :**")
    c_res1, c_res2 = st.columns(2)
    with c_res1:
        touchscreen = st.radio(' Tactile', ['Non', 'Oui'], horizontal=True)
        ips = st.radio(' Dalle IPS', ['Non', 'Oui'], horizontal=True)
    with c_res2:
        screen_size = st.number_input(' Taille (Pouces)', value=15.6)
        resolution = st.selectbox(' Résolution', ['1920x1080','1366x768','1600x900','3840x2160','2560x1600'])
    
    # Stockage Dynamique
    st.write("**Capacité Stockage :**")
    c_st1, c_st2 = st.columns(2)
    with c_st1:
        hdd = st.selectbox(' HDD (GB)', sorted(df['HDD'].unique()))
    with c_st2:
        ssd = st.selectbox(' SSD (GB)', sorted(df['SSD'].unique()))

# 5. RÉSUMÉ AVANT PRÉDICTION
st.markdown("---")
st.markdown("###  Récapitulatif de la configuration")
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("CPU", f"{cpu_freq} GHz")
with m2: st.metric("RAM", f"{ram} GB")
with m3: st.metric("Stockage", f"{hdd + ssd} GB")
with m4: st.metric("Poids", f"{weight} kg")

# 6. LOGIQUE DE PRÉDICTION
if st.button(' CALCULER L\'ESTIMATION'):
    try:
        # A. Feature Engineering (PPI)
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        calculated_ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
        
        t_screen = 1 if touchscreen == 'Oui' else 0
        ips_val = 1 if ips == 'Oui' else 0

        # B. Création du DataFrame (Ordre exact de ton dataset)
        query_df = pd.DataFrame([{
            'Company': company,
            'TypeName': type_name,
            'CPU_Frequency (GHz)': cpu_freq,
            'RAM (GB)': ram,
            'GPU_Company': gpu_brand,
            'Weight (kg)': weight,
            'Touchscreen': t_screen,
            'Ips': ips_val,
            'ppi': calculated_ppi,
            'Cpu brand': cpu_brand,
            'HDD': hdd,
            'SSD': ssd,
            'os': os
        }])

        # C. Prédiction et inversion Log
        prediction_log = model.predict(query_df)[0]
        prediction = np.exp(prediction_log)

        # D. Affichage Fascinnant
        st.markdown(f"""
            <div class="price-box">
                <h2 style='color: #1e3a8a; margin-bottom: 10px;'>Prix de Vente Estimé</h2>
                <h1 style='color: #10b981; font-size: 4em;'>{int(prediction)} €</h1>
                <p style='color: #6b7280;'>Estimation réalisée via l'algorithme <b>{selected_model_name}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Feedback visuel
        if prediction > 1500:
            st.warning("🏷️ **Note :** Ce laptop appartient au segment **Premium**.")
        st.balloons()
    
    except Exception as e:
        st.error(f" Erreur lors du calcul : {e}")
        st.info("💡 Vérifiez que le fichier df.pkl et les modèles sont bien présents dans le dossier.")

# FOOTER
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Application Prédiction Laptop - Soutenance Master 2026</p>", unsafe_allow_html=True)