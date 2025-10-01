import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import datetime
import json

# Configuration de la page
st.set_page_config(
    page_title="🛰️ EuroSAT - Satellite Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-family: 'Orbitron', monospace;
        background: linear-gradient(45deg, #06B6D4, #3B82F6, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #0B1426 0%, #1E3A8A 50%, #7C3AED 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        color: white;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .upload-area {
        border: 2px dashed #3B82F6;
        border-radius: 1rem;
        padding: 3rem;
        text-align: center;
        background: rgba(59, 130, 246, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: #06B6D4;
    }
    
    .class-card {
        border-left: 4px solid;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Configuration des chemins
MODEL_PATH = "resnet50_fast_model.h5"
INPUT_SIZE = (128, 128)

# Définitions des classes
CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

CLASS_DESCRIPTIONS = {
    'AnnualCrop': '🌱 Cultures annuelles : Plantes cultivées et récoltées chaque année (blé, maïs, orge, riz, tournesol). Cycle de vie court.',
    'Forest': "🌳 Forêts : Zones couvertes d'arbres et de végétation dense, naturelles ou plantées. Important pour biodiversité et climat.",
    'HerbaceousVegetation': '🌿 Végétation herbacée : Plantes non ligneuses (herbes, buissons bas, plantes sauvages). Présentes dans prairies ou landes.',
    'Highway': '🛣 Routes et autoroutes : Surfaces asphaltées pour le transport routier (autoroutes, routes principales, carrefours).',
    'Industrial': '🏭 Zones industrielles : Espaces dédiés aux usines, entrepôts, centrales, zones de stockage. Grandes surfaces construites.',
    'Pasture': '🐄 Pâturages : Zones herbeuses utilisées pour nourrir le bétail (vaches, moutons, chevaux). Finalité agricole.',
    'PermanentCrop': '🍇 Cultures permanentes : Plantes cultivées qui repoussent chaque année sans replantation (vignes, vergers, oliviers, caféiers).',
    'Residential': "🏠 Zones résidentielles : Quartiers d'habitations humaines (maisons, immeubles, lotissements).",
    'River': "🌊 Rivières et cours d'eau : Étendues d'eau coulante (rivières, fleuves, canaux). Formes allongées et sinueuses.",
    'SeaLake': "🌊 Mers et lacs : Étendues d'eau statiques (mers, océans, lacs, réservoirs). Grandes surfaces d'eau immobiles."
}

# Fonctions de base de données
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1234567890",
            database="dbtkinter_app",
            autocommit=True
        )
        return connection
    except Error as e:
        st.error(f"Erreur de connexion à la base de données: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    image_name VARCHAR(255) NOT NULL,
                    predicted_class VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
            if cursor.fetchone()[0] == 0:
                hashed_password = generate_password_hash('admin')
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (%s, %s)",
                    ('admin', hashed_password)
                )
                st.success("Utilisateur admin créé avec le mot de passe 'admin'")
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Error as e:
            st.error(f"Erreur lors de l'initialisation de la base de données: {e}")

# Fonctions d'authentification
def register_user(username, email, password):
    conn = get_db_connection()
    if not conn:
        return False, "Erreur de connexion à la base de données"
    
    try:
        cursor = conn.cursor()
        
        # Vérifier si l'utilisateur existe déjà
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return False, "Nom d'utilisateur déjà utilisé"
        
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return False, "Adresse email déjà utilisée"
        
        # Créer l'utilisateur
        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed_password)
        )
        
        conn.commit()
        return True, "Compte créé avec succès"
        
    except Error as e:
        return False, f"Erreur interne: {e}"
    finally:
        if conn:
            conn.close()

def login_user(username, password):
    conn = get_db_connection()
    if not conn:
        return False, None, "Erreur de connexion à la base de données"
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user['password'], password):
            return True, user, "Connexion réussie"
        else:
            return False, None, "Nom d'utilisateur ou mot de passe incorrect"
            
    except Error as e:
        return False, None, f"Erreur interne: {e}"
    finally:
        if conn:
            conn.close()

def reset_password_by_username(username, new_password):
    conn = get_db_connection()
    if not conn:
        return False, "Erreur de connexion à la base de données"
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if not user:
            return False, "Nom d'utilisateur introuvable"
        
        hashed_password = generate_password_hash(new_password)
        cursor.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            (hashed_password, user['id'])
        )
        conn.commit()
        
        return True, "Mot de passe réinitialisé avec succès"
        
    except Error as e:
        return False, f"Erreur interne: {e}"
    finally:
        conn.close()

# Fonctions de prédiction
@st.cache_resource
def load_model_cached():
    try:
        model = load_model(MODEL_PATH)
        st.success("Modèle chargé avec succès")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def predict_with_model(img_array):
    model = load_model_cached()
    if model is None:
        return None, 0, None
        
    try:
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array, verbose=0)
        pred_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        return pred_class, confidence, preds[0]
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None, 0, None

def save_prediction(user_id, image_name, predicted_class, confidence):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO predictions (user_id, image_name, predicted_class, confidence) VALUES (%s, %s, %s, %s)",
                (user_id, image_name, predicted_class, confidence)
            )
            conn.commit()
            conn.close()
            return True
        except Error as e:
            st.error(f"Erreur sauvegarde prédiction: {e}")
            return False
    return False

def get_user_history(user_id):
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT image_name, predicted_class, confidence, timestamp 
            FROM predictions 
            WHERE user_id = %s 
            ORDER BY timestamp DESC
        """, (user_id,))
        
        predictions = cursor.fetchall()
        
        for pred in predictions:
            pred['confidence'] = round(pred['confidence'] * 100, 2)
        
        return predictions
        
    except Error as e:
        st.error(f"Erreur récupération historique: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_user_stats(user_id):
    conn = get_db_connection()
    if not conn:
        return {'total_predictions': 0, 'class_distribution': {}}
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = %s", (user_id,))
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT predicted_class, COUNT(*) as count 
            FROM predictions 
            WHERE user_id = %s 
            GROUP BY predicted_class 
            ORDER BY count DESC
        """, (user_id,))
        
        class_stats = {}
        for row in cursor.fetchall():
            class_stats[row[0]] = row[1]
        
        return {
            'total_predictions': total_predictions,
            'class_distribution': class_stats
        }
        
    except Error as e:
        st.error(f"Erreur récupération statistiques: {e}")
        return {'total_predictions': 0, 'class_distribution': {}}
    finally:
        if conn:
            conn.close()

# Pages de l'application
def login_page():
    st.markdown('<div class="main-header">🛰️ EuroSAT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">SATELLITE INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("### 🔐 MISSION CONTROL")
            st.markdown("Authentification sécurisée • Classification IA avancée")
            
            with st.form("login_form"):
                username = st.text_input("👤 IDENTIFIANT AGENT", placeholder="Votre nom d'utilisateur")
                password = st.text_input("🔒 CODE D'ACCÈS", type="password", placeholder="••••••••")
                
                col1, col2 = st.columns(2)
                with col1:
                    login_submitted = st.form_submit_button("🚀 INITIER CONNEXION", use_container_width=True)
                with col2:
                    if st.form_submit_button("📝 CRÉER PROFIL AGENT", use_container_width=True):
                        st.session_state.current_page = "register"
                        st.rerun()
                
                if st.form_submit_button("🔑 RÉCUPÉRATION CODE", use_container_width=True):
                    st.session_state.current_page = "forgot_password"
                    st.rerun()
            
            if login_submitted and username and password:
                success, user, message = login_user(username, password)
                if success:
                    st.session_state.user = user
                    st.session_state.current_page = "dashboard"
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

def register_page():
    st.markdown('<div class="main-header">🛰️ EuroSAT</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("### 👨‍🚀 NOUVEAU AGENT")
            st.markdown("Intégration au programme EuroSAT")
            
            with st.form("register_form"):
                username = st.text_input("👤 IDENTIFIANT", placeholder="Choisissez un nom d'utilisateur")
                email = st.text_input("📧 GMAIL", placeholder="exemple@gmail.com")
                password = st.text_input("🔒 CODE SÉCURISÉ", type="password", placeholder="Minimum 6 caractères")
                confirm_password = st.text_input("✅ CONFIRMATION", type="password", placeholder="Répétez le code")
                
                col1, col2 = st.columns(2)
                with col1:
                    register_submitted = st.form_submit_button("👨‍🚀 ACTIVER AGENT", use_container_width=True)
                with col2:
                    if st.form_submit_button("🔙 RETOUR", use_container_width=True):
                        st.session_state.current_page = "login"
                        st.rerun()
            
            if register_submitted:
                if not all([username, email, password, confirm_password]):
                    st.error("Tous les champs sont requis")
                elif password != confirm_password:
                    st.error("Les codes d'accès ne correspondent pas")
                elif len(password) < 6:
                    st.error("Le code doit contenir au moins 6 caractères")
                else:
                    success, message = register_user(username, email, password)
                    if success:
                        st.success(message)
                        st.session_state.current_page = "login"
                        st.rerun()
                    else:
                        st.error(message)

def forgot_password_page():
    st.markdown('<div class="main-header">🛰️ EuroSAT</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.container():
            st.markdown("### 🔐 RÉINITIALISATION SÉCURISÉE")
            st.markdown("Récupération code d'accès agent")
            
            with st.form("reset_form"):
                username = st.text_input("👤 IDENTIFIANT AGENT", placeholder="Votre nom d'utilisateur")
                new_password = st.text_input("🔒 NOUVEAU CODE", type="password", placeholder="Nouveau code sécurisé")
                confirm_password = st.text_input("✅ CONFIRMATION CODE", type="password", placeholder="Répétez le nouveau code")
                
                col1, col2 = st.columns(2)
                with col1:
                    reset_submitted = st.form_submit_button("🔄 RÉINITIALISER", use_container_width=True)
                with col2:
                    if st.form_submit_button("🔙 RETOUR", use_container_width=True):
                        st.session_state.current_page = "login"
                        st.rerun()
            
            if reset_submitted:
                if not all([username, new_password, confirm_password]):
                    st.error("Tous les champs sont requis")
                elif new_password != confirm_password:
                    st.error("Les codes ne correspondent pas")
                elif len(new_password) < 6:
                    st.error("Le code doit contenir au moins 6 caractères")
                else:
                    success, message = reset_password_by_username(username, new_password)
                    if success:
                        st.success(message)
                        st.info("Redirection vers Mission Control...")
                        st.session_state.current_page = "login"
                        st.rerun()
                    else:
                        st.error(message)

def dashboard_page():
    user = st.session_state.user
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"# 🛰️ CENTRE DE CONTRÔLE")
        st.markdown(f"**Mission en cours :** `{user['username']}`")
        st.markdown("Système d'analyse satellite • IA de classification avancée • Surveillance globale active")
    with col2:
        if st.button("🚪 DÉCONNEXION", use_container_width=True):
            st.session_state.pop('user', None)
            st.session_state.current_page = "login"
            st.rerun()
    
    st.divider()
    
    # Quick Stats
    stats = get_user_stats(user['id'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 ANALYSES TOTALES",
            value=stats['total_predictions']
        )
    with col2:
        st.metric(
            label="🛰️ SCANS SATELLITE",
            value=stats['total_predictions']
        )
    with col3:
        most_common_class = max(stats['class_distribution'].items(), key=lambda x: x[1])[0] if stats['class_distribution'] else "N/A"
        st.metric(
            label="🏆 CLASSE PRÉDITE",
            value=most_common_class
        )
    with col4:
        st.metric(
            label="📈 ACTIVITÉ",
            value="ACTIVE" if stats['total_predictions'] > 0 else "INACTIVE"
        )
    
    # Quick Actions
    st.subheader("🚀 ACTIONS RAPIDES")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🛰️ SCAN SATELLITE", use_container_width=True):
            st.session_state.current_page = "predict"
            st.rerun()
    with col2:
        if st.button("📊 ARCHIVES MISSION", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()
    with col3:
        if st.button("🗺️ CARTOGRAPHIE", use_container_width=True):
            st.session_state.current_page = "classes"
            st.rerun()
    with col4:
        if st.button("📈 STATISTIQUES", use_container_width=True):
            st.session_state.current_page = "stats"
            st.rerun()
    
    # Recent Activity
    st.subheader("📋 ACTIVITÉ RÉCENTE")
    history = get_user_history(user['id'])[:5]
    
    if not history:
        st.info("Aucune mission récente. Initiez votre première analyse satellite.")
    else:
        for pred in history:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{pred['predicted_class']}**")
                    st.write(f"`{pred['image_name']}`")
                with col2:
                    st.write(f"Confiance: **{pred['confidence']}%**")
                with col3:
                    st.write(pred['timestamp'].strftime("%d/%m/%Y %H:%M"))

def predict_page():
    user = st.session_state.user
    
    st.markdown("# 🛰️ ANALYSE SATELLITE")
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔙 RETOUR AU TABLEAU DE BORD", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    # Upload Area
    st.markdown("### 📤 TÉLÉCHARGEMENT DONNÉES SATELLITE")
    
    uploaded_file = st.file_uploader(
        "Sélectionnez une image satellite",
        type=['png', 'jpg', 'jpeg'],
        help="FORMATS: PNG, JPG • TAILLE MAX: 16MB"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Image satellite chargée", use_column_width=True)
            
            if st.button("🧠 INITIER ANALYSE IA", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    # Process image
                    img = Image.open(uploaded_file).convert("RGB")
                    img = img.resize((128, 128))
                    img_array = image.img_to_array(img)
                    
                    # Make prediction
                    pred_class, confidence, _ = predict_with_model(img_array)
                    
                    if pred_class is not None:
                        class_name = CLASS_NAMES[pred_class]
                        confidence_percent = round(confidence * 100, 2)
                        
                        # Save prediction
                        save_prediction(
                            user['id'],
                            uploaded_file.name,
                            class_name,
                            confidence
                        )
                        
                        # Display results
                        with col2:
                            st.markdown("### 📊 RÉSULTATS CLASSIFICATION")
                            
                            with st.container():
                                st.markdown(f"**Classe prédite:** `{class_name}`")
                                st.markdown(f"**Niveau de confiance:** `{confidence_percent}%`")
                                st.progress(confidence)
                                
                                st.markdown("**Description:**")
                                st.info(CLASS_DESCRIPTIONS.get(class_name, ''))
                                
                                st.markdown(f"**Timestamp:** `{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}`")
                                
                                st.success("✅ Analyse terminée avec succès")

def history_page():
    user = st.session_state.user
    
    st.markdown("# 📊 ARCHIVES MISSION")
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔙 RETOUR AU TABLEAU DE BORD", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    history = get_user_history(user['id'])
    
    if not history:
        st.info("Aucune mission enregistrée. Initiez votre première analyse satellite.")
        if st.button("🛰️ INITIER PREMIÈRE ANALYSE"):
            st.session_state.current_page = "predict"
            st.rerun()
    else:
        for pred in history:
            with st.expander(f"{pred['predicted_class']} - {pred['confidence']}% - {pred['timestamp'].strftime('%d/%m/%Y %H:%M')}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Fichier:** {pred['image_name']}")
                    st.write(f"**Classe:** {pred['predicted_class']}")
                    st.write(f"**Confiance:** {pred['confidence']}%")
                    st.write(f"**Description:** {CLASS_DESCRIPTIONS.get(pred['predicted_class'], '')}")
                with col2:
                    st.write(f"**Date:** {pred['timestamp'].strftime('%d/%m/%Y')}")
                    st.write(f"**Heure:** {pred['timestamp'].strftime('%H:%M:%S')}")

def classes_page():
    st.markdown("# 🗺️ CLASSIFICATION TERRAIN")
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔙 RETOUR AU TABLEAU DE BORD", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    cols = st.columns(2)
    
    class_colors = {
        'AnnualCrop': 'green',
        'Forest': 'darkgreen',
        'HerbaceousVegetation': 'lightgreen',
        'Highway': 'gray',
        'Industrial': 'orange',
        'Pasture': 'yellow',
        'PermanentCrop': 'teal',
        'Residential': 'purple',
        'River': 'blue',
        'SeaLake': 'lightblue'
    }
    
    for i, (class_name, description) in enumerate(CLASS_DESCRIPTIONS.items()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"### {class_name}")
                st.markdown(description)
                st.markdown("---")

def stats_page():
    user = st.session_state.user
    
    st.markdown("# 📈 STATISTIQUES")
    
    # Navigation
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔙 RETOUR AU TABLEAU DE BORD", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    stats = get_user_stats(user['id'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Analyses totales", stats['total_predictions'])
        
        if stats['class_distribution']:
            st.subheader("Distribution des classes")
            for class_name, count in stats['class_distribution'].items():
                st.write(f"{class_name}: {count}")
        else:
            st.info("Aucune donnée statistique disponible")
    
    with col2:
        if stats['class_distribution']:
            # Simple bar chart using st.bar_chart
            import pandas as pd
            df = pd.DataFrame(list(stats['class_distribution'].items()), columns=['Classe', 'Count'])
            st.bar_chart(df.set_index('Classe'))

# Application principale
def main():
    # Initialisation de la base de données
    if 'db_initialized' not in st.session_state:
        init_db()
        st.session_state.db_initialized = True
    
    # Gestion de l'état de l'application
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "login"
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Navigation entre les pages
    if st.session_state.current_page == "login":
        login_page()
    elif st.session_state.current_page == "register":
        register_page()
    elif st.session_state.current_page == "forgot_password":
        forgot_password_page()
    elif st.session_state.current_page == "dashboard" and st.session_state.user:
        dashboard_page()
    elif st.session_state.current_page == "predict" and st.session_state.user:
        predict_page()
    elif st.session_state.current_page == "history" and st.session_state.user:
        history_page()
    elif st.session_state.current_page == "classes" and st.session_state.user:
        classes_page()
    elif st.session_state.current_page == "stats" and st.session_state.user:
        stats_page()
    else:
        st.session_state.current_page = "login"
        st.rerun()

if __name__ == "__main__":
    main()