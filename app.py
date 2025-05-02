import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from quantile_forest import RandomForestQuantileRegressor
import os
import matplotlib.pyplot as plt
import seaborn as sns

quantiles = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]



# Load trained models
rf_classifier = joblib.load("models/rf_classifier.pkl")

qrf_mass = joblib.load("models/qrf_mass_model.pkl")
qrf_rade = joblib.load("models/qrf_rade_model.pkl")


# Load HGB models for mass and radius
hgb_mass_models = {}
hgb_rade_models = {}
for q in quantiles:
    hgb_mass_models[q] = joblib.load(f"models/hgb_mass_q_"+str(q)+".pkl")
    hgb_rade_models[q] = joblib.load(f"models/hgb_rade_q_"+str(q)+".pkl")


# Streamlit app title and description
st.title("ECLIPSE")
st.write("""
    Input the features of your exoplanet, 
    and the app will classify its disposition and predict its mass and radius in quantiles.
""")

# Input form for user to input feature values
st.sidebar.header("Exoplanet Feature Input")

# Example feature inputs (adjust this to match your dataset's feature names)
toi = st.sidebar.number_input("toi", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
tid = st.sidebar.number_input("tid", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_tmag= st.sidebar.number_input("st_tmag", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_tmagerr = st.sidebar.number_input("st_tmagerr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
ra = st.sidebar.number_input("ra", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
dec = st.sidebar.number_input("dec", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pmra = st.sidebar.number_input("pmra", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pmraerr = st.sidebar.number_input("pmraerr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pmdec = st.sidebar.number_input("pmdec", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pmdecerr = st.sidebar.number_input("pmdecerr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_tranepoch = st.sidebar.number_input("pl_tranepoch", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_tranepocherr = st.sidebar.number_input("pl_tranepocherr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_orbper = st.sidebar.number_input("pl_orbper", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_orbpererr = st.sidebar.number_input("pl_orbpererr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_trandurh = st.sidebar.number_input("pl_trandurh", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_trandurherr = st.sidebar.number_input("pl_trandurherr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_trandepmmag = st.sidebar.number_input("pl_trandepmmag", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_trandepmmagerr = st.sidebar.number_input("pl_trandepmmagerr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_trandep = st.sidebar.number_input("pl_trandep", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_trandeperr = st.sidebar.number_input("pl_trandeperr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_insol = st.sidebar.number_input("pl_insol", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_eqt = st.sidebar.number_input("pl_eqt", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
pl_snr = st.sidebar.number_input("pl_snr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_dist = st.sidebar.number_input("st_dist", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_disterr = st.sidebar.number_input("st_disterr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_teff = st.sidebar.number_input("st_teff", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_tefferr = st.sidebar.number_input("st_tefferr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_logg = st.sidebar.number_input("st_logg", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_loggerr = st.sidebar.number_input("st_loggerr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_rad = st.sidebar.number_input("st_rad", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_raderr = st.sidebar.number_input("st_raderr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_feh = st.sidebar.number_input("st_feh", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_feherr = st.sidebar.number_input("st_feherr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_mass = st.sidebar.number_input("st_mass", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)
st_masserr = st.sidebar.number_input("st_masserr", min_value=-1e15, max_value=1e15, step=0.0001, format="%.8f", value=0.0)

# Run Button
run_button = st.button("Run Prediction")

if run_button: 
    # Put input features into a DataFrame for the classifier and regressor
    input_features = pd.DataFrame([[tid,toi,st_tmag,st_tmagerr,ra,dec,pmra,pmraerr,pmdec,pmdecerr,
                                pl_tranepoch,pl_tranepocherr,pl_orbper,pl_orbpererr,pl_trandurh,pl_trandurherr,
                                pl_trandepmmag,pl_trandepmmagerr,pl_trandep,pl_trandeperr,pl_insol,pl_eqt,pl_snr,
                                st_dist,st_disterr,st_teff,st_tefferr,st_logg,st_loggerr,st_rad,st_raderr,st_feh,
                                st_feherr,st_mass,st_masserr]],
                              columns=["tid","toi","st_tmag","st_tmagerr","ra","dec",
                                "pmra","pmraerr","pmdec","pmdecerr","pl_tranepoch","pl_tranepocherr","pl_orbper",
                                "pl_orbpererr","pl_trandurh","pl_trandurherr","pl_trandepmmag","pl_trandepmmagerr",
                                "pl_trandep","pl_trandeperr","pl_insol","pl_eqt","pl_snr",
                                "st_dist","st_disterr","st_teff","st_tefferr","st_logg","st_loggerr","st_rad",
                                "st_raderr","st_feh","st_feherr","st_mass","st_masserr"])

    # Initialize lists to store results
    mean_mass_preds, mean_rade_preds = [], []
    q_5_mass = 0
    q_5_rade = 0
    # Predict mass and radius using Quantile Regression Forest (with quantiles)
    # QRF predictions (shape: [1, 2, len(quantiles)])
    qrf_mass_preds = qrf_mass.predict(input_features, quantiles=quantiles)[0]  # shape: (n_quantiles,)
    qrf_rade_preds = qrf_rade.predict(input_features, quantiles=quantiles)[0]


    # Predict mass and radius using HistGradientBoostingRegressor (mean prediction)
    for i, q in enumerate(quantiles):
    
        hgb_mass_log = hgb_mass_models[q].predict(input_features)[0]
        hgb_rade_log = hgb_rade_models[q].predict(input_features)[0]

        qrf_mass_val = qrf_mass_preds[i]
        qrf_rade_val = qrf_rade_preds[i]

        # Combine (mean) and inverse log10
        mean_mass = (10 ** hgb_mass_log + 10 ** qrf_mass_val) / 2
        mean_rade = (10 ** hgb_rade_log + 10 ** qrf_rade_val) / 2

        mean_mass_preds.append(mean_mass)
        mean_rade_preds.append(mean_rade)

        st.write(f"**Quantile {q}:**")
        st.write(f"- Mass (mean of HGB+QRF): {mean_mass:.4f} Earth masses")
        st.write(f"- Radius (mean of HGB+QRF): {mean_rade:.4f} Earth radii")

        if q == 0.5:
            q_5_mass = mean_mass
            q_5_rade = mean_rade

    # First, add the input features to the model input (adding disposition as a feature)
    #input_features['tfopwg_disp'] = disposition_prediction

    input_features.insert(2, "pl_mass", q_5_mass)
    input_features.insert(21, "pl_rade", q_5_rade)

    st.write(f"Median Mass: {q_5_mass:.4f} Earth masses")
    st.write(f"Median Radius: {q_5_rade:.4f} Earth radii")
    
    # Predict the disposition using the Random Forest Classifier
    disposition_prediction = rf_classifier.predict(input_features)[0]

    disposition_name = "Unknown"
    if disposition_prediction == 0:
        disposition_name = "False Positive"
    elif disposition_prediction == 1:
        disposition_name = "True Planet"

    # Show disposition prediction to user
    st.write(f"Predicted Disposition: {disposition_name}")



    

    # Optionally allow download
    result_df = pd.DataFrame({
        "Quantile": quantiles,
        "Mean_Mass": mean_mass_preds,
        "Mean_Radius": mean_rade_preds
    })
    st.download_button("Download Predictions CSV", result_df.to_csv(index=False), "planet_predictions.csv", "text/csv")

    #plot median and IQR
    # Indexes for quantiles
    q25 = 0.25
    q50 = 0.5
    q75 = 0.75

    i25 = quantiles.index(q25)
    i50 = quantiles.index(q50)
    i75 = quantiles.index(q75)

    # Values
    mass_median = mean_mass_preds[i50]
    mass_lower = mean_mass_preds[i25]
    mass_upper = mean_mass_preds[i75]

    rade_median = mean_rade_preds[i50]
    rade_lower = mean_rade_preds[i25]
    rade_upper = mean_rade_preds[i75]

    yerr = [
    [mass_median - mass_lower, rade_median - rade_lower],  # lower errors
    [mass_upper - mass_median, rade_upper - rade_median]   # upper errors
]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        x=["Mass", "Radius"],
        y=[mass_median, rade_median],
        yerr=yerr,
        fmt="o",
        capsize=5,
        label="Median ± IQR"
    )

    x_coords = [0, 1]  # Mass, Radius
    labels = ["Mass [Earth masses]", "Radius [Earth radii]"]

    # Add numeric labels
    for i, (x, median, lower, upper) in enumerate(zip(x_coords, [mass_median, rade_median], [mass_lower, rade_lower], [mass_upper, rade_upper])):
        ax.text(x, median + 0.03, f"Median: {median:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(x, lower - 0.03, f"↓ {lower:.3f}", ha='center', va='top', fontsize=9, color='gray')
        ax.text(x, upper + 0.03, f"↑ {upper:.3f}", ha='center', va='bottom', fontsize=9, color='gray')

    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels)
    
    ax.set_ylabel("Predicted Value")
    ax.set_title("Predicted Mass and Radius with IQR")
    ax.legend()
    st.pyplot(fig)


    # Distribution of predictions
    st.write("### Distribution of Predicted Mass and Radius")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.kdeplot(mean_mass_preds, ax=axs[0], fill=True, color="skyblue")
    axs[0].set_title("Mass Prediction Distribution")
    axs[0].set_xlabel("Mass")
    axs[0].grid(True)

    sns.kdeplot(mean_rade_preds, ax=axs[1], fill=True, color="salmon")
    axs[1].set_title("Radius Prediction Distribution")
    axs[1].set_xlabel("Radius")
    axs[1].grid(True)

    st.pyplot(fig)