from pathlib import Path
print("RUNNING:", Path(__file__).resolve())

import numpy as np
from .config import DATA_DIR
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv(DATA_DIR / "clinical_data_pbc.csv")
    print(df.head())

    datafram_clinical = df.copy()

    #le nombre d'observations
    n_observations = len(datafram_clinical)
    #le nombre de patients
    n_patients = datafram_clinical['id'].nunique()
    #le nombre moyen de visites par patient
    visit_par_patient = n_observations / n_patients
    print(f"Nombre d'observations: {n_observations}")
    print(f"Nombre de patients: {n_patients}")
    print(f"Nombre moyen de visites par patient: {visit_par_patient:.2f}")

    # Visualiser la distribution du nombre de visites par patient
    plt.figure(figsize=(10, 6))
    sns.histplot(datafram_clinical['id'].value_counts(), bins=20, kde=True)
    plt.title("Distribution du nombre de visites par patient")
    plt.xlabel("Nombre de visites")
    plt.ylabel("Nombre de patients")
    plt.savefig(DATA_DIR / "la distribution du nombre de visites par patient.png")

    #les valeurs manquantes
    missing_values = datafram_clinical.isnull().mean() * 100
    missing_percent = missing_values[missing_values > 0].sort_values(ascending=False)
    print(missing_percent)
    # Visualiser les pourcentages de valeurs manquantes
    if not missing_percent.empty:
        plt.figure(figsize=(10, 8))
        sns.barplot(x=missing_percent.index, y=missing_percent.values)
        plt.title("Pourcentage de données manquantes par variable")
        plt.ylabel("% Manquant")
        plt.xticks(rotation=45)
        plt.savefig(DATA_DIR / "pourcentage_manquants.png")
        plt.close()
    else:
        print("Ce jeu de données ne contient pas de Nan.")
        plt.figure(figsize=(10, 8))
        sns.barplot(x=missing_percent.index, y=missing_percent.values)
        plt.title("Pourcentage de données manquantes par variable")
        plt.ylabel("% Manquant")
        plt.xticks(rotation=45)
        plt.savefig(DATA_DIR / "pourcentage_manquants.png")
        plt.close()
    
    # variables cles de foie
    vars_to_check = ['serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']
    var_existing = [var for var in vars_to_check if var in datafram_clinical.columns]
    for var in var_existing:
        plt.figure(figsize=(12,4))
        # Distribution  brute
        plt.subplot(1,2,1)
        sns.histplot(datafram_clinical[var],kde=True,color='skyblue')
        plt.title(f"Distribution de {var} (brute)")
        # Distribution apres log transformation
        if datafram_clinical[var].min()>0:
            plt.subplot(1,2,2)
            sns.histplot(np.log1p(datafram_clinical[var]),kde = True,color = 'salmon')
            plt.title(f"Distribution de {var} (log transformée)")
            plt.savefig(DATA_DIR / f"distribution_{var}.png")
            plt.close()

    print("\n--- MATRICE DE CORRÉLATION ---")
    # on prend seulement les variables numériques
    numeric_df = datafram_clinical.select_dtypes(include=[np.number])
    #eliminer les patients avec beaucoup de visites:
    corr_matrix = numeric_df.groupby('id').mean().corr()
    plt.figure(figsize=(12,8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix,mask=mask,annot=True,cmap='coolwarm',fmt=".2f",vmin=-1,vmax=1)
    plt.title("Matrice de corrélation des variables cliniques")
    plt.savefig(DATA_DIR / "matrice_de_correlation.png")
    plt.close()


    print("\n--- temps entre les visites ---")
    #Tri des donnees par id et par temps
    datafram_clinical = datafram_clinical.sort_values(by=['id','times'])
    datafram_clinical['delta_time'] = datafram_clinical.groupby('id')['times'].diff()

    #creation d'une figure pour visualiser la distribution des deltas de temps
    suivi_visit = datafram_clinical.dropna(subset=['delta_time'])
    plt.figure(figsize=(10,6))
    sns.histplot(suivi_visit['delta_time'],kde=True,bins=40, color='teal', alpha=0.6)
    plt.axvline(90, color='red', linestyle='--', label='90 jours')
    plt.axvline(180, color='orange', linestyle='--', label='180 jours')
    plt.axvline(365, color='green', linestyle='--', label='365 jours')
    plt.title("Distribution du temps entre les visites")
    plt.xlabel("Temps entre les visites (jours)")
    plt.ylabel("Nombre de visites")
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(DATA_DIR / "distribution_temps_entre_visites.png")
    plt.close()

    print("distribution du temps de survie et des evenements")

    datafram_clinical_last = datafram_clinical.sort_values(by=['id','tte']).groupby('id').tail(1).copy()

    status_map = {
        0: 'Censuré (Vivant)',
        1: 'Transplanté',
        2: 'Décédé'
    }
    col_event = 'label'
    datafram_clinical_last['event_label'] = datafram_clinical_last[col_event].map(status_map)

    #distribution of the statue
    plt.figure(figsize=(10,6))
    ax = sns.countplot(data=datafram_clinical_last, x='event_label', palette= 'viridis',order = ['Censuré (Vivant)','Transplanté','Décédé'])
    plt.xlabel("Statut de l'événement")
    plt.ylabel("Nombre de patients")
    plt.title("Distribution du statut des événements")
    for i in ax.containers:
        ax.bar_label(i,)
    plt.savefig(DATA_DIR / "distribution_statut_evenement.png")
    plt.close()

    #distribution du temps d'evenement
    sns.histplot(datafram_clinical_last,x = 'tte',hue = 'event_label',
                 hue_order=['Censuré (Vivant)','Transplanté','Décédé'],
                 element='step',palette='viridis',alpha=0.3)
    plt.title("Distribution du temps jusqu'à l'événement")
    plt.xlabel("Temps jusqu'à l'événement (jours)")
    plt.ylabel("Nombre de patients")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(DATA_DIR / "distribution_temps_evenement.png")
    plt.close()

    


if __name__ == "__main__":
    main()
