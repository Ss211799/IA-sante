# IA-sante — Survival Analysis for Liver Transplant Prioritization (PBC/CBP)

**Performance clé (test)** : C-index = **0.8621**


## Contexte
Ce projet vise à estimer le risque de mortalité chez des patients atteints de cholangite biliaire primitive (CBP/PBC),
afin d’améliorer la priorisation de la greffe du foie en contexte de pénurie d’organes.

## Méthode
- Modélisation du temps en **pas discrets** (discrete-time survival)
- Réseau **LSTM** pour exploiter les données longitudinales
- Optimisation via **Negative Log-Likelihood (NLL)** adaptée aux données censurées

## Évaluation
- **C-index dépendant du temps** (concordance) calculé à partir du **risque cumulatif**.
- Interprétation: capacité du modèle à classer correctement les patients selon le risque (plus élevé = décès plus précoce).

## Résultats (test set)

- **TEST LOSS (Negative Log-Likelihood)**: 323.6649  
- **C-INDEX dépendant du temps**: **0.8621**

Interprétation :
- Le C-index élevé indique une bonne capacité du modèle à classer correctement les patients selon le risque de mortalité.
- La valeur absolue de la loss dépend de la discrétisation temporelle et sert principalement au suivi de l’apprentissage.


## Installation
```bash
uv sync
