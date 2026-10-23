# Journal de bord — The Hedgehogs of Wall Street

> Ce fichier trace **ce qu'on fait, pourquoi on le fait, ce qu'on a observé et les pistes ouvertes**.
> Objectif : pouvoir justifier chaque choix à la soutenance (11/12).
> Règle : chaque décision = constat chiffré + alternatives envisagées + raison du choix.

---

## 0. Cadre du projet

- **Cours** : Analyse de données et Apprentissage Automatique (IMT Atlantique) — encadrant Bilal Said.
- **Sujet** : prédire la qualité de crédit d'une entreprise à partir de ses ratios financiers → outil de *screening* du risque pour un investisseur.
- **Tâche principale** : classification binaire **Investment Grade (IG) vs Speculative Grade (SG)**.
- **Extension** : classification multiclasse (notes regroupées en niveaux de risque).
- **Dataset** : [Kaggle — Corporate Credit Rating with Financial Ratios](https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios), copie dans ce repo.
- **Livrable** : un notebook Colab unique, structuré par sections, + un notebook séparé « ce qui n'a pas marché ».

### Échéances

| Date | Rendu | Poids |
|---|---|---|
| 16/10 | Équipes & sujet | — |
| **23/10** | EDA & exploration non supervisée | 30 % |
| 20/11 | Baselines ML & modèles d'ensemble | 30 % |
| 27/11 | Modèles améliorés, tuning & analyse | 30 % |
| 04/12 | Rendu final | (+5 % consignes, +5 % organisation) |
| 11/12 | Soutenance | — |

### Contraintes imposées par le sujet (à cocher au fil du projet)

- [ ] Feature engineering (nouvelles variables à partir des existantes)
- [ ] Nested CV, même nombre de folds pour tous les modèles
- [ ] Learning curves
- [ ] Mêmes métriques pour tous les modèles, comparées sur les mêmes graphiques
- [ ] Évaluation finale sur un test set jamais vu (overfitting vs généralisation)
- [ ] Sur/sous-échantillonnage **uniquement sur le train** (après le split)
- [ ] Pas de modèle linéaire/polynomial non régularisé
- [ ] Pas d'arbre de décision seul
- [ ] Tester des versions polynomiales des modèles linéaires
- [ ] Si on part d'un sous-échantillon du train, réentraîner ensuite sur tout le train
- [ ] Ensembles avancés (boosting, stacking…)
- [ ] Tuning sur plusieurs hyperparamètres significatifs
- [ ] Graphes en coordonnées parallèles pour le tuning
- [ ] Recommandé : faire critiquer le travail par un LLM

---

## 1. Ce qu'on sait des données (constats chiffrés)

*Exploration du 23/10.*

### Structure générale
- **7 805 lignes, 25 colonnes**, **aucune valeur manquante**, **aucune ligne dupliquée**.
- Notations entre **avril 2010 et décembre 2016**.
- **16 ratios financiers** : liquidité (Current Ratio), endettement (LT Debt/Capital, Debt/Equity), marges (Gross, Operating, EBIT, EBITDA, Pre-Tax, Net), efficacité (Asset Turnover), rentabilité (ROE, ROTE, ROA, ROI), cash-flows par action (OCF, FCF).
- **Variables catégorielles** : Rating Agency, Sector (12 secteurs), SIC Code (240 valeurs).
- **Identifiants** (non utilisables comme features) : Corporation, CIK, Ticker, Rating Date.

### Cible
- Colonne `Binary Rating` déjà présente : **65 % IG / 35 % SG** → déséquilibre modéré.
- Frontière vérifiée : BBB- = IG, BB+ = SG.
- Notes détaillées : 23 modalités, de AAA à D, très inégalement réparties (AAA : 90, CC+ : 1, D : 5) → l'extension multiclasse imposera des regroupements.

### Entreprises et doublons (point clé pour le leakage)
- **686 CIK uniques**, mais 1 377 noms d'entreprise : **430 CIK apparaissent sous plusieurs noms** (variantes d'écriture, ex. « Inc » / « Inc. »). → **L'identifiant fiable d'une entreprise est le CIK, pas le nom.**
  - ⚠️ À corriger dans la description du projet : on a écrit « 1 377 corporations », c'est en réalité 686 entreprises distinctes.
- Une entreprise apparaît en moyenne **11 fois** (médiane 9, max 42).
- Seulement **2 694 bilans distincts** (CIK + ratios identiques) pour 7 805 lignes : **6 908 lignes ont un « jumeau »** aux ratios strictement identiques (même bilan noté par plusieurs agences ou à plusieurs dates).

### Désaccord entre agences
Mesuré sur les **1 231 bilans notés par au moins deux agences** :

| Paire | Bilans communs | Accord IG/SG | Écart moyen (crans) |
|---|---|---|---|
| S&P – Moody's | 665 | 94,4 % | S&P +0,5 cran plus généreuse |
| S&P – Fitch | 220 | 93,6 % | S&P +0,2 |
| Egan-Jones – S&P | 453 | 90,5 % | EJ +0,5 |
| Moody's – Fitch | 194 | 89,7 % | Fitch +0,4 |
| Egan-Jones – Moody's | 321 | 88,5 % | EJ +1,1 |
| Egan-Jones – Fitch | 115 | 84,3 % | EJ +0,6 |

- **Egan-Jones est systématiquement plus généreuse.** Explication plausible : modèle *investor-pays* (Egan-Jones), contre *issuer-pays* pour les trois grandes.
- **149 bilans** sont classés IG par une agence et SG par une autre. Les désaccords se concentrent autour de la frontière (BBB- et BB+ surtout).
- 766 lignes ont des ratios identiques mais des labels binaires différents → **bruit irréductible** si on ignore l'agence.
- 131 entreprises changent de catégorie IG/SG au cours du temps.
- ⚠️ La part brute d'IG par agence (EJ 77 %, Fitch 71 %, S&P 59 %, Moody's 53 %) mélange biais de notation et **effet de couverture** (les agences ne notent pas les mêmes entreprises). Seule la comparaison sur bilans identiques isole le biais.

### Distributions
- **Queues extrêmes** sur plusieurs ratios : ROE de -11 258 à 7 038, ROTE de -14 023 à 9 883, Debt/Equity jusqu'à -1 473.
- Debt/Equity **négatif** = fonds propres négatifs : c'est un signal de détresse financière, pas une erreur de saisie.
- Plusieurs marges sont quasi redondantes (Operating ≈ EBIT) → multicolinéarité à traiter.

---

## 2. Décisions prises

### D1 — Split au niveau de l'entreprise (CIK) ✅
- **Constat** : 6 908 lignes sur 7 805 partagent leurs ratios avec une autre ligne de la même entreprise.
- **Risque** : avec un split aléatoire, un bilan vu à l'entraînement réapparaît au test → scores artificiellement gonflés (*data leakage*).
- **Décision** : tous les splits (train/test et folds de CV) se font **par CIK** (`GroupShuffleSplit`, `StratifiedGroupKFold`). Le CIK est utilisé plutôt que le nom à cause des 430 variantes de noms.
- **Piste de démonstration** : comparer les scores obtenus avec un split aléatoire et avec un split par CIK, pour **chiffrer le leakage** (argument fort pour la soutenance).

### D2 — Garder toutes les lignes, avec l'agence comme variable ✅
- **Constat** : pour un même bilan, la note dépend en partie de l'agence (biais systématique, voir tableau ci-dessus).
- **Alternatives envisagées** :
  - A. Toutes les lignes, agence en variable ← **retenu (pipeline principal)**
  - B. Toutes les lignes, sans l'agence ← **retenu (pipeline de comparaison)**
  - C. Une seule agence (ex. S&P, 2 813 lignes) : cible plus propre, mais beaucoup de données perdues.
  - D. Une ligne par bilan avec note consensus : plus de doublons, mais l'information sur l'agence est perdue.
- **Pourquoi A** : on garde le maximum de données et le modèle peut apprendre le décalage entre agences au lieu de le subir comme du bruit.
- **Pourquoi aussi B** : l'écart de performance A vs B mesure directement ce qu'apporte l'agence.
- **Usage en screening** : au moment de prédire, on fixe une agence de référence (ex. S&P) → « comment S&P noterait-elle cette entreprise ? ».
- **Préparation** : regrouper les agences marginales (DBRS, JCR, HR Ratings : 53 lignes en tout) dans une modalité « Autre ».

---

## 3. Décisions proposées, à valider

### P1 — Traiter les outliers dans le pipeline plutôt que supprimer des lignes
- Supprimer les valeurs extrêmes reviendrait à retirer les entreprises en difficulté, qui sont justement les plus informatives pour la classe SG.
- **Proposition** : winsorisation ou `QuantileTransformer` / `RobustScaler`, **ajustés sur le train uniquement**, et comparés entre pipelines.

### P2 — Métriques orientées « screening »
- Rater une entreprise SG (la classer IG à tort) coûte plus cher à un investisseur qu'une fausse alerte.
- **Proposition** : ROC-AUC et PR-AUC comme métriques principales, + **recall de la classe SG**, F1 et balanced accuracy. Mêmes métriques pour tous les modèles.

---

## 4. Pistes et idées

### Feature engineering
- Indicateur « fonds propres négatifs » (Debt/Equity < 0).
- Transformations log signées pour les ratios à queues lourdes.
- Écarts entre marges (ex. Gross − Operating = poids des charges d'exploitation).
- Ratio FCF / OCF (part du cash opérationnel absorbée par les investissements).
- Ratios relatifs au secteur (ratio − médiane du secteur, calculée sur le train seulement).
- SIC à 2 chiffres (division industrielle), plutôt que 240 codes.
- Année de notation.

### Modélisation
- Baselines : DummyClassifier, régression logistique régularisée (+ version polynomiale), SVM, KNN.
- Ensembles : Random Forest, gradient boosting (HistGradientBoosting, XGBoost/LightGBM), stacking.
- Déséquilibre : comparer `class_weight` et SMOTE (dans le pipeline, train seulement).
- Tuning : Optuna (fournit directement les graphes en coordonnées parallèles).
- Explicabilité : SHAP, permutation importance → règles lisibles pour l'investisseur.

### Robustesse
- Split temporel en complément (train ≤ 2014, test 2015-2016) pour tester la stabilité dans le temps.

### Extension multiclasse
- Regrouper les 23 notes en 4-5 niveaux (ex. AAA–AA / A / BBB / BB / B et moins).

---

## 5. Questions probables à la soutenance (à préparer)

- Pourquoi un split par entreprise ? Combien le split aléatoire surestime-t-il les scores ?
- Pourquoi l'agence comme variable ? N'est-ce pas une forme de triche ?
- Pourquoi ne pas supprimer les outliers ?
- Pourquoi ces métriques plutôt que l'accuracy ?
- Pourquoi le modèle retenu ? Overfitting ou généralisation ?
- Quelles variables comptent le plus, et est-ce cohérent avec la finance ?

---

## 6. Historique

- **23/10** — Repo cloné. Exploration initiale des données. Décisions D1 (split par CIK) et D2 (agence en variable, pipelines A et B). Propositions P1 (outliers) et P2 (métriques) en attente de validation.
