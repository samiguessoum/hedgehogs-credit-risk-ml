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

**Partie 1 (EDA et exploration non supervisée) : terminée (étapes 1 à 8).**

---

## 1. Ce qu'on sait des données

### Structure générale (étape 2)
- **7 805 lignes, 25 colonnes**, **aucune valeur manquante**, **aucune ligne dupliquée**.
- Notations du **6 avril 2010 au 27 décembre 2016**.
- **16 ratios financiers** : liquidité (Current Ratio), endettement (LT Debt / Capital, Debt/Equity), marges (Gross, Operating, EBIT, EBITDA, Pre-Tax, Net), efficacité (Asset Turnover), rentabilité (ROE, ROTE, ROA, ROI), cash-flows par action (OCF, FCF).
- **Variables catégorielles** : Rating Agency (7), Sector (12), SIC Code (240, haute cardinalité).
- **Cibles** : `Binary Rating` (binaire), `Rating` (23 notes détaillées).
- **Identifiants à exclure des features** : Corporation, Ticker, CIK (gardé pour le split), Rating Date.
- **Échelles très hétérogènes** et **queues extrêmes** (ROE de −11 258 à 7 038 ; kurtosis de 2 753 pour LT Debt / Capital) → mise à l'échelle indispensable pour KNN, SVM et régression logistique.

### Valeurs manquantes et suspectes (étape 2)
- Aucun NaN, aucun zéro, aucun infini : pas de valeur manquante « déguisée » classique.
- **Valeur sentinelle : Gross Margin = 100 % exactement** sur 145 lignes (énergie 83, utilities 24, finance 16), contre 23 au plus pour toute autre valeur. Probablement un coût des ventes non renseigné.
- **Nombreuses valeurs négatives, légitimes** : pertes, fonds propres négatifs (Debt/Equity < 0 : 299 lignes), Return On Tangible Equity < 0 dans 33 % des cas (goodwill supérieur aux fonds propres). Ce ne sont pas des erreurs.

---

### Entreprises et doublons (étape 3)
- **686 CIK** (identifiant SEC) pour **1 377 noms** : **430 CIK apparaissent sous plusieurs noms** (« Inc » / « Inc. », filiales). Le CIK est l'identifiant fiable, pas le nom.
  - ⚠️ À corriger dans la description initiale du projet : « 1 377 corporations » → 686 entreprises distinctes.
  - Quelques rattachements douteux (ex. CIK 1004315 : entités National Grid + National Air Cargo) → rendent le split par CIK plus prudent, sans créer de fuite.
- Une entreprise apparaît en moyenne **11 fois** (médiane 9, max 42).
- **2 694 bilans distincts** (CIK + 16 ratios identiques) pour 7 805 lignes : **89 % des lignes ont un « jumeau »** (même bilan noté par plusieurs agences ou à plusieurs dates).
- Vue par bilan : **57 % IG / 43 % SG** (contre 65 / 35 sur les lignes). Les IG sont notées plus souvent (3,3 notations par bilan contre 2,4) → les doublons surreprésentent les IG.

### Cible (étape 4)
- `Binary Rating` cohérent avec la convention de marché : BBB- et mieux = IG, BB+ et moins = SG.
- **65 % IG / 35 % SG** sur les lignes (57 / 43 sur les bilans) → déséquilibre modéré.
- 23 notes détaillées, en cloche autour de BBB / A. Notes extrêmes rares (AAA : 90 lignes ; CCC- à D : 34).
- Forte masse **de part et d'autre de la frontière** (BBB- : 705, BB+ : 545) → zone où se concentreront les erreurs.

### Désaccord entre agences (étape 4)
Mesuré sur les **1 231 bilans notés par au moins deux agences** :

| Paire | Bilans communs | Accord IG/SG | Écart moyen |
|---|---|---|---|
| S&P – Moody's | 665 | 94,4 % | S&P +0,5 cran plus généreuse |
| S&P – Fitch | 220 | 93,6 % | S&P +0,2 |
| S&P – Egan-Jones | 453 | 90,5 % | EJ +0,5 |
| Moody's – Fitch | 194 | 89,7 % | Fitch +0,4 |
| Moody's – Egan-Jones | 321 | 88,5 % | EJ +1,1 |
| Fitch – Egan-Jones | 115 | 84,3 % | EJ +0,6 |

- **Egan-Jones est systématiquement plus généreuse.** ⚠️ Ce n'est **pas** dû à son mode de rémunération : payée par les investisseurs, elle devrait au contraire noter plus sévèrement si les agences payées par les émetteurs gonflaient leurs notes.
  - Résultat **cohérent avec la littérature** : +0,66 cran EJ vs Moody's (+0,93 en IG) dans [Financial Innovation, 2021](https://link.springer.com/doi/10.1186/s40854-021-00263-z) ; nous trouvons +1,1.
  - Explication (Beaver, Shakespeare & Soliman, [Stanford GSB](https://www.gsb.stanford.edu/insights/research-bond-rating-system-isnt-broken)) : rôle **quasi réglementaire** de Moody's / S&P (les fonds doivent vendre sous la catégorie IG) → notes plus **conservatrices et stables** ; Egan-Jones, plus réactive au marché.
- **149 bilans (12,1 %)** classés IG par une agence et SG par une autre, presque toujours autour de BBB- / BB+.
- **766 lignes** ont un bilan identique à une autre mais un label différent → **bruit irréductible** : un score parfait est impossible.
- **131 entreprises** changent de catégorie IG / SG sur la période.
- ⚠️ La part brute d'IG par agence (EJ 77 %, Fitch 71 %, S&P 59 %, Moody's 53 %) mélange biais de notation et **effet de couverture** : seule la comparaison sur bilans identiques isole le biais.

### Évolution dans le temps (étape 4)
- 2012-2016 concentrent l'essentiel des données ; 2010 n'a que 41 lignes.
- Part d'IG : 69 % en 2014 → 56 % en 2016. **Énergie : 77 % → 43 %** (chute du pétrole 2014-2016) ; autres secteurs : 69 % → 57 %.
- → La relation ratios → note n'est pas parfaitement stable : **split temporel à tester** en partie 3.

### Extension multiclasse (étape 4)
- Regroupement en 5 niveaux : AAA–AA (602), A (2 036), BBB (2 461), BB (1 425), B et moins (1 281).
- Respecte la frontière IG / SG (niveaux 1-3 = IG, 4-5 = SG) → tâches binaire et multiclasse cohérentes.

### Distributions (étape 5)
- Presque tous les ratios sont **très concentrés avec des queues extrêmes** (axe log nécessaire pour les voir). Seules Gross Margin et Asset Turnover ont une forme « raisonnable ».
- Origine économique : **un ratio dont le dénominateur s'approche de zéro explose** (ex. ROE avec fonds propres quasi nuls). Pas des erreurs de saisie.

### Valeurs extrêmes (étape 5, critère de Tukey à 3 IQR)
- **29 % des lignes** ont au moins une valeur extrême, et ces lignes sont à **51 % SG** (contre 35 % en moyenne) → les extrêmes sont **informatifs**.
- Particulièrement associés au SG : Debt/Equity (72 % SG), LT Debt / Capital (73 %), marges (65-73 %).
- Exceptions : Return On Tangible Equity (35 % SG = la moyenne : extrêmes dus au goodwill, ratio bruité) ; Asset Turnover (89 % IG : distributeurs à forte rotation).
- **Fonds propres négatifs** (Debt/Equity < 0, 299 lignes) : **78 % SG** (contre 33 %). Mais les 22 % d'IG sont très rentables (ROA médian 14,3 % contre 4,8 %) → probablement des rachats d'actions massifs.
- ⚠️ Toute transformation monotone range les fonds propres négatifs à côté des entreprises les moins endettées → **indicateur « fonds propres négatifs » nécessaire**.

### Pouvoir discriminant univarié (étape 5, AUC de Mann-Whitney)
- **Rentabilité nette en tête** : Pre-Tax Margin 0,75, Net Margin 0,73, ROI 0,73, ROA 0,71 ; puis endettement (LT Debt / Capital 0,69 dans le sens « plus endetté = plus risqué »).
- **Current Ratio plus élevé chez les SG** (AUC 0,41 ; médiane 1,47 IG contre 1,68 SG ; 22 % des IG sous 1 contre 14 % des SG). Contre-intuitif → analysé en section 6.4 :
  - **Effet réel, pas seulement un effet de secteur** : net dans NoDur (1,45 vs 2,20), télécoms (0,83 vs 1,52), chimie, distribution ; absent dans informatique, santé, finance, énergie.
  - **Explications** : (1) accès au crédit / motif de précaution : les IG empruntent facilement, les fragiles gardent un coussin (Opler, Pinkowitz, Stulz & Williamson, 1999) ; (2) pouvoir de négociation : les grandes entreprises paient leurs fournisseurs tard → ratio bas = signe de force ; (3) utilities : revenus réglementés, ratio structurellement bas.
  - **Conséquence** : effet faible et dépendant du secteur → interactions (arbres, boosting) et variables relatives au secteur.
- **Ne discriminent presque pas** : cash-flows par action (0,53-0,54, montant par action non comparable entre entreprises), Asset Turnover (0,48).
- ⚠️ p-valeurs trop optimistes (lignes non indépendantes) → on classe par AUC (taille d'effet), pas par p-valeur.

### Corrélations entre ratios (étape 6, Spearman)
- **Redondances fortes** : Operating Margin ≈ EBIT Margin (ρ = 0,999, quasi identiques), ROA ≈ ROI (0,957), chaîne des marges Operating → Pre-Tax → Net (0,91-0,93), EBITDA ↔ Operating (0,88).
- Autres liens notables : ROE ↔ ROI (0,80), LT Debt / Capital ↔ Debt/Equity (0,78), OCF ↔ FCF par action (0,75).
- → Moins de 16 « informations » indépendantes. Risque de **multicolinéarité** pour les modèles linéaires.
- Spearman plutôt que Pearson : calcul sur les rangs → insensible aux valeurs extrêmes.

### Information mutuelle (étape 6) — le piège des doublons
- **Sur les 7 805 lignes** : IM quasi identique pour tous les ratios (0,30-0,35), même les inutiles. Cause : l'estimateur repose sur les plus proches voisins, et chaque point trouve son jumeau exact, de même label.
- **Sur les 2 694 bilans distincts** : IM 4 à 30 fois plus faible, et classement cohérent avec l'AUC (Pre-Tax 0,091, Net 0,088, ROI 0,081, ROA 0,079 ; cash-flows et Current Ratio en queue).
- → **Démonstration concrète, avant toute modélisation, que les doublons faussent les méthodes fondées sur la proximité (KNN compris).** Argument fort pour la soutenance.

### Variables catégorielles (étape 6, V de Cramér)
- **Secteur : V ≈ 0,22**, stable quelle que soit la méthode. Utilities 87 % IG, NoDur 79 % ; télécoms 49 %, Other 52 %.
- **Agence : V = 0,21** (confirme l'étape 4).
- **Piège du code SIC** : SIC à 4 chiffres V = 0,66 en apparence, mais **240 codes pour 686 entreprises (médiane 2 entreprises par code)** → il identifie presque l'entreprise et « mémorise » sa note. Au niveau entreprise : 0,32. SIC à 2 chiffres : 0,44 → 0,30.
- Secteurs = classification Fama-French à 12 industries (glossaire dans le notebook, section 7.2).
- **CIK vs SIC** : le CIK dit *qui* est l'entreprise (identifiant unique → groupes de split, jamais une variable) ; le SIC dit *ce qu'elle fait* (code d'activité hiérarchique : 49xx électricité-gaz-eau → 4941 eau potable).
- **SIC à 4 chiffres = identifiant déguisé** : 108 codes sur 240 ne contiennent qu'une seule entreprise. Avec un split par entreprise, **≈ 20 % des entreprises du test ont un code SIC jamais vu dans le train** (2 % pour le SIC à 2 chiffres, 0 % pour le secteur).

### Réduction de dimension (étape 7, sur les 2 694 bilans, ratios passés par QuantileTransformer + StandardScaler)
- **Pas de dimension dominante** : PC1 = 38 % de la variance, PC2 = 16 % ; 2 composantes = 55 %, 5 = 81 %, **8 composantes pour 90 %**.
- **Sens des axes (loadings)** :
  - **PC1 = rentabilité** (toutes les marges, ROA, ROI, ROE) ;
  - **PC2 = modèle économique** : rotation des actifs et rendements élevés, contre marge brute, EBITDA et endettement élevés (distributeurs « légers » vs industries lourdes en capital) ;
  - **PC3 = endettement et cash-flows** ;
  - **PC4 = cash-flows par action contre endettement et ROE**.
- **Lien avec le risque** : PC1 corrélée à la note (Spearman −0,45) ; PC2 presque pas (−0,17). Les SG sont à gauche (faible rentabilité), avec un **fort recouvrement** au centre.
- **t-SNE** : pas de groupes nets, des zones à dominante IG ou SG et de petits îlots homogènes.
- → **Classes non linéairement séparables** : avantage attendu pour les modèles non linéaires.

### Anomalies (étape 7, Isolation Forest, contamination 5 %)
- 135 bilans atypiques, **60 % SG** (contre 42 % pour les autres).
- Profil : marges et rendements très bas, endettement élevé, marge brute haute → coûts fixes lourds en perte. **Énergie surreprésentée (+17 points)** : cohérent avec la crise pétrolière.
- Situés aux **deux extrémités de PC1** : grandes difficultés d'un côté, rentabilités exceptionnelles de l'autre.

### Clustering (étape 8, KMeans et GMM sur les bilans transformés)
- **Pas de groupes naturels** : silhouette maximale 0,21 (k = 2) puis 0,13-0,16 ; inertie sans coude ; BIC du GMM minimal vers k = 8 (nuage continu approximé par morceaux).
- **Faible accord avec la cible** : ARI ≤ 0,13, NMI < 0,09, pureté 68 % (contre 57 % pour la classe majoritaire) → le clustering ne retrouve pas IG / SG à lui seul.
- **k = 4 retenu pour la lisibilité** (k = 2 coupe juste le long de la rentabilité). Profils :
  - **Cluster 0 — très rentables** (marges et rendements élevés, peu de dette) : 77 % IG ; informatique, santé.
  - **Cluster 1 — modèle « volume »** (forte rotation des actifs, marge brute faible) : 62 % IG ; industrie, distribution.
  - **Cluster 2 — « capital lourd »** (endettement élevé, faible liquidité, faible rotation) : 62 % IG ; utilities, télécoms.
  - **Cluster 3 — peu ou pas rentables** : **25 % IG**, note médiane BB-.
- **Lecture** : les clusters capturent des **modèles économiques** plus que le risque. Clusters 1 et 2 : même part d'IG, ratios très différents → **un même ratio n'a pas le même sens selon le modèle économique**.

## 2. Décisions prises

*Chaque décision : constat chiffré, alternatives envisagées, raison du choix.*

### Gross Margin = 100 % — garder la valeur ✅ (étape 2)
- **Constat** : 145 lignes à exactement 100 %, concentrées dans l'énergie, les utilities et la finance.
- **Alternatives** : laisser tel quel ; remplacer par NaN puis imputer ; ajouter un indicateur « marge brute = 100 % ».
- **Choix** : garder la valeur, et tester l'indicateur en feature engineering, car « ce poste n'existe pas pour cette entreprise » peut être informatif.

### D1 — Split au niveau de l'entreprise (CIK) ✅ (étape 3)
- **Constat** : 89 % des lignes ont un jumeau aux ratios identiques.
- **Risque** : avec un split aléatoire, un bilan vu à l'entraînement réapparaît au test → le modèle « reconnaît » au lieu de généraliser → scores artificiellement gonflés (*data leakage*).
- **Alternatives** : split aléatoire (rejeté : fuite) ; split par nom d'entreprise (rejeté : 430 entreprises ont plusieurs noms) ; dédoublonner en ne gardant qu'une ligne par bilan (possible, mais perd l'information sur l'agence, voir étape 4).
- **Décision** : tous les splits (train / test et folds de CV) se font **par CIK** (`GroupShuffleSplit`, `StratifiedGroupKFold`).
- **À faire en partie 2** : chiffrer le leakage en comparant split aléatoire et split par CIK.

### D2 — Garder toutes les lignes, avec l'agence comme variable ✅ (étape 4)
- **Constat** : à bilan identique, la note dépend en partie de l'agence (biais systématique, tableau ci-dessus).
- **Alternatives** :
  - A. Toutes les lignes, agence en variable ← **pipeline principal**
  - B. Toutes les lignes, sans l'agence ← **pipeline de comparaison**
  - C. Une seule agence (S&P, 2 813 lignes) : cible plus propre, mais 64 % des données perdues.
  - D. Une ligne par bilan avec note consensus : plus de doublons, mais information sur l'agence perdue.
- **Pourquoi A** : maximum de données ; le modèle apprend le décalage entre agences au lieu de le subir comme du bruit.
- **Pourquoi aussi B** : l'écart A vs B mesure ce que l'agence apporte.
- **Usage en screening** : on fixe une agence de référence (ex. S&P) → « comment S&P noterait-elle cette entreprise ? ».
- **Préparation** : les 3 agences marginales (DBRS, JCR, HR Ratings : 53 lignes) regroupées en « Autre ».

### D3 — Métriques orientées « screening » ✅ (étape 4)
- **Constat** : 65 % d'IG → un modèle qui répond toujours « IG » ferait déjà 65 % d'accuracy. Et classer IG une entreprise risquée coûte plus cher à un investisseur qu'une fausse alerte.
- **Décision** : **ROC-AUC** et **PR-AUC** (indépendantes du seuil) comme métriques principales, + **rappel de la classe SG**, F1 et balanced accuracy. **Mêmes métriques pour tous les modèles** (exigence du sujet).
- **Déséquilibre** : découpages stratifiés ; corrections à comparer (pondération des classes, SMOTE **sur le train uniquement**).
- Approche de l'équipe : on teste plusieurs variantes et on compare.

### D4 — Garder les valeurs extrêmes et les traiter dans le pipeline ✅ (étape 5)
- **Constat** : 29 % des lignes ont une valeur extrême, à 51 % SG. Les supprimer retirerait une partie des entreprises en difficulté, celles que l'outil doit détecter.
- **Alternatives comparées** :

| Transformation | Effet | Limites |
|---|---|---|
| Brut + `StandardScaler` | aucun traitement | moyenne et écart-type écrasés par les extrêmes |
| `RobustScaler` (médiane, IQR) | centrage robuste | ne réduit pas les queues |
| Winsorisation 1-99 % | coupe les queues, garde l'ordre | pics aux bornes ; seuil arbitraire |
| Log signé | compresse les queues, garde le signe | bimodalité (révèle les ROE négatifs → piste de feature) |
| Quantile → normale | distribution normale, robuste | ne garde que le rang |

- **Décision** : garder toutes les lignes ; comparer **au moins deux pipelines** : winsorisation + `StandardScaler`, et `QuantileTransformer`, **ajustés sur le train uniquement**, + indicateurs de signe. Les modèles à arbres seront aussi testés sur les ratios bruts (insensibles aux échelles).
- Approche de l'équipe : on teste plusieurs variantes et on compare.

### D5 — Sélection et encodage des variables ✅ (étape 6)
- **Ratios** : aucun supprimé à ce stade. Pour les modèles linéaires, comparer une **version réduite** (une seule de Operating / EBIT Margin, ROA ou ROI…) à la version complète, avec régularisation L1 / L2. Arbres et boosting : toutes les variables (ils tolèrent la redondance).
- **Secteur et agence** : **one-hot** (12 et 5 modalités, raisonnable).
- **SIC à 4 chiffres : exclu** (mémorise l'entreprise → sur-apprentissage).
- **SIC à 2 chiffres** : à tester en *target encoding* calculé dans les folds, en comparaison du simple secteur.
- **Alternatives écartées** : supprimer les ratios peu discriminants (un ratio faible seul peut servir en interaction) ; garder le SIC à 4 chiffres en one-hot (240 colonnes, sur-apprentissage).

### D6 — PCA et anomalies : pistes de pipelines, pas de suppression ✅ (étape 7)
- **PCA** : testée comme **étape de réduction de dimension** dans un pipeline (ajustée sur le train), en comparaison des variables brutes. Intérêt surtout pour les modèles linéaires et KNN (redondance des marges).
- **Anomalies** : **gardées** (cas réels, informatifs pour la classe SG). Pistes : score d'anomalie comme variable (Isolation Forest ajustée sur le train) ; pipeline qui les retire du train seulement, pour mesurer leur impact.
- **Alternatives écartées** : supprimer les anomalies avant le split (perte d'information sur les SG et risque de fuite) ; remplacer les 16 ratios par 2-3 composantes (55 % de variance seulement → trop de perte).

### D7 — Conséquences du clustering pour la modélisation ✅ (étape 8)
- Privilégier des modèles capables de capter des **interactions** (arbres, boosting).
- Tester des variables **relatives au secteur** (ratio − médiane du secteur, calculée sur le train).
- Tester l'**appartenance à un cluster** ou la distance aux centres (KMeans ajusté sur le train) comme variable supplémentaire.

### Table « bilans » pour les analyses non supervisées (étape 3)
- Une ligne par bilan distinct, label majoritaire (ex aequo → SG, par prudence).
- Utilisée pour l'information mutuelle, la PCA et le clustering, que les doublons fausseraient (un bilan noté 10 fois pèserait 10 fois plus).

### Stratégies prévues s'il y avait eu des NaN (étape 2)
- Suppression de lignes ou de colonnes, imputation (médiane, médiane du secteur, KNN), indicateur de valeur manquante. Toujours ajustées sur le train uniquement, dans le pipeline. Sans objet ici.

---

## 3. Questions probables à la soutenance

- Pourquoi charger les données depuis une URL figée sur un commit précis ?
- Vous n'avez aucune valeur manquante : avez-vous vérifié les valeurs « déguisées » ? (oui : zéros, infinis, valeur la plus fréquente → Gross Margin = 100 %)
- Les valeurs négatives sont-elles des erreurs ?
- Pourquoi exclure Corporation, Ticker et CIK des variables explicatives ?
- C'est quoi le data leakage, et pourquoi votre dataset y est-il particulièrement exposé ?
- Pourquoi le CIK plutôt que le nom de l'entreprise pour le split ?
- Pourquoi la répartition IG / SG n'est-elle pas la même sur les lignes et sur les bilans ?
- Pourquoi l'agence comme variable ? N'est-ce pas une forme de triche ? (non : c'est une information connue au moment de la notation, et on fixe l'agence de référence en usage)
- Pourquoi ne pas se fier à la part d'IG par agence pour comparer leur générosité ?
- Pourquoi Egan-Jones note-t-elle plus haut ? (pas le mode de paiement : conservatisme réglementaire de Moody's / S&P)
- Pourquoi ces métriques plutôt que l'accuracy ?
- Pourquoi un modèle ne peut-il pas atteindre 100 % sur ce dataset ?
- Pourquoi ne pas supprimer les valeurs extrêmes ?
- Quelle transformation avez-vous choisie et pourquoi ? (on en compare plusieurs dans des pipelines)
- Pourquoi les arbres n'ont-ils pas besoin de mise à l'échelle ?
- Pourquoi le Current Ratio est-il plus élevé chez les entreprises risquées ? (accès au crédit, pouvoir de négociation, secteurs ; vérifié secteur par secteur en 6.4)
- Vos p-valeurs sont toutes nulles : est-ce fiable ? (non, lignes non indépendantes → on utilise l'AUC)
- Pourquoi Spearman plutôt que Pearson ?
- C'est quoi la multicolinéarité, et pourquoi est-ce un problème ?
- Pourquoi l'information mutuelle est-elle différente sur les lignes et sur les bilans ?
- Pourquoi avoir exclu le code SIC alors qu'il semblait très lié à la cible ? (identifiant déguisé : 108 codes à une seule entreprise, 20 % de codes inconnus en test)
- Quelle différence entre CIK et SIC ?
- Comment se calcule le V de Cramér, et pourquoi l'avoir corrigé ?
- C'est quoi une PCA ? Qu'est-ce qu'un loading ?
- Pourquoi la PCA est-elle faite sur les bilans distincts et pas sur toutes les lignes ?
- Pourquoi transformer les ratios (quantiles) avant la PCA ?
- Que montre le t-SNE que la PCA ne montre pas ? Quelles sont ses limites ?
- Comment fonctionne l'Isolation Forest ? Pourquoi ne pas supprimer les anomalies ?
- Comment avez-vous choisi le nombre de clusters ? Pourquoi 4 et pas 2 ?
- C'est quoi la silhouette, l'ARI, la NMI ?
- Pourquoi le clustering ne retrouve-t-il pas IG / SG ? Qu'en avez-vous tiré ?

---

## 4. Historique

- **Étape 1** — Base du notebook : description du projet et du dataset, travaux existants, imports, chargement des données depuis GitHub.
- **Étape 2** — Métadonnées (types, rôle de chaque colonne, statistiques descriptives) et valeurs manquantes (aucun NaN, valeur sentinelle Gross Margin = 100 %).
- **Étape 3** — Structure des entreprises et doublons : CIK comme identifiant, 89 % de lignes jumelles, décision D1 (split par CIK), table des bilans distincts.
- **Étape 4** — Étude de la cible : équilibre des classes, désaccord entre agences sur bilans identiques, évolution dans le temps, regroupement multiclasse. Décisions D2 (agence en variable, pipelines A et B) et D3 (métriques).
- **Étape 5** — Distributions, valeurs extrêmes, comparaison de 5 transformations, pouvoir discriminant univarié (AUC). Décision D4 (extrêmes gardés, traités dans le pipeline). Zoom sur le Current Ratio par secteur (section 6.4).
- **Étape 6** — Corrélations (Spearman), information mutuelle (piège des doublons), variables catégorielles (V de Cramér, piège du SIC). Décision D5 (sélection et encodage). Zoom CIK / SIC (identifiant déguisé).
- **Étape 7** — PCA (sens économique des axes, 8 composantes pour 90 %), t-SNE, anomalies (Isolation Forest). Décision D6 (PCA en pipeline, anomalies gardées).
- **Étape 8** — Clustering (KMeans, GMM ; silhouette, BIC, ARI, NMI, pureté), profils des 4 clusters, conclusions de l'exploration (D1 à D6 récapitulées). Décision D7. **Partie 1 terminée.**
