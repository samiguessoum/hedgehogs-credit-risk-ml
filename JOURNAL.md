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

- [x] Feature engineering (nouvelles variables à partir des existantes) — section 17
- [x] Nested CV, même nombre de folds pour tous les modèles (5 × 3, mêmes plis — section 11.3)
- [x] Learning curves — section 14.2
- [x] Mêmes métriques pour tous les modèles, comparées sur les mêmes graphiques — section 14.1
- [x] Évaluation finale sur un test set jamais vu (overfitting vs généralisation) — section 15
- [x] Sur/sous-échantillonnage **uniquement sur le train** (après le split) — SMOTE dans le fit, section 18.2
- [x] Pas de modèle linéaire/polynomial non régularisé
- [x] Pas d'arbre de décision seul
- [x] Tester des versions polynomiales des modèles linéaires
- [x] Si on part d'un sous-échantillon du train, réentraîner ensuite sur tout le train — courbes d'apprentissage jusqu'à 100 %, modèles finaux sur tout le train
- [x] Ensembles avancés (boosting, stacking…) — section 13
- [x] Tuning sur plusieurs hyperparamètres significatifs (étape 16 : 6 pour le GB, intervalles continus)
- [x] Graphes en coordonnées parallèles pour le tuning (étape 16, section 19)
- [ ] Recommandé : faire critiquer le travail par un LLM

**Partie 1 (EDA et exploration non supervisée) : terminée (étapes 1 à 8).**
**Partie 2 (modèles de base et d'ensemble) : terminée (étapes 9 à 13).**
**Partie 3 (modèles améliorés, optimisation, analyse) : en cours (étapes 14 à 19).**

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

### Partie 2 — Découpage et data leakage chiffré (étape 9)
- **Fuite résiduelle découverte et corrigée** : 34 jeux de 16 ratios identiques apparaissent sous **plusieurs CIK** (178 lignes, 31 entreprises). Deux cas : même groupe sous deux identifiants (FMC Technologies, Kronos, Southwest Gas…) et **erreurs du dataset** (AAR / Air Products, EMCOR / Edison Mission, Apple / PartnerRe : bilans identiques pour des entreprises sans rapport, noms souvent voisins alphabétiquement → erreur de fusion probable).
  - Correction : **groupes étendus** = CIK reliés par un bilan identique (union-find) → 686 CIK → **670 groupes**. Lignes conservées (on ne sait pas quelle entreprise porte les bons chiffres ; 2,3 % des données).
- **Split train / test par groupe** : 6 250 lignes / 560 entreprises (train) et **1 555 lignes / 126 entreprises (test)**. 0 entreprise et 0 jeu de ratios en commun. Part de SG : 34,7 % / 34,6 %. Agences et années réparties de la même façon.
- **CV imbriquée** : 5 plis externes × 3 plis internes, par entreprise. Plis externes : ~1 250 lignes et ~100-115 entreprises en validation, part de SG stable (34,5-34,7 %).
- **Démonstration du leakage** (CV à 5 plis sur le train, modèles non réglés) :

| | ROC-AUC aléatoire | ROC-AUC par entreprise | Rappel SG aléatoire | Rappel SG par entreprise |
|---|---|---|---|---|
| Random Forest | 0,980 | 0,843 | 89 % | 67 % |
| KNN (k = 5) | 0,947 | 0,756 | 79 % | 56 % |

- → Un découpage aléatoire **surestime les performances de 14 à 28 points**. Le KNN est le plus touché (son plus proche voisin est souvent son propre jumeau). **Argument central pour la soutenance.**
- Niveau de référence : une Random Forest sans réglage atteint **0,84 de ROC-AUC** en validation honnête (variation de ± 0,02 selon le découpage).
- **Essai rapide (hors notebook)** : régression logistique, Random Forest et gradient boosting plafonnent tous vers **0,82-0,83** de ROC-AUC sans feature engineering, et le réglage des hyperparamètres ne fait gagner que quelques millièmes. Une régression logistique sur les seuls ratios atteint déjà 0,816 → le signal (rentabilité + endettement) est fort et simple. **Les gains viendront plutôt des variables (feature engineering) que du réglage.**

### Partie 2 — Modèles de base (étape 10, CV imbriquée 5 × 3, ROC-AUC optimisée)

| Modèle | ROC-AUC | PR-AUC | Rappel SG | Précision SG | Écart train − val |
|---|---|---|---|---|---|
| Dummy (prior) | 0,500 | 0,347 | 0 % | — | — |
| Régression logistique (L1/L2) | 0,825 ± 0,028 | 0,745 | 60 % | 76 % | +0,04 |
| Rég. log. polynomiale (degré 2) | 0,829 ± 0,023 | 0,764 | 60 % | 78 % | +0,07 |
| **SVM (RBF)** | **0,848 ± 0,036** | **0,780** | 59 % | 77 % | +0,05 |
| KNN | 0,831 ± 0,032 | 0,745 | 47 % | 80 % | +0,15 (non interprétable) |

- **Modèles très proches** : écarts (≤ 0,02) du même ordre que la variation entre plis (± 0,02-0,04) → **le choix du modèle compte peu**, le signal est déjà capté par un modèle linéaire.
- **SVM légèrement devant** (frontière courbe → non-linéarités : fonds propres négatifs, effets sectoriels).
- **Polynomial** : +0,004 seulement, mais écart train − val doublé → plus de liberté que les données ne le justifient.
- **Hyperparamètres** : L1 presque toujours choisie (redondance des ratios) ; `C` de la rég. log. instable (0,1 à 100) → performance insensible à la régularisation ; SVM `C` faible (0,25-0,5) → frontière lisse, cohérente avec une cible bruitée ; KNN k = 121 → lissage fort ; winsor et quantile se valent (winsor un peu plus souvent).
- **Méthode** : grilles élargies jusqu'à ce que les valeurs retenues ne soient plus en bord de grille (SVM `C` et KNN `k` étaient en bord au premier essai).
- **KNN** : écart train − val de 0,15 non interprétable (pondération `distance` → chaque point d'entraînement est son propre voisin).
- **Point faible commun : rappel SG ≈ 60 %** (47 % KNN) au seuil 0,5 : bonne précision mais trop peu d'entreprises signalées (déséquilibre 65/35). → Pondération des classes et seuil de décision en partie 3.
- **Référence à battre par les ensembles : 0,848 de ROC-AUC (SVM).**
- Temps de calcul de la section 12 : ≈ 4 min sur 2 cœurs (comparable à Colab gratuit).

### Partie 2 — Modèles d'ensemble (étape 11, même procédure de CV imbriquée)

| Modèle | ROC-AUC | PR-AUC | Rappel SG | Précision SG | Écart train − val |
|---|---|---|---|---|---|
| SVM (meilleur modèle de base) | 0,848 ± 0,036 | 0,780 | 59 % | 77 % | +0,05 |
| Random Forest | 0,843 ± 0,031 | 0,764 | 64 % | 72 % | +0,15 |
| Gradient boosting (HistGB) | 0,839 ± 0,024 | 0,764 | 64 % | 71 % | +0,14 |
| **Stacking** (LR + SVM + RF + GB) | **0,855 ± 0,032** | **0,789** | 62 % | 75 % | +0,12 |

- **Stacking = meilleur modèle à ce stade**, mais +0,007 seulement sur le SVM pour ± 0,03 de variation entre plis → **différence non significative**. Tous les bons modèles plafonnent vers **0,84-0,86**.
- **Les ensembles d'arbres ne battent pas le SVM** : le signal (rentabilité, endettement) est assez simple pour un modèle plus simple.
- **Arbres plus « prudents » au seuil 0,5** (rappel 64 % vs 59-60 %) : réglage du compromis rappel / précision différent, pas meilleure discrimination (ROC-AUC équivalente) → seuil en partie 3.
- **Écart train − val des arbres (+0,12-0,15) non alarmant** : une RF mémorise par construction (ROC-AUC train 0,995) mais généralise aussi bien que les autres → courbes d'apprentissage à l'étape 12.
- **Poids dans le stacking** (version corrigée, étape 12) : RF 0,54, SVM 0,52, rég. log. 0,39, GB 0,24. GB le moins pondéré car **corrélé à 0,92 avec la RF** (mêmes erreurs) ; la rég. log. garde du poids car la **moins corrélée** aux arbres (0,77). → Le stacking valorise la **diversité des erreurs**, pas seulement la performance individuelle.
- **Réglages finaux des modèles de base** (grille sur tout le train, CV par entreprise) : LR L1 `C` = 0,1 (winsor) ; SVM `C` = 0,25, `gamma` = scale (quantile) ; RF `max_features` = 0,1 ; GB 0,03 × 150 arbres, 8 feuilles, ≥ 80 lignes par feuille.
- **Hyperparamètres** : RF `max_features` 0,10-0,15 et feuilles ≥ 5 lignes (forêt robuste au bruit) ; GB : taux d'apprentissage et nombre d'arbres varient ensemble (0,01 × 300, 0,03 × 150, 0,1 × 75 : réglages équivalents). Grilles élargies après un premier essai en bord de grille.
- Temps : section 13 ≈ 10 min sur 2 cœurs (RF ≈ 4 min, réglage final ≈ 3 min).

### ⚠️ Erreur corrigée à l'étape 12 — réglages des modèles de base du stacking (→ notebook « ce qui n'a pas marché »)
- **Première version** : pour chaque hyperparamètre, on prenait la valeur **la plus fréquente** sur les 5 plis. Pour le gradient boosting, ça donnait `learning_rate = 0,01` avec `max_iter = 150`, une combinaison **qu'aucun pli n'avait choisie** (0,01 n'était retenu qu'avec 300 arbres) → boosting **sous-entraîné** (ROC-AUC 0,829 sur la courbe d'apprentissage au lieu de 0,848) et poids faussé dans le stacking (0,13).
- **Détection** : incohérence entre la courbe d'apprentissage du GB (0,829) et sa CV imbriquée (0,839).
- **Correction** : (1) pour **évaluer** le stacking, chaque modèle de base reprend sur chaque pli externe les réglages choisis par **sa** boucle interne sur ce pli ; (2) pour le **modèle final**, réglages choisis par grille sur tout le train (CV par entreprise).
- **Leçon** : des hyperparamètres qui vont **par paires** (taux d'apprentissage × nombre d'arbres) ne se choisissent pas séparément.

### Partie 2 — Comparaison, courbes d'apprentissage, effet de l'agence (étape 12)
- **Comparaison appariée** (mêmes plis) : le stacking bat RF, GB, rég. log. et KNN sur **5 plis sur 5** (+0,01 à +0,03) et le SVM sur **4 sur 5** (+0,007 en moyenne, entre −0,002 et +0,013). Avantage **petit mais régulier**.
- **Métriques à seuil** : différences surtout dues à la position du seuil 0,5 (arbres : rappel 64 % ; KNN : 47 % mais précision 80 %). F1 ≈ 0,67-0,68 et balanced accuracy ≈ 0,75 pour tous les bons modèles.
- **Courbes ROC / précision-rappel** (scores hors pli) : courbes quasi superposées ; stacking légèrement au-dessus partout.
- **Courbes d'apprentissage** : validation **quasi plate** (gain de 0,001 à 0,005 de 55 % à 100 % du train) → **plus de données n'aiderait pas** ; les modèles plafonnent à cause des **variables**. Rég. log. : écart train − val 0,03 (limitée par sa forme linéaire). RF : écart 0,16 mais validation au niveau du SVM (pas de sur-apprentissage préjudiciable). GB : progresse au début puis plafonne (0,848).
- **Effet de l'agence (pipeline A vs B, CV imbriquée)** : +0,003 à +0,010 de ROC-AUC, +0,7 à +2 points de rappel SG ; SVM meilleur avec l'agence sur 5 plis sur 5, rég. log. et GB sur 3 sur 5. **Gain réel mais petit** : le biais d'agence ne joue que près de la frontière BBB- / BB+.
- **Le modèle a appris le biais de chaque agence** : mêmes 6 250 bilans notés « comme si » par une seule agence → **25,6 % classés SG (Egan-Jones), 28,1 % (Fitch), 31,3 % (S&P), 33,4 % (Moody's)**. Même ordre que l'exploration (section 5.2). Près de 8 points d'écart entre EJ et Moody's → l'outil doit fixer une **agence de référence** (S&P par défaut).
- **Modèle retenu pour le test** : **stacking** (modèle principal), avec le **SVM** comme référence plus simple.
- Temps : notebook complet ≈ 19 min sur 2 cœurs.

### Partie 2 — Évaluation finale sur le test (étape 13, test ouvert une seule fois)

| Modèle | ROC-AUC entraînement | ROC-AUC CV | **ROC-AUC test** | PR-AUC test | Rappel SG test | Précision SG test |
|---|---|---|---|---|---|---|
| **Stacking (retenu)** | 0,990 | 0,855 | **0,820** | 0,713 | 61 % | 67 % |
| SVM (référence) | 0,921 | 0,848 | 0,811 | 0,699 | 58 % | 66 % |
| Gradient boosting | 0,929 | 0,839 | 0,828 | 0,737 | 61 % | 70 % |
| Rég. log. polynomiale | 0,893 | 0,829 | 0,826 | 0,713 | 53 % | 70 % |
| Random Forest | 1,000 | 0,843 | 0,803 | 0,694 | 67 % | 61 % |
| Régression logistique | 0,854 | 0,825 | 0,796 | 0,652 | 60 % | 60 % |
| KNN | 1,000 | 0,831 | 0,783 | 0,624 | 41 % | 62 % |

- **Performance réelle du modèle retenu : 0,820** (IC 95 % bootstrap par entreprise : **[0,758 ; 0,874]**, l'estimation de CV 0,855 est dedans). 326 SG détectées sur 538 (61 %) ; 162 fausses alertes sur 1 017 IG (16 %).
- **Tous les modèles baissent un peu** (−0,003 à −0,047), y compris la rég. log. → test un peu plus difficile pour tous (variabilité d'échantillon avec 126 entreprises), pas de fuite ni de sur-apprentissage de la sélection.
- **Sur-apprentissage vs généralisation** : les modèles qui mémorisent (RF, KNN : train = 1,000) perdent le plus (−0,04 / −0,05) ; les plus contraints (polynomial `C` = 0,01, GB 8 feuilles / 80 lignes) le moins (−0,003 / −0,011).
- **GB et polynomial légèrement devant le stacking sur le test** → on ne change **pas** de modèle (choix fait avant le test ; écarts < 0,01 ≪ incertitude ± 0,06).
- **Stacking vs SVM** : meilleur dans 80 % des tirages bootstrap, mais l'IC de l'écart [−0,013 ; +0,030] contient 0.
- **Où sont les erreurs** : 37 % d'erreur à ≤ 2 crans de la frontière (BBB à BB) contre 17 % plus loin ; 52 % des erreurs sur un tiers des lignes. **Asymétrie** : BB+ 55 %, BB 66 %, BB- 50 % d'erreur contre BBB- 35 % → le modèle classe IG des SG modérées (penche vers la classe majoritaire) → **seuil et pondération en partie 3**.
- **Par agence** : Egan-Jones 17 % d'erreur mais rappel SG 50 % (SG plus proches de la frontière) ; S&P rappel 67 %. **Par secteur** : utilities 6 %, finance 13 % ; chimie 43 %, distribution 35 %, Other 34 % → variables relatives au secteur.

### Partie 3 — Règle méthodologique
- Le test a déjà été ouvert une fois (étape 13). En partie 3, **toutes les comparaisons se font en CV sur le train** (mêmes plis). Le test ne sera rouvert **qu'une fois**, à la fin, pour le modèle final amélioré, en le signalant comme un second regard.

### ⚠️ Anomalie du dataset : les colonnes « cash-flow par action » (étape 14)
- **40 % des lignes** ont un OCF par action **négatif**, dont **un tiers des lignes bénéficiaires** (87 % des lignes ont une marge nette > 0). Médiane de l'OCF par action : 0,24 $. FCF < 0 : 46 % des lignes.
- Or l'OCF est normalement **supérieur** au bénéfice (il réintègre les amortissements) : une grande entreprise notée et bénéficiaire a presque toujours un OCF positif.
- → Ces colonnes ne mesurent **probablement pas le niveau** du cash-flow par action (plutôt une **variation** annuelle ?). Cohérent avec leur AUC univariée de 0,53.
- **Décision** : gardées telles quelles comme variables, mais **aucune variable construite** à partir d'elles (un ratio FCF / OCF n'aurait pas de sens). Limite du dataset à signaler.

### Partie 3 — Feature engineering (étape 14)
- **11 nouvelles variables** : 4 indicateurs (fonds propres négatifs, perte, ROE négatif, marge brute = 100 %), 1 ratio construit (écart marge brute − marge opérationnelle), 6 **centiles sectoriels** (ROA, marge nette, LT Debt / Capital, Current Ratio, EBITDA Margin, Asset Turnover ; distribution sectorielle apprise **sur le pli d'entraînement**), + variante **réduite** (sans EBIT Margin ni ROI).
- **Pouvoir univarié** : indicateurs très ciblés (90 % de SG si fonds propres négatifs, 78 % si perte) mais **rares** (3-13 % des lignes) → AUC 0,54-0,62. Marge brute 100 % : AUC 0,50 (inutile). Centiles sectoriels de rentabilité / endettement : |AUC − 0,5| ≈ 0,21-0,23, **comme les ratios bruts**.
- **Ablation** (hyperparamètres fixes, mêmes plis ; écarts toujours < 0,011) :

| Modèle | A (partie 2) | B + indicateurs | C + ratio construit | D + centiles sectoriels | E réduite |
|---|---|---|---|---|---|
| Rég. log. | 0,829 | 0,829 | 0,829 | **0,833** | 0,829 |
| SVM | 0,850 | 0,851 | 0,848 | 0,848 | 0,847 |
| RF | 0,843 | 0,843 | 0,838 | 0,842 | 0,842 |
| GB | 0,847 | 0,847 | 0,844 | 0,841 | 0,836 |

- **CV imbriquée avec la variante D** : rég. log. **0,825 → 0,832 (+0,007, 5 plis sur 5**, +0,012 PR-AUC, +1,4 pt de rappel) ; SVM +0,001 (3/5), RF −0,001 (1/5), GB +0,003 (3/5) → **bruit**.
- **Pourquoi c'est logique** : un indicateur n'est qu'un **seuil** et un centile sectoriel une **interaction** ratio × secteur ; arbres et SVM à noyau les construisent **eux-mêmes**. Seul le modèle linéaire, qui ne peut ni créer de seuil ni combiner, en profite.
- **Conclusion** : les modèles non linéaires extraient déjà l'essentiel de l'information des 16 ratios. Pour progresser nettement, il faudrait des **informations nouvelles** (historique, taille, marché), absentes du dataset. **Résultat négatif mais instructif**, à présenter comme tel.
- Temps : section 17 ≈ 11 min (CV imbriquée des 4 modèles enrichis ≈ 10 min). Notebook complet ≈ 31 min.

### Partie 3 — Déséquilibre des classes et seuil de décision (étape 15)
- **Rééquilibrage pendant l'entraînement** (mêmes hyperparamètres, mêmes plis, seuil 0,5) :

| Modèle | Rappel SG sans → pondération / SMOTE | Précision SG sans → pondération | ROC-AUC |
|---|---|---|---|
| Rég. log. + FE | 60 % → 73 % / 72 % | 75 % → 66 % | 0,833 inchangée |
| SVM | 62 % → 74 % / 74 % | 77 % → 68 % | 0,850 inchangée |
| GB | 63 % → 72 % / 70 % | 73 % → 65 % | 0,847 → 0,842 |

- → **Rééquilibrer ≈ déplacer le seuil** : +10 à +13 pts de rappel, −9 pts de précision, ROC-AUC inchangée. **Pondération ≈ SMOTE** (SMOTE fabrique souvent des points proches de jumelles) → pondération préférable (plus simple, pas de données artificielles).
- **SMOTE implémenté à la main** (`SMOTESimple`, imbalanced-learn non installable ici), appliqué **dans le fit uniquement** → jamais sur validation / test (exigence du sujet).
- **Seuil choisi selon le coût des erreurs** (hypothèse : SG ratée = 3 × fausse alerte), **en CV imbriquée** (seuil choisi sur les scores hors pli internes, appliqué au pli externe), sur le **stacking final** (rég. log. + FE, SVM, RF, GB) :

| | Seuil 0,5 | Seuil optimisé (coût 3:1) |
|---|---|---|
| Seuil | 0,50 | **0,21** (0,16-0,24 selon le pli) |
| Rappel SG | 62 % | **86 %** |
| Précision SG | 74 % | 59 % |
| F1 SG | ≈ 0,68 | 0,70 |
| Balanced accuracy | ≈ 0,75 | 0,77 |
| Coût moyen | 0,47 | **0,35 (−24 %)** |
| ROC-AUC / PR-AUC | 0,856 / 0,789 | inchangées |

- Seuil **stable** d'un pli à l'autre ; proche du seuil théorique 1/(1+3) = 0,25 → scores bien calibrés.
- **Sensibilité au ratio de coût** (scores hors pli regroupés) : ratio 1 → seuil 0,43, 68 % détectées, 33 % signalées ; ratio 2 → 0,29, 78 %, 42 % ; **ratio 3 → 0,20, 85 %, 51 %** ; ratio 5 → 0,16, 90 %, 60 %. Gains qui ralentissent au-delà de 3.
- **Usage recommandé** : seuil **0,20** → l'outil signale ~la moitié des entreprises, qui contient **85 % des SG**. Une SG ratée sur sept.
- Temps : section 18 ≈ 2 min. Notebook complet ≈ 32 min.

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

### D8 — Classe positive = Speculative Grade ✅ (étape 9)
- Cible codée **SG = 1**, IG = 0 (inverse de `Binary Rating`).
- **Pourquoi** : la classe positive est celle qu'on veut détecter. Rappel, précision, F1 et PR-AUC de scikit-learn portent ainsi directement sur les entreprises risquées.

### D9 — Jeu de test : 20 % des entreprises, figé et sous clé ✅ (étape 9)
- `StratifiedGroupKFold(5)`, premier pli = test ; graine fixée (`SEED = 42`).
- **Utilisé une seule fois**, à la fin de la partie 2. Toute sélection (modèles, hyperparamètres, transformations) se fait sur le train.
- **Alternative écartée** : choisir les modèles en regardant le test → le test ne serait plus « jamais vu » (sur-apprentissage du test).

### D1 bis — Groupes étendus pour le découpage ✅ (étape 9)
- Les découpages se font par **groupe** = CIK reliés par au moins un bilan identique (686 CIK → 670 groupes), et non par CIK seul.
- **Pourquoi** : 34 bilans identiques sont rattachés à des CIK différents ; un découpage par CIK pourrait mettre le même bilan dans le train et dans le test.

### D10 — Validation croisée imbriquée 5 × 3, mêmes plis pour tous les modèles ✅ (étape 9)
- **Boucle interne (3 plis)** : choix des hyperparamètres, en optimisant la **ROC-AUC**. **Boucle externe (5 plis)** : estimation de la performance.
- Plis calculés **une seule fois** et réutilisés pour tous les modèles (exigence du sujet) ; par entreprise à tous les niveaux.
- **Pourquoi imbriquée** : utiliser la même CV pour régler et pour évaluer donnerait un score optimiste (on retiendrait la configuration qui a eu de la chance sur ces plis).
- **Pourquoi la ROC-AUC pour régler** : indépendante du seuil, elle juge la qualité du classement par niveau de risque ; les autres métriques sont rapportées pour tous les modèles.

### D11 — Pipelines : prétraitement choisi comme un hyperparamètre ✅ (étape 10)
- Chaque modèle est un `Pipeline` (prétraitement + modèle) : toutes les transformations sont ajustées **dans chaque pli, sur l'entraînement uniquement** → pas de fuite.
- Ratios : **winsor 1-99 % + StandardScaler** ou **QuantileTransformer**, choisi par la boucle interne (au lieu d'un choix arbitraire a priori). Catégorielles : one-hot.
- `Winsorizer` écrit comme un transformateur scikit-learn (percentiles appris au `fit`).
- **Alternative écartée** : transformer les données une fois pour toutes avant la CV (fuite : les percentiles verraient la validation).

### D12 — Modèles de base retenus et grilles ✅ (étape 10)
- Dummy (plancher), régression logistique L1/L2 (`C` de 0,01 à 100), version polynomiale de degré 2 sur les ratios seulement (`C` de 0,001 à 1), SVM RBF (`C`, `gamma`), KNN (`k` de 31 à 301, pondération).
- Pas de modèle linéaire non régularisé, pas d'arbre seul (exigences du sujet).
- SVM : score = distance à la frontière (pas de probabilité) → suffit pour ROC-AUC et PR-AUC.

### D13 — Modèles d'ensemble et précautions anti-fuite ✅ (étape 11)
- **Random Forest** (bagging, 300 arbres ; `max_features`, `min_samples_leaf`), **gradient boosting** (`HistGradientBoostingClassifier` ; `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`), **stacking** (méta-modèle = régression logistique sur les scores de LR, SVM, RF, GB réglés avec leurs hyperparamètres les plus fréquents).
- **Ratios passés bruts aux arbres** : seul l'ordre compte, la mise à l'échelle n'apporte rien.
- **Early stopping du gradient boosting désactivé** : sa validation interne est tirée par ligne → jumeaux → arrêt trop tardif. Nombre d'arbres réglé par la CV par entreprise.
- **Stacking réécrit (`StackingParGroupe`)** : le `StackingClassifier` de scikit-learn produit ses scores hors pli en découpant par ligne → fuite. Notre version découpe **par groupe d'entreprises**.
- **KNN exclu du stacking** : moins bon et redondant avec les autres.
- **Alternative écartée** : XGBoost / LightGBM (non installables dans notre environnement de travail ; `HistGradientBoosting` repose sur le même principe).

### D14 — Choix du modèle final et de la méthode de réglage ✅ (étape 12)
- **Modèle principal : stacking** (meilleur sur toutes les métriques sans seuil, devant les autres sur 4-5 plis sur 5). **Référence : SVM** (0,007 derrière, bien plus simple).
- **Réglage du modèle final** : recherche par grille sur tout le train (CV par entreprise) ; l'évaluation par CV imbriquée reste celle de référence.
- **Agence gardée (D2 confirmée)** : petit gain de performance, et le modèle apprend le biais de chaque agence au lieu de le confondre avec le risque.
- **Partie 3** : les gains viendront des variables (courbes d'apprentissage plates) et du seuil / déséquilibre (rappel SG ≈ 62 %).

### D15 — Évaluation finale : une seule ouverture du test, pas de re-sélection ✅ (étape 13)
- Modèles finaux entraînés sur **tout le train** avec leurs réglages finaux (grille sur le train, CV par entreprise), puis appliqués **une fois** au test.
- **Pas de changement de modèle** après le test, même si deux modèles y font légèrement mieux : sinon le test devient un jeu de validation.
- **Incertitude du test** quantifiée par **bootstrap par entreprise** (1 000 tirages) plutôt que par ligne (lignes non indépendantes).

### D16 — Nouvelles variables uniquement pour la régression logistique ✅ (étape 14)
- Variante retenue : **D** (indicateurs + écart de marges + centiles sectoriels), **pour la régression logistique seulement** (gain régulier 5/5).
- SVM, RF, GB : variables de la partie 2 (aucun gain, calculs plus longs, modèles moins lisibles).
- Dans le stacking final : rég. log. **avec** les nouvelles variables, autres modèles **sans**.
- **Alternatives écartées** : imposer les nouvelles variables à tous les modèles (aucun gain) ; version réduite (dégrade le GB de −0,011) ; variables construites à partir des cash-flows (colonnes douteuses).
- Pour le notebook « ce qui n'a pas marché » : indicateur marge brute = 100 %, écart de marges, version réduite, nouvelles variables pour les modèles non linéaires.

### D17 — Seuil de décision choisi selon le coût des erreurs ✅ (étape 15)
- **Hypothèse de coût : SG ratée = 3 × fausse alerte** (investisseur : une perte en capital coûte plus qu'une analyse de trop). Ratio **paramétrable** : sensibilité étudiée de 1 à 5.
- **Seuil retenu : ≈ 0,20**, choisi en CV imbriquée (jamais sur les données évaluées).
- **Pondération des classes préférée à SMOTE** (résultats équivalents, plus simple) ; mais le seuil explicite est le levier principal, car rééquilibrer revient à déplacer le seuil.
- **Alternatives écartées** : seuil 0,5 (suppose des erreurs de même coût) ; choisir le seuil sur les données d'évaluation (fuite) ; SMOTE avant le split (fuite, interdit par le sujet).

### D18 — Recherche aléatoire sur intervalles pour le gradient boosting ✅ (étape 16)
- **Méthode** : `RandomizedSearchCV`, lois log-uniformes pour les hyperparamètres qui agissent par ordre de grandeur (Bergstra et Bengio, 2012). Exploration (100 tirages GB, 60 SVM) pour comprendre, puis **CV imbriquée** (40 et 30 tirages par pli) pour l'estimation honnête.
- **GB : +0,009 de ROC-AUC** (0,839 → 0,848, 4 plis sur 5). Hyperparamètres qui comptent : `learning_rate` (optimum 0,012 à 0,023, forme en cloche) et `min_samples_leaf` (petites feuilles meilleures, ρ = −0,41). `max_iter` se compense avec le pas ; `l2_regularization` sans effet ; `max_features` faible (0,2 à 0,4) prometteur.
- **SVM : aucun gain** (0,848 → 0,845) : `gamma="scale"` était déjà dans la bonne zone (0,009 à 0,05).
- **Malédiction du gagnant observée** : en exploration le SVM semblait battre la grille (0,851), en CV imbriquée non. Justifie de ne comparer qu'en CV imbriquée.
- **Décision** : GB réglé par recherche aléatoire dans le modèle final ; SVM garde la grille.
- **Alternatives écartées** : grille plus fine (coût exponentiel avec 6 hyperparamètres) ; optimisation bayésienne (Optuna, non disponible et gain marginal attendu vu la convergence en ~20 tirages).

### D19 — Explicabilité par méthodes agnostiques, raisonnement par famille ✅ (étape 17)
- **Méthodes** : importance par permutation **en CV** (jamais sur le test), variable par variable **et par famille** (même permutation pour toutes les colonnes d'une famille) ; dépendance partielle + ICE (stacking entraîné sur tout le train, usage descriptif) ; effet agence / secteur par substitution ; explications locales par **neutralisation** des familles (valeur de référence = médiane du train).
- **Résultats** : endettement (8,3 points de ROC-AUC) et marges (7,0) dominent pour les 4 modèles ; puis rentabilité, liquidité, secteur, agence ; cash-flows et efficacité ≈ 0. Effets dans le sens attendu, seuil de rentabilité autour de 0, compensation dette / rentabilité. À ratios égaux : agence ±8 points (Egan-Jones 0,30 → Moody's 0,39), secteur ±16 points (utilities 0,30 → télécoms 0,46).
- **Point de méthode** : la permutation individuelle sous-estime les variables redondantes (Operating Margin ≈ 0 car ≈ EBIT Margin ; dette : 4,9 en individuel contre 8,3 en famille).
- **Alternatives écartées** : `feature_importances_` des arbres (calculée sur le train, biaisée) ; coefficients (illisibles pour un stacking) ; SHAP / KernelSHAP (très coûteux sur un stacking de 4 modèles, bibliothèque non disponible partout) ; permutation sur le test (évitée : le test reste réservé au regard final).

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

**Partie 2**
- Comment avez-vous découpé train et test ? Pourquoi 20 % ? Comment avez-vous vérifié qu'il n'y a pas de fuite ?
- De combien le data leakage aurait-il faussé vos résultats ? (14 à 28 points ; démonstration section 11.5)
- Le découpage par CIK suffit-il ? (non : 34 bilans identiques sous des CIK différents, dont des erreurs du dataset → groupes étendus)
- 0,84 de ROC-AUC, c'est bien ? Pourquoi pas plus ? (bon niveau ; cible bruitée : 12 % de désaccord entre agences ; les agences utilisent des informations absentes des ratios)
- Pourquoi le KNN est-il plus sensible au leakage que la Random Forest ?
- C'est quoi une validation croisée imbriquée ? Pourquoi deux boucles ?
- Pourquoi SG comme classe positive ?
- Pourquoi optimiser la ROC-AUC plutôt que l'accuracy ou le F1 ?
- C'est quoi un pipeline, et pourquoi est-ce indispensable ici ?
- Pourquoi le prétraitement est-il traité comme un hyperparamètre ?
- C'est quoi la régularisation ? Différence entre L1 et L2 ?
- Pourquoi la version polynomiale n'apporte-t-elle presque rien ?
- Comment savez-vous que vos grilles d'hyperparamètres sont assez larges ? (valeurs retenues hors des bords)
- Les modèles sont très proches : lequel choisir, et sur quel critère ?
- Pourquoi le rappel SG est-il si bas alors que la ROC-AUC est bonne ? (seuil 0,5 + déséquilibre)
- Pourquoi l'écart train − validation du KNN n'est-il pas interprétable ?
- C'est quoi le bagging, le boosting, le stacking ? Quelle différence ?
- Pourquoi la Random Forest ne fait-elle pas mieux que le SVM ?
- Pourquoi avoir désactivé l'early stopping du gradient boosting ?
- Pourquoi avoir réécrit le stacking au lieu d'utiliser celui de scikit-learn ?
- Pourquoi le gradient boosting a-t-il si peu de poids dans le stacking alors qu'il est aussi bon que la RF ?
- Votre meilleur modèle est-il vraiment meilleur que les autres ? (non significatif : +0,007 pour ± 0,03)
- La Random Forest a une ROC-AUC d'entraînement de 0,995 : sur-apprend-elle ?
- Comment avez-vous comparé les modèles entre eux ? (mêmes plis → comparaison pli par pli)
- Le stacking est-il vraiment meilleur que le SVM ? (4 plis sur 5, +0,007 : régulier mais faible)
- Que montrent vos courbes d'apprentissage ? Faudrait-il plus de données ? (non : plateau → variables)
- Que vous apporte l'agence ? (+0,01 de ROC-AUC, et le modèle retrouve seul l'ordre de générosité des agences)
- Pourquoi ne pas avoir pris les hyperparamètres « les plus fréquents » pour le stacking ? (paires d'hyperparamètres : erreur détectée et corrigée)
- Quel modèle retenez-vous et pourquoi ?
- Quelle est la performance réelle de votre modèle ? (0,82 sur le test, IC 95 % [0,76 ; 0,87])
- Pourquoi le score de test est-il plus bas que la validation croisée ? (variabilité : 126 entreprises ; baisse générale à tous les modèles ; IC contient l'estimation de CV)
- Le gradient boosting fait mieux que le stacking sur le test : pourquoi ne pas le choisir ? (choix fait avant le test)
- C'est quoi un bootstrap ? Pourquoi par entreprise ?
- Où votre modèle se trompe-t-il ? (frontière BBB-/BB+, asymétrie vers IG, certains secteurs)
- Votre modèle sur-apprend-il ? (comparaison train / CV / test ; les modèles contraints généralisent mieux)

**Partie 3**
- Quelles variables avez-vous créées, et pourquoi chacune ?
- Pourquoi vos nouvelles variables n'améliorent-elles pas les arbres ? (seuils et interactions déjà construits par les arbres)
- Pourquoi la régression logistique en profite-t-elle ? (ne peut ni créer de seuil ni combiner avec le secteur)
- Comment calculez-vous les centiles sectoriels sans fuite ? (distribution apprise sur le pli d'entraînement, dans le pipeline)
- Que vous a appris l'étude par ajout successif (ablation) ?
- Les colonnes de cash-flow sont-elles fiables ? (non : 40 % d'OCF négatifs, dont des entreprises bénéficiaires)
- Comment pourrait-on dépasser 0,85 ? (informations nouvelles : historique, taille, données de marché)
- C'est quoi le seuil de décision ? Pourquoi pas 0,5 ?
- Pourquoi un coût de 3:1 ? (hypothèse métier, sensibilité étudiée de 1 à 5)
- Comment avez-vous choisi le seuil sans fuite ? (CV imbriquée : seuil choisi sur les plis internes)
- Pondération ou SMOTE ? (équivalents ; rééquilibrer ≈ déplacer le seuil ; pondération plus simple)
- Pourquoi la ROC-AUC ne change-t-elle pas quand on rééquilibre ? (le classement reste le même)
- Comment avez-vous évité que SMOTE crée une fuite ? (appliqué dans le fit, donc sur le train uniquement)
- Combien d'entreprises risquées votre outil détecte-t-il ? (86 % au seuil 0,21, avec 59 % de précision)
- Pourquoi une recherche aléatoire plutôt qu'une grille ? (peu d'hyperparamètres comptent ; à budget égal, plus de valeurs distinctes testées par hyperparamètre)
- Pourquoi une loi log-uniforme ? (hyperparamètres qui agissent par ordre de grandeur)
- Comment lire vos graphes en coordonnées parallèles ?
- Le meilleur score de l'exploration n'est-il pas optimiste ? (si : malédiction du gagnant, d'où la CV imbriquée ; exemple du SVM)
- Pourquoi les hyperparamètres choisis changent-ils d'un pli à l'autre ? (compensation pas × nombre d'arbres : performances équivalentes)
- Pourquoi l'importance par permutation plutôt que feature_importances_ ? (calculée sur des données non vues, mesure la dépendance de la performance)
- Operating Margin a une importance nulle : faut-il la supprimer ? (non : redondante avec EBIT Margin, ρ = 0,999 ; l'importance mesure l'apport EN PLUS des autres)
- Pourquoi l'endettement passe-t-il devant la rentabilité alors que la rentabilité discriminait mieux seule ? (rentabilité répartie sur 10 ratios corrélés ; dette = information complémentaire)
- Pourquoi l'EBITDA Margin augmente-t-elle le risque ? (à marge nette fixée : amortissements et intérêts lourds)
- Le modèle utilise l'agence : est-ce gênant ? (effet de 8 points à ratios égaux, cohérent avec la sévérité des agences ; ajustement, pas raccourci)
- Pourquoi le modèle a-t-il raté GameStop ? (aucune dette, rentable : la note reflète un risque d'activité absent des ratios)

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
- **Étape 9** — Partie 2, préparation : variables et cible (SG = 1), split train / test par entreprise, CV imbriquée 5 × 3, métriques communes, démonstration chiffrée du data leakage, correction d'une fuite résiduelle (bilans identiques sous plusieurs CIK). Décisions D1 bis, D8, D9, D10.
- **Étape 10** — Pipelines de prétraitement (winsor / quantile comme hyperparamètre), procédure de CV imbriquée commune, 5 modèles de base (Dummy, rég. log. L1/L2, polynomiale, SVM RBF, KNN), bilan. Grilles élargies après détection de valeurs en bord de grille. Décisions D11, D12.
- **Étape 11** — Modèles d'ensemble : Random Forest, gradient boosting, stacking par groupe d'entreprises. Stacking en tête (0,855) mais non significativement. Grilles RF et GB élargies (valeurs en bord au premier essai). Décision D13.
- **Étape 12** — Comparaison des 8 modèles (6 métriques, mêmes graphiques, courbes ROC / PR, comparaison appariée), courbes d'apprentissage, pipelines A vs B (effet de l'agence). **Correction** de la méthode de réglage des modèles de base du stacking. Décision D14.
- **Étape 13** — Évaluation finale sur le test (ouvert une seule fois) : stacking 0,820 (IC 95 % [0,758 ; 0,874]), comparaison train / CV / test pour tous les modèles, bootstrap par entreprise, matrice de confusion, analyse des erreurs (frontière, asymétrie, agences, secteurs). Conclusions de la partie 2. Décision D15. **Partie 2 terminée.**
- **Étape 14** — Partie 3, feature engineering : 11 nouvelles variables, ablation sur 4 modèles × 5 variantes, CV imbriquée des modèles enrichis. Résultat : gain uniquement pour la régression logistique (+0,007). Anomalie des colonnes de cash-flow documentée. Décision D16.
- **Étape 15** — Déséquilibre et seuil : pondération vs SMOTE (implémenté à la main, train uniquement), seuil optimisé selon un coût 3:1 en CV imbriquée sur le stacking final → **rappel SG 62 % → 86 %**, coût −24 %. Sensibilité au ratio de coût. Décision D17.
- **Étape 16** — Optimisation des hyperparamètres : recherche aléatoire sur intervalles (GB : 6 hyperparamètres, 100 tirages ; SVM : 60 tirages), graphes en coordonnées parallèles, effet de chaque hyperparamètre (Spearman), convergence, CV imbriquée. GB +0,009, SVM sans gain, malédiction du gagnant observée. Décision D18.
- **Étape 17** — Explicabilité : importance par permutation en CV (variables et familles, 4 modèles), PDP / ICE des 6 ratios les plus importants, carte rentabilité × endettement, effet de l'agence et du secteur à ratios égaux, explications locales de 3 entreprises (SG détectée, SG ratée, fausse alerte). Décision D19.
