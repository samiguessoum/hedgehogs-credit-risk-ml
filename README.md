# The Hedgehogs of Wall Street

**From Financial Fundamentals to Credit Risk: An Explainable Machine Learning Screener**

Projet du cours *Analyse de données et Apprentissage Automatique* — IMT Atlantique
Équipe : **Baptiste Gauthier**, **Sami Guessoum**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samiguessoum/hedgehogs-credit-risk-ml/blob/main/hedgehogs_credit_risk.ipynb)

---

## Objectif

Prédire la **qualité de crédit d'une entreprise** à partir de ses ratios financiers, pour aider un investisseur à évaluer rapidement son niveau de risque.

- **Tâche principale** : classification binaire **Investment Grade** (note BBB- ou mieux) vs **Speculative Grade** (BB+ ou moins).
- **Extension** : classification multiclasse en 5 niveaux de risque.
- **Point d'attention** : une même entreprise apparaît plusieurs fois dans les données. Tous les découpages entraînement / test se font **par entreprise** pour éviter le *data leakage*.

## Données

[Corporate Credit Rating with Financial Ratios](https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios) (Kaggle)

- 7 805 notations de crédit émises entre 2010 et 2016 par 7 agences ;
- 686 entreprises distinctes, 2 694 bilans distincts ;
- 16 ratios financiers (rentabilité, liquidité, endettement, cash-flows, efficacité) ;
- variables catégorielles : secteur, agence de notation, code SIC.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `hedgehogs_credit_risk.ipynb` | Notebook principal (Google Colab) |
| `corporateCreditRatingWithFinancialRatios.csv` | Dataset |
| `JOURNAL.md` | Journal de bord : constats, décisions et leurs justifications, pistes, questions de soutenance |

## Lancer le notebook

Cliquer sur le bouton **Open in Colab** ci-dessus. Aucune installation n'est nécessaire : toutes les bibliothèques sont préinstallées sur Colab, et les données sont chargées directement depuis ce dépôt (avec repli sur une copie locale).

En local : cloner le dépôt et ouvrir le notebook avec Jupyter (Python ≥ 3.10, pandas, numpy, matplotlib, seaborn, scipy, scikit-learn).

```bash
git clone https://github.com/samiguessoum/hedgehogs-credit-risk-ml
```

## Avancement

| Partie | Contenu | Statut |
|---|---|---|
| 1 | Analyse exploratoire et exploration non supervisée | ✅ Terminée |
| 2 | Modèles de base et modèles d'ensemble | ⏳ À venir |
| 3 | Modèles améliorés, optimisation des hyperparamètres et analyse | ⏳ À venir |

## Principaux enseignements de l'exploration

- **89 % des lignes ont un « jumeau »** (même bilan noté par plusieurs agences ou à plusieurs dates) → split obligatoire par entreprise.
- **Cible modérément déséquilibrée** (65 % IG / 35 % SG) et partiellement bruitée : les agences sont en désaccord sur 12 % des bilans qu'elles notent en commun.
- **La rentabilité est le premier facteur de risque**, suivie de l'endettement.
- **Les valeurs extrêmes sont informatives** (plus souvent des entreprises risquées) : elles sont gardées et traitées dans le pipeline.
- **Les clusters capturent des modèles économiques plus que le risque** : un même ratio n'a pas le même sens selon l'activité.

Le détail de chaque choix et de ses alternatives est dans [`JOURNAL.md`](JOURNAL.md).
