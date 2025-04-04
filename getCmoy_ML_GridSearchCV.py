from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from IPython.display import display


TF_ENABLE_ONEDNN_OPTS=0

X = pd.read_csv('C:/Users/samad/OneDrive/Bureau/ml_2/Dataset_numerique_20000_petites_machines.csv', sep=';')
X.drop('l_cr', axis=1,inplace=True)
X.drop('l_axe', axis=1,inplace=True)
X.drop('l_cs', axis=1,inplace=True)
y = X.pop('Cmoy')
X_test = pd.read_csv('C:/Users/samad/OneDrive/Bureau/ml_2/Dataset_numerique_10000_petites_machines.csv', sep=';')
X_test.drop('l_cr', axis=1,inplace=True)
X_test.drop('l_axe', axis=1,inplace=True)
X_test.drop('l_cs', axis=1,inplace=True)
y_test = X_test.pop('Cmoy')
X_Without_FE = X.copy()

X_test_Without_FE = X_test.copy()
# Feature Engineering
X['log_Cmax'] = np.log1p(X['Cmax'])  # log(1 + Cmax) pour éviter log(0)
X['Cdiff'] = X['Cmax'] - X['Cmin']
X['C_ratio'] = X['Cmax'] / (X['Cmin'] + 1e-6)  # Évite la division par 0
X_test['log_Cmax'] = np.log1p(X_test['Cmax'])  # log(1 + Cmax) pour éviter log(0)
X_test['Cdiff'] = X_test['Cmax'] - X_test['Cmin']
X_test['C_ratio'] = X_test['Cmax'] / (X_test['Cmin'] + 1e-6)  # Évite la division par 0

X['log_Cmax'] = np.log1p(X['Cmax'])
X['Cdiff'] = X['Cmax'] - X['Cmin']
X['C_ratio'] = X['Cmax'] / (X['Cmin'] + 1e-6)
X['Cmax_squared'] = X['Cmax']**2
X['log_C_ratio'] = np.log1p(X['Cmax'] / (X['Cmin'] + 1e-6))
X['Cmax_Cmin_product'] = X['Cmax'] * X['Cmin']

X_test['log_Cmax'] = np.log1p(X_test['Cmax'])
X_test['Cdiff'] = X_test['Cmax'] - X_test['Cmin']
X_test['C_ratio'] = X_test['Cmax'] / (X_test['Cmin'] + 1e-6)
X_test['Cmax_squared'] = X_test['Cmax']**2
X_test['log_C_ratio'] = np.log1p(X_test['Cmax'] / (X_test['Cmin'] + 1e-6))
X_test['Cmax_Cmin_product'] = X_test['Cmax'] * X_test['Cmin']
imputer = SimpleImputer(strategy='mean')  # ou 'median', 'constant', etc.
X[:] = imputer.fit_transform(X)
X_test[:] = imputer.transform(X_test)
X_Without_FE[:] = imputer.fit_transform(X_Without_FE)
X_test_Without_FE[:] = imputer.transform(X_test_Without_FE)
print("X.shape:", X.shape)
print("y.shape:", y.shape)
print("X_test.shape:", X_test.shape)
print("y_test.shape:", y_test.shape)
print("X___.shape:", X_Without_FE.shape)
print("X____test.shape:", X_test_Without_FE.shape)
print("Indices X_test:", X_test.index[:5])
print("Indices y_test:", y_test.index[:5])

print(X.columns)
print(X.head())
'''
z_scores = np.abs(zscore(X.select_dtypes(include=[np.number])))
X = X[(z_scores < 3).all(axis=1)]  # Garde les lignes où toutes les features sont < 3 sigmas
y = y.loc[X.index]  # Synchroniser y avec les nouvelles lignes de X 
'''


if X_Without_FE.shape[0] > 0:
    X_Without_FE += np.random.normal(0, 0.01, X_Without_FE.shape)
else:
    print("❌ X_Without_FE est vide après le filtrage des outliers.")
if X_test_Without_FE.shape[0] > 0:
    X_test_Without_FE += np.random.normal(0, 0.01, X_test_Without_FE.shape)
else:
    print("❌ X_test_Without_FE est vide après le filtrage des outliers.")
print("train set sans FE dim",X_Without_FE.shape)
#******************/////////////////////////////////////////////////////////////////
if X.shape[0] > 0:
    X += np.random.normal(0, 0.01, X.shape)
else:
    print("❌ X est vide après le filtrage des outliers.")
if X_test.shape[0] > 0:
    X_test += np.random.normal(0, 0.01, X_test.shape)
else:
    print("❌ X est vide après le filtrage des outliers.")
print("train set dim",X.shape)

# Liste des modèles à tester
regressors = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=0),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=0),
        "params": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "HistGradient Boosting": {
        "model": HistGradientBoostingRegressor(random_state=0),
        "params": {
            "max_iter": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "Extra Trees": {
        "model": ExtraTreesRegressor(random_state=0),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "KNeighbors": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        }
    },
    "Ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0]
        }
    },
    "Lasso": {
        "model": Lasso(),
        "params": {
            "alpha": [0.0001, 0.001, 0.01]
        }
    }
}
for name, config in regressors.items():
    print(f"Optimisation de {name}...")
    
    grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring='r2', n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    
    regressors[name] = grid_search.best_estimator_
    
    print(f"Meilleurs paramètres pour {name}: {grid_search.best_params_}")
    print(f"Meilleur score R² sur validation: {grid_search.best_score_}\n")

from sklearn.impute import SimpleImputer


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')
# Bundle preprocessing for numerical data
preprocessor_Without_FE = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X_Without_FE.columns)])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X.columns)])


my_pipline_Without_FE = [ Pipeline(steps=[('preprocessor', preprocessor_Without_FE),
                      ('model', model)]) for model in regressors.values() ]
my_pipline = [ Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)]) for model in regressors.values() ]

X_test_ =  preprocessor.fit_transform(X_test)
X_test_ = pd.DataFrame(X_test_, columns=X_test.columns)

X_test_Without_FE_ =  preprocessor_Without_FE.fit_transform(X_test_Without_FE)
X_test_Without_FE_ = pd.DataFrame(X_test_Without_FE_, columns=X_test_Without_FE.columns)

scoree_Without_FE = {}
importances_Without_FE = {}
#********************************************************************************************
for piplinee in my_pipline_Without_FE:
    model_name = next(k for k, v in regressors.items() if v == piplinee.named_steps['model'])

    print(f"\n➡️ Training sans FE model: {model_name}")
    with tqdm(total=1, desc=f"Fitting {model_name}", unit="model") as pbar:
        piplinee.fit(X_Without_FE, y)     
        pbar.update(1)

    preds_Without_FE = piplinee.predict(X_test_Without_FE_)
    score_Without_FE = r2_score(y_test, preds_Without_FE)

    scoree_Without_FE[model_name] = score_Without_FE
    print(f"✅ Score sans FE of model {model_name}: {score_Without_FE:.4f}")

    try:
        perm = PermutationImportance(piplinee.named_steps['model'], random_state=42)
        perm.fit(piplinee.named_steps['preprocessor'].transform(X_test_Without_FE_), y_test)
        weights = eli5.explain_weights_df(perm, feature_names=X_Without_FE.columns.tolist())
        importances_Without_FE[model_name] = weights
        display(weights.head(10))  # top 10
    except Exception as e:
        print(f"⚠️ Importance non dispo pour {model_name} : {e}")
    
#********************************************************************************************
scoree = {}
importances = {}
for piplinee in my_pipline:
    model_name = next(k for k, v in regressors.items() if v == piplinee.named_steps['model'])

    print(f"\n➡️ Training model: {model_name}")
    with tqdm(total=1, desc=f"Fitting {model_name}", unit="model") as pbar:
        piplinee.fit(X, y)     
        pbar.update(1)

    preds = piplinee.predict(X_test_)
    score = r2_score(y_test, preds)

    scoree[model_name] = score
    print(f"✅ Score of model {model_name}: {score:.4f}")

    try:
        perm = PermutationImportance(piplinee.named_steps['model'], random_state=42)
        perm.fit(piplinee.named_steps['preprocessor'].transform(X_test_), y_test)
        weights = eli5.explain_weights_df(perm, feature_names=X.columns.tolist())
        importances[model_name] = weights
        display(weights.head(10))  # top 10
    except Exception as e:
        print(f"⚠️ Importance non dispo pour {model_name} : {e}")
    


# Tri des scores du meilleur au moins bon
sorted_scores_Without_FE = dict(sorted(scoree_Without_FE.items(), key=lambda item: item[1], reverse=True))
sorted_scores = dict(sorted(scoree.items(), key=lambda item: item[1], reverse=True))
fig, ax = plt.subplots(2, 1, figsize=(16, 6))

# **1. Affichage des R² Scores**

# Graphique des scores R²
ax[0].barh(list(sorted_scores.keys()), list(sorted_scores.values()), color='skyblue')
ax[0].set_xlabel("R² Score")
ax[0].set_title("Performance des régressions")
ax[0].invert_yaxis()  # Le meilleur modèle en haut
for i, (model, score) in enumerate(sorted_scores.items()):
    ax[0].text(score + 0.01, i, f"{score:.4f}", va='center')
ax[1].barh(list(sorted_scores_Without_FE.keys()), list(sorted_scores_Without_FE.values()), color='skyblue')
ax[1].set_xlabel("R² Score")
ax[1].set_title("Performance des régressions sans FE")
ax[1].invert_yaxis()  # Le meilleur modèle en haut
for i, (model, score) in enumerate(sorted_scores_Without_FE.items()):
    ax[1].text(score + 0.01, i, f"{score:.4f}", va='center')


# **2. Affichage des Importances des Features dans une même figure**
# Nombre de modèles avec importance disponible
n_models = len(importances)

# Déterminer la grille de subplots (par exemple 2 lignes si plus de 3 modèles)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()  # Pour un accès facile

for idx, (model_name, importance_df) in enumerate(importances.items()):
    ax = axes[idx]
    top_features = importance_df.head(10)
    ax.barh(top_features['feature'], top_features['weight'], color='lightgreen')
    ax.set_title(f"{model_name}")
    ax.invert_yaxis()
    #ax.set_xlabel("Poids")

# Supprimer les axes vides s'il y en a
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Top 10 Features Importantes par Modèle", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  

#*************************************SANS FE*********************************************************
# **2. Affichage des Importances des Features dans une même figure**
# Nombre de modèles avec importance disponible
n_models = len(importances_Without_FE)

# Déterminer la grille de subplots (par exemple 2 lignes si plus de 3 modèles)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()  # Pour un accès facile

for idx, (model_name, importance_df) in enumerate(importances_Without_FE.items()):
    ax = axes[idx]
    top_features = importance_df.head(10)
    ax.barh(top_features['feature'], top_features['weight'], color='green')
    ax.set_title(f"{model_name}")
    ax.invert_yaxis()
    #ax.set_xlabel("Poids")

# Supprimer les axes vides s'il y en a
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Top 10 Features Importantes sans FE par Modèle", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Pour ne pas couper le titre
plt.show() 