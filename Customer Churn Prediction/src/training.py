import os
import mlflow
import joblib
import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import tqdm as notebook_tqdm


class Trainer():

    def __init__(self, X_train, X_val, y_train, y_val, path_model, dv):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.path_model = path_model
        self.dv = dv

        os.makedirs(self.path_model, exist_ok=True)

    def LogReg(self):
        base_model = LogisticRegression()

        param_grid = [
            {"penalty":["l2"],
             "C" : np.logspace(-4,4,20),
            "solver": ["lbfgs","newton-cg","liblinear","sag","saga"],
            "max_iter"  : [100,1000,2500,5000]}
        ]

        grid_search = GridSearchCV(base_model, param_grid=param_grid, cv=4, verbose=True, n_jobs=-1)
        mlflow.set_experiment("Logistic Regression")
        with mlflow.start_run():

            grid_search.fit(self.X_train, self.y_train)

            best_model = grid_search.best_estimator_
            params = grid_search.best_params_

            y_pred = best_model.predict(self.X_val)
            
            acc = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("f1_score", f1)

            model_file = os.path.join(self.path_model,"log_reg.pkl")
            joblib.dump((self.dv, best_model), model_file)

            mlflow.sklearn.log_model(best_model, "LogisticRegressionModel")

        return acc, f1

    def RandForest(self):
        base_model = RandomForestClassifier()
        param_grid = [{
            "n_estimators": [100, 200, 300, 400, 600, 800],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }]
        grid_search = GridSearchCV(base_model, param_grid=param_grid, cv=4, verbose=True, n_jobs=-1)

        mlflow.set_experiment("Random Forest")
        with mlflow.start_run():

            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            params = grid_search.best_params_

            y_pred = best_model.predict(self.X_val)

            acc = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("f1_score", f1)

            model_file = os.path.join(self.path_model,"rand_forest.pkl")
            joblib.dump((self.dv, best_model), model_file)

            mlflow.sklearn.log_model(best_model, "RandomForestModel")

        return acc, f1
    
    
    def XGB(self):

        def objective(trial):
        
            params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0)
                }
            base_model = XGBClassifier(**params, eval_metric="logloss")
            base_model.fit(self.X_train, self.y_train)

            y_pred = base_model.predict(self.X_val)

            acc = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)

            
            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("f1_score", f1)

            return acc

        mlflow.set_experiment("XGBoost")
        with mlflow.start_run():

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=100)

            best_params = study.best_params
            mlflow.log_params(best_params)

            best_model = XGBClassifier(**best_params, eval_metric="logloss")
            best_model.fit(self.X_train, self.y_train)
            
            model_file = os.path.join(self.path_model,"xgboost.pkl")
            joblib.dump((self.dv, best_model), model_file)

            mlflow.sklearn.log_model(best_model, "XGBoostModel")

        return best_model