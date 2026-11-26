"""
Entrenamiento del modelo - Grupo 04 MDS (UPC)
Pregunta: ¿ocurrirá al menos un sismo M >= 5.0 en la zona durante el trimestre siguiente (T+1)?

Flujo (sin fuga temporal):
  1. Se usa la partición cronológica ya definida en el dataset (train / valid / test).
  2. Se comparan dos modelos entrenados SOLO con train y evaluados en valid.
  3. El umbral de decisión se ajusta en valid (no en test).
  4. Se reporta el desempeño final en test (2021-2026), datos que el modelo nunca vio.
  5. Se guarda model.joblib con el pipeline, el umbral y las métricas.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "Dataset_Final_Modelamiento.csv"
MODEL_PATH = "model.joblib"
SEED = 42

ID_COLS = ["zona_id", "trimestre", "split"]
TARGET = "target_m5_prox_trim"


def cargar_datos(path=DATA_PATH):
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    assert len(features) == 26, f"Se esperaban 26 variables, hay {len(features)}"
    return df, features


def construir_modelos():
    # El imputer (mediana) se ajusta dentro del pipeline, es decir, solo con train.
    # El dataset ya no tiene nulos, pero queda como protección para datos nuevos.
    return {
        "Regresión logística": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=6, min_samples_leaf=5,
                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        ]),
    }


def metricas(y, prob, umbral):
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(roc_auc_score(y, prob), 3),
        "pr_auc": round(average_precision_score(y, prob), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred, zero_division=0), 3),
        "f1": round(f1_score(y, pred, zero_division=0), 3),
        "matriz_confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def mejor_umbral(y, prob):
    """Umbral que maximiza F1 en validación."""
    umbrales = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y, (prob >= u).astype(int), zero_division=0) for u in umbrales]
    return float(round(umbrales[int(np.argmax(f1s))], 2))


def main():
    df, features = cargar_datos()
    train, valid, test = (df[df.split == s] for s in ["train", "valid", "test"])
    X_tr, y_tr = train[features], train[TARGET]
    X_va, y_va = valid[features], valid[TARGET]
    X_te, y_te = test[features], test[TARGET]

    print(f"Filas -> train: {len(train)} | valid: {len(valid)} | test: {len(test)}")
    print(f"% positivos -> train: {y_tr.mean():.1%} | valid: {y_va.mean():.1%} | test: {y_te.mean():.1%}\n")

    # 1) Selección de modelo en validación (métrica principal: PR-AUC, adecuada para desbalance)
    resultados = {}
    for nombre, pipe in construir_modelos().items():
        pipe.fit(X_tr, y_tr)
        prob_va = pipe.predict_proba(X_va)[:, 1]
        u = mejor_umbral(y_va, prob_va)
        resultados[nombre] = {"pipe": pipe, "umbral": u, "valid": metricas(y_va, prob_va, u)}
        print(f"[VALID] {nombre}: umbral={u} -> {resultados[nombre]['valid']}")

    ganador = max(resultados, key=lambda n: resultados[n]["valid"]["pr_auc"])
    pipe, umbral = resultados[ganador]["pipe"], resultados[ganador]["umbral"]
    print(f"\nModelo elegido: {ganador} (umbral {umbral})")

    # 2) Evaluación final en test: datos 2021-2026 que el modelo nunca vio
    prob_te = pipe.predict_proba(X_te)[:, 1]
    m_test = metricas(y_te, prob_te, umbral)
    print(f"[TEST]  {ganador}: {m_test}")

    # Línea base: predecir con la tasa histórica de M>=5 de la zona (sin modelo)
    base = metricas(y_te, test["tasa_hist_m5_zona"].values, umbral)
    print(f"[TEST]  Línea base (tasa histórica de la zona): PR-AUC={base['pr_auc']} ROC-AUC={base['roc_auc']}")

    # 3) Guardar artefacto: pipeline + umbral + metadatos
    artefacto = {
        "pipeline": pipe,
        "umbral": umbral,
        "features": features,
        "modelo": ganador,
        "metricas_valid": resultados[ganador]["valid"],
        "metricas_test": m_test,
        "linea_base_test": {"pr_auc": base["pr_auc"], "roc_auc": base["roc_auc"]},
        "prevalencia_train": round(float(y_tr.mean()), 3),
    }
    joblib.dump(artefacto, MODEL_PATH)
    with open("metricas.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in artefacto.items() if k != "pipeline"}, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {MODEL_PATH} y metricas.json")


if __name__ == "__main__":
    main()
