# Riesgo sísmico por zona y trimestre — Grupo 04 MDS (UPC)

Aplicación web que estima si en una zona del Perú ocurrirá **al menos un sismo de magnitud ≥ 5.0 en el trimestre siguiente (T+1)**, a partir de la actividad sísmica histórica (catálogo USGS, 1990–2026).

## Cómo se usa
El usuario elige **zona** (9 zonas de 2° × 2°) y **trimestre de referencia**. La app toma las 26 variables históricas ya calculadas para esa fila del dataset y el modelo devuelve:
- un puntaje de riesgo (0 a 1) y una alerta de riesgo **ALTO / bajo** según el umbral elegido en validación;
- un ranking de las 9 zonas para ese trimestre;
- el desempeño en prueba y un monitoreo por año.

## Dataset
`Dataset_Final_Modelamiento.csv`: 1,125 filas (zona × trimestre), 26 variables predictoras, 3 identificadores (`zona_id`, `trimestre`, `split`) y la variable objetivo `target_m5_prox_trim`. Sin nulos. Partición cronológica: train 1995–2016, valid 2017–2020, test 2021–2026.

## Modelo
`train_model.py` compara una regresión logística y un Random Forest (ambos con pesos balanceados por el desbalance del 13.7%), los entrena solo con train, elige el mejor por PR-AUC en validación, ajusta ahí el umbral y reporta el resultado final en test. Las métricas quedan en `metricas.json`.

## Ejecutar localmente
```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows  |  source venv/bin/activate  (Mac/Linux)
pip install -r requirements.txt
python train_model.py            # genera model.joblib y metricas.json
streamlit run app.py
```

## Estructura
```
sismos-streamlit/
├── Dataset_Final_Modelamiento.csv
├── train_model.py
├── app.py
├── model.joblib
├── metricas.json
├── requirements.txt
└── README.md
```

> Es una estimación estadística con fines académicos, no un pronóstico sísmico oficial.
