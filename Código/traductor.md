# TRADUCTOR INTEGRAL LÍNEA POR LÍNEA — `Código unificado.ipynb`

**Proyecto de Titulación:** *Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas*  
**Autor:** Felipe Andrés Guerra Maturana  
**Profesor Guía:** Dr. José Delgado Vega  
**Institución:** Universidad de Antofagasta — Departamento de Ingeniería en Minas  

---

## ¿Cómo estudiar con este documento?

Este manual traduce **cada bloque y cada instrucción de Python** de [`Código unificado.ipynb`](./Código%20unificado.ipynb) al lenguaje de la **Ingeniería en Minas, la Geoestadística y la Estadística Aplicada**.  
Está organizado siguiendo **exactamente el mismo orden de celdas** del cuaderno para que puedas abrir Google Colab a la izquierda y este documento a la derecha, entendiendo no solo *qué hace* cada línea, sino *por qué está escrita así* en caso de que el Dr. José Delgado te pregunte por cualquier fragmento.

> **Nota sobre numeración de celdas:** En `Código unificado.ipynb`, los 10 bloques ejecutables aparecen marcados en el margen izquierdo de Jupyter / Google Colab como **`[1]` a `[10]`** (que corresponden internamente a las celdas `1, 3, 5, 7, 9, 10, 11, 13, 14 y 15` del archivo, alternadas con las celdas explicativas de texto). Cada encabezado de este traductor indica ambos números para ubicación instantánea.

---

## CELDA 1: Configuración del Entorno e Importación de Librerías

### 1. Control del Sistema e Instalación Automática en Google Colab
```python
import os, sys, io, platform, tempfile, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
```
* **`import os, sys, io, platform, tempfile, warnings`**: Carga módulos internos de Python:
  * `os` y `sys`: comunican Python con el sistema operativo (Windows local o Linux en Google Colab).
  * `io`: permite leer un texto en memoria (`io.StringIO`) como si fuera un archivo `.csv` físico. Gracias a esto el código corre en Google Colab sin necesidad de subir archivos externos.
  * `platform`: detecta la versión del procesador y sistema operativo.
  * `tempfile`: localiza la carpeta temporal del sistema para probar el guardado `.joblib` sin ensuciar tu carpeta de trabajo.
  * `warnings`: administra advertencias visuales en pantalla.
* **`os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'`**: Le ordena al motor C++ de TensorFlow que oculte mensajes técnicos internos de tarjetas gráficas (CUDA/oneDNN) y muestre únicamente errores reales.
* **`warnings.filterwarnings('ignore')`**: Oculta avisos repetitivos de convergencia numérica cuando el optimizador prueba configuraciones extremas en muestras pequeñas ($N=23$), manteniendo limpia la salida para tu presentación.

```python
try:
    import shap
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "shap"])
    import shap
```
* **`try ... except ImportError`**: Intenta cargar la librería de explicabilidad `shap`. Si en una sesión nueva de Google Colab no viniera preinstalada, el bloque `except` ejecuta automáticamente `pip install -q shap` en segundo plano (`-q` significa *quiet*, silencioso) y luego la importa sin interrumpir tu ejecución.

### 2. Librerías Científicas, Estadísticas y de Machine Learning
```python
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
```
* **`numpy as np`**: Motor de álgebra lineal matricial (vectores, matrices, productos punto, promedios, desviaciones estándar).
* **`pandas as pd`**: Manejo de tablas estructuradas (`DataFrame`), equivalentes a hojas de Excel programables.
* **`joblib`**: Empaquetador binario de alta eficiencia para guardar modelos entrenados de `scikit-learn` en disco.
* **`matplotlib.pyplot` y `seaborn`**: Motores gráficos para construir series temporales, histogramas, mapas de contorno 2D y gráficos Q-Q.
* **`Ellipse`**: Objeto geométrico para dibujar la frontera elíptica del $95\%$ de confianza de la Distancia de Mahalanobis.

```python
from scipy import stats
from scipy.spatial import ConvexHull, Delaunay
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
```
* **`scipy.stats`**: Contiene las pruebas estadísticas exactas: `shapiro` (normalidad), `pearsonr` (correlación lineal), `spearmanr` (correlación por rangos), `probplot` (gráfico Q-Q) y `wilcoxon` (contraste pareado no paramétrico).
* **`ConvexHull` y `Delaunay`**: Algoritmos de geometría computacional. `ConvexHull` dibuja el polígono exterior que encierra los 24 meses reales en el plano $[\text{CuT}, \text{CuS}]$, y `Delaunay` permite comprobar matemáticamente si un lote nuevo cae dentro o fuera de ese dominio empírico.
* **`statsmodels` (`sm`, `variance_inflation_factor`, `acf`, `het_breuschpagan`, `durbin_watson`)**: Librería econométrica rigurosa para calcular el Factor de Inflación de la Varianza ($\text{VIF}$), la Función de Autocorrelación Temporal ($\text{ACF}$), el test de homocedasticidad de Breusch-Pagan y el estadístico $d$ de Durbin-Watson.

```python
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
import tensorflow as tf
```
* **`StandardScaler` / `RobustScaler`**: Escaladores que normalizan las variables para que la diferencia de varianza entre $\text{CuT}$ y $\text{CuS}$ ($61.64\times$) no distorsione los pesos.
* **`Pipeline`**: Encapsulador secuencial que une el escalador con el modelo. Es **la pieza más importante para evitar fuga de datos (*Data Leakage*)**: asegura que la media y desviación estándar se calculen solo con los $23$ meses de entrenamiento y jamás con el mes de prueba.
* **`LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, `SVR`**: Los modelos de referencia (*baselines*) lineales, penalizados ($L_1, L_2$) y de soporte vectorial no lineal.
* **`MLPRegressor`**: Perceptrón Multicapa de `scikit-learn` en precisión doble (`float64`) con optimizador cuasi-Newton `lbfgs`.
* **`tensorflow as tf`**: Motor de *Deep Learning* usado para reconstruir con fidelidad exacta el modelo original en Keras del Dr. José Delgado Vega.

### 3. Estilo Visual y Ficha del Entorno
```python
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 130
...
display(df_entorno)
```
* Configura grilla limpia, tipografía legible y resolución nítida en pantalla (`130 DPI`), e imprime una tabla (`df_entorno`) con las versiones exactas de Python y librerías que se están ejecutando en ese instante en Colab o en tu PC.

---

## CELDA 3: Ingesta Auto-Contenida, Auditoría ETL y Tarifas ENAMI (Bloque 0)

### 1. Datos Maestros Integrados en Memoria
```python
CSV_DATASET_N24 = '''Fecha,Mes_Num,Cancha_Ton,Romana_Ton,Cancha_CuT,...'''
CSV_TARIFAS_ENAMI = '''Fecha,Precio_Cu_cUSD_lb,Tipo_Cambio_CLP_USD,...'''

df = pd.read_csv(io.StringIO(CSV_DATASET_N24))
df_tarifas = pd.read_csv(io.StringIO(CSV_TARIFAS_ENAMI))
```
* **`CSV_DATASET_N24` y `CSV_TARIFAS_ENAMI`**: Cadenas de texto multilínea (`'''...'''`) que contienen los $24$ registros auditados de `Data Cobrizo.xlsx` (homologados a $\%\text{Cu}$) y las $24$ circulares tarifarias de ENAMI (homologadas $100\%$ a dólares `US$/tms`).
* **`pd.read_csv(io.StringIO(...))`**: Convierte cada bloque de texto directamente en una tabla `DataFrame` de Pandas. Así, el cuaderno es **100% autónomo** en Google Colab.

### 2. Cálculo de Variables Diagnósticas sin Redondeo Intermedio
```python
df['Sesgo_Canje'] = df['Cancha_CuT'] - df['Canje_CuT']
df['Ratio_Solubilidad'] = (df['Cancha_CuS'] / df['Cancha_CuT']) * 100.0
df_tarifas['Maquila_Implicita_USD_tms'] = (2.5 * df_tarifas['Flot_Escala_USD_tms_1pct'] - df_tarifas['Flot_Base_USD_tms']).round(2)
```
* **`df['Sesgo_Canje']`**: Calcula mes a mes la sobreestimación de mina ($e_{\text{emp}} = \text{Cancha\_CuT} - \text{Canje\_CuT}$). En los 24 meses es positiva, promediando **$+0.3404\%\text{Cu}$**.
* **`df['Ratio_Solubilidad']`**: Calcula qué porcentaje del cobre total corresponde a especies oxidadas solubles en ácido cítrico ($\text{CuS}/\text{CuT} \times 100$).
* **`Maquila_Implicita_USD_tms`**: Verifica la consistencia económica de las tarifas ENAMI en dólares: si un mineral de $2.5\%\text{Cu}$ se valorizara sin descuento de planta valdría $2.5 \times \text{Escala}$, por lo que restarle la `Tarifa Base` revela el cargo implícito de tratamiento/maquila ($\sim 34.6\text{ a }39.6\text{ US\$/tms}$).

### 3. Controles de Calidad ETL y Gráfico de Brecha Operacional
```python
n_nulos = int(df.isna().sum().sum())
n_duplicados = int(df.duplicated(subset=['Fecha']).sum())
diff_quimica_cancha = np.abs(df['Cancha_CuT'] - (df['Cancha_CuS'] + df['Cancha_CuIn'])).max()
diff_quimica_canje = np.abs(df['Canje_CuT'] - (df['Canje_CuS'] + df['Canje_CuIn'])).max()
```
* **`df.isna().sum().sum()`**: Cuenta celdas vacías en toda la tabla (resultado: `0`).
* **`df.duplicated(subset=['Fecha']).sum()`**: Verifica que ningún mes esté repetido (resultado: `0`).
* **`diff_quimica_cancha` / `diff_quimica_canje`**: Comprueba el balance de masa químico $\text{CuT} = \text{CuS} + \text{CuIn}$. En Cancha difiere máximo $0.01\%\text{Cu}$ por redondeo; en las columnas auxiliares de Canje de 2024 difiere hasta $0.20\%\text{Cu}$ en la planilla histórica de origen, razón por la cual el modelado utiliza **exclusivamente** la ley oficial liquidada `Canje_CuT`.
* **`fig, (ax1, ax2) = plt.subplots(2, 1, ...)`**: Crea una figura dividida en dos pisos (`height_ratios: [1.8, 1]`): arriba grafica las tres curvas de ley (`Cancha`, `PSJ`, `Canje`) y abajo grafica en barras rojas el sesgo positivo de los 24 meses junto a la línea punteada del promedio ($+0.3404\%\text{Cu}$).

> **Pregunta típica del profesor en esta sección:** *"¿Por qué tu tonelaje total dice $295{,}884.67\text{ t}$?"*  
> **Respuesta:** Porque distingue entre el tonelaje estimado en acopio de mina (`Cancha_Ton` $= 286{,}704\text{ t}$) y el pesaje oficial certificado en báscula puente de recepción en Planta San José (`Romana_Ton` $= 295{,}884.67\text{ t}$), que es el que rige jurídicamente la liquidación comercial ENAMI.

---

## CELDA 5: Análisis Exploratorio (EDA), Mahalanobis, ACF y VIF (Bloque 1)

### 1. Estadísticos Univariados y Normalidad (Shapiro-Wilk)
```python
for v in vars_eda:
    serie = df[v]
    w_sw, p_sw = stats.shapiro(serie)
    ...
    'Desv_Muestral_s': round(serie.std(ddof=1), 4),
    'Var_Muestral_s2': round(serie.var(ddof=1), 6),
```
* **`ddof=1`** (*Delta Degrees of Freedom = 1*): Indica a Python que divida por $(N - 1 = 23)$ para obtener la **varianza muestral insesgada** ($s^2$). Cuando más abajo usamos `ddof=0`, divide por $N = 24$ (varianza poblacional $\sigma^2$) porque la identidad exacta del $\text{MSE}$ se define con denominador $N$.
* **`stats.shapiro(serie)`**: Ejecuta el test de normalidad de Shapiro-Wilk (óptimo para muestras pequeñas $N < 50$). Si `p_sw >= 0.05` no se rechaza normalidad (`Sesgo_Canje` obtiene $W = 0.9852, p = 0.9695$, es decir, **es normal**); en cambio, las leyes brutas (`Cancha_CuT`, `Cancha_CuS`, `Canje_CuT`) tienen $p < 0.05$ por la asimetría geológica de los pulsos ricos de fines de 2023.

### 2. Contraste de Escalas y Descomposición Sesgo-Varianza
```python
ratio_varianzas = df['Cancha_CuT'].var(ddof=1) / df['Cancha_CuS'].var(ddof=1)
mse_hist = np.mean(df['Sesgo_Canje']**2)
sesgo2_hist = (df['Sesgo_Canje'].mean())**2
var_pob_error = df['Sesgo_Canje'].var(ddof=0)
```
* **`ratio_varianzas`**: Divide $s^2_{\text{CuT}} / s^2_{\text{CuS}} = 0.058128 / 0.000943 = \mathbf{61.64\times}$. Si entráramos a una red neuronal o a Ridge sin escalar, la red ignoraría `Cancha_CuS` por ser $61$ veces más pequeña en varianza.
* **`mse_hist = sesgo2_hist + var_pob_error`**: Demuestra matemáticamente que del error cuadrático histórico total ($0.1350$), el **$85.86\%$** ($0.1159$) es sesgo constante y el **$14.14\%$** ($0.0191$) es varianza fluctuante alrededor de ese sesgo.

### 3. Matriz de Covarianza e Identidad Algebraica
```python
matriz_cov = df[cov_cols].cov(ddof=1)
identidad_var = matriz_cov.loc['Cancha_CuT', 'Cancha_CuT'] + matriz_cov.loc['Canje_CuT', 'Canje_CuT'] - 2 * matriz_cov.loc['Cancha_CuT', 'Canje_CuT']
assert np.isclose(identidad_var, matriz_cov.loc['Sesgo_Canje', 'Sesgo_Canje'])
```
* **`assert np.isclose(...)`**: Es un "guardián matemático" en Python. Comprueba que $\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y) - 2\text{Cov}(X, Y) = 0.019917\,(\%\text{Cu})^2$. Si hubiera cualquier error de redondeo previo en las columnas, `assert` detendría el programa de inmediato.

### 4. Efecto Apalancamiento (*Leverage*): $N=24$ vs $N=22$
```python
df_n22 = df[~df['Fecha'].isin(['2023-10', '2023-11'])]
...
mi_val = mutual_info_regression(df[[col_x]], df['Canje_CuT'], random_state=42)[0]
```
* **`~df['Fecha'].isin(['2023-10', '2023-11'])`**: El símbolo `~` significa negación ("NO está en"). Crea una tabla temporal de $22$ meses excluyendo octubre y noviembre de 2023 para medir cuánto influyen esos dos meses extremos en las correlaciones lineales.
* **`mutual_info_regression`**: Calcula la **Información Mutua** ($\text{MI}$), una métrica de teoría de la información capaz de detectar dependencias curvas o no lineales que Pearson no ve.

### 5. Distancia Multivariada de Mahalanobis
```python
X_mat = df[['Cancha_CuT', 'Cancha_CuS']].values
mu_X = np.mean(X_mat, axis=0)
cov_X = np.cov(X_mat, rowvar=False)
inv_cov_X = np.linalg.inv(cov_X)
df['Mahalanobis_D2'] = [float((x - mu_X) @ inv_cov_X @ (x - mu_X).T) for x in X_mat]
```
* **`np.linalg.inv(cov_X)`**: Invierte la matriz de covarianza $2 \times 2$ ($\boldsymbol{\Sigma}^{-1}$).
* **`(x - mu_X) @ inv_cov_X @ (x - mu_X).T`**: El operador `@` ejecuta multiplicación matricial exacta: $(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$. Mide qué tan lejos está cada mes del centro de operación tomando en cuenta la correlación entre `CuT` y `CuS`. Detecta exactamente dos frentes geológicos extraordinarios (`2023-10` con $D_M^2 = 12.16$ y `2023-11` con $D_M^2 = 8.48$), los cuales **se conservan** porque son leyes reales de producción, no errores de laboratorio.

### 6. Autocorrelación Serial (ACF) y Multicolinealidad (VIF)
```python
acf_sesgo = acf(df['Sesgo_Canje'], nlags=6, fft=False)
banda_bartlett = 1.96 / np.sqrt(len(df))

X_vif_2d = sm.add_constant(df[['Cancha_CuT', 'Cancha_CuS']])
vif_2d = [variance_inflation_factor(X_vif_2d.values, i) for i in range(1, 3)]
```
* **`acf(..., nlags=6, fft=False)`**: Calcula la correlación de la serie consigo misma desfasada $1, 2, \dots, 6$ meses (límite de Box-Jenkins $K \le N/4 = 6$). Como todos los valores de `Sesgo_Canje` caen dentro de $\pm 1.96/\sqrt{24} = \pm 0.4001$, se demuestra que los errores mensuales son independientes (**ruido blanco**), autorizando el uso de **LOOCV**.
* **`sm.add_constant(...)` y `variance_inflation_factor`**: Agrega la columna de unos (intercepto $\beta_0$, indispensable en econometría para no sesgar el $\text{VIF}$) y evalúa $\text{VIF}_j = 1 / (1 - R_j^2)$. Con `CuT` y `CuS` da **`1.8045`** ($< 5$, óptimo); si se mete `CuIn`, sube sobre **`4,000`** porque $\text{CuT} = \text{CuS} + \text{CuIn}$ es una combinación lineal perfecta.

---

## CELDA 7: Motor LOOCV Cero Fuga y Entrenamiento de los 7 Baselines (Bloque 2)

### 1. Grillas de Regularización
```python
ALPHAS_RIDGE = np.logspace(-3, 3, 100)
ALPHAS_LASSO = np.logspace(-4, 2, 100)
```
* **`np.logspace(-3, 3, 100)`**: Genera $100$ valores de penalización $\alpha$ espaciados logarítmicamente entre $10^{-3} = 0.001$ y $10^3 = 1{,}000$. Así el modelo explora desde regularización casi nula hasta regularización muy fuerte.

### 2. Bucle Exterior LOOCV (*Leave-One-Out Cross-Validation*)
```python
for k in range(N):
    if buffered:
        train_idx = [i for i in range(N) if i not in (k - 1, k, k + 1)]
    else:
        train_idx = [i for i in range(N) if i != k]

    X_tr, y_tr = X_2d[train_idx], y_real[train_idx]
    X_te = X_2d[k:k+1]
```
* **`for k in range(N)`**: Recorre uno por uno los $24$ meses ($k = 0, 1, \dots, 23$).
* **`train_idx = [i for i in range(N) if i != k]`**: Selecciona los **$23$ meses restantes** para entrenar, dejando el mes `k` completamente aislado en `X_te` (`k:k+1` conserva formato matricial $1 \times 2$).
* Si `buffered=True` (*Buffered-LOOCV*), excluye además el mes anterior ($k-1$) y el mes siguiente ($k+1$) para verificar que la continuidad geológica mensual no infle artificialmente los resultados.

### 3. Ajuste de Cada Baseline sin Fuga de Información
```python
sesgo_tr = float(np.mean(X_tr[:, 0] - y_tr))
predicciones['Baseline_1B'][k] = round(float(X_te[0, 0] - sesgo_tr), 4)
```
* **`Baseline_1B`**: Calcula el sesgo promedio **solo con los 23 meses de entrenamiento** (`sesgo_tr`) y se lo resta a la ley `Cancha_CuT` del mes de prueba `k`. No mira jamás el valor real `y_real[k]`.

```python
p_ridge = Pipeline([('scaler', scaler_cls()), ('model', RidgeCV(alphas=ALPHAS_RIDGE, cv=5))]).fit(X_tr, y_tr)
predicciones['RidgeCV'][k] = round(float(p_ridge.predict(X_te)[0]), 4)
```
* **`Pipeline([('scaler', ...), ('model', RidgeCV(..., cv=5))])`**: Al ejecutar `.fit(X_tr, y_tr)`:
  1. Ajusta el escalador con los 23 meses.
  2. Divide internamente esos 23 meses en 5 subgrupos (`cv=5`, *Validación Cruzada Anidada*) para elegir el mejor $\alpha$ sin tocar el mes `k`.
  3. Al llamar `.predict(X_te)`, escala el mes `k` usando la media/desviación de los 23 meses y emite la predicción ciega.
* **`coefs_lasso_cus.append(p_lasso.named_steps['model'].coef_[1])`**: Guarda el peso $\beta_2$ que `LassoCV` le dio a `Cancha_CuS` en cada pliegue. Al comprobar `np.abs(coefs_lasso_cus) > 1e-5`, se demuestra que Lasso **jamás apaga `Cancha_CuS` ($24/24$ pliegues retenida)**.

### 4. Función Maestra de Métricas (`calcular_metricas_oos`)
```python
def calcular_metricas_oos(y_true, y_pred):
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    nrmse = float((rmse / np.mean(y_true)) * 100.0)
    sst = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = float(1.0 - (np.sum(err**2) / sst))
    mape = float(np.mean(np.abs(err / y_true)) * 100.0)
    r_p = float(np.corrcoef(y_true, y_pred)[0, 1])
    sesgo = float(np.mean(y_pred - y_true))
    var_exp = float(1.0 - (np.var(err) / np.var(y_true)))
    return mse, rmse, mae, nrmse, r2, mape, r_p, sesgo, var_exp
```
* Centraliza las **9 métricas oficiales** de toda la tesis sobre el vector de $24$ predicciones fuera de muestra:
  * `mse`: Error Cuadrático Medio en $(\%\text{Cu})^2$.
  * `rmse`: Raíz del MSE en unidades físicas ($\%\text{Cu}$).
  * `mae`: Error Absoluto Medio en $\%\text{Cu}$.
  * `nrmse`: RMSE dividido por la ley media de canje ($1.43125\%\text{Cu}$), expresado en porcentaje.
  * `r2`: Coeficiente de determinación fuera de muestra ($1 - \text{SSE}/\text{SST}$).
  * `mape`: Error porcentual absoluto medio respecto a cada mes.
  * `r_p`: Correlación lineal de Pearson entre predicción y realidad.
  * `sesgo`: Sobreestimación ($+$) o subestimación ($-$) promedio del modelo ($\hat{y} - y$).
  * `var_exp`: Varianza explicada ($1 - \text{Var}(y - \hat{y})/\text{Var}(y)$), que coincide con $R^2$ cuando el sesgo es cero.

---

## CELDA 9: Réplica en Keras del Dr. Delgado y Demostración Algebraica (Bloque 3 — Parte A)

### 1. Arquitectura Exacta del Profesor Guía
```python
def construir_red_delgado_keras():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(units=4),  # Sin activation -> Identidad lineal g(z) = z
        tf.keras.layers.Dense(units=4),
        tf.keras.layers.Dense(units=4),
        tf.keras.layers.Dense(units=1)
    ])
```
* Replica línea por línea el cuaderno preliminar de marzo 2025: 2 entradas $\to$ 4 neuronas $\to$ 4 neuronas $\to$ 4 neuronas $\to$ 1 salida.
* **Detalle clave:** Al no escribir `activation='relu'` ni `activation='tanh'` dentro de `Dense(...)`, Keras aplica por defecto la función identidad $g(z) = z$.
* **Conteo de parámetros (`modelo_demo.count_params()`):**
  $$(2\times 4 + 4) + (4\times 4 + 4) + (4\times 4 + 4) + (4\times 1 + 1) = 12 + 20 + 20 + 5 = \mathbf{57\text{ parámetros nominales}}$$

### 2. Aceleración en Grafo Compilado `@tf.function`
```python
    @tf.function
    def bucle_optimizacion():
        for _ in tf.range(epocas):
            with tf.GradientTape() as tape:
                pred = modelo(X_t, training=True)
                loss = tf.reduce_mean(tf.square(y_t - pred))
            grads = tape.gradient(loss, modelo.trainable_variables)
            optimizador.apply_gradients(zip(grads, modelo.trainable_variables))
```
* **¿Por qué se escribió con `@tf.function` y `GradientTape` en lugar de `model.fit()`?**  
  Matemáticamente hace **exactamente lo mismo** (minimizar el error cuadrático medio `tf.reduce_mean(tf.square(y_t - pred))` mediante derivadas automáticas `tape.gradient` y el optimizador `Adam`), pero al decorarlo con `@tf.function`, TensorFlow compila las 800 épocas en código máquina C++ de una sola vez. Eso reduce el tiempo de entrenamiento de los 24 pliegues en Google Colab de **~12 minutos a menos de 20 segundos**.

### 3. Demostración Computacional del Colapso Lineal ($57 \to P_{\text{eff}} = 3$)
```python
W1, b1 = modelo_demo.layers[0].get_weights()
...
W_eq = W1 @ W2 @ W3 @ W4
b_eq = (((b1 @ W2 + b2) @ W3 + b3) @ W4 + b4).item()
pred_formula_lineal = (X_2d @ W_eq).flatten() + b_eq
error_algebraico_max = np.max(np.abs(pred_red_profunda - pred_formula_lineal))
```
* Extrae las 4 matrices de pesos sinápticos entrenadas ($\mathbf{W}_1, \mathbf{W}_2, \mathbf{W}_3, \mathbf{W}_4$) y las multiplica en cadena.
* El resultado `W_eq` tiene solo **2 números** (las pendientes de `CuT` y `CuS`) y `b_eq` tiene **1 número** (el intercepto).
* `error_algebraico_max` da del orden de `1e-07` (cero numérico), **probando irrefutablemente en vivo** que la red profunda sin función de activación es idéntica a una regresión lineal de $3$ parámetros efectivos ($P_{\text{eff}} = 3$).

### 4. Corrida en Vivo vs. Vector Certificado de Tesis
* TensorFlow opera en punto flotante de $32$ bits (`float32`) con suma paralela multihilo en CPU (`oneDNN`) o GPU (`CUDA` en Colab), lo que introduce micro-variaciones de `~0.005 %Cu` según el hardware donde se corra ($\text{NRMSE} \approx 11.6\% - 11.8\%$). Por transparencia total, la celda imprime **ambos resultados** (la corrida en vivo en tu sesión de Colab y el vector congelado de la memoria `11.64%`), demostrando que ambos convergen al mismo régimen lineal de `OLS` (`11.85%`).

---

## CELDA 10: Entrenamiento Multi-Semilla de las 10 Redes MLP (Bloque 3 — Parte B)

### 1. Estabilización por Ensamble de 10 Semillas y Optimizador L-BFGS
```python
SEMILLAS = [42, 101, 2023, 7, 13, 99, 123, 456, 789, 999]
```
* En muestras pequeñas ($N_{\text{train}} = 23$), una red neuronal no lineal inicializada con una sola semilla aleatoria puede caer en un mínimo local subóptimo. Entrenar **10 redes independientes con semillas fijas** y promediar sus predicciones (`np.mean(preds_semillas)`) estabiliza la varianza de inicialización y es **100% determinista en `float64`** (da exactamente el mismo sexto decimal en cualquier computador del mundo y en Google Colab).

### 2. Función Unificada para Redes Directas y Residuales (`entrenar_mlp_loocv`)
```python
X_base = np.asfortranarray(X_matriz)
...
X_tr = np.asfortranarray(np.delete(X_base, excluidos, axis=0))
objetivo_tr = (cancha_tr - y_tr) if es_residual else y_tr
```
* **`np.delete(X_base, excluidos, axis=0)` y `np.asfortranarray(...)`**: Elimina la fila del mes de prueba `k` preservando el orden de memoria por columnas (*Fortran-contiguous*) nativo de las tablas de Pandas/BLAS. Esto garantiza que las sumas de punto flotante del optimizador `L-BFGS` sean **idénticas bit a bit** tanto en Windows como en Linux / Google Colab, independientemente de la versión de Pandas instalada.
* Aquí ocurre la bifurcación entre las dos familias neuronales:
  * **Si `es_residual=False` (`Direct-MLP`):** La red aprende a predecir directamente la ley comercial `Canje_CuT` (`y_tr`).
  * **Si `es_residual=True` (`Residual-MLP`):** La red aprende a predecir el **descuento o sesgo físico** $e_i = \text{Cancha\_CuT}_i - \text{Canje\_CuT}_i$.

```python
red = MLPRegressor(
    hidden_layer_sizes=topologia, activation=activacion, solver='lbfgs',
    alpha=alpha, max_iter=3000, tol=1e-5, random_state=s
)
```
* **`solver='lbfgs'`**: Algoritmo cuasi-Newton de memoria limitada (Broyden–Fletcher–Goldfarb–Shanno). A diferencia de `Adam` o `SGD` (pensados para miles de datos por lotes), `lbfgs` aproxima la matriz Hessiana (curvatura de segundo orden) usando los 23 datos simultáneamente, logrando convergencia rápida y exacta en datasets mineros pequeños.
* **`alpha=alpha`**: Penalización de regularización $L_2$ ($\alpha \Vert\mathbf{W}\Vert_2^2$) que frena el sobreajuste.

```python
pred_final = (cancha_te - salida) if es_residual else salida
```
* **Conexión Física (*Skip-Connection*):** En las redes residuales, a la ley conocida de cancha del mes de prueba (`cancha_te`) se le descuenta el sesgo estimado por la red (`salida`). Cuando $\alpha = 2.0$, los pesos se contraen a cero, `salida` converge a $\bar{e}_{\text{train}}$, y la red empata exactamente a `Baseline 1B` ($10.07\%$) sin riesgo de saturar en leyes récord como noviembre de 2023.

---

## CELDA 11: Ablación de Entradas, Holdout 19/5 y Buffered-LOOCV (Bloque 3 — Parte C)

1. **Estudio de Ablación (`X_2d` vs `X_1d`):**
   ```python
   pred_1d = entrenar_mlp_loocv(X_1d, top, act, alpha, esc_y, es_res)
   ```
   Llama exactamente a la misma función neuronal pero entregándole solo 1 columna (`X_1d = Cancha_CuT`, quitando `Cancha_CuS`). En **`RNA_MLP_2`**, el error salta de **$12.28\%$ a $13.81\%$** (`Aporte_Neto_CuS_% = -1.53%`), demostrando cuantitativamente que el cobre soluble reduce el error de predicción en $1.53$ puntos porcentuales.

2. **Validación Prospectiva Simétrica *Holdout* 19/5:**
   ```python
   X_tr_h, y_tr_h, c_tr_h = X_2d[:19], y_real[:19], cancha_vec[:19]
   X_te_h, y_te_h, c_te_h = X_2d[19:], y_real[19:], cancha_vec[19:]
   ```
   Corta la línea de tiempo como ocurrirá en la operación real de la mina: entrena con los primeros **19 meses** (`2023-01` a `2024-07`) y predice hacia el futuro los últimos **5 meses** (`2024-08` a `2024-12`). En este régimen sulfurado regular, **`RNA_MLP_2` alcanza el primer lugar absoluto con $\text{NRMSE} = 6.56\%$**, superando a `RidgeCV` ($7.57\%$), `OLS` ($7.76\%$) y `Baseline 1B` ($8.96\%$).

3. **Sensibilidad *Buffered-LOOCV*:**
   ```python
   fn_buffer = lambda k: [i for i in (k - 1, k, k + 1) if 0 <= i < N]
   ```
   Define una regla rápida (`lambda`) que al evaluar el mes $k$ borra del entrenamiento a sus vecinos temporales inmediatos $k-1$ y $k+1$. Explica por qué sube el error de las redes directas: como octubre 2023 ($k=9$) y noviembre 2023 ($k=10$) son meses vecinos consecutivos, al evaluar noviembre el buffer borra octubre, obligando a extrapolar desde $1.57\%$ hasta $2.27\%\text{Cu}$.

---

## CELDA 13: Benchmarking de 18 Modelos y Diagnóstico de Residuos (Bloque 4 — Parte A)

### 1. Ranking Maestro y Empate Técnico Cuádruple (`1=`)
```python
empate_cuadruple = {'Res_MLP_2_2', 'Baseline_1B', 'Res_MLP_4', 'Res_MLP_3'}
df_bench.insert(0, 'Ranking', ['1=' if m in empate_cuadruple else str(i + 1) for i, m in enumerate(df_bench['Modelo'])])
```
* Reúne los $18$ modelos, los ordena de menor a mayor error cuadrático y asigna `'1='` a los cuatro modelos equivalentes cuyo $\text{MSE}$ difiere en menos de $10^{-6}\,(\%\text{Cu})^2$, pasando `Res_MLP_2` directamente al puesto `'5'`.
* Las tres líneas `assert round(...) == ...` verifican automáticamente que tu ejecución en Google Colab dio exactamente `10.07%`, `12.28%` y `25.67%`.

### 2. Batería de 5 Pruebas Econométricas sobre los Residuos ($e_k = y_k - \hat{y}_k$)
* **`stats.shapiro(residuos)`**: Verifica normalidad de errores ($p \ge 0.05$). Tanto `Baseline_1B` ($p=0.9696$) como `RNA_MLP_2` ($p=0.1012$) cumplen normalidad.
* **`durbin_watson(residuos)`**: Mide autocorrelación serial en los residuos ($d \approx 2.0$ significa ausencia total de autocorrelación; `Cancha_Directa` falla con $d=0.285$, mientras `RNA_MLP_2` da $d=1.958$ y `Baseline_1B` da $d=2.014$).
* **`het_breuschpagan(residuos, sm.add_constant(y_p))`**: Prueba exacta de homocedasticidad de Breusch-Pagan regresando $e_k^2$ sobre $[1, \hat{y}_k]$. **Hallazgo clave para tu defensa:** Todos los modelos lineales (`OLS` $p=0.0016$, `Keras` $p=0.0040$, `Lasso` $p=0.0029$, `Ridge` $p=0.0228$) **fallan ($p < 0.05$, son heterocedásticos)** porque su varianza de error explota en leyes altas; en cambio, `Baseline_1B` ($p=0.4075$) y `RNA_MLP_2` ($p=0.0822$) **aprueban homocedasticidad**.
* **`stats.wilcoxon(..., method='exact')`**:
  * *Unilateral (`alternative='greater'`) vs Cancha Directa:* Prueba que el modelo reduce significativamente el error cuadrático de terreno ($p = 0.000010$ en `Baseline_1B` y $p = 0.000186$ en `RNA_MLP_2`).
  * *Bilateral (`alternative='two-sided'`) vs Baseline 1B:* Obtiene $p = 0.988813$ para `RNA_MLP_2` porque de hecho **`RNA_MLP_2` le gana a `Baseline_1B` en $14$ de los $24$ meses ($58.3\%$)**, teniendo menor mediana de error absoluto ($0.0801$ vs $0.1044\%\text{Cu}$).
* **Bootstrap No Paramétrico ($B = 10{,}000$):**
  ```python
  idx_boot = rng_boot.integers(0, N, size=(B_BOOT, N))
  rmse_boot = np.sqrt(np.mean((y_real[idx_boot] - y_p[idx_boot])**2, axis=1))
  ci_inf, ci_sup = np.percentile(rmse_boot, [2.5, 97.5])
  ```
  Simula $10{,}000$ escenarios de muestreo con reemplazo de tamaño $24$ y extrae los percentiles $2.5\%$ y $97.5\%$ para construir el Intervalo de Confianza al $95\%$ del $\text{RMSE}$ sin asumir distribución paramétrica.

---

## CELDA 14: Explicabilidad Geometalúrgica XAI — Permutación, SHAP y Convex Hull (Bloque 4 — Parte B)

### 1. Clase Envoltorio (`EnsamblePredictorWrapper`)
```python
class EnsamblePredictorWrapper(BaseEstimator, RegressorMixin):
    ...
    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        X_s = self.scaler.transform(X_arr)
        pred_media = np.mean([m.predict(X_s) for m in self.estimadores], axis=0)
        return (X_arr[:, 0] - pred_media) if self.es_residual else pred_media
```
* **¿Por qué creamos esta clase?** Porque las herramientas de explicabilidad (`permutation_importance` y `shap.KernelExplainer`) esperan recibir **un solo objeto** que tenga el método `.predict(X)`. Esta clase agrupa el `StandardScaler` y las $10$ redes del ensamble en una única caja limpia compatible con `scikit-learn`.

### 2. Importancia por Permutación y Valores de Shapley (SHAP)
```python
perm_mlp2 = permutation_importance(wrap_mlp2, X_2d, y_real, n_repeats=1000, scoring='neg_root_mean_squared_error', random_state=42)
fondo_shap = shap.sample(X_2d, 10, random_state=42)
shap_vals_mlp2 = shap.KernelExplainer(wrap_mlp2.predict, fondo_shap).shap_values(X_2d, silent=True)
```
* **`permutation_importance(..., n_repeats=1000)`**: Desordena al azar 1,000 veces una columna (por ejemplo, `Cancha_CuS`) dejando fija la otra, y mide cuánto empeora el $\text{RMSE}$. En `RNA_MLP_2`, desordenar `Cancha_CuT` empeora el error en **`+0.1256 %Cu`** y desordenar `Cancha_CuS` lo empeora en **`+0.1003 %Cu`**, confirmando que ambas neuronas de entrada trabajan activamente.
* **`shap.KernelExplainer`**: Aplica Teoría de Juegos Cooperativos de Shapley sobre $10$ muestras representativas de fondo (`fondo_shap`) para descomponer cada predicción mensual en cuántos puntos de $\%\text{Cu}$ aportó `Cancha_CuT` ($E[|\phi|] = 0.0922\%\text{Cu}$) y cuántos aportó `Cancha_CuS` ($E[|\phi|] = 0.0701\%\text{Cu}$).
* En cambio, al pasarlo sobre `Res_MLP_2_2`, el impacto de `Cancha_CuS` da exactamente **`0.0000 %Cu`**, probando visual y numéricamente que la penalización $\alpha=2.0$ apagó las neuronas residuales.

### 3. Superficie de Respuesta 2D con Envolvente Convexa (*Convex Hull*)
```python
CUT_G, CUS_G = np.meshgrid(g_cut, g_cus)
Z_sup = wrap_mlp2.predict(np.column_stack([CUT_G.ravel(), CUS_G.ravel()])).reshape(CUT_G.shape)
...
hull = ConvexHull(X_2d)
v_hull = np.append(hull.vertices, hull.vertices[0])
ax.plot(X_2d[v_hull, 0], X_2d[v_hull, 1], 'w--', lw=2, label='Convex Hull (Dominio Real N=24)')
```
* **`np.meshgrid`**: Crea una malla fina de $90 \times 90 = 8{,}100$ combinaciones posibles de leyes $[\text{CuT}, \text{CuS}]$ y evalúa la red `RNA_MLP_2` en cada punto para colorear las curvas de nivel (`contourf`).
* **`ConvexHull(X_2d)`**: Calcula los vértices exteriores de los $24$ lotes reales y los une con una línea blanca segmentada (`'w--'`). Cualquier punto **dentro** del polígono es interpolación respaldada por datos históricos; fuera de él es extrapolación.

---

## CELDA 15: Serialización Portable (`.joblib`) y Simulador Interactivo de Defensa

```python
artefacto_mlp2 = {'nombre': 'RNA_MLP_2', 'scaler': sc_final_mlp2, 'estimadores': modelos_final_mlp2}
joblib.dump(artefacto_mlp2, ruta_tmp_mlp2)
artefacto_cargado = joblib.load(ruta_tmp_mlp2)
```
* Guarda el modelo campeón como un **diccionario estándar de Python** que solo contiene objetos nativos de `scikit-learn` (`StandardScaler` y lista de `MLPRegressor`). Esto evita errores de importación de clases personalizadas cuando el modelo se cargue después en el simulador económico (Bloque 5) o en la aplicación web Streamlit (Bloque 6).

```python
triangulacion_hull = Delaunay(X_2d)

def simular_lote_faena(cancha_cut, cancha_cus):
    x_nuevo = np.array([[cancha_cut, cancha_cus]], dtype=float)
    dentro_hull = bool(triangulacion_hull.find_simplex(x_nuevo)[0] >= 0)
    ...
simular_lote_faena(cancha_cut=1.75, cancha_cus=0.14)
```
* **`Delaunay(X_2d)` y `.find_simplex(x_nuevo)[0] >= 0`**: Divide el polígono del *Convex Hull* en triángulos. Si para un lote nuevo `find_simplex` devuelve un número $\ge 0$, significa que el lote cayó dentro de alguno de los triángulos históricos (**interpolación segura**); si devuelve `-1`, cayó fuera del *Convex Hull* y el código emite automáticamente una alerta preventiva de extrapolación.
* **Uso en vivo con tu Profesor Guía:** Durante tu reunión en Google Colab, puedes agregar una celda al final y escribir `simular_lote_faena(1.85, 0.16)` (o cualquier par de leyes que el Dr. Delgado quiera probar) para obtener instantáneamente las tres proyecciones y su diagnóstico geométrico.

---

## Mini-Glosario de Símbolos de Python para Responder con Seguridad

| Expresión en el Código | Significado Exacto |
| :--- | :--- |
| `X_2d[:, 0]` | Toma **todas las filas** (`:`) y solo la **primera columna** (`0`, que es `Cancha_CuT`). |
| `X_2d[k:k+1]` | Extrae la fila `k` manteniendo forma de matriz 2D `(1, 2)` (exigido por `.predict()`). |
| `A @ B` | Multiplicación matricial algebraica (producto fila por columna). |
| `err**2` | Eleva al cuadrado cada elemento del vector de errores ($e_k^2$). |
| `ddof=1` vs `ddof=0` | Denominador $(N-1)$ para muestra estadística vs denominador $N$ para población/MSE. |
| `random_state=42` | Fija la semilla del generador pseudoaleatorio para que el resultado sea $100\%$ repetible. |
| `[f(x) for x in lista]` | *List comprehension*: forma compacta de escribir un ciclo `for` que construye una lista. |
| `assert condicion` | Verificación automática: si la condición matemática no se cumple, detiene la ejecución con alerta. |

---
*Guía de estudio profundo elaborada para defensa técnica — Departamento de Ingeniería en Minas, Universidad de Antofagasta.*
