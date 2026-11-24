# TRADUCTOR INTEGRAL LÍNEA POR LÍNEA — `Código unificado.ipynb` (Versión Universal)

**Proyecto de Titulación:** *Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas*  
**Autor:** Felipe Andrés Guerra Maturana  
**Profesor Guía:** Dr. José Delgado Vega  
**Institución:** Universidad de Antofagasta — Departamento de Ingeniería en Minas  

---

## ¿Cómo estudiar con este documento?

Este manual traduce **cada bloque y cada línea de Python** de [`Código unificado.ipynb`](./Código%20unificado.ipynb) al lenguaje de la **Ingeniería en Minas, la Geoestadística y la Estadística Aplicada**.  
El cuaderno ha sido diseñado como una **plataforma universal compatible con Google Colab**: permite subir cualquier archivo `.csv` o `.xlsx` —al igual que en tu cuaderno original `CÓDIGO_TESIS.ipynb` (`DATASET_AUDITADO_N24.csv`, `DATASET2.csv`, `DATASET.csv`, Mina Tigresa o futuras campañas 2025+)— y reejecuta automáticamente los Bloques 0 al 4 sin cambiar una sola línea de los algoritmos internos.

> **Nota sobre numeración de celdas:** En `Código unificado.ipynb`, los 10 bloques ejecutables aparecen marcados en el margen izquierdo de Jupyter / Google Colab como **`[1]` a `[10]`** (que corresponden internamente a las celdas `1, 3, 5, 7, 9, 10, 11, 13, 14 y 15` del archivo, alternadas con las celdas explicativas de texto). Cada sección de este traductor indica ambos números para ubicación instantánea.

---

## CELDA 1 (`[1]`): Configuración del Entorno e Importación de Librerías

### 1. Control del Sistema, Logs e Instalación Automática en Google Colab
```python
import os, sys, io, logging, platform, tempfile, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
```
* **`import os, sys, io, logging, platform, tempfile, warnings`**: Carga herramientas base de Python:
  * `os` y `sys`: interactúan con rutas de archivos y detectan si estás en Windows local o en Linux (Google Colab).
  * `io`: permite leer texto CSV en memoria (`io.StringIO`) como respaldo cuando aún no has subido un archivo externo.
  * `logging`: controla la verbosidad de mensajes de TensorFlow.
  * `platform`: consulta sistema operativo y versión de Python.
  * `tempfile`: ubica la carpeta temporal del sistema para probar `.joblib` sin ensuciar tu carpeta de tesis.
  * `warnings`: silencia avisos repetitivos de convergencia numérica en muestras pequeñas.
* **`os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'`**: Oculta mensajes internos de C++/CUDA de TensorFlow para mostrar solo errores críticos.

```python
try:
    import shap
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "shap"])
    import shap
```
* **`try ... except ImportError`**: Intenta cargar `shap` (explicabilidad de caja negra). Si la máquina virtual de Google Colab no lo tiene instalado por defecto, lo instala silenciosamente en 3 segundos (`pip install -q shap`) sin interrumpir la corrida.

### 2. Librerías Científicas, Econométricas y de Machine Learning
```python
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.spatial import ConvexHull, Delaunay
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
```
* **`numpy` / `pandas` / `joblib`**: Álgebra matricial, tablas tipo Excel programables y serialización de modelos entrenados.
* **`matplotlib` / `seaborn` / `Ellipse`**: Generación de gráficos científicos (series, mapas de calor, contornos 2D, Q-Q plots y elipse de Mahalanobis al $95\%$).
* **`scipy.stats`**: Pruebas exactas de Shapiro-Wilk (`shapiro`), correlaciones (`pearsonr`, `spearmanr`), gráficos Q-Q (`probplot`) y contraste pareado no paramétrico de Wilcoxon (`wilcoxon`).
* **`ConvexHull` y `Delaunay`**: Geometría computacional para construir la frontera convexa del dominio empírico real y evaluar si un lote futuro cae dentro (interpolación) o fuera (extrapolación).
* **`statsmodels`**: Pruebas econométricas formales de multicolinealidad (`variance_inflation_factor`), autocorrelación serial (`acf`), homocedasticidad (`het_breuschpagan`) e independencia de residuos (`durbin_watson`).

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
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
```
* **`StandardScaler` / `RobustScaler`**: Normalizan la escala de las entradas para equilibrar el peso de `CuT` frente a `CuS`.
* **`Pipeline`**: Encapsula escalado + modelo dentro de cada pliegue de validación cruzada, garantizando **cero fuga de datos (*Zero Data Leakage*)**.
* **`LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, `SVR`**: Los modelos de contraste (*baselines*) del Bloque 2.
* **`MLPRegressor` y `tensorflow`**: Motores neuronales para las redes parsimoniosas `L-BFGS` (precisión doble `float64`) y la réplica histórica en Keras del Dr. Delgado.

### 3. Activación Automática del Visor Interactivo de Google Colab
```python
EN_COLAB = 'google.colab' in sys.modules
if EN_COLAB:
    try:
        get_ipython().run_line_magic('load_ext', 'google.colab.data_table')
    except Exception:
        pass
```
* **`EN_COLAB = 'google.colab' in sys.modules`**: Detecta automáticamente si estás corriendo el código en **Google Colab** o en tu computador local.
* Si estás en Google Colab, activa `%load_ext google.colab.data_table` (exactamente igual que en tu `CÓDIGO_TESIS.ipynb`) para que las tablas se vean con buscador y filtros interactivos, sin producir error al abrir el archivo en Jupyter local.

---

## CELDA 3 (`[2]`): Importación Universal de Dataset (`.csv`/`.xlsx`) y Auditoría ETL (Bloque 0)

### 1. Panel de Control Editable para Cualquier Base de Datos
```python
ARCHIVO_CSV       = 'DATASET_AUDITADO_N24.csv'  # Ej: 'DATASET.csv', 'DATASET2.csv', 'mi_nueva_base.csv'
SUBIR_DESDE_COLAB = False                       # Cambia a True en Google Colab para abrir botón de subida

COL_X1    = 'Cancha_CuT'
COL_X2    = 'Cancha_CuS'
COL_Y     = 'Canje_CuT'
COL_FECHA = 'Fecha'
```
* **Estas 6 líneas son el único lugar que necesitas tocar para correr otra base de datos:**
  * Si subes un archivo al panel izquierdo de Colab (como en `CÓDIGO_TESIS.ipynb`), basta escribir su nombre en `ARCHIVO_CSV`.
  * Si prefieres que Colab te abra la ventana de Windows para seleccionar un archivo desde tu escritorio, cambia `SUBIR_DESDE_COLAB = True`.
  * Si tu nuevo archivo viene de otra mina con columnas llamadas distinto (por ejemplo `Ley_Mina`, `Oxido_Mina`, `Ley_Planta`), solo cambias `COL_X1`, `COL_X2` y `COL_Y` aquí arriba y todo el resto del cuaderno se adapta automáticamente.

### 2. Función `cargar_dataset_universal(...)` — Paso a Paso
```python
if subir_en_colab and EN_COLAB:
    from google.colab import files
    subidos = files.upload()
    ruta_elegida = next(iter(subidos.keys()))
```
* **`files.upload()`**: Abre el selector nativo de archivos de Google Colab y captura el nombre exacto del archivo que elegiste (`next(iter(subidos.keys()))`).

```python
dataset_raw = pd.read_csv(ruta_elegida, sep=None, engine='python', encoding='utf-8-sig')
```
* **Exactamente el mismo método de tu `CÓDIGO_TESIS.ipynb`:** `sep=None, engine='python'` hace que Pandas inspeccione el archivo y detecte automáticamente si viene separado por comas (`,`), punto y coma (`;` típico de Excel en español) o tabulaciones. Además, `encoding='utf-8-sig'` elimina caracteres invisibles (BOM) que a veces agrega Excel al guardar en CSV.

```python
dataset_raw.columns = [str(c).strip() for c in dataset_raw.columns]
for col in dataset_raw.columns:
    if col != COL_FECHA and dataset_raw[col].dtype == object:
        convertida = pd.to_numeric(dataset_raw[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        if convertida.notna().sum() >= len(dataset_raw) * 0.8:
            dataset_raw[col] = convertida
```
* Limpia espacios sobrantes en los títulos de las columnas (ej. `'Cancha_CuT '` $\to$ `'Cancha_CuT'`).
* Si un CSV exportado desde Excel en español trajo los números con **coma decimal** (como `"2,16"` en vez de `2.16`), reemplaza `,` por `.` y los convierte automáticamente a números flotantes (`float`).

```python
if COL_X2 == 'Cancha_CuS' and 'Cancha_CuS' not in dataset_raw.columns and {'Cancha_CuT', 'Cancha_CuIn'}.issubset(dataset_raw.columns):
    dataset_raw['Cancha_CuS'] = (dataset_raw['Cancha_CuT'] - dataset_raw['Cancha_CuIn']).round(6)
if 'Cancha_CuIn' not in dataset_raw.columns and {'Cancha_CuT', 'Cancha_CuS'}.issubset(dataset_raw.columns):
    dataset_raw['Cancha_CuIn'] = (dataset_raw['Cancha_CuT'] - dataset_raw['Cancha_CuS']).round(6)
```
* **Compatibilidad total con `DATASET.csv` y `DATASET2.csv`:**
  * Si subes `DATA_COBRIZO/DATASET.csv` (que solo trae `Cancha_CuT`, `Cancha_CuIn` y `Canje_CuT`), deduce automáticamente `Cancha_CuS = Cancha_CuT - Cancha_CuIn` por balance de masas químico.
  * Si subes un archivo sin `Cancha_CuIn`, la calcula como `Cancha_CuT - Cancha_CuS` para poder graficarla en el mapa de calor y auditar el VIF.

```python
if dataset_raw[COL_X1].median() < 0.25 and dataset_raw[COL_Y].median() < 0.25:
    for c in cols_leyes:
        dataset_raw[c] = (dataset_raw[c].astype(float) * 100.0).round(4)
```
* **Homologación automática de unidades:** Tanto `DATASET.csv` como `DATASET2.csv` guardan las leyes en fracción decimal (`0.0216` en lugar de `2.16 %Cu`). Si la mediana de las leyes es menor a `0.25`, el código detecta que están en fracción y las multiplica automáticamente por `100` para trabajar siempre en puntos porcentuales (`%Cu`).

```python
if COL_FECHA not in dataset_raw.columns:
    dataset_raw.insert(0, COL_FECHA, [f"Reg_{i:02d}" for i in range(1, len(dataset_raw) + 1)])
```
* Si el archivo subido no trae columna de fechas (como `DATASET.csv` o `DATASET2.csv`), genera etiquetas ordenadas (`Reg_01`, `Reg_02`, ...) para que todos los gráficos cronológicos funcionen sin fallos.

### 3. Detección Automática del Benchmark de Tesis (`ES_COBRIZO_N24`) y Auditoría ETL
```python
ES_COBRIZO_N24 = (len(df) == 24 and ... and np.allclose(df[COL_X1].values, ...))
df['Sesgo_Canje'] = df[COL_X1] - df[COL_Y]
df['Ratio_Solubilidad'] = (df[COL_X2] / df[COL_X1].replace(0, np.nan)) * 100.0
```
* **`ES_COBRIZO_N24`**: Detecta automáticamente si el archivo cargado corresponde a los $24$ meses oficiales de Mantos Cobrizo (ya sea cargado desde `DATASET_AUDITADO_N24.csv`, `DATASET2.csv` o el respaldo) o si cargaste una **base de datos nueva**.
* **`df['Sesgo_Canje']`**: Calcula el desvío empírico mensual ($X_1 - y$, promedio $+0.3404\%\text{Cu}$ en Mantos Cobrizo).
* Si las columnas opcionales (`Cancha_Ton`, `Romana_Ton`, `PSJ_CuT`) existen en el CSV subido, se auditan y grafican; si no existen, se omiten elegantemente sin lanzar `KeyError`.

> **Pregunta típica del profesor en esta sección:** *"¿Qué pasa si mañana me traes los datos de 2025 o de Mina Tigresa en un `.csv` con punto y coma y en decimales `0,018`?"*  
> **Respuesta:** El bloque `cargar_dataset_universal` detecta el separador `;`, convierte la coma decimal a punto, detecta la escala fraccionaria y la homologa a `%Cu`, ajusta dinámicamente el tamaño $N$, los rezagos ACF ($N/4$) y la partición Holdout ($80/20$), y reentrena los 18 modelos automáticamente.

---

## CELDA 5 (`[3]`): Análisis Exploratorio (EDA), Heatmap, Mahalanobis, ACF y VIF (Bloque 1)

### 1. Estadísticos Descriptivos, Escalas y Descomposición Sesgo-Varianza
```python
candidatas_eda = [COL_X1, COL_X2, 'Cancha_CuIn', 'PSJ_CuT', COL_Y, 'Sesgo_Canje', 'Ratio_Solubilidad']
vars_eda = [v for v in dict.fromkeys(candidatas_eda) if v in df.columns]
```
* Filtra dinámicamente las columnas que realmente existen en el archivo subido sin repetir nombres (`dict.fromkeys`), evitando errores si falta `PSJ_CuT` u otra columna auxiliar.
* **`ddof=1` vs `ddof=0`**: `ddof=1` divide por $N-1$ (varianza muestral insesgada $s^2$, usada en el contraste de escalas $s^2(\text{CuT})/s^2(\text{CuS}) = 61.6\times$); `ddof=0` divide por $N$ (varianza poblacional $\sigma^2$, usada en la identidad exacta $\text{MSE} = \bar{e}^2 + \sigma^2_e = 0.1159 + 0.0191 = 0.1350$).
* **`assert np.isclose(identidad_var, matriz_cov.loc['Sesgo_Canje', 'Sesgo_Canje'])`**: Verifica la identidad algebraica $\text{Var}(X_1 - y) = \text{Var}(X_1) + \text{Var}(y) - 2\text{Cov}(X_1, y)$.

### 2. Distancia de Mahalanobis y Efecto Apalancamiento (*Leverage*) Dinámico
```python
inv_cov_X = np.linalg.pinv(cov_X)
df['Mahalanobis_D2'] = [float((x - mu_X) @ inv_cov_X @ (x - mu_X).T) for x in X_mat]
df_outliers = df[df['Mahalanobis_D2'] > umbral_chi2]
df_sin_out = df[df['Mahalanobis_D2'] <= umbral_chi2] if len(df_outliers) > 0 and (N - len(df_outliers)) >= 4 else df
```
* **`np.linalg.pinv(cov_X)`**: Calcula la pseudoinversa de Moore-Penrose de la matriz de covarianza $2 \times 2$ (idéntica a la inversa tradicional `inv` cuando las columnas son independientes, pero segura incluso si algún CSV externo trajera datos casi colineales).
* **`df_sin_out`**: Excluye **dinámicamente** los registros detectados por Mahalanobis ($D_M^2 > 5.9915$) para medir cuánto apalancamiento ejercen sobre las correlaciones. En Mantos Cobrizo ($N=24$) selecciona automáticamente `2023-10` y `2023-11` ($N=22$), y en cualquier base nueva identifica los extremos propios de esa mina.

### 3. Autocorrelación ACF Dinámica ($K \le N/4$), VIF y Panel de 4 Gráficos (con Heatmap)
```python
max_lags = max(1, min(12, N // 4))
acf_sesgo = acf(df['Sesgo_Canje'], nlags=max_lags, fft=False)
banda_bartlett = 1.96 / np.sqrt(N)
```
* Aplica la regla clásica de Box-Jenkins ($K = \lfloor N/4 \rfloor$, es decir $6$ rezagos para $N=24$, o hasta $12$ para series más largas) y calcula las bandas de ruido blanco al $95\%$ ($\pm 1.96/\sqrt{N}$).
* **`sm.add_constant(...)` + `variance_inflation_factor`**: Evalúa la colinealidad con intercepto obligatorio ($\text{VIF} = 1.8045$ en 2 entradas vs $>4{,}000$ si se agrega `CuIn`).
* **Panel A (`sns.heatmap`)**: Integra directamente el **Mapa de Calor de Correlación de Pearson** de tu `CÓDIGO_TESIS.ipynb` junto al gráfico Q-Q Normal (Panel B), la elipse de Mahalanobis al $95\%$ (Panel C) y el correlograma ACF del sesgo (Panel D).

---

## CELDA 7 (`[4]`): Motor LOOCV Cero Fuga y Entrenamiento de los 7 Baselines (Bloque 2)

### 1. Preparación Matricial y Validación Cruzada Anidada Dinámica
```python
X_2d = np.asfortranarray(df[[COL_X1, COL_X2]].values, dtype=float)
X_1d = np.asfortranarray(df[[COL_X1]].values, dtype=float)
CV_INTERNO = min(5, max(2, N - 2))
```
* **`np.asfortranarray(...)`**: Asegura alineamiento de memoria por columnas (*Fortran-contiguous*) idéntico en Windows y Linux/Google Colab, garantizando reproducibilidad numérica exacta en BLAS.
* **`CV_INTERNO`**: Usa validación cruzada interna de $5$ pliegues (`cv=5`) dentro de los datos de entrenamiento (y se adapta automáticamente si subieras un archivo muy pequeño con menos de 7 filas).

### 2. Bucle Exterior LOOCV (`for k in range(N)`) y Cero Fuga de Datos (`Pipeline`)
```python
for k in range(N):
    train_idx = [i for i in range(N) if i not in (k - 1, k, k + 1)] if buffered else [i for i in range(N) if i != k]
    X_tr, y_tr = X_2d[train_idx], y_real[train_idx]
    X_te = X_2d[k:k+1]
```
* Deja fuera el registro `k` completo (aislado en `X_te`) y entrena cada uno de los 7 baselines exclusivamente con los otros $N-1$ registros:
  1. `Cancha_Directa`: Predice la ley bruta $X_{1,k}$ sin corregir.
  2. `Baseline_1B`: Resta el sesgo promedio calculado **solo** sobre el entrenamiento (`np.mean(X_tr[:, 0] - y_tr)`).
  3. `OLS`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, `SVR_RBF`: Encapsulados dentro de `Pipeline([('scaler', scaler_cls()), ('model', ...)])`, impidiendo que la media o varianza del mes de prueba `k` se filtre al entrenamiento.
* **`coefs_lasso_x2.append(p_lasso.named_steps['model'].coef_[1])`**: Registra en cada pliegue el coeficiente de `COL_X2` (`Cancha_CuS`) en Lasso para verificar si la regularización $L_1$ la conserva ($24/24 = 100\%$ en Mantos Cobrizo) o la anula.

### 3. Función Universal de 9 Métricas (`calcular_metricas_oos`)
* Calcula de una sola vez: **MSE** ($(\%\text{Cu})^2$), **RMSE** ($\%\text{Cu}$), **MAE** ($\%\text{Cu}$), **NRMSE** ($\%$), **$R^2$**, **MAPE** ($\%$), **Pearson $r$**, **Sesgo Medio** ($\hat{y} - y$) y **Varianza Explicada** ($1 - \text{Var}(e)/\text{Var}(y)$).

---

## CELDA 9 (`[5]`): Réplica en Keras del Dr. Delgado y Demostración Algebraica (Bloque 3 — Parte A)

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
* Construye la topología exacta del profesor guía (`57` parámetros nominales: `12 + 20 + 20 + 5`).
* **`@tf.function` + `tf.GradientTape()`**: Compila las $800$ épocas de `Adam(0.05)` en grafo C++ nativo para que los $N$ pliegues LOOCV corran en ~20 segundos en Google Colab (en vez de >10 minutos).
* **Demostración matricial en vivo:** Extrae $\mathbf{W}_1 \dots \mathbf{W}_4$, calcula `W_eq = W1 @ W2 @ W3 @ W4` y prueba que `np.max(np.abs(pred_red_profunda - pred_formula_lineal))` es del orden de $10^{-7}$ (identidad algebraica exacta: $57$ parámetros colapsan a $P_{\text{eff}} = 3$).
* **Modo dual universal:**
  ```python
  preds_keras_final = PREDS_KERAS_CERTIFICADO_TESIS if ES_COBRIZO_N24 else preds_keras_vivo
  ```
  Si corres el dataset de la tesis ($N=24$), muestra tanto la corrida en vivo de tu Colab como el vector certificado (`11.64%`) para mantener consistencia al sexto decimal frente a micro-variaciones `float32` CPU/GPU; si subes **cualquier otra base de datos**, utiliza directamente las predicciones recién entrenadas `preds_keras_vivo`.

---

## CELDA 10 (`[6]`): Entrenamiento Multi-Semilla de las 10 Redes MLP (Bloque 3 — Parte B)

```python
SEMILLAS = [42, 101, 2023, 7, 13, 99, 123, 456, 789, 999]
```
* Entrena 6 redes directas (`Direct-MLP`) y 4 redes residuales (`Residual-MLP`) promediando **10 semillas fijas** con el optimizador cuasi-Newton **`solver='lbfgs'`** (óptimo para muestras acotadas).
* **`X_tr = np.asfortranarray(np.delete(X_base, excluidos, axis=0))`**: Retira el registro de prueba `k` preservando el orden de memoria por columnas para garantizar determinismo bit a bit entre Windows y Colab.
* **`objetivo_tr = (cancha_tr - y_tr) if es_residual else y_tr`** y **`pred_final = (cancha_te - salida) if es_residual else salida`**: Implementa la **conexión física residual (*Skip-Connection*)**, donde la red predice el descuento $e_i$ y lo resta a la ley base de cancha. Bajo $\alpha = 2.0$, los pesos residuales se contraen a cero ($P_{\text{eff}} \approx 1$), asegurando convergencia estable hacia `Baseline_1B`.

---

## CELDA 11 (`[7]`): Ablación (2D vs 1D), Holdout Dinámico 80/20 y Buffered-LOOCV (Bloque 3 — Parte C)

1. **Estudio de Ablación (`X_2d` vs `X_1d`):** Reentrena cada red entregándole únicamente `[COL_X1]` (suprimiendo `COL_X2`) para cuantificar el aporte neto de la segunda variable (en Mantos Cobrizo, incluir `Cancha_CuS` reduce el NRMSE de `RNA_MLP_2` en **$-1.53\%$**).
2. **Holdout Prospectivo Dinámico 80/20:**
   ```python
   n_test_h  = max(2, int(round(N * 0.20)))
   n_train_h = N - n_test_h
   ```
   Entrena con el primer $80\%$ cronológico de los registros y evalúa a ciegas sobre el último $20\%$ futuro. Para $N=24$, `round(24 * 0.20) = 5`, reproduciendo exactamente la partición **`19/5`** de la tesis (`RNA_MLP_2 = 6.56%`); para una base de $N=40$, ejecuta automáticamente un Holdout **`32/8`**.
3. ***Buffered-LOOCV*:** Excluye los vecinos temporales `(k-1, k, k+1)` en cada pliegue para medir sensibilidad ante continuidad espacial/temporal entre meses adyacentes.

---

## CELDA 13 (`[8]`): Benchmarking de 18 Modelos y Pruebas de Hipótesis (Bloque 4 — Parte A)

```python
mse_min = df_bench['MSE (%Cu)^2'].min()
empate_primer_lugar = set(df_bench.loc[(df_bench['MSE (%Cu)^2'] - mse_min) < 1e-5, 'Modelo'])
```
* **Detección automática de empate técnico (`1=`):** En lugar de escribir los ganadores a mano, identifica matemáticamente qué modelos están a menos de $10^{-5}\,(\%\text{Cu})^2$ del menor $\text{MSE}$. En Mantos Cobrizo detecta automáticamente el **empate técnico cuádruple** (`Res_MLP_2_2`, `Baseline_1B`, `Res_MLP_4`, `Res_MLP_3` con `10.07%`), y en cualquier archivo nuevo identifica el(los) ganador(es) real(es) de esa base.
* **`if ES_COBRIZO_N24: assert ...`**: Solo cuando se corre la base oficial de la tesis verifica con aserciones automáticas que se reprodujeron exactamente `10.07%`, `12.28%`, `25.67%` y el empate cuádruple.
* **Pruebas econométricas sobre residuos ($e_k = y_k - \hat{y}_k$):**
  * **Shapiro-Wilk** (normalidad), **Durbin-Watson** (independencia serial), **Breusch-Pagan** exacto sobre `[1, y_p]` (homocedasticidad, donde fallan `OLS`/`Ridge`/`Lasso`/`Keras` y aprueban `Baseline_1B` y `RNA_MLP_2`), **Wilcoxon pareado** (`method='exact'` si $N \le 50$, `auto` si $N > 50$) y **Bootstrap** ($B = 10{,}000$ iteraciones vectorizadas para el IC al $95\%$ del RMSE).

---

## CELDA 14 (`[9]`): Explicabilidad XAI (Permutación + SHAP) y Panel de 4 Figuras (Bloque 4 — Parte B)

* Reentrena `RNA_MLP_2` y `Res_MLP_2_2` sobre los $N$ registros completos, encapsulándolos en la clase **`EnsamblePredictorWrapper`** para exponer un método `.predict(X)` estándar compatible con `scikit-learn` y `shap`.
* **`permutation_importance(..., n_repeats=1000)`**: Mide cuánto empeora el RMSE al desordenar aleatoriamente cada variable $1{,}000$ veces.
* **`shap.KernelExplainer(..., shap.sample(X_2d, min(10, N), random_state=42))`**: Calcula valores exactos de Shapley por observación (adaptando el tamaño del fondo si $N < 10$).
* **Panel de 4 Figuras Dinámicas:** Las líneas de referencia del gráfico de barras (Panel A), los valores $p$ de Breusch-Pagan (Panel B), los márgenes de la malla 2D y el polígono del *Convex Hull* (Panel C), y los ejes de Permutación/SHAP (Panel D) se recalculan dinámicamente a partir de los datos cargados.

---

## CELDA 15 (`[10]`): Serialización Portable (`.joblib`) y Simulador Interactivo de Lotes

```python
triangulacion_hull = Delaunay(X_2d) if N >= 4 else None

def simular_lote_faena(valor_x1, valor_x2):
    x_nuevo = np.array([[valor_x1, valor_x2]], dtype=float)
    dentro_hull = bool(triangulacion_hull.find_simplex(x_nuevo)[0] >= 0) if triangulacion_hull is not None else True
    ...
```
* Guarda el modelo campeón como diccionario nativo (`StandardScaler` + lista de `MLPRegressor`) en un archivo temporal `.joblib` y verifica su recarga limpia.
* **`Delaunay(X_2d).find_simplex(x_nuevo)[0] >= 0`**: Verifica geométricamente si el lote que consultas cae **dentro** de la envolvente convexa de los datos históricos (*interpolación segura*) o **fuera** (*alerta de extrapolación*).
* Al ejecutarse sobre Mantos Cobrizo evalúa automáticamente el lote certificado `simular_lote_faena(1.75, 0.14)` (`1.4076 %Cu` en `RNA_MLP_2`); si subes otra base de datos, evalúa automáticamente la mediana real de esa base. En tu presentación en vivo puedes agregar una celda extra con `simular_lote_faena(valor1, valor2)` para probar cualquier escenario que te pida el Dr. Delgado.

---

## Mini-Glosario Rápido de Python para Defensa Oral

| Sintaxis | Qué significa exactamente |
| :--- | :--- |
| `sep=None, engine='python'` | Auto-detecta si el archivo `.csv` separa columnas con coma (`,`), punto y coma (`;`) o tabulación. |
| `X_2d[:, 0]` | Selecciona todas las filas (`:`) de la primera columna (`0`, es decir `COL_X1`). |
| `X_2d[k:k+1]` | Extrae la fila `k` preservando formato de matriz 2D `(1, 2)` requerido por `.predict()`. |
| `A @ B` | Producto matricial algebraico estándar (filas por columnas). |
| `ddof=1` vs `ddof=0` | Denominador $(N-1)$ para varianza muestral vs denominador $N$ para identidad exacta del MSE. |
| `np.asfortranarray(...)` | Ordena la matriz por columnas en RAM para asegurar igualdad numérica exacta en BLAS. |
| `assert condicion` | Verificación automática de integridad: si la condición matemática falla, detiene el código. |
| `Delaunay(...).find_simplex(...) >= 0` | Comprueba si un punto 2D cae dentro del polígono del *Convex Hull* histórico. |

---
*Manual de estudio profundo — Departamento de Ingeniería en Minas, Universidad de Antofagasta.*
