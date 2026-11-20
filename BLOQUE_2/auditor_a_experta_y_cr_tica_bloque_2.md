# AUDITORÍA EXPERTO-CRÍTICA — BLOQUE 2
## Verificación Numérica, Análisis Geometalúrgico de los Modelos de Referencia y Directrices para la Red Neuronal

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas  
**Documento Auditado:** `BLOQUE_2/INFORME_ENTREGABLE_BLOQUE_2.md` (Versión 4.0)  
**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)  
**Estándar de Evaluación:** Rigor académico de Ingeniería Civil en Minas (Universidad de Antofagasta — Comisión con especialidad en Geoestadística y Ciencia de Datos)  

---

## 1. Dictamen Ejecutivo de Auditoría

> **Dictamen:** **BLOQUE 2 APROBADO Y SELLADO PARA AVANZAR AL BLOQUE 3** *(con ajustes menores de estandarización documental para su integración en el Bloque 4).*

La ejecución del **Bloque 2** cumple rigurosamente con los cuatro compromisos sellados en la auditoría del Bloque 1 y con la **Metodología v4.0**:

1. **Cero fuga de información (*Zero Data Leakage*):** Todas las transformaciones de escalado y búsquedas de hiperparámetros se encapsularon dentro de `sklearn.pipeline.Pipeline` en cada uno de los $24$ pliegues externos LOOCV ($23$ entrenamiento / $1$ prueba ciega).
2. **Exactitud matemática verificada al $100\%$:** Se recalcularon de forma independiente todas las métricas de la Tabla 3.1 a partir de los $24$ registros de la Tabla 3.2 (`PREDICCIONES_OOS_BASELINES`), confirmando coincidencia exacta hasta el cuarto decimal.
3. **Hallazgo clave sobre `Cancha_CuS`:** `LassoCV` retuvo un coeficiente positivo para $\text{Cancha\_CuS}$ en **$24$ de $24$ pliegues ($100\%$)**, refutando la hipótesis de que la solubilidad fuera una variable espuria.
4. **Blindaje de escala y geoestadística:** Las pruebas de sensibilidad con `RobustScaler` ($|\Delta\text{NRMSE}| \le 0.09\%$) y `Buffered-LOOCV` ($\Delta\text{NRMSE} = +0.04\%$ en Lasso) demuestran que los modelos no dependen de artefactos de escala ni de memoria temporal autorregresiva ($r_1 = +0.4467$ en $\text{Cancha\_CuT}$).

---

## 2. Verificación Numérica Independiente (Tabla 3.1 vs. Tabla 3.2)

Se auditó la consistencia aritmética entre las predicciones fuera de muestra mes a mes (Tabla 3.2) y los indicadores globales reportados (Tabla 3.1):

| Modelo Evaluado ($N = 24$ OOS) | $\text{RMSE}$ Reportado ($\%\text{Cu}$) | $\text{RMSE}$ Verificado ($\%\text{Cu}$) | $\text{NRMSE}$ Reportado ($\%$) | $\text{NRMSE}$ Verificado ($\%$) | $R^2$ Reportado | $R^2$ Verificado | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Línea Base 1A (Cancha Directa)** | $0.3674$ | $0.3674$ | $25.67\%$ | $25.67\%$ | $-1.2709$ | $-1.2709$ | Exacto |
| **Baseline 1B (Sesgo LOOCV)** | $0.1442$ | $0.1442$ | $10.07\%$ | $10.07\%$ | $+0.6503$ | $+0.6503$ | Exacto |
| **OLS (Regresión Lineal)** | $0.1697$ | $0.1697$ | $11.85\%$ | $11.85\%$ | $+0.5156$ | $+0.5156$ | Exacto |
| **RidgeCV ($L_2$)** | $0.1685$ | $0.1685$ | $11.78\%$ | $11.78\%$ | $+0.5221$ | $+0.5221$ | Exacto |
| **LassoCV ($L_1$)** | $0.1760$ | $0.1760$ | $12.30\%$ | $12.30\%$ | $+0.4789$ | $+0.4789$ | Exacto |
| **ElasticNetCV ($L_1 + L_2$)** | $0.1769$ | $0.1769$ | $12.36\%$ | $12.36\%$ | $+0.4735$ | $+0.4735$ | Exacto |
| **SVR (Kernel RBF)** | $0.2387$ | $0.2387$ | $16.68\%$ | $16.68\%$ | $+0.0412$ | $+0.0412$ | Exacto |

---

## 3. Análisis Experto de los Resultados: Por Qué Ocurre Cada Fenómeno

Más allá de reportar la tabla, debes dominar la explicación física y matemática de **cuatro comportamientos críticos** revelados en este bloque, ya que constituirán el núcleo de tu discusión ante el Dr. José Delgado y la comisión evaluadora:

### 3.1. ¿Por qué Baseline 1B ($\text{NRMSE} = 10.07\%$) supera a OLS ($11.85\%$) y Ridge ($11.78\%$)?
Un evaluador atento notará de inmediato que restar el sesgo constante en LOOCV (`Baseline 1B`) logra un error menor que ajustar una regresión lineal múltiple (`OLS` o `RidgeCV`). La explicación matemática y metalúrgica es contundente:

1. **Atenuación de pendiente (*Slope Attenuation*) en modelos lineales:**
   * `Baseline 1B` impone implícitamente una pendiente unitaria ($\beta_{\text{CuT}} = 1.0$) sobre el cobre total: $\hat{y} = 1.0 \cdot \text{CuT}_{\text{cancha}} - \bar{e}_{-k}$.
   * En cambio, `OLS` estima un coeficiente estandarizado promedio $\bar{\beta}_{\text{CuT}} = +0.1782$, lo que en unidades físicas equivale a una pendiente achatada ($\approx 0.74 < 1.0$) arrastrada hacia abajo por los meses de leyes medias/bajas con alto sesgo (como marzo–mayo 2024).
2. **El impacto concentrado de Octubre y Noviembre 2023 ($40.5\%$ del error total de OLS):**
   * **En Octubre 2023** ($\text{Cancha\_CuT} = 2.21\%$, $\text{Cancha\_CuS} = 0.26\%$, $\text{Canje\_CuT} = 1.79\%$): Al dejar este mes fuera, el modelo lineal aprende una pendiente positiva para $\text{CuS}$ ($\bar{\beta}_{\text{CuS}} = +0.0366$) a partir de los meses sulfurados normales. Cuando evalúa Octubre 2023 con su pico anómalo de cobre soluble ($0.26\%$), **OLS suma un ajuste positivo excesivo y predice $2.1184\%$** frente a un canje real de $1.79\%$ (error de $+0.3284\%\text{Cu}$). Metalúrgicamente, en una planta de flotación de sulfuros (Planta San José), un exceso de óxidos solubles no se recupera linealmente en el concentrado, provocando un quiebre no lineal que un plano rígido no puede doblar.
   * **En Noviembre 2023** ($\text{Cancha\_CuT} = 2.47\%$, $\text{Cancha\_CuS} = 0.20\%$, $\text{Canje\_CuT} = 2.27\%$): Al dejar este mes fuera, la pendiente achatada de OLS subestima fuertemente este frente sulfurado de alta ley, prediciendo **$1.8555\%$** frente a $2.27\%$ (error de $-0.4145\%\text{Cu}$), mientras que `Baseline 1B` predice $2.1235\%$ (error de solo $-0.1465\%\text{Cu}$).
   * En conjunto, **solo estos dos meses explican el $40.5\%$ de todo el error cuadrático de OLS** ($0.2797$ de $0.6909$).

### 3.2. ¿Por qué Ridge ($L_2$) aumenta el peso de $\text{CuS}$ mientras contrae $\text{CuT}$?
En la Sección 4.1 se observa un fenómeno muy interesante:
* En `OLS`: $\bar{\beta}_{\text{CuT}} = +0.1782$ y $\bar{\beta}_{\text{CuS}} = +0.0366$.
* En `RidgeCV` ($\alpha \in [5.72, 17.48]$): $\bar{\beta}_{\text{CuT}}$ se contrae un $34.5\%$ hasta $+0.1167$, mientras que $\bar{\beta}_{\text{CuS}}$ **aumenta un $46.7\%$** hasta $+0.0537$.

**Explicación técnica:** Debido a que $\text{CuT}$ y $\text{CuS}$ presentan colinealidad moderada positiva ($r = +0.6677$, $\text{VIF} = 1.8045$), la penalización cuadrática $L_2$ castiga el cuadrado del coeficiente grande ($\beta_{\text{CuT}}^2$) y redistribuye parte de la carga explicativa hacia la variable correlacionada ($\beta_{\text{CuS}}$). Esto confirma que ambas variables trabajan de forma acoplada.

### 3.3. ¿Por qué SVR con Kernel RBF fracasa ($\text{NRMSE} = 16.68\%$, $R^2 = +0.0412$)?
El bajo desempeño de `SVR (RBF)` es un argumento experimental valioso para defender la elección de una **Red Neuronal MLP parsimoniosa**:
* El kernel gaussiano $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ es un **aproximador local**: cuando un punto de prueba cae en una región de baja densidad de vecinos dentro del plano $(\text{CuT}, \text{CuS})$, la función decae rápidamente hacia el intercepto medio ($\approx 1.35\%\text{–}1.40\%\text{Cu}$).
* Esto se observa con claridad en **Septiembre 2023** ($\text{CuT} = 2.01\%, \text{CuS} = 0.12\% \implies \text{SVR predice } 1.0603\%$ vs. real $1.55\%$), **Noviembre 2023** ($\text{SVR predice } 1.6132\%$ vs. real $2.27\%$) y **Diciembre 2024** ($\text{SVR predice } 1.4063\%$ vs. real $1.06\%$).
* Por el contrario, un **Perceptrón Multicapa (MLP) con activación `relu`** construye hiperplanos globales continuos por tramos, permitiéndole extrapolar sin colapsar hacia la media en los bordes del dominio.

### 3.4. Interpretación de `Pearson r` en la Tabla 3.1 (Advertencia Metodológica)
En la Tabla 3.1 se incluye `Pearson r` como columna comparativa, donde `Cancha Directa` muestra el valor más alto ($r = +0.8347$) pese a ser el peor predictor ($R^2 = -1.2709$). Al trasladar esta tabla al **Bloque 4** y a la memoria final, mantén la advertencia explícita de que `Pearson r` mide únicamente sincronía de tendencia y es insensible al sesgo de nivel ($+0.3404\%\text{Cu}$), por lo que las métricas rectoras de precisión son **$\text{RMSE}$ ($\text{MSE}$), $\text{NRMSE}$ ($\text{MAPE}$), $R^2$ y Varianza Explicada**.

---

## 4. Observaciones Menores para Trazabilidad hacia los Bloques 3 y 4

No se requiere reejecutar el Bloque 2, pero deja registradas estas **4 observaciones** para mantener consistencia perfecta en los siguientes módulos:

1. **Métricas faltantes que deberán incluirse en la Tabla Unificada del Bloque 4:**
   Para cumplir íntegramente con la Sección 4 de `METODOLOGIA_TRABAJO_TESIS.md` (v4.0) y el estándar de la Universidad de Antofagasta, cuando el **Bloque 4** lea `PREDICCIONES_OOS_BASELINES.csv` junto a `PREDICCIONES_OOS_RNA.csv`, deberá añadir las columnas:
   * **`MSE` ($(\%\text{Cu})^2$):** Función de pérdida directa ($0.1350$ en Cancha, $0.0208$ en Baseline 1B, $0.0288$ en OLS, $0.0284$ en Ridge).
   * **`Sesgo Medio / Bias` ($\%\text{Cu}$):** Para verificar que los modelos eliminan el $+0.3404\%\text{Cu}$ histórico.
   * **`Varianza Explicada` (`explained_variance_score`):** Indicador institucional utilizado en memorias recientes de la UA.
2. **Estabilidad de la Validación Cruzada Interna (`cv=5` vs. `LeaveOneOut` en $N_{\text{train}} = 23$):**
   En el Bloque 2 se utilizó `cv=5` (5 particiones secuenciales sobre $23$ meses ordenados cronológicamente), lo que provoca que un pliegue interno concentre simultáneamente octubre y noviembre de 2023, generando saltos en el $\alpha$ óptimo de `LassoCV` ($\alpha \in [0.0001, 0.0351]$). En el **Bloque 3**, para sintonizar la regularización $L_2$ ($\alpha$) de la red neuronal dentro de cada pliegue de $23$ datos, se recomienda emplear **LOOCV interno (`cv=LeaveOneOut()`)** o `KFold(n_splits=5, shuffle=True, random_state=42)` para evitar particiones cronológicas desbalanceadas.
3. **Corrección de errata tipográfica (Sección 2.2, ítem 4):**
   Cambiar `(GKB/LOOCV interno)` por `(GCV/LOOCV interno)` (*Generalized Cross-Validation*, el método analítico interno de `RidgeCV`).
4. **Estandarización de nombres de archivos para el Bloque 3 (Sección 8.2):**
   En la Sección 8.2 se menciona `03_red_neuronal_loocv.ipynb` y `LOG_ENTRENAMIENTO_RNA.csv`. Mantén los nombres definidos en el tablero de `METODOLOGIA_TRABAJO_TESIS.md` (v4.0): **`03_modelado_rna_loocv.ipynb`** y **`LOG_HIPERPARAMETROS_RNA.csv`** (junto con `PREDICCIONES_OOS_RNA.csv`), asegurando que el Bloque 4 los importe sin conflictos de ruta.

---

## 5. Directrices Técnicas para Ejecutar el Bloque 3 (`03_modelado_rna_loocv.ipynb`)

Con la valla competitiva claramente establecida en **$\text{NRMSE} = 10.07\%$ ($\text{RMSE} = 0.1442\%\text{Cu}$, $R^2 = +0.6503$)** y la meta objetivo en **$\text{NRMSE} \approx 8.63\%$**, `Antigravity` debe ejecutar el **Bloque 3** siguiendo estas especificaciones:

1. **Arquitecturas a entrenar bajo el mismo protocolo LOOCV ($N = 24$):**
   * **RNA Referencia (Dr. José Delgado Vega):** Réplica fiel de la topología del profesor guía.
   * **Grilla parsimoniosa MLP (`MLPRegressor`):** Evaluar `(2,)` ($9$ parámetros), `(3,)` ($13$ parámetros), `(4,)` ($17$ parámetros), `(2, 2)` ($15$ parámetros) y `(3, 2)` ($20$ parámetros).
2. **Motor de optimización y activación:**
   * Usar `solver='lbfgs'` (cuasi-Newton determinista de segundo orden, ideal para $N = 23$) con `max_iter=2000` y `tol=1e-6`.
   * Evaluar activación **`relu`** frente a **`tanh`** (prestándole especial atención a su comportamiento de extrapolación en **Noviembre 2023**, donde `tanh` tiende a saturar).
   * Sintonizar regularización $L_2$ ($\alpha \in \{0.001, 0.01, 0.05, 0.1, 0.5, 1.0\}$) mediante validación cruzada interna sin fuga de datos.
   * Aplicar **ensamble/promediado multi-semilla** (por ejemplo, $10$ semillas aleatorias fijas por pliegue) para eliminar cualquier varianza por inicialización de pesos sinápticos.
3. **Verificaciones complementarias comprometidas en los Bloques 1 y 2:**
   * **Estudio de ablación de entradas:** Comparar la topología MLP ganadora con $2$ entradas $[\text{Cancha\_CuT}, \text{Cancha\_CuS}]$ frente a la misma arquitectura con $1$ sola entrada $[\text{Cancha\_CuT}]$, cuantificando la reducción neta de error aportada por el cobre soluble.
   * **Sensibilidad temporal y geoestadística:** Evaluar validación prospectiva (*Holdout* cronológico: $19$ primeros meses train $\to$ $5$ últimos meses test) y *Buffered-LOOCV* (excluyendo $t-1$ y $t+1$).
4. **Contrato de salida del Bloque 3:**
   * Exportar `PREDICCIONES_OOS_RNA.csv/.xlsx`, `LOG_HIPERPARAMETROS_RNA.csv/.xlsx` y `SENSIBILIDAD_Y_ABLACION_RNA.csv/.xlsx`.