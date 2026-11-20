# AUDITORÍA EXPERTO-CRÍTICA — BLOQUE 3

## Verificación Forense de Predicciones, Análisis Matemático de las Redes Neuronales y Parche Técnico Obligatorio Previo al Bloque 4

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas

**Documento Auditado:** `BLOQUE_3/INFORME_ENTREGABLE_BLOQUE_3.md` (Versión 4.0)

**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)

**Estándar de Evaluación:** Rigor académico de Ingeniería Civil en Minas (Universidad de Antofagasta — Comisión con especialidad en Geoestadística y Ciencia de Datos)

---

## 1. Dictamen Ejecutivo de Auditoría

> **Dictamen:** **NO SELLAR TODAVÍA — REQUIERE CORRECCIÓN CRÍTICA Y RE-SINCRONIZACIÓN EN `BLOQUE_3` ANTES DE EJECUTAR EL BLOQUE 4.**

Al auditar en profundidad las tablas, formulaciones y código reportados en `INFORME_ENTREGABLE_BLOQUE_3.md`, se detectaron **3 hallazgos críticos de alto impacto** y **3 observaciones metodológicas relevantes**. El hallazgo más grave es una **inconsistencia numérica masiva entre la Tabla 4.1 (métricas resumen) y la Tabla 4.2 (predicciones mes a mes)** que afecta a las **7 redes neuronales** e incluso invierte el ranking de modelos. De pasar sin corrección al Bloque 4, el benchmarking unificado contradirá directamente al Bloque 3.

Al mismo tiempo, esta auditoría descubre **dos hallazgos científicos de enorme valor** para tu defensa de título:
1. La demostración algebraica de que el código original de Keras del Dr. José Delgado colapsa matemáticamente a una **Regresión Lineal (OLS)** por carecer de funciones de activación no lineales.
2. La explicación física de por qué las redes MLP directas siguen perdiendo frente a `Baseline 1B` ($10.07\%$) en LOOCV, junto con la solución exacta de ingeniería (**MLP con Aprendizaje Residual / *Skip-Connection***) para perforar limpiamente el piso de `Baseline 1B`.

### Matriz Resumen de Hallazgos

| ID | Severidad | Sección Auditada | Hallazgo Principal | Acción Requerida |
| :---: | :---: | :--- | :--- | :--- |
| **H-01** | **CRÍTICO** | Tablas 4.1 vs. 4.2 | **Discrepancia aritmética grave en los 7 modelos:** Las métricas reportadas en la Tabla 4.1 ($\text{RMSE} \in [0.1699, 0.1929]$) **no corresponden** a las 24 predicciones de la Tabla 4.2 ($\text{RMSE} \in [0.1498, 0.1727]$). Además, el ranking entre topologías se invierte completamente. | Auditar `03_modelado_rna_loocv.ipynb`, verificar el origen del desfase (probable promedio de métricas por semilla vs. métrica del ensamble) y recalcular la Tabla 4.1 directamente desde `PREDICCIONES_OOS_RNA.csv`. |
| **H-02** | **CRÍTICO** | Sec. 1, 2.3 y 3.1 | **Colapso lineal de `RNA_Delgado_Keras` + errata de comparación:** El código original en Keras omite el argumento `activation` (usa activación lineal por defecto), reduciendo $49$ pesos a **$3$ grados de libertad efectivos (equivalente a OLS)**. Además, se entrena sobre `psj` y no `Canje_CuT`, y la Sec. 1 afirma erróneamente que $11.87\%$ *"supera a OLS ($11.85\%$)"*. | Documentar este hallazgo algebraico como justificación formal de la tesis, corregir la errata ($11.87\% > 11.85\%$), reportar el valor *in-sample* exacto en `PSJ_CuT` y `Canje_CuT`, y respetar `Adam(0.1)` / `1000` épocas en la réplica exacta. |
| **H-03** | **CRÍTICO** | Sec. 4 y 8 | **Ninguna RNA directa vence aún a `Baseline 1B` ($10.07\%$) en LOOCV:** Incluso con las métricas reales de la Tabla 4.2 (`MLP(3,2)` $= 10.46\%$, `MLP(2,)` $= 10.53\%$), las redes siguen detrás del sesgo constante ($10.07\%$) debido a la extrapolación en **Noviembre 2023** ($27.3\%$ del error total de `MLP(2,)`). | Incorporar en el Bloque 3 una **Red Neuronal con Aprendizaje Residual (*Residual-MLP*)** que prediga la corrección $\hat{e}(\text{CuT}, \text{CuS})$ sobre la ley de cancha ($\hat{y} = \text{CuT}_{\text{cancha}} - \hat{e}_{\text{MLP}}$), eliminando la atenuación hacia la media. |
| **H-04** | **ALTO** | Sec. 1 y 6.1 | **Sobre-interpretación del Holdout ($n = 5$) y comparación asimétrica:** Se compara el `6.56%` de `RNA_MLP_2` sobre solo **5 meses** contra el `10.07%` de `Baseline 1B` sobre **24 meses** (en esos mismos 5 meses, `Baseline 1B` logra `8.96%`). | Evaluar los 7 modelos del Bloque 2 sobre la misma ventana exacta de 5 meses (`2024-08` a `2024-12`) y presentar el Holdout solo como prueba prospectiva complementaria (reconociendo alta varianza por $n = 5$). |
| **H-05** | **ALTO** | Sec. 4.1 y 9 | **Posible fuga de hiperparámetros y violación de Metodología v4.0:** La Tabla 4.1 fija una sola activación por topología y fija $\alpha = 2.0$ en `RNA_Delgado_MLP` sin aclarar si se eligieron dentro de Nested CV o mirando LOOCV; además, corona ganador en el Bloque 3 cuando eso corresponde al Bloque 4. | Verificar que la selección de activación y $\alpha$ ocurra estrictamente dentro de CV interna ($23$ datos) sin mirar el pliegue de prueba, y retirar la etiqueta `(Ganadora / Campeón)` hasta la evaluación unificada del Bloque 4. |
| **H-06** | **MEDIO** | Sec. 5 y 6.2 | **Interpretación exagerada en Ablación y *Buffered-LOOCV*:** En Ablación, `MLP(3,)` ($+0.02\%$) y `MLP(4,)` ($0.00\%$) son neutras a `CuS`, contradiciendo el tono de "prueba irrefutable" basado solo en `MLP(2,)` ($-1.53\%$); y en *Buffered-LOOCV*, un alza de $+2.25\%$ en `NRMSE` requiere explicar el efecto de excluir mutuamente Oct-23 y Nov-23. | Recalcular la ablación sobre las predicciones consistentes del CSV, matizar el análisis por arquitectura y explicar el fenómeno de exclusión mutua de los dos pulsos contiguos en *Buffered-LOOCV*. |

---

## 2. Análisis Forense y Crítico Detallado

### 2.1. [H-01] Inconsistencia Numérica Masiva entre la Tabla 4.1 y la Tabla 4.2

Al recalcular independientemente los indicadores de error ($\text{RMSE}$, $\text{NRMSE}$, $R^2$, $\text{MAE}$ y $\text{Sesgo}$) directamente sobre los **$24$ valores exactos** publicados en la **Tabla 4.2 (`PREDICCIONES_OOS_RNA`)** frente a `Canje Real` ($\bar{y} = 1.43125\%\text{Cu}$, $\text{SST} = 1.42646$), **ninguno de los 7 modelos coincide con la Tabla 4.1**:

| Modelo Evaluado ($N = 24$) | $\text{RMSE}$ Tabla 4.1 | $\text{RMSE}$ Real Tabla 4.2 | $\text{NRMSE}$ Tabla 4.1 | $\text{NRMSE}$ Real Tabla 4.2 | $R^2$ Tabla 4.1 | $R^2$ Real Tabla 4.2 | Discrepancia $\Delta\text{NRMSE}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RNA_Delgado_Keras`** | $0.1699$ | **$0.1727$** | $11.87\%$ | **$12.06\%$** | $+0.5144$ | **$+0.4985$** | $+0.19\%$ |
| **`RNA_MLP_2`** | $0.1758$ | **$0.1507$** | $12.28\%$ | **$10.53\%$** | $+0.4803$ | **$+0.6180$** | **$-1.75\%$** |
| **`RNA_MLP_4`** | $0.1773$ | **$0.1600$** | $12.39\%$ | **$11.18\%$** | $+0.4709$ | **$+0.5695$** | **$-1.21\%$** |
| **`RNA_MLP_3`** | $0.1787$ | **$0.1597$** | $12.49\%$ | **$11.16\%$** | $+0.4624$ | **$+0.5709$** | **$-1.33\%$** |
| **`RNA_Delgado_MLP`** | $0.1805$ | **$0.1607$** | $12.61\%$ | **$11.23\%$** | $+0.4516$ | **$+0.5653$** | **$-1.38\%$** |
| **`RNA_MLP_2_2`** | $0.1861$ | **$0.1549$** | $13.00\%$ | **$10.82\%$** | $+0.4173$ | **$+0.5961$** | **$-2.18\%$** |
| **`RNA_MLP_3_2`** | $0.1929$ | **$0.1498$** | $13.48\%$ | **$10.46\%$** | $+0.3737$ | **$+0.6226$** | **$-3.02\%$** |

#### Diagnóstico de la Causa Raíz
1. **Consecuencia crítica en el ranking:** Según la Tabla 4.1, **todas** las redes MLP de `scikit-learn` perdían frente a `OLS` ($11.85\%$) y `RidgeCV` ($11.78\%$). En cambio, según las predicciones reales de la Tabla 4.2, **todas las 6 redes MLP de `scikit-learn` superan holgadamente a `OLS` y `RidgeCV`** ($\text{NRMSE}$ entre $10.46\%$ y $11.23\%$, $R^2$ entre $+0.5653$ y $+0.6226$).
2. **Explicación técnica más probable (Desigualdad de Jensen en el ensamble multi-semilla):**
   * En el código de Antigravity, probablemente se entrenaron $10$ semillas por pliegue y en la **Tabla 4.1** se reportó el *promedio de los RMSE individuales* de cada semilla ($\frac{1}{10}\sum_{s=1}^{10} \text{RMSE}_s$, penalizado por 1 o 2 semillas que cayeron en mínimos locales malos), mientras que en la **Tabla 4.2** se guardó la *predicción promediada del ensamble* ($\bar{\hat{y}}_k = \frac{1}{10}\sum_{s=1}^{10} \hat{y}_{k,s}$), cuyo error cuadrático siempre es estrictamente menor por efecto de reducción de varianza (*bagging / ensemble averaging*).
   * Adicionalmente, en `RNA_MLP_2` incluso cambia el signo del sesgo medio: la Tabla 4.1 reporta $-0.0033\%\text{Cu}$ y $\text{MAE} = 0.1318\%\text{Cu}$, mientras la Tabla 4.2 arroja un sesgo exacto de $+0.0123\%\text{Cu}$ y $\text{MAE} = 0.1213\%\text{Cu}$.
   * En el caso de `RNA_Delgado_Keras` (donde `0.1699` vs `0.1727` difiere casi por completo en `2023-01`: `1.8384` vs `~1.79`), es señal de una falta de control de semilla determinista (`tf.random.set_seed`) entre ejecuciones o de transcripción desfasada al informe Markdown.
3. **Regla intransable:** Las métricas de la Tabla 4.1 deben calcularse **exacta y únicamente** sobre los vectores exportados en `PREDICCIONES_OOS_RNA.csv` (la predicción final del ensamble multi-semilla). Jamás pueden coexistir dos conjuntos de números distintos entre el CSV y el informe.

---

### 2.2. [H-02] Hallazgo Matemático Clave en la Réplica del Dr. Delgado (`RNA_Delgado_Keras`)

Al revisar el código fuente preservado del profesor guía en la **Sección 3.1**:
```python
capaEntrada = tf.keras.layers.Dense(units=4, input_shape=[2])
capaOculta1 = tf.keras.layers.Dense(units=4)
capaOculta2 = tf.keras.layers.Dense(units=4)
capaSalida  = tf.keras.layers.Dense(units=1)
historial = modelo.fit(cancha, psj, epochs=1000, verbose=False)
```
Surgen **cuatro precisiones técnicas fundamentales** que elevan radicalmente el nivel de tu tesis:

1. **Colapso algebraico a Regresión Lineal (OLS):** En TensorFlow/Keras, el parámetro por defecto de `Dense` es `activation=None` (función identidad $g(z) = z$). La composición de cuatro capas densas lineales:
   $$\hat{y} = \mathbf{W}_4\Big(\mathbf{W}_3\big(\mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2\big) + \mathbf{b}_3\Big) + \mathbf{b}_4 = \mathbf{w}_{\text{eq}}^T \mathbf{x} + b_{\text{eq}}$$
   **es algebraicamente idéntica a una regresión lineal múltiple (OLS)** sin ninguna capacidad no lineal.
2. **Parámetros nominales ($P = 49$) vs. grados de libertad efectivos ($P_{\text{eff}} = 3$):** Aunque la Tabla 2.3 contabiliza correctamente $49$ pesos y sesgos nominales, la transformación resultante tiene rango máximo $2$ y solo **$3$ grados de libertad efectivos** ($\beta_0, \beta_1, \beta_2$). Esto explica exactamente *por qué* la red de 49 parámetros no colapsó por sobreajuste en LOOCV y por qué sus predicciones mes a mes (`2023-10`: `2.1152` vs OLS `2.1184`; `2023-11`: `1.8496` vs OLS `1.8555`) replican casi idénticamente a `OLS` del Bloque 2.
3. **Corrección de errata en Sección 1 y consistencia de parámetros:**
   * La Sección 1 afirma que `RNA_Delgado_Keras` ($\text{NRMSE} = 11.87\%$) *"supera a OLS ($11.85\%$)"*. Eso es aritméticamente falso ($11.87\% > 11.85\%$, y en la Tabla 4.2 es $12.06\%$); corresponde redactar que **converge prácticamente a OLS por carecer de activación no lineal**.
   * El código del Dr. Delgado entrenaba contra `psj` (`PSJ_CuT`) usando `Adam(0.1)` y `epochs=1000`, mientras que la réplica usó `Canje_CuT`, `Adam(0.05)` y `epochs=800`. Documentar explícitamente el ajuste de la variable objetivo (`Canje_CuT` para homologación comercial) y reportar además el desempeño *in-sample* de la réplica sobre `PSJ_CuT` y `Canje_CuT` cerrará de manera impecable el contraste con el $8.63\%$ preliminar.

---

### 2.3. [H-03] Por Qué las MLP Directas Pierden Frente a `Baseline 1B` ($10.07\%$) y la Solución de Ingeniería: *Residual-Learning MLP*

Incluso después de corregir las métricas hacia la Tabla 4.2, la mejor red neuronal (`RNA_MLP_3_2`: $\text{RMSE} = 0.1498\%\text{Cu}$, $\text{NRMSE} = 10.46\%$; `RNA_MLP_2`: $\text{RMSE} = 0.1507\%\text{Cu}$, $\text{NRMSE} = 10.53\%$) **todavía no logra batir el piso empírico de `Baseline 1B` ($\text{RMSE} = 0.1442\%\text{Cu}$, $\text{NRMSE} = 10.07\%$, $R^2 = +0.6503$)**.

#### Diagnóstico físico-matemático exacto
Observa lo que ocurre en el pliegue de **Noviembre 2023** (`Cancha_CuT = 2.47%`, `Cancha_CuS = 0.20%`, `Canje_CuT = 2.27%`):
* `Baseline 1B` predice `2.1235%` (error $= -0.1465\%\text{Cu}$, error cuadrático $= 0.0215$).
* `RNA_MLP_2` predice `1.8845%` (error $= -0.3855\%\text{Cu}$, error cuadrático $= 0.1486$ — **casi $7$ veces más error cuadrático en un solo mes**, explicando el **$27.3\%$ de todo el SSE** de la red).

**¿Por qué sucede esto?** Cuando entrenas una red MLP *directa* ($\hat{y} = f_{\text{MLP}}(\text{CuT}, \text{CuS})$) con regularización $L_2$ ($\alpha$) y/o activación `tanh`, la penalización contrae los pesos hacia cero y la función `tanh` satura en los extremos. En consecuencia, ante un punto de extrapolación como Noviembre 2023 (fuera del rango máximo de $2.21\%$ del pliegue de entrenamiento), la red **jala la predicción hacia el promedio global de entrenamiento ($\approx 1.40\%$)**.

#### Solución de alto impacto para tu tesis: Red Neuronal con Aprendizaje Residual (*Residual-Learning MLP*)
En lugar de obligar a la red a reaprender desde cero la relación base $1:1$ entre la ley de cancha y la ley de canje, se incorpora una **conexión residual física (*skip-connection*)** inspirada en la ecuación de conciliación minera de Pierre Gy:
$$\hat{y}_i = \text{Cancha\_CuT}_i - \hat{e}_{\text{MLP}}(\text{Cancha\_CuT}_i,\; \text{Cancha\_CuS}_i)$$
Es decir, la variable objetivo interna de entrenamiento de la red es la **discrepancia empírica** $e_i = \text{Cancha\_CuT}_i - \text{Canje\_CuT}_i$, y la ley de canje predicha se reconstruye restando el sesgo predicho a la ley de cancha:
1. **Límite inferior garantizado bajo regularización $L_2$:** Cuando $\alpha$ contrae los pesos de la red hacia cero, $\hat{e}_{\text{MLP}} \to \bar{e}_{\text{train}, -k}$, por lo que la red colapsa suavemente hacia **`Baseline 1B` ($10.07\%$)** en lugar de colapsar hacia la media global ($1.40\%$).
2. **Sin saturación catastrófica en extrapolación:** En Noviembre 2023, la red solo corrige el desvío físico esperado en torno a $0.25\%\text{–}0.35\%\text{Cu}$, evitando subestimar en $-0.39\%\text{Cu}$ un frente sulfurado de alta ley.
3. **Captura pura de la no linealidad de `CuS`:** Todos los parámetros libres de la red se dedican exclusivamente a modelar la variación no lineal del sesgo ($14.17\%$ del $\text{MSE}$ histórico) provocada por el cobre soluble y la ley total.

Incorporar y contrastar las arquitecturas parsimoniosas bajo **Aprendizaje Directo vs. Aprendizaje Residual (`Residual-MLP`)** en el Bloque 3 constituye un aporte de ingeniería aplicada de primer nivel ante cualquier tribunal.

---

### 2.4. [H-04] Corrección sobre la Validación Prospectiva (*Holdout* 19/5)

El informe destaca el $\text{NRMSE} = 6.56\%$ ($\text{RMSE} = 0.0861\%\text{Cu}$, $R^2 = +0.6550$) de `RNA_MLP_2` en la ventana `2024-08` a `2024-12` ($n = 5$) como un hito que *"pulveriza la meta del Dr. Delgado ($8.63\%$) y bate a Baseline 1B ($10.07\%$)"*. Hay **dos errores metodológicos** en esa afirmación:
1. **Comparación asimétrica (manzanas con naranjas):** El $10.07\%$ de `Baseline 1B` corresponde al periodo completo de **24 meses**, que incluye los atípicos extremos de 2023. Si se evalúa `Baseline 1B` sobre los **mismos 5 meses** (`2024-08` a `2024-12`, usando $\bar{e}_{\text{train}, 19} = 0.3395\%\text{Cu}$), su desempeño real en esa ventana es $\text{RMSE} = 0.1175\%\text{Cu}$ y **$\text{NRMSE} = 8.96\%$** ($R^2 = +0.3578$), no $10.07\%$.
2. **Inestabilidad inherente a $n = 5$:** Sobre los mismos 5 meses finales, la predicción LOOCV de `RNA_MLP_2` en la Tabla 4.2 (entrenada con 23 meses en vez de 19) arroja $\text{RMSE} = 0.1118\%\text{Cu}$ ($\text{NRMSE} = 8.52\%$). El salto desde $8.52\%$ hasta $6.56\%$ refleja la alta varianza muestral de una prueba de solo 5 puntos.

**Corrección requerida:** Incluir en la Tabla 6.1 el desempeño de **Cancha Directa, Baseline 1B, OLS y Ridge** sobre exactamente la misma ventana prospectiva `2024-08` a `2024-12`, y redactar el resultado con rigor científico: evidencia favorable de estabilidad operativa en régimen sulfurado regular, pero **subordinada al LOOCV global ($N = 24$)** como criterio rector de selección.

---

### 2.5. [H-05] Auditoría de Posible Fuga de Hiperparámetros y Respeto a Metodología v4.0

1. **Verificación de *Nested CV* sin fuga:** En la Tabla 4.1, cada arquitectura aparece con una sola función de activación fija (`tanh` para `(2,)` y `(3,2)`; `relu` para `(3,)`, `(4,)` y `(2,2)`), y `RNA_Delgado_MLP` aparece con un valor fijo $\alpha = 2.0$. Si esas activaciones o el $\alpha = 2.0$ se eligieron mirando cuál obtenía menor error en los $24$ pliegues LOOCV, existe **fuga de selección de hiperparámetros (*hyperparameter leakage*)**. Se debe garantizar (y dejar evidencia explícita en `LOG_HIPERPARAMETROS_RNA.csv`) que tanto $\alpha$ como la función de activación o bien se definen como variantes separadas a priori en la grilla, o bien se seleccionan **automáticamente dentro de la CV interna sobre los $23$ datos de entrenamiento** en cada pliegue.
2. **Respeto a la separación de bloques (v4.0):** Retirar las etiquetas `(Ganadora)` y `(Campeón)` del Bloque 3. Según la Metodología v4.0, el Bloque 3 es estrictamente entrenamiento y exportación de predicciones ciegas; la declaración formal del *Champion Model* ocurre únicamente en el **Bloque 4**, tras cruzar todas las métricas y pruebas de hipótesis contra los baselines del Bloque 2.

---

### 2.6. [H-06] Matices Críticos en Ablación (2D vs. 1D) y *Buffered-LOOCV*

1. **Estudio de Ablación (Sección 5):** Afirmar que el resultado constituye una *"validación irrefutable"* solo porque `RNA_MLP_2` mejoró en $-1.53\%$ resulta vulnerable ante la comisión, ya que en la misma tabla `RNA_MLP_3` ($+0.02\%$) y `RNA_MLP_4` ($0.00\%$) muestran efecto nulo, y el deterioro de `RNA_MLP_2` en 1D ($13.81\%$) puede haberse originado por saturación de `tanh` con solo 2 neuronas. Tras corregir la inconsistencia numérica de H-01 (y evaluar también la ablación en la **Residual-MLP**), reporta con objetividad el aporte medio de `CuS` a través de toda la grilla y complementa el veredicto en el Bloque 4 con SHAP y Permutación.
2. **Explicación de *Buffered-LOOCV* (Sección 6.2):** Un aumento de $+2.25$ puntos porcentuales en `RNA_MLP_2` (de $12.28\%$ a $14.53\%$) no debe calificarse simplemente como *"estable sin fallas"*. La explicación técnica correcta —y defendible ante un geoestadístico— es que los dos únicos meses extremos de alta ley (**Octubre 2023** y **Noviembre 2023**) son **cronológicamente adyacentes**. Al aplicar la ventana de exclusión $(t-1, t+1)$, cuando se evalúa Noviembre 2023 se elimina automáticamente Octubre 2023 del entrenamiento (y viceversa), dejando al modelo sin ningún soporte cercano en la cola superior del dominio.

---

## 3. Plan de Acción y Parche Exacto para `Antigravity` (Re-ejecución Controlada de `BLOQUE_3`)

Antes de abrir el Bloque 4, ejecuta las siguientes correcciones en `BLOQUE_3/03_modelado_rna_loocv.ipynb` y actualiza `BLOQUE_3/INFORME_ENTREGABLE_BLOQUE_3.md`:

### Paso 1: Eliminar de Raíz la Inconsistencia Tabla 4.1 vs. Tabla 4.2
* Guardar primero las predicciones finales de cada modelo (promedio del ensamble de $10$ semillas fijas en cada pliegue $k = 1, \dots, 24$) en el dataframe maestro `df_preds_rna` y exportar `PREDICCIONES_OOS_RNA.csv`.
* Calcular **todas** las métricas de la Tabla 4.1 (incluyendo `MSE` y `Varianza Explicada`) ejecutando una única función sobre las columnas de `df_preds_rna` (`PREDICCIONES_OOS_RNA.csv`). Fijar semillas globales (`np.random.seed(42)`, `tf.random.set_seed(42)`) para reproducibilidad exacta bit a bit.

### Paso 2: Incorporar la Familia `Residual-MLP` (Aprendizaje del Sesgo con *Skip-Connection*)
Añadir el siguiente esquema dentro del mismo bucle LOOCV sin fuga de datos (evaluando topologías `(2,)`, `(3,)`, `(4,)` y `(2, 2)`):

```python
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Objetivo residual: e = Cancha_CuT - Canje_CuT
# Predicción final: y_hat = Cancha_CuT - e_hat_MLP
def run_residual_mlp_loocv(X, y_canje, hidden_sizes, activation='relu', seeds=range(10)):
    loo = LeaveOneOut()
    y_pred_oos = np.zeros(len(y_canje))
    alpha_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    chosen_alphas = []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        # Target interno: sesgo empírico calculado SOLO en entrenamiento
        e_tr = X_tr[:, 0] - y_canje[train_idx]

        # Selección interna de alpha vía LOOCV interno (cero fuga de datos)
        pipe_base = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=hidden_sizes,
                                 activation=activation,
                                 solver='lbfgs',
                                 max_iter=2000,
                                 random_state=42))
        ])
        gs = GridSearchCV(pipe_base, {'mlp__alpha': alpha_grid},
                          cv=LeaveOneOut(), scoring='neg_mean_squared_error')
        gs.fit(X_tr, e_tr)
        best_alpha = gs.best_params_['mlp__alpha']
        chosen_alphas.append(best_alpha)

        # Ensamble determinista de 10 semillas con el best_alpha interno
        preds_seeds = []
        for s in seeds:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(hidden_layer_sizes=hidden_sizes,
                                     activation=activation,
                                     solver='lbfgs',
                                     alpha=best_alpha,
                                     max_iter=2000,
                                     random_state=s))
            ])
            pipe.fit(X_tr, e_tr)
            e_hat = pipe.predict(X_te)[0]
            preds_seeds.append(X_te[0, 0] - e_hat)

        y_pred_oos[test_idx[0]] = np.mean(preds_seeds)

    return y_pred_oos, chosen_alphas
```

### Paso 3: Completar las Correcciones Documentales en `INFORME_ENTREGABLE_BLOQUE_3.md`
1. **Sección 3 (`RNA_Delgado_Keras`):**
   * Documentar explícitamente que la omisión de `activation` equivale a activación lineal ($P_{\text{eff}} = 3$ grados de libertad efectivos), explicando su convergencia a OLS.
   * Reportar la evaluación *in-sample* exacta sobre `PSJ_CuT` y `Canje_CuT` para cerrar el origen del $8.63\%$.
   * Corregir la frase de la Sec. 1 (*"supera a OLS"*) por *"converge al límite lineal de OLS"*.
2. **Sección 4 (Tablas 4.1 y 4.2):**
   * Garantizar coincidencia exacta al cuarto decimal entre ambas tablas y `PREDICCIONES_OOS_RNA.csv`, incorporando las columnas de `Residual-MLP`.
   * Retirar la designación prematura de `(Ganadora / Campeón)` hasta el Bloque 4.
3. **Sección 5 (Ablación 2D vs. 1D):**
   * Recalcular la tabla de ablación sobre las predicciones de ensamble consistentes (incluyendo `Residual-MLP`), analizando con rigor por qué algunas topologías ganan más que otras al incluir `Cancha_CuS`.
4. **Sección 6 (Holdout 19/5 y *Buffered-LOOCV*):**
   * Agregar `Cancha Directa`, `Baseline 1B`, `OLS` y `Ridge` a la Tabla 6.1 (Holdout `2024-08` a `2024-12`) para una comparación simétrica y justa.
   * Explicar el incremento en *Buffered-LOOCV* por la adyacencia temporal de los dos meses extremos (`2023-10` y `2023-11`).