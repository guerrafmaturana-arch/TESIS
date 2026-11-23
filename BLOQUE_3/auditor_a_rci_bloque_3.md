# AUDITORÍA EXPERTO-CRÍTICA (MODELO RCI) — BLOQUE 3
## Verificación Forense de Resultados, Coherencia Geometalúrgica y Directrices de Cierre Previo al Bloque 4

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas  
**Documento Auditado:** `BLOQUE_3/INFORME_ENTREGABLE_BLOQUE_3.md` (Versión 4.0)  
**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)  
**Institución:** Universidad de Antofagasta — Facultad de Ingeniería, Departamento de Ingeniería en Minas  
**Destinatario Técnico:** Entorno de desarrollo y ejecución (`Antigravity`)  

---

## 1. Marco de Auditoría RCI (Rol, Contexto, Instrucción)

* **Rol (R):** Auditor Técnico Senior en Geoestadística Minera, Conciliación Mina–Planta (estándar comercial ENAMI) y Machine Learning aplicado a muestras pequeñas ($N = 24$).
* **Contexto (C):** Evaluación crítica del entregable `INFORME_ENTREGABLE_BLOQUE_3.md` (Metodología v4.0), correspondiente al diseño, entrenamiento, estudio de ablación y validación LOOCV de $11$ arquitecturas de Redes Neuronales Artificiales (`Direct-MLP`, `Residual-MLP` y réplica Keras del Dr. José Delgado Vega) para Faena Mantos Cobrizo con entradas $[\text{Cancha\_CuT}, \text{Cancha\_CuS}]$ y objetivo $\text{Canje\_CuT}$.
* **Instrucción (I):** Auditar el rigor aritmético, la validez econométrica, la ausencia real de fuga de información (*zero data leakage*), la coherencia metalúrgica y la solidez de los insumos que alimentarán el Bloque 4 (Validación Unificada, Diagnóstico de Residuos y Explicabilidad XAI), clasificando los hallazgos por severidad y entregando acciones correctivas concretas.

---

## 2. Dictamen Ejecutivo de Auditoría

> **Veredicto:** **APROBADO CON OBSERVACIONES DE CIERRE (apto para avanzar al Bloque 4, previo ajuste documental sin necesidad de re-entrenar modelos).**

El avance consolidado en el Bloque 3 cumple con los pilares centrales de la **Metodología v4.0**:
1. **Consistencia aritmética exacta:** Todas las métricas reportadas en la Tabla 4.1 (`MSE`, `RMSE`, `MAE`, `NRMSE`, $R^2$, `MAPE`, Pearson $r$, Sesgo y Varianza Explicada) coinciden al cuarto decimal con los $24$ pronósticos fuera de muestra de la Tabla 4.2 (`PREDICCIONES_OOS_RNA.csv`).
2. **Validación prospectiva simétrica:** El contraste *Holdout 19/5* (`2024-08` a `2024-12`) incluye los baselines del Bloque 2 sobre la misma ventana temporal, contextualizando con honestidad el $\text{NRMSE} = 6.56\%$ de `RNA_MLP_2` frente al $8.96\%$ de `Baseline 1B` y $7.57\%$ de `RidgeCV`.
3. **Aporte algebraico clave:** La demostración matricial del colapso lineal de la red original en Keras ($\mathbf{W}_{\text{eq}} = \mathbf{W}_4 \mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1 \implies P_{\text{eff}} = 3$) explica científicamente por qué dicha arquitectura converge al límite de OLS ($\text{NRMSE} = 11.64\%$ vs. $11.85\%$).

No obstante, se identificaron **5 hallazgos técnicos y estratégicos (`A-01` a `A-05`)** que deben quedar documentados y corregidos antes de ejecutar el **Bloque 4** y trasladar los resultados a `INFORME_TESIS.docx`.

---

## 3. Verificación Independiente de Cálculos

| Dimensión Auditada | Resultado Reportado en Bloque 3 | Auditoría Independiente | Estado |
| :--- | :--- | :--- | :---: |
| **Tabla 4.1 vs. Tabla 4.2 ($11$ redes, $N = 24$)** | `Res_MLP_2_2` ($\text{MSE}=0.020781$, $\text{NRMSE}=10.07\%$, $R^2=+0.6504$) hasta `RNA_MLP_3_2` ($\text{MSE}=0.037222$, $\text{NRMSE}=13.48\%$, $R^2=+0.3737$). | Coincidencia exacta al sexto decimal en $\text{MSE}$ ($\Delta = 0.000000$) y al cuarto decimal en $\text{RMSE}$, $\text{MAE}$, $R^2$ y Sesgo. | **Exacto** |
| **Tabla 6.1 (*Holdout* 19/5, $n = 5$)** | `RNA_MLP_2`: $\text{MSE}=0.007417$, $\text{RMSE}=0.0861$, $\text{NRMSE}=6.56\%$, $R^2=+0.6549$; `Baseline 1B`: $8.96\%$; `RidgeCV`: $7.57\%$. | Verificado contra $\bar{y}_{5} = 1.3120\%\text{Cu}$ y $\sigma^2_{5} = 0.021496\,(\%\text{Cu})^2$: todas las conversiones $\text{MSE} \to \text{RMSE} \to \text{NRMSE} \to R^2$ cuadran exactamente. | **Exacto** |
| **Tabla 6.2 (*Buffered-LOOCV*)** | `Res_MLP_3`: $10.04\%$; `RNA_MLP_4`: $13.43\%$; `RNA_MLP_2`: $14.53\%$; `RNA_MLP_2_2`: $15.28\%$. | Verificado contra $\bar{y}_{24} = 1.4313\%\text{Cu}$ y $\sigma^2_{24} = 0.059436\,(\%\text{Cu})^2$: todas las métricas cuadran exactamente. | **Exacto** |
| **Equivalencia lineal de `RNA_Delgado_Keras`** | Demuestra $\mathbf{W}_{\text{eq}} = \mathbf{W}_4\mathbf{W}_3\mathbf{W}_2\mathbf{W}_1 \implies P_{\text{eff}} = 3$. | Confirmado: sus pronósticos difieren en promedio solo `0.0023 %Cu` respecto a `OLS` del Bloque 2. | **Válido** |

---

## 4. Matriz de Hallazgos Críticos y Acciones Correctivas

| ID | Severidad | Hallazgo Detectado | Impacto ante la Comisión | Acción Correctiva Concreta |
| :---: | :---: | :--- | :--- | :--- |
| **A-01** | **ALTA** *(Estratégico)* | **Colapso a pesos nulos en `Residual-MLP` ($\alpha = 2.0$):** Al comparar la Tabla 4.2 contra `Baseline 1B` del Bloque 2, `Res_MLP_3`, `Res_MLP_4` y `Res_MLP_2_2` entregan pronósticos idénticos en **$24$ de $24$ meses** ($\max\vert\Delta\vert \le 0.0001\%\text{Cu}$). | Si en el Bloque 4 se declara a `Res_MLP_2_2` como "red neuronal ganadora" y se ejecuta **SHAP** o **Permutación**, el aporte de `Cancha_CuS` dará `0.0000` (neuronas apagadas). Un evaluador notará que el modelo colapsó a una resta de constante. | Documentar explícitamente que $\alpha = 2.0$ comprime $\Vert\mathbf{W}\Vert \to 0$ (demostrando que en $N=24$ el piso empírico de `Baseline 1B` es el límite global ante extremos) y **ejecutar XAI (SHAP, Superficie 2D y Permutación) en el Bloque 4 sobre `RNA_MLP_2`**, donde las neuronas sí están activas. |
| **A-02** | **MEDIA–ALTA** *(Metodológico)* | **Posible sesgo de selección en $\alpha$ fijo (Tabla 4.1):** Cada arquitectura reporta un único valor fijo de $\alpha$ (`0.001` en `RNA_MLP_2`, `5.0` en `RNA_MLP_3/4`, `2.0` en `RNA_MLP_2_2/3_2`, `1.0` en `Res_MLP_2`). | Si ese $\alpha$ único se escogió comparando directamente el error LOOCV final de los $24$ meses en vez de ajustarse dentro de cada pliegue (*Nested CV*), existe *model selection bias* (fuga suave de hiperparámetros). | Auditar `LOG_HIPERPARAMETROS_RNA.csv` y documentar explícitamente si $\alpha$ corresponde a configuraciones fijas predefinidas de la grilla o selección anidada (*Nested CV*), evitando sobreoptimizar mirando los $24$ errores de prueba. |
| **A-03** | **MEDIA** *(Aritmético)* | **Conteo erróneo de parámetros en `RNA_Delgado_Keras`:** Las Secciones 1, 2.4, 3.1 y 4.1 declaran **$49$ parámetros nominales** para la arquitectura `(4, 4, 4)` con `input_shape=[2]` y salida `1`. | Error directo de verificación básica en defensa de tesis. El cálculo capa por capa es: $(2\times 4 + 4) + (4\times 4 + 4) + (4\times 4 + 4) + (4\times 1 + 1) = 12 + 20 + 20 + 5 = \mathbf{57}$. | Corregir `$49$` por **`$57$`** en las Secciones 1, 2.4, 3.1 y 4.1 (manteniendo intacto **$P_{\text{eff}} = 3$**). |
| **A-04** | **MEDIA** *(Terminológico)* | **Confusión entre `MECM`, `MAPE` y `NRMSE` en la Sección 3.2:** Se reportan *in-sample* `MECM = 7.97%` sobre `PSJ_CuT` (`RMSE = 0.1418`) y `MECM = 7.38%` sobre `Canje_CuT` (`RMSE = 0.1315`). | Contradice la definición sellada en el Bloque 0 (`MECM` $\equiv$ `NRMSE`). En realidad, `7.97%` y `7.38%` son **MAPE *in-sample***; los **NRMSE *in-sample*** exactos son **$9.91\%$** en `PSJ_CuT` ($\frac{0.1418}{1.4304}$) y **$9.19\%$** en `Canje_CuT` ($\frac{0.1315}{1.4313}$). | Corregir la Sección 3.2 reportando ambas métricas explícitamente: `MAPE = 7.97% (NRMSE = 9.91%)` en `PSJ_CuT` y `MAPE = 7.38% (NRMSE = 9.19%)` en `Canje_CuT`. |
| **A-05** | **BAJA** *(Editorial)* | **Inconsistencias menores en Secciones 5 y 6.1:** (a) La Sección 6.1 indica que se evaluaron *"los 7 baselines del Bloque 2"*, pero la tabla lista solo **4** (`Cancha Directa`, `Baseline 1B`, `OLS`, `RidgeCV`); (b) Microdesfase de redondeo en $\text{MSE}$ entre Tabla 4.1 (`0.030890` en `RNA_MLP_2`, `0.020781` en `Res_MLP_2_2`) y Tabla 5 (`0.030887` y `0.020783`). | Detalles que restan prolijidad ante revisores meticulosos, aunque no alteran ninguna conclusión cuantitativa. | Cambiar la frase en Sección 6.1 a *"los 4 baselines principales del Bloque 2"* (o sumar `LassoCV`, `ElasticNetCV` y `SVR` en la tabla del Bloque 4) y unificar el cálculo de $\text{MSE}$ de Tabla 5 sin redondeo intermedio. |

---

## 5. Análisis Profundo de Hallazgos Críticos

### 5.1. [A-01] El Fenómeno de Contracción en `Residual-MLP` y su Manejo en el Bloque 4
Al entrenar una red residual $\hat{y}_k = \text{Cancha\_CuT}_k - \hat{e}_{\text{MLP}}(\text{CuT}_k, \text{CuS}_k)$ contra el sesgo empírico $e_i$ (cuya varianza poblacional es acotada: $\sigma^2_e = 0.0191\,(\%\text{Cu})^2$), una penalización $L_2$ de $\alpha = 2.0$ contrae la norma de los pesos ocultos prácticamente a cero ($\Vert\mathbf{W}_1\Vert \approx 0.0005$). En consecuencia, $\hat{e}_{\text{MLP}} \approx \bar{e}_{\text{train}, -k}$, haciendo que `Res_MLP_2_2`, `Res_MLP_3` y `Res_MLP_4` reproduzcan exactamente a `Baseline 1B` ($\text{NRMSE} = 10.07\%$).

* **Por qué es una fortaleza científica si se explica bien:** Demuestra matemáticamente que, en presencia de los dos pulsos extremos adyacentes (`2023-10` y `2023-11`), cualquier curvatura no lineal agresiva sobre el residuo incrementa la varianza fuera de muestra ($\alpha = 0.5 \to 10.73\%$; $\alpha = 0.1 \to 11.94\%$), por lo que la regularización óptima converge hacia una corrección de sesgo robusta.
* **Implicancia operativa para el Bloque 4:** En la etapa de explicabilidad (**XAI**), no se debe aplicar SHAP sobre `Res_MLP_2_2` (donde los pesos están apagados y `CuS` tendría importancia nula), sino sobre **`RNA_MLP_2`** (arquitectura `(2,)`, $9$ parámetros, activación `tanh`), donde ambos predictores tienen pesos activos, reducen el error en $-1.53\%$ en ablación y alcanzan $\text{NRMSE} = 6.56\%$ en validación prospectiva (*Holdout* 19/5).

### 5.2. [A-03] Verificación del Conteo de Parámetros en Keras
Para cualquier perceptrón multicapa totalmente conectado con vector de entrada de dimensión $d_0 = 2$ y capas de tamaños $[d_1, d_2, d_3, d_4] = [4, 4, 4, 1]$:
$$P_{\text{nominal}} = \sum_{l=1}^{4} (d_{l-1} \cdot d_l + d_l) = (2 \cdot 4 + 4) + (4 \cdot 4 + 4) + (4 \cdot 4 + 4) + (4 \cdot 1 + 1) = 12 + 20 + 20 + 5 = 57$$
Corregir $49 \to 57$ en el texto asegura coherencia total con el código Keras presentado en la Sección 3.1.

---

## 6. Instrucciones Directas para `Antigravity` (Cierre Bloque 3 y Arranque Bloque 4)

1. **Parche documental rápido en `BLOQUE_3/INFORME_ENTREGABLE_BLOQUE_3.md` (sin re-entrenar modelos):**
   * Reemplazar `49` parámetros nominales por `57` en `RNA_Delgado_Keras` (Secciones 1, 2.4, 3.1 y 4.1).
   * Corregir en Sección 3.2: `MAPE = 7.97% (NRMSE = 9.91%)` en `PSJ_CuT` y `MAPE = 7.38% (NRMSE = 9.19%)` en `Canje_CuT`.
   * Ajustar en Sección 6.1 *"los 4 baselines principales del Bloque 2"* y homogenizar el $\text{MSE}$ de Tabla 5 con Tabla 4.1.
2. **Ejecución inmediata del `BLOQUE_4` (`./BLOQUE_4/04_validacion_residuos_xai.ipynb`):**
   * **Tabla Maestra Unificada ($18$ modelos):** Consolidar `BLOQUE_2/PREDICCIONES_OOS_BASELINES.csv` ($7$ baselines) y `BLOQUE_3/PREDICCIONES_OOS_RNA.csv` ($11$ redes) ordenados por $\text{MSE}$ / $\text{NRMSE}$ con las $9$ métricas completas (`MSE`, `RMSE`, `MAE`, `NRMSE%`, `R2`, `MAPE%`, `Pearson_r`, `Bias`, `Varianza_Explicada`).
   * **Selección Dual Transparente:** Contrastar el mejor predictor global ante extremos (`Baseline 1B` / `Res_MLP_2_2`, $\text{NRMSE} = 10.07\%$, $R^2 = +0.6504$, explicando el colapso de pesos en $\alpha = 2.0$) frente a la mejor red neuronal no lineal con pesos activos (`RNA_MLP_2`, $\text{NRMSE}_{\text{LOOCV}} = 12.28\%$, $\text{NRMSE}_{\text{Holdout}} = 6.56\%$, aporte de `CuS` $= -1.53\%$).
   * **Diagnóstico de Residuos y Significancia ($N = 24$ cronológicos):** Ejecutar Shapiro-Wilk, Breusch-Pagan, Durbin-Watson y **Wilcoxon unilateral pareado** (errores cuadráticos contra `Cancha Directa`, reportando $p$-valores exactos para ambas familias y contraste frente a `Baseline 1B`), más intervalo *bootstrap* ($B = 10{,}000$).
   * **Explicabilidad (XAI) y Serialización:** Ejecutar **Superficie de Respuesta 2D**, **SHAP (`KernelExplainer`)** e **Importancia por Permutación** sobre `RNA_MLP_2` (donde `CuT` y `CuS` están activos), mostrar la evidencia de contracción de pesos en `Res_MLP` y exportar los pipelines reentrenados en $N = 24$ (`.joblib`) para alimentar el **Bloque 5** y el **Bloque 6**.