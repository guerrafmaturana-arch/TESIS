# AUDITORÍA EXPERTO-CRÍTICA — BLOQUE 1
## Verificación Matemática, Rigor Econométrico y Estrategia ante Comisión Evaluadora

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas  
**Documento Auditado:** `BLOQUE_1/INFORME_ENTREGABLE_BLOQUE_1.md` (Versión 4.0)  
**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)  
**Estándar de Evaluación:** Rigor académico de Ingeniería Civil en Minas (Universidad de Antofagasta — Comisión con especialidad en Geoestadística y Ciencia de Datos)

---

## 1. Dictamen Ejecutivo de Auditoría

> **Dictamen:** **APROBADO CON OBSERVACIONES CRÍTICAS (CORREGIR ANTES DE CONGELAR PARA MEMORIA FINAL).**

La reestructuración del **Bloque 1** bajo la **Metodología v4.0** representa un avance muy sólido: integra correctamente el estudio de varianzas, desviaciones estándar, matriz de covarianza, depuración de la tabla de VIF y la necesidad de *Buffered-LOOCV*. Sin embargo, al realizar una **verificación independiente de cada cálculo sobre los $N = 24$ datos maestros** y evaluar los argumentos bajo el escrutinio de un tribunal experto en geoestadística y aprendizaje automático, se detectaron **3 hallazgos críticos (1 error numérico y 2 vulnerabilidades conceptuales que debilitarían tu defensa)** y **3 observaciones de severidad media**.

### Matriz Resumen de Hallazgos

| ID | Severidad | Sección Auditada | Hallazgo Principal | Acción Requerida |
| :---: | :---: | :--- | :--- | :--- |
| **H-01** | **CRÍTICO** | Sec. 3.2 (Tabla $N=22$) | **Error numérico en correlaciones de `Cancha_CuS` sin atípicos:** Se reporta $r = +0.4210$ y $\rho = +0.4380$, pero el cálculo exacto sobre los $22$ meses certificados arroja **$r = +0.3688$ ($p = 0.091$)** y **$\rho = +0.3520$ ($p = 0.108$)** (no significativos al $5\%$). | Recalcular directamente desde `DATASET_AUDITADO_N24.csv` en `01_eda_caracteristicas.ipynb`, corregir la tabla y diseñar prueba de ablación en Bloques 2–4. |
| **H-02** | **CRÍTICO** | Sec. 2.3 (Sesgo-Varianza) | **Error conceptual peligroso para la defensa:** Se etiqueta el $14.17\%$ restante del $\text{MSE}$ ($\sigma^2_{\text{error}} = 0.01914$) como *"ruido estocástico no explicable"*. Si fuera no explicable, restar una constante fija ($-0.34\%\text{Cu}$) sería el óptimo teórico ($\text{NRMSE} = 9.66\%$) y haría innecesaria la Red Neuronal. | Reformular como **varianza de la discrepancia empírica**, compuesta por señal geometalúrgica no lineal explicable (objetivo de la RNA para bajar hacia $\sim 8.6\%$) más ruido analítico irreducible. |
| **H-03** | **CRÍTICO** | Sec. 1.5 y Sec. 4 | **Imprecisión econométrica sobre normalidad:** Afirmar que el rechazo de normalidad en $X$ e $Y$ *"invalida modelos lineales gaussianos"* es incorrecto bajo el Teorema de Gauss-Markov (OLS exige supuestos sobre los **residuos condicionales**, no sobre $X$ ni $Y$ marginales). | Corregir la redacción: la no normalidad de $X$ e $Y$ alerta sobre *leverage* (puntos de palanca) y justifica contrastes no paramétricos, escalado robusto y exploración no lineal, no invalida OLS a priori. |
| **H-04** | **MEDIO** | Sec. 3 (Correlaciones) | **Omisión de entregables de Metodología v4.0:** Desaparecieron la **Información Mutua ($\text{MI}$)**, los **$p$-valores** de Pearson/Spearman y el análisis numérico de **Ratio de Solubilidad vs. Sesgo Empírico**. | Restituir las columnas de $p$-valor y $\text{MI}$ en la Tabla 3.2 y el párrafo cuantitativo del Ratio de Solubilidad. |
| **H-05** | **MEDIO** | Sec. 2.2 y Sec. 5 | **Limitaciones de `StandardScaler` y Mahalanobis bajo no normalidad:** Con curtosis de $+5.862$ en `CuS`, `StandardScaler` comprime los $22$ meses normales; además, el umbral $\chi^2_2$ de Mahalanobis asume normalidad bivariada (ya rechazada). | Añadir nota técnica sobre el carácter heurístico del umbral $\chi^2_2$ y ordenar en Bloques 2–3 un contraste de sensibilidad entre `StandardScaler` y `RobustScaler`. |
| **H-06** | **BAJO** | Sec. 3.1 y Sec. 6 | **Micro-inconsistencia en $\text{Var}(\text{Sesgo})$ y rango ACF:** $\text{Var}(\text{CuT}) + \text{Var}(\text{Canje}) - 2\text{Cov} = 0.019996 \neq 0.019969$ (redondeo intermedio); y el inventario reporta ACF hasta rezago $10$ (excesivo para $N=24$, donde Box-Jenkins recomienda máx. $N/4 = 6$). | Verificar cálculo exacto de `Sesgo_Canje` sin redondeos intermedios y acotar los gráficos ACF a rezagos $0$ a $6$. |

---

## 2. Análisis Experto y Crítico de los Hallazgos

### 2.1. [H-01] Error Numérico en Correlaciones $N = 22$ y el Desafío Real de la Segunda Neurona (`Cancha_CuS`)

En la **Sección 3.2** del informe se reporta que, al excluir los dos meses atípicos (**Octubre 2023** y **Noviembre 2023**), las correlaciones entre `Cancha_CuS` y `Canje_CuT` son $r = +0.4210$ y $\rho = +0.4380$.

Al recalcular independientemente los estadísticos sobre los $22$ registros oficiales de `DATASET_AUDITADO_N24.csv`:

| Par de Variables ($N = 22$, sin Oct/Nov-23) | Reportado en Bloque 1 | Cálculo Exacto Verificado | $p$-valor Real ($N = 22$) | Significancia ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: |
| **`Cancha_CuT` vs. `Canje_CuT` (Pearson $r$)** | $+0.6181$ | **$+0.6181$** | $0.0022$ | Sí (Significativa) |
| **`Cancha_CuT` vs. `Canje_CuT` (Spearman $\rho$)** | $+0.6253$ | **$+0.6253$** | $0.0019$ | Sí (Significativa) |
| **`Cancha_CuS` vs. `Canje_CuT` (Pearson $r$)** | $+0.4210$ *(Erróneo)* | **$+0.3688$** | **$0.0912$** | **NO significativa** |
| **`Cancha_CuS` vs. `Canje_CuT` (Spearman $\rho$)** | $+0.4380$ *(Erróneo)* | **$+0.3520$** | **$0.1081$** | **NO significativa** |

#### ¿Por qué esto es una revelación estratégica para tu tesis?
1. **Riesgo de credibilidad:** Si un miembro de la comisión reproduce tus tablas con el CSV adjunto en el repositorio, encontrará inmediatamente que $+0.4210$ y $+0.4380$ no corresponden a los datos.
2. **Lectura geometalúrgica profunda:** Fuera de los dos pulsos extremos (cuando `CuS` salta a $0.26\%$ y $0.20\%$), la fracción soluble oscila en una banda muy estrecha ($0.12\%\text{--}0.17\%\text{CuS}$) y **no tiene correlación lineal marginal significativa** con `Canje_CuT` ($p = 0.091$).
3. **Implicancia directa para los Bloques 2, 3 y 4:** Esto significa que el aporte de la segunda neurona (`Cancha_CuS`) **no puede darse por sentado solo por el $r = +0.6406$ global**, sino que deberá **demostrarse empíricamente**:
   * En el **Bloque 2**, observar si `LassoCV` mantiene o lleva a cero el coeficiente de `CuS`.
   * En el **Bloque 3 y 4**, incluir un **estudio de ablación complementario** (comparar la red ganadora con 2 entradas `[CuT, CuS]` frente a 1 entrada `[CuT]`) y cuantificar el aporte real de `CuS` mediante **Importancia por Permutación** y **SHAP**. Así demostrarás si `CuS` ayuda específicamente a corregir la curvatura cuando aparecen frentes mixtos/oxidados (como Oct-2023) sin degradar la predicción en los $22$ meses sulfurados regulares.

---

### 2.2. [H-02] La Trampa de la Descomposición Sesgo-Varianza y el Techo del $9.66\%$

En la **Sección 2.3**, el informe afirma textualmente:
> *"Este resultado prueba que casi el $86\%$ del error histórico en Mantos Cobrizo es un sesgo determinista de sobreestimación, mientras que solo el $14\%$ es **ruido estocástico no explicable**."*

#### Por qué esta frase puede destruir tu tesis en la defensa oral
Apliquemos álgebra elemental sobre tus propios datos:
* Supongamos que un supervisor de mina, sin saber nada de Inteligencia Artificial, simplemente resta el sesgo histórico constante en Excel:
  $$\hat{y}_{\text{trivial}, i} = \text{Cancha\_CuT}_i - 0.3404$$
* ¿Cuál sería el error de esa fórmula trivial?
  * $\text{MSE}_{\text{trivial}} = \sigma^2_{\text{error}} = \mathbf{0.01914\,(\%\text{Cu})^2}$ (reducción inmediata del $85.83\%$ del $\text{MSE}$).
  * $\text{RMSE}_{\text{trivial}} = \sqrt{0.01914} = \mathbf{0.1383\%\text{Cu}}$.
  * $\text{NRMSE}_{\text{trivial}} = \frac{0.1383}{1.4313} \times 100\% = \mathbf{9.66\%}$.
  * $R^2_{\text{trivial}} = 1 - \frac{0.01914}{0.05945} = \mathbf{+0.6781}$.

Si tú declaras en el Bloque 1 que ese $14.17\%$ restante ($\sigma^2_{\text{error}} = 0.01914$) es *"ruido estocástico no explicable"*, cualquier evaluador te hará la pregunta letal: ***"Candidato, si el $85.8\%$ se arregla restando $0.34\%$ en una calculadora y el $14.2\%$ restante es ruido no explicable, ¿para qué entrenó una Red Neuronal?"***

#### La formulación técnica correcta que blinda tu investigación
Ese $14.17\%$ ($\sigma^2_{\text{error}} = 0.01914\,(\%\text{Cu})^2$, $\text{NRMSE} = 9.66\%$) **NO es ruido puramente irreducible**. Es la **varianza total de la discrepancia empírica**, la cual contiene dos subcomponentes:
1. **Varianza geometalúrgica estructurada (explicable):** Variación del sesgo en función del nivel de ley ($\text{Cov}(\text{CuT}, \text{Sesgo}) = +0.0080$) y del régimen mineralógico óxido/sulfuro (`CuS`). Es exactamente esta fracción la que explica por qué el modelo de referencia del Dr. Delgado logra perforar el piso del $9.66\%$ y bajar hasta **$\text{NRMSE} \approx 8.63\%$** ($\text{MSE} \approx 0.0153\,(\%\text{Cu})^2$, $R^2 > 0.74$).
2. **Error analítico y de muestreo irreducible:** Piso físico acotado por la varianza fundamental de Pierre Gy y las tolerancias de canje reglamentarias de ENAMI ($\pm 0.04\%\text{ a }\pm 0.06\%\text{Cu}$).

> **Recomendación de oro para el Bloque 2:** Agrega como modelo de referencia **Baseline 1B: Corrección de Sesgo Constante (LOOCV)** ($\hat{y}_k = \text{Cancha\_CuT}_k - \bar{e}_{\text{train}, -k}$). Demostrar que tu Red Neuronal supera tanto a la Cancha directa ($25.67\%$) como a la corrección por sesgo constante ($\sim 10\%$) y a la regresión OLS/Ridge será un argumento irrefutable ante el tribunal.

---

### 2.3. [H-03] Rigor Econométrico sobre Normalidad y Modelos Lineales (Teorema de Gauss-Markov)

En las **Secciones 1.5 y 4**, el informe sostiene que el rechazo de normalidad en `Cancha_CuT`, `Cancha_CuS` y `Canje_CuT` *"invalida modelos lineales gaussianos"*.

* **Crítica experta:** En regresión lineal clásica (MCO / OLS), **NO se exige normalidad en las variables de entrada ($X$) ni en la distribución marginal de la respuesta ($Y$)**. Los supuestos clásicos recaen sobre los **errores condicionales** ($\varepsilon_i \mid X_i \sim \mathcal{N}(0, \sigma^2)$). De hecho, el sesgo empírico de tus datos **sí aceptó la normalidad** en Shapiro-Wilk ($W = 0.9782, p = 0.8491$).
* **Redacción correcta para tu tesis:**
  > *"El rechazo de normalidad en las leyes brutas ($p < 0.05$ en Shapiro-Wilk), caracterizado por asimetría positiva y curtosis elevada en $\text{CuS}$ ($+5.862$), refleja la naturaleza lognormal/mixta de los depósitos estratoligados tipo manto. Aunque la no normalidad marginal de los predictores no invalida algebraicamente la estimación por MCO, sí evidencia la presencia de observaciones con alto apalancamiento (*leverage*) capaces de distorsionar hiperplanos lineales en muestras acotadas ($N = 24$). Esto fundamenta: (i) el uso de pruebas no paramétricas (Spearman y Wilcoxon), (ii) la evaluación de escalado robusto frente a Z-score, y (iii) el contraste empírico entre modelos lineales regularizados y aproximadores no lineales (SVR y MLP), cuya validez distribucional se auditará formalmente sobre los residuos condicionales en el Bloque 4."*

---

### 2.4. [H-04] Restitución de Entregables Omitidos en la Sección 3

Al pasar de la primera versión del Bloque 1 a esta versión v4.0, se perdieron tres elementos cuantitativos que sí estaban comprometidos en `METODOLOGIA_TRABAJO_TESIS.md` (v4.0):
1. **Información Mutua ($\text{MI}$):** Reincorporar los valores de `mutual_info_regression` (`0.2156` para `Cancha_CuT` y `0.1437` para `Cancha_CuS`), aclarando que con $N = 24$ el estimador basado en vecinos cercanos ($k\text{-NN}$) tiene varianza muestral y sirve como diagnóstico complementario de dependencia no lineal.
2. **$p$-valores de Correlación:** Agregar explícitamente las columnas de $p$-valor en la Tabla 3.2 tanto para $N = 24$ como para $N = 22$.
3. **Análisis Numérico del Ratio de Solubilidad vs. Sesgo:** Reincorporar en el texto de la Sección 3 los coeficientes ($r = -0.2164, p = 0.3097$; $\rho = -0.3430, p = 0.1008$) que explican la tendencia negativa moderada no significativa y justifican no añadir este ratio como tercera entrada sintética.

---

### 2.5. [H-05] Sensibilidad de `StandardScaler` y Advertencia Estadística sobre Mahalanobis

1. **Limitación de `StandardScaler` frente a `Cancha_CuS`:** Dado que `Cancha_CuS` presenta asimetría $+2.277$ y curtosis $+5.862$ por Octubre 2023 ($0.26\%$, ubicado a $+3.68$ desviaciones estándar de la media), el escalado Z-score comprime el $92\%$ de los meses normales dentro de un rango muy estrecho ($[-0.88, +0.75]$). Por ello, el Bloque 1 debe concluir que `StandardScaler` es el escalador base, pero **ordenar explícitamente en los Bloques 2 y 3 una prueba comparativa con `RobustScaler`** (centrado en mediana e $\text{IQR}$) dentro de cada pliegue LOOCV.
2. **Supuesto de Mahalanobis:** Agregar una línea técnica en la Sección 5 señalando que, al haberse rechazado la normalidad univariada de $\text{CuT}$ y $\text{CuS}$, el contraste de $D_M^2$ contra $\chi^2_{2, 0.95} = 5.9915$ se emplea como **criterio diagnóstico heurístico de distancia multivariada** y no como prueba probabilística paramétrica exacta.

---

### 2.6. [H-06] Detalles Técnicos Menores (Verificación de Precisión y Rezagos ACF)

1. **Exactitud algebraica de la matriz de covarianza (Tabla 3.1):** Verificar en `01_eda_caracteristicas.ipynb` por qué $\text{Var}(\text{Cancha\_CuT}) + \text{Var}(\text{Canje\_CuT}) - 2\text{Cov}(\text{Cancha\_CuT}, \text{Canje\_CuT}) = 0.058128 + 0.062038 - 2(0.050085) = 0.019996$, mientras la diagonal reporta `0.019969` (diferencia de $2.7 \times 10^{-5}$, probablemente por haber leído una columna pre-redondeada frente a otra con más decimales). Asegurar que `Sesgo_Canje = df['Cancha_CuT'] - df['Canje_CuT']` se calcule directamente sobre las mismas columnas exactas del CSV auditado.
2. **Rango máximo de rezagos en ACF (Sección 6 y 8):** En el inventario (Sección 8) se menciona *"Gráficos ACF para rezagos 0 a 10"*, mientras la Tabla 6 muestra solo hasta el rezago $4$. Con $N = 24$, estimar autocorrelaciones más allá del rezago $6$ ($N/4 = 6$) utiliza menos de $18$ pares y pierde fiabilidad estadística. Estandarizar tanto la tabla como el gráfico ACF a **rezagos $1$ a $6$** (dejando visible el comportamiento de `Canje_CuT` hasta Lag $6$).

---

## 3. Implicancias Estratégicas de los Resultados del Bloque 1 para el Resto de la Tesis

Más allá de las correcciones de forma, los números reales del Bloque 1 ya dibujan el **panorama técnico exacto** al que te enfrentarás en los Bloques 2, 3 y 4:

1. **El verdadero desafío predictivo no es bajar del $26\%$ al $10\%$, sino romper la barrera del $9.66\%$:**
   * Eliminar el sesgo constante ($-0.34\%\text{Cu}$) resuelve por sí solo el $85.8\%$ del $\text{MSE}$ histórico. El mérito real de tu modelo de Machine Learning se jugará en reducir el $\text{RMSE}$ por debajo de $0.138\%\text{Cu}$ ($\text{NRMSE} < 9.66\%$, $R^2 > 0.68$) de manera estable y fuera de muestra (*out-of-sample*).
2. **Alto riesgo de sobreajuste en los dos meses extremos (Oct-2023 y Nov-2023):**
   * En los pliegues LOOCV donde se deja fuera Noviembre 2023 (`Cancha_CuT = 2.47%`, `Canje_CuT = 2.27%`), el modelo entrenado con los otros $23$ meses estará **extrapolando** fuera del rango máximo de entrenamiento ($2.21\%$).
   * Las redes neuronales con activación `tanh` saturan en los extremos y tienden a subestimar fuertemente puntos de extrapolación, mientras que `relu` extrapola linealmente pero puede amplificar ruido. Evaluar **ambas activaciones (`relu` vs. `tanh`)** con regularización $L_2$ en el Bloque 3 será decisivo para manejar Noviembre 2023 sin disparar el $\text{MSE}_{\text{LOOCV}}$.
3. **Transparencia ante la comisión:**
   * Documentar con honestidad tanto el rol crítico de los dos pulsos extremos como el comportamiento de `Cancha_CuS` en meses regulares ($N = 22$) te posiciona como un ingeniero crítico que entiende sus datos en profundidad, evitando que la comisión te sorprenda en la defensa.

---

## 4. Instrucciones Exactas de Corrección para `Antigravity`

Antes de iniciar la ejecución del **Bloque 2**, aplicar este parche quirúrgico sobre `./BLOQUE_1/`:

1. **Corregir Tabla 3.2 en `01_eda_caracteristicas.ipynb` e `INFORME_ENTREGABLE_BLOQUE_1.md`:**
   * Recalcular dinámicamente `pearsonr` y `spearmanr` (con sus $p$-valores) para $N = 24$ y $N = 22$ (excluyendo `2023-10` y `2023-11`) directamente desde `DATASET_AUDITADO_N24.csv`. Confirmar que `Cancha_CuS` vs. `Canje_CuT` ($N = 22$) arroje $r = +0.3688$ ($p = 0.0912$) y $\rho = +0.3520$ ($p = 0.1081$).
   * Reincorporar las columnas de **Información Mutua ($\text{MI}$)** y el análisis numérico de **Ratio de Solubilidad vs. Sesgo Empírico**.
2. **Corregir redacción técnica en Secciones 1.5, 2.3, 4 y 5:**
   * **Sec. 2.3:** Reemplazar *"ruido estocástico no explicable"* por *"varianza de la discrepancia empírica (que combina estructura geometalúrgica no lineal explicable y error analítico/de muestreo irreducible)"*.
   * **Sec. 1.5 y 4:** Reemplazar *"invalida modelos lineales gaussianos"* por *"evidencia asimetría positiva y observaciones de alto apalancamiento (leverage), justificando pruebas no paramétricas, escalado robusto y contraste empírico entre modelos lineales y no lineales (auditando los residuos condicionales en el Bloque 4)"*.
   * **Sec. 2.2 y 5:** Añadir nota sobre evaluar `RobustScaler` como prueba de sensibilidad frente a `StandardScaler` en los Bloques 2–3, y declarar el umbral $\chi^2_2$ de Mahalanobis como criterio diagnóstico heurístico bajo no normalidad bivariada.
3. **Pulido numérico en Tabla 3.1 y Sección 6:**
   * Calcular `df['Sesgo_Canje'] = df['Cancha_CuT'] - df['Canje_CuT']` directamente sobre las series exactas para eliminar la diferencia de $2.7 \times 10^{-5}$ en la identidad de varianza.
   * Homologar la tabla y el gráfico ACF estrictamente a los rezagos **$k = 1, \dots, 6$** ($N/4$).
4. **Directrices adicionales que pasan al Bloque 2 (`02_baselines_loocv.ipynb`):**
   * Incluir **Baseline 1B: Corrección de Sesgo Constante (LOOCV)** ($\hat{y}_k = \text{Cancha\_CuT}_k - \bar{e}_{\text{train}, -k}$) junto a Cancha directa, OLS, Ridge, Lasso, ElasticNet y SVR (RBF).
   * Incluir prueba comparativa de escalado (`StandardScaler` vs. `RobustScaler`) y verificación **Buffered-LOOCV**.