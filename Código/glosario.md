# GLOSARIO TÉCNICO-ACADÉMICO MAESTRO DE TESIS
## De los Fundamentos Mineros y Estadísticos hasta el Machine Learning, Redes Neuronales y Explicabilidad (XAI)

**Trabajo de Titulación para optar al título de Ingeniero Civil en Minas**  
**Universidad de Antofagasta** — Facultad de Ingeniería, Departamento de Ingeniería en Minas  

* **Autor:** Felipe Andrés Guerra Maturana  
* **Profesor Guía:** Dr. José Delgado Vega (*Doctor en Geología, Especialista en Geoestadística y Explotaciones Mineras, École des Mines de Paris*)  
* **Título de Tesis:** *Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas*  
* **Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) $\to$ Planta San José (ENAMI)  
* **Versión Metodológica:** 4.0 (Cifras Auditadas y Selladas RCI — Bloques 0 al 4 y `Código unificado.ipynb`)  

---

## Guía de Estudio: Cómo Dominar Este Glosario (De Básico a Avanzado)

Este documento está diseñado para que domines **cada concepto, fórmula y decisión de tu tesis desde cero hasta el nivel de defensa ante comisión evaluadora**. Cada término sigue siempre la misma triple estructura pedagógica:
1. **Explicación intuitiva desde cero:** Qué significa en lenguaje claro de terreno e ingeniería, sin jerga innecesaria.
2. **Formulación matemática:** La ecuación exacta en notación limpia para escribir en pizarra o diapositiva.
3. **Aplicación y valor exacto en tu tesis (Mantos Cobrizo):** El número oficial auditado (Metodología v4.0) y el argumento defensivo ante el Dr. José Delgado Vega.

### Ruta de Estudio Recomendada en 3 Niveles

| Nivel | Enfoque Pedagógico | Secciones Clave | Objetivo de Dominio |
| :---: | :--- | :--- | :--- |
| **Nivel 1: Base Operacional y Estadística** | Entender el problema físico en cancha/ENAMI y los estadísticos esenciales de los 24 meses. | **Sección 1** (completa) + **Sección 2** (2.1 a 2.13) + **Sección 5** (5.1 a 5.6) | Explicar qué son `CuT`, `CuS`, `Canje_CuT`, Pierre Gy, el sesgo $+0.3404\%\text{Cu}$, el $\text{MECM} \approx 26\%$, la varianza y el $\text{MSE}/\text{RMSE}$. |
| **Nivel 2: Protocolo Experimental y Modelos** | Dominar el diseño sin fuga de datos, los 7 *baselines* y las 11 redes neuronales. | **Sección 2** (2.14 a 2.25) + **Sección 3** (completa) + **Sección 4** (completa) | Defender por qué LOOCV y no `80/20`, cómo actúa `StandardScaler`, por qué `L-BFGS` supera a `Adam` en $N=23$, por qué Keras colapsa de $57 \to 3$ y cómo funciona `Residual-MLP`. |
| **Nivel 3: Defensa Avanzada y Blindaje Crítico** | Responder preguntas trampa de econometría, geometría de dominio y explicabilidad XAI. | **Sección 5** (5.7 a 5.10) + **Sección 6** (completa) + **Tablas Comparativas Finales** | Explicar la paradoja $r=+0.8347$ vs. $R^2=-1.2709$, la Varianza Explicada ($0.6789$ vs. $0.6503$ por $(24/23)^2$), Breusch–Pagan, SHAP, *Convex Hull* y la Estrategia Dual. |

> **Regla de Oro Metalúrgica (Memorizar antes de la defensa):**  
> **`Canje_CuT` es la ley de cabeza química acordada en recepción ENAMI mediante el protocolo reglamentario de tres sobres, *ANTES* de que el mineral entre a molienda y flotación.** Jamás atribuyas la brecha entre `Cancha_CuT` y `Canje_CuT` a cinética de flotación, reactivos xantatos ni pérdidas en celdas concentradoras. La causa física real reside aguas arriba en el **muestreo del acopio** bajo la **Teoría de Pierre Gy** (friabilidad diferencial y segregación gravitacional de finos oxidados).

---

## Índice Temático Estructurado

* [**Sección 1: Fundamentos Mineros, Metalúrgicos y Comercialización ENAMI**](#sección-1-fundamentos-mineros-metalúrgicos-y-comercialización-enami)
  * 1.1 Cobre Total (`CuT`) | 1.2 Cobre Soluble (`CuS`) | 1.3 Cobre Insoluble (`CuIn`) y Balance Químico | 1.4 Ratio de Solubilidad | 1.5 `Cancha_Ton` vs. `Romana_Ton` | 1.6 `PSJ_CuT` | 1.7 `Canje_CuT` | 1.8 Protocolo Reglamentario de 3 Sobres | 1.9 Tarifa Base $2.5\%\text{Cu}$ | 1.10 Escala $1\%\text{Cu}$ | 1.11 Maquila Implícita | 1.12 Fondo de Estabilización (Cargos Reglamentarios) | 1.13 Teoría de Muestreo de Pierre Gy | 1.14 Heterogeneidad Constitucional y de Distribución | 1.15 Friabilidad Diferencial y Dureza de Mohs | 1.16 Segregación Gravitacional de Finos
* [**Sección 2: Estadística Descriptiva, Geoestadística y Econometría Clásica**](#sección-2-estadística-descriptiva-geoestadística-y-econometría-clásica)
  * 2.1 Media Aritmética | 2.2 Mediana | 2.3 Media Truncada | 2.4 Desviación Estándar y Varianza (`ddof=1` vs. `ddof=0`) | 2.5 Coeficiente de Variación ($\text{CV}\%$) | 2.6 Asimetría | 2.7 Curtosis | 2.8 Covarianza y Matriz de Covarianza | 2.9 Descomposición Sesgo–Varianza del MSE | 2.10 Prueba de Shapiro–Wilk | 2.11 Gráfico Q–Q Normal | 2.12 Teorema de Gauss–Markov | 2.13 Correlación de Pearson ($r$) vs. Spearman ($\rho$) vs. Información Mutua ($\text{MI}$) | 2.14 Efecto Apalancamiento (*Leverage*) | 2.15 Regla de Tukey (IQR) | 2.16 Distancia de Mahalanobis y Elipse $95\%$ | 2.17 Autocorrelación Serial ($\text{ACF}$) | 2.18 Bandas de Bartlett | 2.19 Ruido Blanco | 2.20 Multicolinealidad ($\text{VIF}$ con Constante) | 2.21 Homocedasticidad vs. Heterocedasticidad | 2.22 Prueba de Breusch–Pagan | 2.23 Estadístico de Durbin–Watson | 2.24 Test Pareado Exacto de Wilcoxon | 2.25 Bootstrap No Paramétrico
* [**Sección 3: Protocolo Experimental y Validación en Muestras Pequeñas**](#sección-3-protocolo-experimental-y-validación-en-muestras-pequeñas)
  * 3.1 Fuga de Datos (*Data Leakage*) | 3.2 `Pipeline` | 3.3 `StandardScaler` (Z-score) vs. `RobustScaler` | 3.4 LOOCV | 3.5 Validación Cruzada Anidada (*Nested CV*) | 3.6 Holdout Cronológico 19/5 | 3.7 *Buffered-LOOCV* | 3.8 Compromiso Sesgo–Varianza | 3.9 *Overfitting* vs. *Underfitting* | 3.10 Principio de Occam / Parsimonia | 3.11 Ratio $N_{\text{train}} / P$
* [**Sección 4: Modelos Predictivos, Hiperparámetros, Optimizadores y Funciones de Activación**](#sección-4-modelos-predictivos-hiperparámetros-optimizadores-y-funciones-de-activación)
  * 4.1 Baseline 1A (Cancha Directa) | 4.2 Baseline 1B (Corrección Sesgo Constante) | 4.3 OLS / MCO | 4.4 Regularización $L_1$ (`Lasso`) vs. $L_2$ (`Ridge`) vs. `ElasticNet` | 4.5 SVR Kernel RBF ($C, \epsilon, \gamma$) | 4.6 Perceptrón Multicapa (MLP) | 4.7 Capas Ocultas | 4.8 Pesos Sinápticos ($\mathbf{W}$) y Sesgos ($\mathbf{b}$) | 4.9 Parámetros Nominales ($P$) vs. Efectivos ($P_{\text{eff}}$) | 4.10 Colapso Lineal Algebraico de Keras ($57 \to 3$) | 4.11 Funciones de Activación (`Linear` vs. `ReLU` vs. `Tanh`) | 4.12 Optimizadores (`L-BFGS` vs. `Adam` vs. `SGD`) | 4.13 Penalización *Weight Decay* ($\alpha$) | 4.14 Escalado de $Y$ (`scale_y`) | 4.15 Ensamble Multi-Semilla | 4.16 Aprendizaje Residual (`Residual-MLP` / *Skip-Connection*) | 4.17 Estudio de Ablación (2D vs. 1D)
* [**Sección 5: Métricas de Evaluación Predictiva**](#sección-5-métricas-de-evaluación-predictiva)
  * 5.1 MSE | 5.2 RMSE | 5.3 MAE | 5.4 NRMSE ($\%$) | 5.5 MECM Histórico | 5.6 MAPE ($\%$) | 5.7 $R^2$ Score y Por Qué Puede Ser Negativo ($-1.2709$) | 5.8 Varianza Explicada vs. $R^2$ y Factor $(24/23)^2$ | 5.9 Sesgo Medio | 5.10 Paradoja $r = +0.8347$ vs. $R^2 = -1.2709$
* [**Sección 6: Explicabilidad XAI, Geometría de Dominio e Ingeniería de Software**](#sección-6-explicabilidad-xai-geometría-de-dominio-e-ingeniería-de-software)
  * 6.1 Caja Negra vs. XAI | 6.2 *Permutation Importance* | 6.3 Valores SHAP / Shapley / `KernelExplainer` | 6.4 Superficie de Respuesta 2D | 6.5 Envolvente Convexa (*Convex Hull*) | 6.6 Triangulación de Delaunay | 6.7 Serialización `.joblib` y Diccionarios Nativos | 6.8 Determinismo `float64` (BLAS/Fortran) vs. `float32` (`oneDNN`/CUDA) | 6.9 `@tf.function` y `tf.GradientTape`
* [**Anexo 1: Tabla Comparativa Rápida de Funciones de Activación**](#anexo-1-tabla-comparativa-rápida-de-funciones-de-activación)
* [**Anexo 2: Tabla Comparativa Rápida de Optimizadores**](#anexo-2-tabla-comparativa-rápida-de-optimizadores)
* [**Anexo 3: Hoja de Referencia Rápida — Cifras Oficiales Auditadas (Metodología v4.0)**](#anexo-3-hoja-de-referencia-rápida--cifras-oficiales-auditadas-metodología-v40)

---

## Sección 1: Fundamentos Mineros, Metalúrgicos y Comercialización ENAMI

### 1.1. Cobre Total (`CuT`)
* **(a) Explicación intuitiva desde cero:** Es el porcentaje total de cobre metálico contenido en una muestra de roca o mineral, sin importar en qué especie mineralógica se encuentre (óxidos, sulfuros secundarios o sulfuros primarios). Se determina en laboratorio químico mediante digestión ácida completa (ataque multiácido / agua regia) y espectrofotometría de absorción atómica.
* **(b) Fórmula matemática:**
  $$\text{CuT}\;(\%) = \frac{m_{\text{Cu total}}}{m_{\text{mineral seco}}} \times 100\% = \text{CuS} + \text{CuIn}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** `Cancha_CuT` es el **primer predictor ($X_1$)** de todos los modelos de la tesis, medido en la cancha de acopio antes del despacho a ENAMI. En los $N = 24$ balances mensuales auditados (Ene-2023 a Dic-2024), presenta media $\mu = 1.7717\%\text{Cu}$, mediana $\tilde{x} = 1.7350\%\text{Cu}$, rango entre $1.41\%\text{Cu}$ (`2024-12`) y $2.47\%\text{Cu}$ (`2023-11`), y varianza muestral $s^2 = 0.058128\,(\%\text{Cu})^2$. En el Bloque 0 se homologó de fracción decimal (`0.0216`) a puntos porcentuales (`2.16 %Cu`) con exactamente 2 decimales.

### 1.2. Cobre Soluble (`CuS`)
* **(a) Explicación intuitiva desde cero:** Es la fracción del cobre total que se disuelve en una solución estándar de ácido sulfúrico diluido (lixiviación analítica en frío). Cuantifica principalmente las especies oxidadas y de alteración supergena como crisocola, malaquita, atacamita o brocantita. En un yacimiento de sulfuros como Mantos Cobrizo, un alza en `CuS` delata frentes más meteorizados, oxidados y mecánicamente más friables.
* **(b) Fórmula matemática:**
  $$\text{CuS}\;(\%) = \frac{m_{\text{Cu disuelto en }\text{H}_2\text{SO}_4\text{ diluido}}}{m_{\text{mineral seco}}} \times 100\%$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** `Cancha_CuS` es el **segundo predictor ($X_2$)** de las redes neuronales y modelos 2D. Presenta media $\mu = 0.1471\%\text{Cu}$, mediana $\tilde{x} = 0.1400\%\text{Cu}$, desviación estándar $s = 0.0307\%\text{Cu}$, varianza $s^2 = 0.000943\,(\%\text{Cu})^2$, rango de $0.12\%\text{Cu}$ a un pico oxidado extremo de $0.26\%\text{Cu}$ en `2023-10` ($+3.68\sigma$, curtosis $+5.8621$). Aunque `LassoCV` lo retuvo en **$24/24$ pliegues ($100\%$)** y la ablación en `RNA_MLP_2` probó que reduce el NRMSE en **$-1.53\%$**, su aporte real proviene de modular la segregación física del acopio (Pierre Gy), no de la flotación.

### 1.3. Cobre Insoluble (`CuIn`) y Balance Químico
* **(a) Explicación intuitiva desde cero:** Es la fracción de cobre que *no* se disuelve en ácido sulfúrico diluido; representa a los minerales sulfurados primarios y secundarios (calcopirita, bornita, calcosina/calcocina) que requieren flotación para concentrarse. Por conservación de masa química, el cobre total es la suma exacta del cobre soluble más el cobre insoluble.
* **(b) Fórmula matemática:**
  $$\text{CuIn}\;(\%) = \text{CuT} - \text{CuS} \qquad \Longleftrightarrow \qquad \text{CuT} = \text{CuS} + \text{CuIn}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** En cancha (`Cancha_CuIn`: $\mu = 1.6246\%\text{Cu}$ en tabla Bloque 1; $1.6250\%$ por celda exacta de `DATASET_AUDITADO_N24.csv`), el balance químico se cumple con una brecha máxima de solo $0.01\%\text{Cu}$ (redondeo analítico a 2 decimales). **No se ingresa como tercer predictor** porque genera colinealidad exacta ($\text{VIF} > 4{,}000$). En la Auditoría RCI `B0-02` se detectó además que en 2024 las columnas auxiliares `Canje_CuS` y `Canje_CuIn` del Excel original presentaban inconsistencias de registro de hasta $0.20\%\text{Cu}$ respecto a `Canje_CuT`; por ello, la tesis modela y valoriza exclusivamente sobre el ensayo oficial certificado **`Canje_CuT`**.

### 1.4. Ratio de Solubilidad ($\text{CuS}/\text{CuT}$)
* **(a) Explicación intuitiva desde cero:** Indica qué porcentaje del cobre total de un lote se encuentra en forma oxidada/soluble. Permite distinguir rápidamente si un frente de explotación es eminentemente sulfurado fresco (bajo porcentaje soluble) o si corresponde a una zona de transición/enriquecimiento supergeno más meteorizada.
* **(b) Fórmula matemática:**
  $$\text{Ratio\_Solubilidad}\;(\%) = \frac{\text{Cancha\_CuS}}{\text{Cancha\_CuT}} \times 100\%$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** En los $N = 24$ meses exhibe media $\mu = 8.30\%$ (`8.3047%`), mediana $\tilde{x} = 8.33\%$, desviación estándar $s = 1.16\%$, $\text{CV} = 13.98\%$ y distribución normal aceptada (Shapiro–Wilk $W = 0.9463$, $p = 0.2251$). Su correlación con `Sesgo_Canje` resultó no significativa ($r = -0.2164$, $p = 0.3097$; $\rho = -0.3430$, $p = 0.1008$), por lo cual **se usa solo como variable diagnóstica en el EDA y no se añade como entrada sintética** para proteger la parsimonia del modelo.

### 1.5. Tonelaje de Cancha (`Cancha_Ton`) vs. Tonelaje de Romana (`Romana_Ton`)
* **(a) Explicación intuitiva desde cero:** `Cancha_Ton` es la estimación interna de tonelaje acopiado en mina (basada en conteo de baldes/camiones o levantamiento topográfico de pilas), mientras que `Romana_Ton` es el peso oficial certificado en la báscula puente (*romana*) al ingresar los camiones a Planta San José de ENAMI. La romana siempre rige la liquidación financiera comercial.
* **(b) Fórmula matemática:**
  $$T_{\text{Cancha, total}} = \sum_{t=1}^{24} \text{Cancha\_Ton}_t, \qquad T_{\text{Romana, total}} = \sum_{t=1}^{24} \text{Romana\_Ton}_t$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** En el bienio 2023–2024 se auditaron exactamente **$286{,}704\text{ t}$** en `Cancha_Ton` y **$295{,}884.67\text{ t}$** en `Romana_Ton` (media mensual $12{,}328.53\text{ t/mes}$, calculada dinámicamente con `df['Romana_Ton'].sum()`). **Regla operativa v4.0:** Los tonelajes **NO entran como variables predictoras** en los Bloques 1–4 (los algoritmos operan exclusivamente sobre leyes en $\%\text{Cu}$), pero `Romana_Ton` es el insumo maestro del **Bloque 5** (valorización en USD) y demuestra ante el tribunal que los $N = 24$ registros son balances mensuales industriales masivos, no muestras puntuales.

### 1.6. Ley Planta San José (`PSJ_CuT`)
* **(a) Explicación intuitiva desde cero:** Corresponde al ensayo químico propio realizado por el laboratorio de ENAMI Planta San José sobre la muestra tomada en la torre de recepción al momento de recibir el mineral de Mantos Cobrizo. Es el punto de partida de la contraparte compradora antes de cotejarse con el ensayo del productor.
* **(b) Fórmula matemática:**
  $$e_{\text{PSJ}, t} = \text{Cancha\_CuT}_t - \text{PSJ\_CuT}_t$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Presenta media $\mu = 1.4304\%\text{Cu}$, mediana $\tilde{x} = 1.3900\%\text{Cu}$ (`1.3950%` en `DATASET_AUDITADO_N24.csv` sin redondeo intermedio) y desviación estándar $s = 0.2496\%\text{Cu}$. Su brecha histórica frente a `Cancha_CuT` alcanza $\text{NRMSE} = 25.98\%$ ($\text{MAPE} = 25.05\%$, $\text{RMSE} = 0.3717\%\text{Cu}$), reconciliando el **$\approx 26\%$** de la lámina inicial del Dr. Delgado. Además, fue el objetivo preliminar usado en la réplica *in-sample* original de Keras del profesor ($\text{MAPE} = 7.97\%$, $\text{NRMSE} = 9.91\%$–$9.92\%$), antes de consolidar `Canje_CuT` como objetivo comercial definitivo.

### 1.7. Ley de Canje (`Canje_CuT`)
* **(a) Explicación intuitiva desde cero:** Es la **ley oficial y definitiva de compra-venta** sobre la cual ENAMI emite la liquidación monetaria al productor minero. Se acuerda químicamente en recepción de planta **antes de que el mineral ingrese a los circuitos de chancado fino, molienda o flotación**.
* **(b) Fórmula matemática:**
  $$y_t = \text{Canje\_CuT}_t \in \mathbb{R}, \qquad \text{con } t \in \{1, 2, \dots, 24\}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Es la **única variable objetivo ($y$)** de los $18$ modelos de la tesis. En el periodo auditado exhibe media $\bar{y} = 1.4313\%\text{Cu}$ (`1.43125%Cu`), mediana $1.3950\%\text{Cu}$, desviación muestral $s = 0.2491\%\text{Cu}$ (`0.2490%`), varianza muestral $s^2 = 0.062020\,(\%\text{Cu})^2$, varianza poblacional $\sigma_y^2 = 0.059436\,(\%\text{Cu})^2$, mínimo $1.06\%\text{Cu}$ (`2024-12`) y máximo $2.27\%\text{Cu}$ (`2023-11`). **Alerta crítica (Auditoría B3-01 / C-01):** `Canje_CuT` es una **ley de cabeza en recepción**, jamás una ley recuperada tras flotación.

### 1.8. Protocolo Reglamentario de Tres Sobres (ENAMI)
* **(a) Explicación intuitiva desde cero:** Procedimiento normativo de comercialización minera mediante el cual la muestra final preparada en la torre de muestreo automático de ENAMI se divide en **tres sobres sellados e idénticos**: uno para el laboratorio de ENAMI (`PSJ_CuT`), otro para el laboratorio del productor (Mantos Cobrizo) y un tercer sobre lacrado de **dirimencia** (*umpire sample*). Si la diferencia entre productor y ENAMI está dentro de la tolerancia reglamentaria (límite de partición, típicamente $\pm 0.04\%$ a $\pm 0.06\%\text{Cu}$), la ley de canje es el promedio simple; si la supera, el tercer sobre se envía a un laboratorio árbitro independiente.
* **(b) Fórmula matemática:**
  $$\text{Canje\_CuT} = \begin{cases} \dfrac{\text{Ley}_{\text{Productor}} + \text{Ley}_{\text{ENAMI}}}{2}, & \text{si } |\text{Ley}_{\text{Productor}} - \text{Ley}_{\text{ENAMI}}| \le \delta_{\text{tol}} \\[8pt] \text{Resultado Dirimencia (3}^{\text{er}}\text{ sobre)}, & \text{si } |\text{Ley}_{\text{Productor}} - \text{Ley}_{\text{ENAMI}}| > \delta_{\text{tol}} \end{cases}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Este protocolo explica dos hechos fundamentales de tu tesis: (1) prueba irrefutablemente que `Canje_CuT` se cierra **en portería/recepción antes de molienda y flotación**; y (2) las tolerancias analíticas del canje ($\pm 0.04\%$ a $\pm 0.06\%\text{Cu}$) constituyen parte del piso físico de ruido irreducible que ningún algoritmo de Machine Learning puede eliminar por completo.

### 1.9. Tarifa Base de Flotación ($2.5\%\text{Cu}$)
* **(a) Explicación intuitiva desde cero:** Es el precio neto en dólares por tonelada métrica seca (USD/tms) que ENAMI paga por un mineral de concentración (sulfuros) que tenga exactamente una ley de referencia de $2.5\%\text{Cu}$, ya descontados los cargos base de tratamiento (maquila) y las recuperaciones metalúrgicas contractuales fijadas por la estatal en sus circulares mensuales.
* **(b) Fórmula matemática:**
  $$\text{Tarifa Unitaria (US\$/tms)} = \text{Tarifa Base}_{2.5\%} - \text{Escala}_{1\%} \times (2.5 - \text{Canje\_CuT}) - \text{Cargos Reglamentarios}$$
  $$\text{Valor Lote (USD)} = \text{Romana\_Ton (tms)} \times \text{Tarifa Unitaria (US\$/tms)}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** En las $24$ circulares oficiales de ENAMI Sulfuros (2023–2024), la `Tarifa Base 2.5%` osciló entre **$121.684\text{ USD/tms}$** (`2023-11`, cobre LME $360.137\text{ ¢USD/lb}$) y **$166.963\text{ USD/tms}$** (`2024-06`, cobre LME $459.447\text{ ¢USD/lb}$). Gracias a la Auditoría RCI `B0-01`, se corrigió una mezcla silenciosa de monedas en los PDFs originales (22 meses en `MCLP/tms` vs. 2 en `USD/tms`), homologando el $100\%$ de las tarifas a dólares mediante $\text{Valor}_{\text{USD}} = (\text{Valor}_{\text{MCLP}} \times 1{,}000)/\text{Tipo\_Cambio}$.

### 1.10. Escala por $1\%\text{Cu}$ ($\text{Escala}_{1\%}$)
* **(a) Explicación intuitiva desde cero:** Es el monto monetario (en USD/tms por cada $1\%\text{Cu}$) que ENAMI descuenta de la Tarifa Base cuando el mineral entregado tiene una ley inferior a $2.5\%\text{Cu}$, o que suma como premio cuando la ley supera $2.5\%\text{Cu}$. Representa el valor marginal pagable de cada punto porcentual de cobre dentro de la tarifa.
* **(b) Fórmula matemática:**
  $$\Delta \text{Pago}\;(\text{US\$/tms}) = \text{Escala}_{1\%} \times (\text{Canje\_CuT} - 2.5)$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Como todos los lotes de Mantos Cobrizo presentan `Canje_CuT` $< 2.5\%\text{Cu}$ (máximo $2.27\%\text{Cu}$), la escala actúa siempre como descuento proporcional respecto a la base de $2.5\%$. Durante 2023–2024 fluctuó entre **$62.934\text{ USD/tms}\cdot 1\%$** (`2023-11`) y **$82.625\text{ USD/tms}\cdot 1\%$** (`2024-06`). Esto significa que el sesgo sistemático de $+0.3404\%\text{Cu}$ en cancha equivale a una sobrestimación económica directa de $\approx 21$ a $28\text{ USD}$ por cada tonelada despachada.

### 1.11. Maquila Implícita de Tratamiento
* **(a) Explicación intuitiva desde cero:** Es el cargo fijo por tonelada que ENAMI cobra de manera implícita dentro de su tarifa por procesar el mineral (chancado, molienda, flotación y gastos de planta), independientemente de cuánto cobre contenga. Aunque la circular publica directamente la `Tarifa Base` neta, la maquila implícita se recupera exactamente restando la Tarifa Base al valor bruto de $2.5$ unidades de escala.
* **(b) Fórmula matemática:**
  $$\text{Maquila Implícita (US\$/tms)} = 2.5 \times \text{Escala}_{1\%} - \text{Tarifa Base}_{2.5\%}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Fue la **prueba forense definitiva de la Auditoría RCI `B0-01`** que certificó la homologación $100\%$ en dólares de las 24 tarifas ENAMI. Al aplicar la identidad, la maquila implícita converge con exactitud matemática a los tres tramos tarifarios oficiales vigentes: **$34.66\text{ USD/tms}$** (Enero 2023), **$35.65\text{ USD/tms}$** (Febrero 2023 a Enero 2024) y **$39.60\text{ USD/tms}$** (Febrero a Diciembre 2024).

### 1.12. Fondo de Estabilización de Precios (Cargos Reglamentarios)
* **(a) Explicación intuitiva desde cero:** Mecanismo estatal administrado por ENAMI para suavizar los ciclos del precio del cobre para la pequeña minería. Cuando el precio internacional LME supera el precio de sustentación definido por la autoridad, ENAMI aplica una retención reglamentaria por cada punto porcentual de cobre pagado (que sirve para devolver recursos o sostener la tarifa cuando el precio cae).
* **(b) Fórmula matemática:**
  $$\text{Cargos Reglamentarios (US\$/tms)} = \text{Fondo\_Estab}_{1\%} \times \text{Ley Pagable}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** En la tabla tarifaria auditada (`BLOQUE_0`, Auditoría `B0-03`), `Flot_Fondo_Estab_USD_tms_1pct` fue **$0.000\text{ USD/tms}\cdot 1\%$** en los periodos de precio moderado (Ene-2023 y Jun-2023 a Abr-2024) y se activó en los ciclos alcistas del cobre: entre $2.824$ y $4.125\text{ USD/tms}\cdot 1\%$ en Feb–May 2023, y entre $2.020$ y **$5.789\text{ USD/tms}\cdot 1\%$** de May-2024 a Dic-2024 (insumo exacto para la simulación financiera del Bloque 5).

### 1.13. Teoría de Muestreo de Pierre Gy (1998)
* **(a) Explicación intuitiva desde cero:** Es el cuerpo matemático y físico universal que gobierna cómo tomar una muestra representativa de un material particulado heterogéneo (como una pila de mineral chancado) sin introducir sesgos sistemáticos ni varianzas inaceptables. Pierre Gy demostró que todo error de muestreo nace de la combinación de propiedades intrínsecas de la roca, la segregación espacial del acopio y las deficiencias geométricas del equipo o procedimiento de muestreo.
* **(b) Fórmula matemática (Varianza del Error Fundamental de Muestreo — FSE):**
  $$\sigma^2_{\text{FSE}} = \left(\frac{1}{M_S} - \frac{1}{M_L}\right) \, c \, f \, g \, \ell \, d^3$$
  donde $M_S$ es la masa de la muestra, $M_L$ es la masa del lote, $c$ el factor mineralógico, $f$ el factor de forma, $g$ el factor granulométrico, $\ell$ el factor de liberación y $d$ el diámetro nominal máximo de partícula ($d_{95}$).
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Es el **fundamento físico-metalúrgico central de la tesis** (sellado en Auditorías `C-01` y `B3-01`). Explica por qué la ley muestreada en la cancha de la mina sobreestima sistemáticamente en **$+0.3404\%\text{Cu}$** a la ley liquidada en recepción por ENAMI (`Canje_CuT`) en $24$ de $24$ meses: la cancha de acopio presenta errores de delimitación, extracción y segregación mucho mayores que la torre de muestreo automático de Planta San José.

### 1.14. Heterogeneidad Constitucional ($CH_L$) y Heterogeneidad de Distribución ($DH_L$)
* **(a) Explicación intuitiva desde cero:**
  * **Heterogeneidad Constitucional ($CH_L$):** Es la diferencia intrínseca de composición química entre los distintos fragmentos de roca del lote (unos fragmentos son sulfuros ricos en cobre, otros son óxidos friables y otros son ganga estéril). No se elimina mezclando el acopio; solo se reduce chancando más fino.
  * **Heterogeneidad de Distribución ($DH_L$):** Es la forma desigual en que esos fragmentos se reparten en el espacio dentro de la pila o camión (por ejemplo, los finos ricos en óxidos concentrados en el núcleo o base del cono de acopio y los bolones gruesos rodando hacia los bordes).
* **(b) Fórmula matemática:**
  $$CH_L = N_F \sum_{i=1}^{N_F} \left(\frac{a_i - a_L}{a_L}\right)^2 \frac{M_i^2}{M_L^2}, \qquad DH_L = \frac{1 + \gamma \xi}{1 + \gamma} \, CH_L$$
  donde $a_i$ y $M_i$ son la ley y masa del fragmento $i$, $a_L$ y $M_L$ la ley y masa del lote, $\gamma$ el factor de agrupamiento y $\xi$ el factor de segregación.
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** La coexistencia geológica de sulfuros competentes y óxidos supergenos en el Distrito Minero Los Cóndores maximiza $CH_L$, mientras que el vaciado en pilas cónicas en cancha activa un alto factor de segregación $\xi$ ($DH_L$), haciendo que el muestreo superficial de acopio sobreestime la ley real promedio del lote completo ($+0.3404\%\text{Cu}$, con picos hasta $+0.61\%\text{Cu}$ en `2023-01`).

### 1.15. Friabilidad Diferencial y Escala de Dureza de Mohs
* **(a) Explicación intuitiva desde cero:** La **dureza de Mohs** clasifica la resistencia de un mineral a ser rayado del $1$ (talco) al $10$ (diamante), mientras que la **friabilidad** mide la facilidad con que un mineral se fractura y desmenuza en polvo fino al recibir impactos (tronadura, carguío, chancado). Cuando en una misma roca conviven minerales blandos/quebradizos con minerales duros/competentes, al chancarlos los blandos se pulverizan mucho más rápido: eso es la **friabilidad diferencial**.
* **(b) Fórmula matemática (Relación conceptual de partición granulométrica):**
  $$\text{Fracción Fina } (< d_{\text{corte}}) \propto \frac{1}{\text{Dureza Mohs}} \implies \text{Ley}_{\text{finos}} \neq \text{Ley}_{\text{gruesos}}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Los minerales oxidados supergenos cuantificados por `Cancha_CuS` —como **crisocola** (dureza Mohs $2.5$–$3.5$, muy friable/terrosa) y **malaquita** ($3.5$–$4.0$)— son significativamente más blandos y quebradizos que los sulfuros competentes alojados en roca encajadora silicatada (calcopirita $3.5$–$4.0$ tenaz, pirita $6.0$–$6.5$, ganga cuarzosa $7.0$). Durante el manejo y chancado en faena, esta friabilidad diferencial concentra el cobre oxidado en la fracción fina, alterando la representatividad del muestreo en cancha y explicando el peso no lineal activo de `Cancha_CuS` en `RNA_MLP_2` (SHAP $= 0.0701\%\text{Cu}$, ablación $-1.53\%$).

### 1.16. Segregación Gravitacional de Finos (Cono de Acopio vs. Torre Automática ENAMI)
* **(a) Explicación intuitiva desde cero:** Cuando un cargador frontal o camión descarga mineral triturado formando un cono en la cancha, las partículas gruesas y pesadas ruedan por la pendiente hacia la base exterior del cono, mientras que las partículas finas (enriquecidas en especies friables de cobre) se acumulan en el centro y en los intersticios superiores. Si un operador muestrea la superficie o fracciones accesibles de la cancha, captura desproporcionadamente esos finos ricos; en cambio, la **torre de muestreo automático de ENAMI** corta transversalmente la totalidad del flujo cayendo en cinta, eliminando el sesgo de segregación.
* **(b) Fórmula matemática (Sesgo de Muestreo Sistémico):**
  $$\bar{e}_{\text{emp}} = \mathbb{E}[\text{Cancha\_CuT} - \text{Canje\_CuT}] = \text{Error Delimitación (IDE)} + \text{Error Extracción (IEE)} + \text{Error Preparación (IPE)}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Este mecanismo explica la totalidad de la física del modelo predictivo con rigor metalúrgico impecable:
  * **CUIDADO CRÍTICO DE DEFENSA:** Como `Canje_CuT` se acuerda en recepción mediante los **3 sobres antes de molienda y flotación**, la brecha de $+0.3404\%\text{Cu}$ y el efecto de `Cancha_CuS` **NO ocurren en las celdas de flotación ni por baja afinidad a xantatos**, sino por la segregación gravitacional de finos friables en el acopio frente a la torre cortadora de Planta San José.

---

## Sección 2: Estadística Descriptiva, Geoestadística y Econometría Clásica

### 2.1. Media Aritmética ($\mu$ o $\bar{x}$)
* **(a) Explicación intuitiva desde cero:** Es el promedio clásico o "centro de gravedad" de un conjunto de datos. Suma todos los valores y los reparte equitativamente entre la cantidad de observaciones. Es muy intuitiva, pero sensible a valores extremos (*outliers*), que la arrastran hacia arriba o hacia abajo.
* **(b) Fórmula matemática:**
  $$\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo ($N = 24$):**
  * `Cancha_CuT`: $\bar{x} = 1.7717\%\text{Cu}$ | `Cancha_CuS`: $\bar{x} = 0.1471\%\text{Cu}$ | `Canje_CuT`: $\bar{y} = 1.4313\%\text{Cu}$ (`1.43125%Cu`, denominador oficial del NRMSE) | `Sesgo_Canje`: $\bar{e} = +0.3404\%\text{Cu}$ (`+0.340417%Cu`). Note que en las leyes brutas la media supera siempre a la mediana por la asimetría positiva geológica (Oct/Nov 2023).

### 2.2. Mediana ($\tilde{x}$)
* **(a) Explicación intuitiva desde cero:** Es el valor central exacto cuando se ordenan los datos de menor a mayor: la mitad de las observaciones queda por debajo y la otra mitad por encima. A diferencia de la media, la mediana es **robusta**: si el mes más rico del año se duplicara, la mediana no cambiaría en absoluto.
* **(b) Fórmula matemática (para $N$ par):**
  $$\tilde{x} = \frac{x_{(N/2)} + x_{(N/2 + 1)}}{2}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** `Cancha_CuT` $\tilde{x} = 1.7350\%\text{Cu}$, `Cancha_CuS` $\tilde{x} = 0.1400\%\text{Cu}$, `Canje_CuT` $\tilde{y} = 1.3950\%\text{Cu}$, `Sesgo_Canje` $\tilde{e} = +0.3600\%\text{Cu}$. Además, en el Bloque 4 la mediana del error absoluto fuera de muestra revela una ventaja crucial de **`RNA_MLP_2`** (**$0.0801\%\text{Cu}$**) frente a **`Baseline 1B`** (**$0.1044\%\text{Cu}$**), confirmando que la red neuronal es más precisa en un mes típico regular.

### 2.3. Media Truncada (*Trimmed Mean*, $10\%$)
* **(a) Explicación intuitiva desde cero:** Es un punto intermedio entre la media y la mediana: elimina un porcentaje fijo de los valores más pequeños y más grandes (por ejemplo, el $10\%$ inferior y el $10\%$ superior) y promedia únicamente el $80\%$ central. Sirve para diagnosticar cuánto distorsionan los extremos al promedio clásico sin perder la información del cuerpo central.
* **(b) Fórmula matemática (truncamiento simétrico $\alpha$, quitando $k = \lfloor \alpha N \rfloor$ datos en cada cola):**
  $$\bar{x}_{\text{trunc}, \alpha} = \frac{1}{N - 2k} \sum_{i = k+1}^{N - k} x_{(i)}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo ($N=24$, $k=2$ por cola):** `Cancha_CuT` $= 1.7470\%\text{Cu}$ (cae $0.0247\%$ respecto a la media), `Cancha_CuS` $= 0.1415\%\text{Cu}$, `Canje_CuT` $= 1.4045\%\text{Cu}$, `Sesgo_Canje` $= +0.3430\%\text{Cu}$ (casi idéntico a la media $+0.3404\%$, probando simetría del error).

### 2.4. Desviación Estándar y Varianza Muestral (`ddof=1`) vs. Poblacional (`ddof=0`)
* **(a) Explicación intuitiva desde cero:** La **varianza** mide qué tan dispersos están los datos alrededor de su promedio (en unidades al cuadrado), y la **desviación estándar** es su raíz cuadrada (devuelta a las unidades físicas originales, $\%\text{Cu}$).
  * Con **`ddof=1`** (divide por $N-1$, corrección de Bessel) se obtiene el estimador **insesgado muestral** ($s^2, s$), usado en inferencia estadística y matrices de covarianza.
  * Con **`ddof=0`** (divide por $N$) se obtiene la **varianza poblacional/empírica** ($\sigma^2, \sigma$), indispensable para identidades algebraicas exactas sobre el conjunto de $N$ registros como la descomposición del MSE y el $R^2$.
* **(b) Fórmula matemática:**
  $$s^2\;(\text{ddof}=1) = \frac{1}{N - 1} \sum_{i=1}^{N} (x_i - \bar{x})^2, \qquad \sigma^2\;(\text{ddof}=0) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2, \qquad s = \sqrt{s^2}, \quad \sigma = \sqrt{\sigma^2}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * Contraste de escalas (`ddof=1`): $s^2_{\text{Cancha\_CuT}} = 0.058128\,(\%\text{Cu})^2$ ($s = 0.2411\%$) vs. $s^2_{\text{Cancha\_CuS}} = 0.000943\,(\%\text{Cu})^2$ ($s = 0.0307\%$) $\implies$ ratio de **$61.64\times$** ($61.62\times$ sin redondeo intermedio), lo que obliga a escalar entradas.
  * En `Sesgo_Canje`: $s^2 = 0.019917\,(\%\text{Cu})^2$ (`ddof=1`) vs. $\sigma^2 = 0.019087\,(\%\text{Cu})^2$ (`ddof=0`), valor exacto que cierra la identidad $\text{MSE} = \bar{e}^2 + \sigma^2_e = 0.115884 + 0.019087 = 0.134971\,(\%\text{Cu})^2$.

### 2.5. Coeficiente de Variación ($\text{CV}\%$)
* **(a) Explicación intuitiva desde cero:** Expresa la desviación estándar como porcentaje de la media. Como es adimensional, permite comparar la volatilidad relativa de dos variables que viven en escalas muy distintas (por ejemplo, una ley total cercana a $1.8\%$ frente a una ley soluble cercana a $0.15\%$).
* **(b) Fórmula matemática:**
  $$\text{CV}\;(\%) = \frac{s}{|\bar{x}|} \times 100\%$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Mientras en varianza absoluta `Cancha_CuT` domina $61.6\times$, en volatilidad relativa **`Cancha_CuS` es más variable** (**$\text{CV} = 20.87\%$**, `20.88%` sin redondeo) que `Cancha_CuT` (**$13.61\%$**) y `Canje_CuT` (**$17.40\%$**). A su vez, `Sesgo_Canje` presenta $\text{CV} = 41.46\%$, demostrando alta dispersión relativa en la brecha de conciliación mensual.

### 2.6. Asimetría (*Skewness*, $g_1$)
* **(a) Explicación intuitiva desde cero:** Mide hacia qué lado se alarga la "cola" de la distribución de los datos. Un valor cercano a $0$ indica simetría (campana equilibrada); un valor positivo ($g_1 > 0$) indica cola larga hacia la derecha (presencia de leyes excepcionalmente altas, típico de depósitos minerales lognormales); uno negativo ($g_1 < 0$) indica cola hacia la izquierda.
* **(b) Fórmula matemática (momento estandarizado muestral de Fisher–Pearson ajustado):**
  $$g_1 = \frac{N}{(N-1)(N-2)} \sum_{i=1}^{N} \left(\frac{x_i - \bar{x}}{s}\right)^3$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Todas las leyes brutas exhiben fuerte asimetría geológica positiva: `Cancha_CuT` ($+1.2682$), `Canje_CuT` ($+1.5414$) y especialmente `Cancha_CuS` (**$+2.2772$**). En contraste, `Sesgo_Canje` es prácticamente simétrico (**$g_1 = -0.1652$**), coherente con un error de muestreo centrado y bien comportado.

### 2.7. Curtosis (*Kurtosis* de Exceso, $g_2$)
* **(a) Explicación intuitiva desde cero:** Mide qué tan "pesadas" son las colas y qué tan picudo es el centro de la distribución respecto a una campana de Gauss normal (donde el exceso de curtosis es $0$). Valores muy positivos ($g_2 \gg 0$, **leptocúrtica**) alertan sobre observaciones extremas aisladas en las colas; valores negativos ($g_2 < 0$, **platicúrtica**) indican colas más livianas y aplanadas.
* **(b) Fórmula matemática (curtosis de exceso muestral insesgada):**
  $$g_2 = \frac{N(N+1)}{(N-1)(N-2)(N-3)} \sum_{i=1}^{N} \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(N-1)^2}{(N-2)(N-3)}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** `Cancha_CuS` muestra una curtosis leptocúrtica extrema de **$+5.8621$** causada por el pulso oxidado de Octubre 2023 ($0.26\%\text{Cu}$, ubicado a $+3.68\sigma$ de la media). Este hallazgo del Bloque 1 motivó ejecutar en el Bloque 2 el contraste obligatorio de sensibilidad entre `StandardScaler` y `RobustScaler`. Por su parte, `Sesgo_Canje` tiene $g_2 = -0.4804$ (sin colas extremas).

### 2.8. Covarianza ($\text{Cov}(X, Y)$) y Matriz de Covarianza ($\Sigma$)
* **(a) Explicación intuitiva desde cero:** Mide si dos variables tienden a subir y bajar juntas (covarianza positiva), moverse en sentidos opuestos (covarianza negativa) o variar independientemente (cercana a cero). La **matriz de covarianza $\Sigma$** organiza en la diagonal las varianzas de cada variable y fuera de la diagonal las covarianzas entre todos los pares.
* **(b) Fórmula matemática:**
  $$\text{Cov}(X, Y) = \frac{1}{N - 1} \sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y}), \qquad \text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y) - 2\,\text{Cov}(X, Y)$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 1, Tabla 3.1):**
  * $\text{Cov}(\text{Cancha\_CuT}, \text{Canje\_CuT}) = +0.050115\,(\%\text{Cu})^2$.
  * $\text{Cov}(\text{Cancha\_CuT}, \text{Sesgo\_Canje}) = +0.008012\,(\%\text{Cu})^2$ (prueba que el sesgo co-varía con el tenor de ley).
  * Verificación exacta de la identidad matricial: $0.058128 + 0.062020 - 2(0.050115) = 0.019918 \approx 0.019917 = s^2_{\text{Sesgo\_Canje}}$, verificada automáticamente con `assert` en el cuaderno unificado.

### 2.9. Descomposición Sesgo–Varianza del MSE ($\text{MSE} = \bar{e}^2 + \sigma^2_e$)
* **(a) Explicación intuitiva desde cero:** Todo error cuadrático medio total se divide exactamente en dos partes aditivas: (1) el error por apuntar sistemáticamente corrido del blanco (**sesgo al cuadrado**, $\bar{e}^2$), y (2) la dispersión aleatoria o estructural alrededor de ese desvío promedio (**varianza del error**, $\sigma^2_e$). Imagina un tirador que agrupa sus tiros muy juntos, pero todos $30\text{ cm}$ arriba del centro: la mayor parte de su error es sesgo corregible, no dispersión.
* **(b) Fórmula matemática:**
  $$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} e_i^2 = \left(\frac{1}{N}\sum_{i=1}^{N} e_i\right)^2 + \frac{1}{N}\sum_{i=1}^{N}(e_i - \bar{e})^2 = \bar{e}^2 + \sigma^2_e$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Sobre el error histórico de `Cancha_Directa` ($\text{MSE} = 0.134971\,(\%\text{Cu})^2$, $\text{NRMSE} = 25.67\%$):
  * **Sesgo al cuadrado:** $\bar{e}^2 = (0.340417)^2 = 0.115884\,(\%\text{Cu})^2$ (**$85.86\%$** del MSE total).
  * **Varianza empírica de la discrepancia:** $\sigma^2_e = 0.019087\,(\%\text{Cu})^2$ (**$14.14\%$** del MSE total), que define el piso teórico *in-sample* de deducción constante ($\text{RMSE} = 0.1382\%$, $\text{NRMSE} = 9.66\%$). Esta descomposición desenmascara la "falsa valla del $26\%$" y origina a `Baseline 1B` ($10.07\%$ en LOOCV).

### 2.10. Prueba de Normalidad de Shapiro–Wilk ($W$ y $p$-valor)
* **(a) Explicación intuitiva desde cero:** Es la prueba estadística más potente para verificar en muestras pequeñas ($N \le 50$) si un conjunto de datos proviene de una distribución normal (campana de Gauss). Contrasta la hipótesis nula $H_0$: *"los datos siguen una distribución normal"*. Si $W$ está cerca de $1$ y $p \ge 0.05$, se acepta la normalidad; si $p < 0.05$, se rechaza.
* **(b) Fórmula matemática:**
  $$W = \frac{\left(\sum_{i=1}^{N} a_i \, x_{(i)}\right)^2}{\sum_{i=1}^{N} (x_i - \bar{x})^2}$$
  donde $x_{(i)}$ son los datos ordenados y $a_i$ son coeficientes óptimos derivados de las medias y covarianzas de estadísticos de orden normales.
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * **Leyes brutas (Bloque 1):** Todas rechazan normalidad (`Cancha_CuT` $W=0.8871, p=0.0116$; `Cancha_CuS` $W=0.7472, p<0.0001$; `Canje_CuT` $W=0.8787, p=0.0078$).
  * **Errores y residuos (unificado en Auditoría `B1-01`):** `Sesgo_Canje` y residuos de `Cancha_Directa` aceptan normalidad holgadamente (**$W = 0.9852, p = 0.9695$**). En Bloque 4, `Baseline_1B` y `Res_MLP_2_2` obtienen $W = 0.9852, p = 0.9696$, **`RNA_MLP_2`** aprueba con **$W = 0.9307, p = 0.1012$**, mientras que `RidgeCV` ($p = 0.0316$), `RNA_MLP_2_2` ($p = 0.0024$) y `RNA_MLP_3_2` ($p = 0.0130$) fallan la normalidad residual.

### 2.11. Gráfico Q–Q Normal (*Quantile–Quantile Plot*)
* **(a) Explicación intuitiva desde cero:** Es la radiografía visual de la prueba de normalidad: grafica los cuantiles reales de tus datos en el eje vertical contra los cuantiles teóricos que tendría una distribución normal perfecta en el eje horizontal. Si los puntos se alinean sobre la recta diagonal de $45^\circ$, los datos son normales; si se curvan en los extremos en forma de "S" o "J", existen colas pesadas o asimetría.
* **(b) Fórmula matemática (cuantil teórico $i$-ésimo):**
  $$q_i^{\text{teórico}} = \Phi^{-1}\left(\frac{i - 0.5}{N}\right) \quad \text{graficado contra } x_{(i)}$$
  donde $\Phi^{-1}$ es la función inversa de la distribución normal estándar.
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Se utiliza en el Panel B del Bloque 1 (`Sesgo_Canje`, puntos casi pegados a la recta con $W = 0.9852$) y en la Figura 2 del Bloque 4 (residuos OOS de `RNA_MLP_2` y `Baseline 1B`), mostrando visualmente cómo `2023-11` aparece como el único punto alejado en la cola derecha de los modelos directos.

### 2.12. Teorema de Gauss–Markov
* **(a) Explicación intuitiva desde cero:** Establece las condiciones exactas bajo las cuales la regresión lineal por Mínimos Cuadrados Ordinarios (OLS/MCO) es el **Mejor Estimador Lineal Insesgado** (*BLUE*, *Best Linear Unbiased Estimator*), es decir, el de mínima varianza entre todos los estimadores lineales insesgados. **Trampa clásica de examen:** el teorema **NO exige que las variables $X$ ni $Y$ sean normales**; sus supuestos recaen estrictamente sobre el término de error condicional $\varepsilon \mid X$.
* **(b) Fórmula matemática (supuestos fundamentales):**
  $$\mathbb{E}[\varepsilon_i \mid X] = 0, \qquad \text{Var}(\varepsilon_i \mid X) = \sigma^2\;\text{(homocedasticidad)}, \qquad \text{Cov}(\varepsilon_i, \varepsilon_j \mid X) = 0\;\forall\, i \neq j\;\text{(no autocorrelación)}$$
  $$\implies \text{Var}(\hat{\beta}_{\text{OLS}}) \preceq \text{Var}(\tilde{\beta}) \quad \text{para todo estimador lineal insesgado }\tilde{\beta}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Este rigor econométrico (Bloque 1 Sec. 4.2 y Bloque 4 Sec. 3) te blinda ante la comisión: aunque las leyes brutas no son normales, eso por sí solo no invalida a OLS; **lo que sí invalida la optimalidad de OLS en tu tesis es la violación de homocedasticidad** demostrada en el Bloque 4 mediante Breusch–Pagan ($p = 0.0016 < 0.05$), donde la varianza del error explota en leyes altas.

### 2.13. Correlación de Pearson ($r$) vs. Correlación de Spearman ($\rho$) vs. Información Mutua ($\text{MI}$)
* **(a) Explicación intuitiva desde cero:** Son tres lupas distintas para medir asociación entre dos variables:
  * **Pearson ($r$):** Mide únicamente fuerza de relación **en línea recta**. Muy sensible a valores extremos.
  * **Spearman ($\rho$):** Reemplaza cada dato por su posición en el ranking ($1^\circ, 2^\circ, \dots, N^\circ$) y calcula Pearson sobre los rangos; detecta relaciones **monótonas** (curvas crecientes/decrecientes) y es robusto frente a *outliers*.
  * **Información Mutua ($\text{MI} \ge 0$):** Proveniente de la teoría de la información de Shannon; mide cuánta incertidumbre sobre $Y$ desaparece al conocer $X$, capturando **cualquier dependencia no lineal** (incluso no monótona).
* **(b) Fórmula matemática:**
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2}\sqrt{\sum (y_i - \bar{y})^2}}, \qquad \rho = 1 - \frac{6 \sum d_i^2}{N(N^2 - 1)}, \qquad \text{MI}(X; Y) = \iint p(x, y) \ln \frac{p(x, y)}{p(x)p(y)}\,dx\,dy$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 1, Tabla 3.2):**
  * En $N = 24$: `Cancha_CuT` alcanza $r = +0.8347$ ($p = 3.96\times 10^{-7}$), $\rho = +0.7116$, $\text{MI} = 0.2156$; `Cancha_CuS` logra $r = +0.6406$ ($p = 7.46\times 10^{-4}$), $\rho = +0.5024$, $\text{MI} = 0.1437$.
  * En diagnóstico sin extremos ($N = 22$, excluyendo temporalmente Oct/Nov 2023): `CuT` cae a $r = +0.6181 < \rho = +0.6253$, y `CuS` cae a **$r = +0.3688$ ($p = 0.0912$)** y **$\rho = +0.3520$ ($p = 0.1082$)** (no significativo al $5\%$), lo que justificó someter `CuS` a prueba de fuego con `LassoCV`, ablación y SHAP.

### 2.14. Efecto Apalancamiento (*Leverage*)
* **(a) Explicación intuitiva desde cero:** Imagina una balanza o palanca: un peso colocado muy lejos del centro ejerce mucho más torque que el mismo peso cerca del centro. En estadística, una observación con valores de entrada $X$ muy alejados del promedio del resto de los datos posee alto *leverage*: es capaz de inclinar por sí sola una recta de regresión o inflar artificialmente el coeficiente de correlación de Pearson.
* **(b) Fórmula matemática (elemento diagonal $h_{ii}$ de la matriz sombrero $\mathbf{H}$):**
  $$\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T, \qquad h_{ii} = \mathbf{x}_i^T (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{x}_i \in \left[\frac{1}{N}, 1\right]$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Los frentes ricos de **Octubre 2023** (`CuT` $= 2.21\%$, `CuS` $= 0.26\%$) y **Noviembre 2023** (`CuT` $= 2.47\%$, `CuS` $= 0.20\%$) ejercen enorme *leverage*: (1) inflan la correlación Pearson de `CuT` de $+0.6181$ a $+0.8347$ y la de `CuS` de $+0.3688$ a $+0.6406$; y (2) concentran el **$40.5\%$ del error cuadrático total de `OLS`** en LOOCV porque al dejar fuera cualquiera de ellos la pendiente lineal cambia bruscamente.

### 2.15. Regla de Tukey (Rango Intercuartílico, IQR)
* **(a) Explicación intuitiva desde cero:** Método univariado clásico (usado en los gráficos de caja y bigotes o *boxplots*) para marcar valores atípicos sin asumir normalidad. Toma el ancho de la caja central que contiene el $50\%$ medio de los datos ($\text{IQR} = Q_3 - Q_1$) y traza cercas a $1.5$ veces ese ancho hacia arriba y hacia abajo; todo punto fuera de esas cercas es un candidato a *outlier*.
* **(b) Fórmula matemática:**
  $$\text{IQR} = Q_3 - Q_1, \qquad \text{Atípico si } x_i < Q_1 - 1.5\,\text{IQR} \;\text{ o }\; x_i > Q_3 + 1.5\,\text{IQR}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** En `Cancha_CuT` ($\text{IQR} = 0.1900\%$) marca como atípicos a `2023-01` ($2.16\%$), `2023-10` ($2.21\%$) y `2023-11` ($2.47\%$); en `Cancha_CuS` ($\text{IQR} = 0.0225\%$) marca a `2023-10` ($0.26\%$) y `2023-11` ($0.20\%$). Sin embargo, **ninguno se elimina**: son frentes reales facturados por ENAMI, y borrarlos introduciría un sesgo geológico hacia la esterilidad.

### 2.16. Distancia de Mahalanobis ($D_M^2$) y Elipse de Confianza al $95\%$
* **(a) Explicación intuitiva desde cero:** A diferencia de Tukey (que mira una variable a la vez) o la distancia euclidiana (que ignora escalas y correlaciones), la distancia de Mahalanobis mide qué tan lejos está un lote del "centro" de la nube de datos **en múltiples variables simultáneamente**, tomando en cuenta tanto las diferencias de varianza como la correlación entre ellas. Geométricamente, todos los puntos a igual distancia crítica forman una **elipse inclinada** alrededor de los datos.
* **(b) Fórmula matemática:**
  $$D_M^2(\mathbf{x}_i) = (\mathbf{x}_i - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})$$
  Con $p = 2$ predictores, la frontera heurística al $95\%$ viene dada por $\chi^2_{2, 0.95} = 5.9915$.
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 1 Sec. 5):** Utilizada como criterio diagnóstico heurístico multivariado (por no normalidad bivariada estricta), detecta exactamente **2 registros fuera de la elipse al $95\%$**:
  * **`2023-10`** (enriquecimiento supergeno/oxidado): **$D_M^2 = 14.2481$** ($p = 0.0008$).
  * **`2023-11`** (frente competente sulfurado de alta ley): **$D_M^2 = 8.4700$** ($p = 0.0145$).
  * En cambio, `2023-09` ($D_M^2 = 5.2671$, $p = 0.0718$) y `2023-01` ($D_M^2 = 2.7900$, $p = 0.2478$, auditado en `B1-03`) permanecen **dentro** de la elipse. Todos los $24$ meses se retienen íntegramente.

### 2.17. Autocorrelación Serial ($\text{ACF}$, $r_k$)
* **(a) Explicación intuitiva desde cero:** Mide la "memoria temporal" de una serie de datos: correlaciona la serie consigo misma desplazada $k$ meses hacia atrás (rezago o *lag* $k$). Si la ley o el error de un mes depende fuertemente del mes anterior ($r_1$ alto), existe inercia o continuidad espacial/temporal que puede generar fuga temporal si se valida sin cuidado.
* **(b) Fórmula matemática (rezago $k$, regla de Box–Jenkins $K \le \lfloor N/4 \rfloor$):**
  $$r_k = \frac{\sum_{t=k+1}^{N} (x_t - \bar{x})(x_{t-k} - \bar{x})}{\sum_{t=1}^{N} (x_t - \bar{x})^2}, \qquad k = 1, 2, \dots, 6$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo ($K = 24/4 = 6$):**
  * **`Sesgo_Canje`:** $r_1 = -0.0866$, $r_2 = -0.2729$, $r_3 = -0.0625$, $r_4 = +0.0967$, $r_5 = +0.2610$, $r_6 = -0.1036$ (todos dentro de bandas $\pm 0.4001 \implies$ independencia temporal total).
  * **`Cancha_CuT`:** $r_1 = +0.4467 > 0.4001$ (inercia geológica por continuidad de frentes de explotación contiguos $\implies$ justifica *Buffered-LOOCV*).
  * **`Canje_CuT`:** $r_6 = -0.4276$ (Auditoría `B1-04`: refleja el contraste semestral entre el pulso rico Oct/Nov 2023 y los frentes regulares de mediados de 2024, sin comprometer la independencia del sesgo).

### 2.18. Bandas de Bartlett ($\pm 1.96/\sqrt{N}$)
* **(a) Explicación intuitiva desde cero:** Son las dos líneas horizontales punteadas que aparecen en todo correlograma ACF. Marcan el margen de fluctuación aleatoria al $95\%$ de confianza que tendría una serie puramente aleatoria sin memoria: si una barra queda dentro de las bandas, esa autocorrelación es estadísticamente indistinguible de cero.
* **(b) Fórmula matemática:**
  $$\text{Banda de Bartlett}_{95\%} = \pm \frac{z_{0.975}}{\sqrt{N}} = \pm \frac{1.96}{\sqrt{N}}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Para $N = 24$, el umbral exacto es $\pm 1.96 / \sqrt{24} = \mathbf{\pm 0.4001}$ (`±0.40`). El $100\%$ de los rezagos $1$ a $6$ del `Sesgo_Canje` caen holgadamente dentro de $[-0.4001, +0.4001]$.

### 2.19. Ruido Blanco (*White Noise*)
* **(a) Explicación intuitiva desde cero:** Es una secuencia temporal sin memoria ni patrones predecibles en el tiempo: conocer el error del mes pasado no ayuda a adivinar si el error del mes siguiente será mayor o menor al promedio. Su media es constante, su varianza es constante y sus autocorrelaciones son nulas en todos los rezagos $k \ge 1$.
* **(b) Fórmula matemática:**
  $$\mathbb{E}[\varepsilon_t] = \mu_{\varepsilon}, \qquad \text{Var}(\varepsilon_t) = \sigma^2_{\varepsilon}, \qquad \text{Cov}(\varepsilon_t, \varepsilon_{t-k}) = 0 \quad \forall\, k \neq 0$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** El diagnóstico ACF del Bloque 1 demostró que **`Sesgo_Canje` se comporta como ruido blanco puro** ($r_1 = -0.0866 \dots r_6 = -0.1036$). Esta propiedad es el sustento estadístico que valida el uso de **LOOCV** de 24 pliegues independientes, refutando la idea de que exista deriva temporal en el error de conciliación.

### 2.20. Multicolinealidad y Factor de Inflación de Varianza ($\text{VIF}$ con Constante)
* **(a) Explicación intuitiva desde cero:** La **multicolinealidad** ocurre cuando dos o más variables de entrada contienen casi la misma información (una puede predecirse linealmente a partir de las otras). Al suceder esto, el modelo no sabe cuánto mérito asignar a cada variable, y la varianza de los coeficientes (o los gradientes de una red neuronal) se vuelve inestable. El **$\text{VIF}$** mide cuánto se infla la varianza del predictor $j$ por esa redundancia: $\text{VIF} < 5$ es óptimo y $\text{VIF} > 10$ es problemático. Siempre debe calcularse **incluyendo la constante (intercepto)** en la matriz de diseño (`sm.add_constant`), de lo contrario el $R_j^2$ sale artificialmente distorsionado.
* **(b) Fórmula matemática:**
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
  donde $R_j^2$ es el coeficiente de determinación al regresar el predictor $X_j$ sobre los demás predictores **con intercepto**.
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 1 Sec. 7):**
  * **Espacio seleccionado $[\text{Cancha\_CuT}, \text{Cancha\_CuS}]$:** Como $r(\text{CuT}, \text{CuS}) = 0.6677$, se obtiene $\text{VIF} = 1 / (1 - 0.6677^2) = \mathbf{1.8045} < 5.0$ (óptimo, baja colinealidad).
  * **Espacio rechazado $[\text{CuT}, \text{CuS}, \text{CuIn}]$:** Por la identidad química $\text{CuT} = \text{CuS} + \text{CuIn}$, el VIF explota a **$4{,}754.0$** (`CuT`), **$83.2$** (`CuS`) y **$4{,}004.1$** (`CuIn`). Incluso usar solo $[\text{CuT}, \text{CuIn}]$ daría $\text{VIF} = 86.85$ ($r = 0.994$). Esto sella definitivamente el espacio de entrada en 2D: $[\text{Cancha\_CuT}, \text{Cancha\_CuS}]$.

### 2.21. Homocedasticidad vs. Heterocedasticidad
* **(a) Explicación intuitiva desde cero:**
  * **Homocedasticidad:** El tamaño (dispersión) de los errores del modelo se mantiene parejo tanto en lotes de baja ley como en lotes de alta ley. El modelo es igualmente confiable en todo su rango operativo.
  * **Heterocedasticidad:** La varianza de los errores cambia según el nivel predicho (típicamente forma un "abanico" donde los errores explotan en leyes altas), invalidando los intervalos de confianza clásicos y evidenciando que la forma funcional del modelo no se adapta bien a los extremos.
* **(b) Fórmula matemática:**
  $$\text{Homocedasticidad: }\text{Var}(e_k \mid \hat{y}_k) = \sigma^2\;\text{(constante)} \qquad \text{vs.} \qquad \text{Heterocedasticidad: }\text{Var}(e_k \mid \hat{y}_k) = \sigma_k^2(\hat{y}_k)$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 4 Sec. 3):** Es uno de los grandes hallazgos diferenciadores de tu tesis: **todos los modelos lineales 2D resultaron heterocedásticos** porque sus errores se disparan en los meses de alta ley (`2023-10` y `2023-11`), mientras que los dos pilares de tu Estrategia Dual (**`Baseline 1B` / `Res_MLP_2_2`** y **`RNA_MLP_2`**) preservan **homocedasticidad** verificada formalmente.

### 2.22. Prueba de Breusch–Pagan (Exacta sobre $\hat{y}$)
* **(a) Explicación intuitiva desde cero:** Es el test econométrico estándar para detectar heterocedasticidad. Toma los residuos al cuadrado ($e_k^2$) y los regresa linealmente contra las predicciones $\hat{y}_k$ (más intercepto). Si $e_k^2$ crece significativamente con $\hat{y}_k$ ($p < 0.05$), se rechaza la hipótesis nula $H_0$ de homocedasticidad.
* **(b) Fórmula matemática (estadístico Multiplicador de Lagrange):**
  $$e_k^2 = \gamma_0 + \gamma_1 \hat{y}_k + u_k, \qquad \text{LM} = N \cdot R^2_{\text{aux}} \sim \chi^2_1$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (`het_breuschpagan(residuos, sm.add_constant(y_pred))`):**
  * **Homocedásticos ($p \ge 0.05$):** `Baseline_1B` ($p = 0.4075$), `Res_MLP_2_2` ($p = 0.4075$), `Res_MLP_2` ($p = 0.2140$) y **`RNA_MLP_2`** (**$p = 0.0822$**).
  * **Heterocedásticos ($p < 0.05$):** `OLS` (**$p = 0.0016$**), `LassoCV` ($p = 0.0029$), `ElasticNetCV` ($p = 0.0035$), `RNA_Delgado_Keras` (**$p = 0.0040$**), `RidgeCV` ($p = 0.0228$).
  * **Matiz de defensa para `SVR_RBF` ($p = 0.7243$):** Aunque marca homocedasticidad lineal, es el peor modelo entrenado ($\text{NRMSE} = 16.68\%$) porque sus errores gigantes aparecen en *ambos* extremos de $\hat{y}$ (`2023-09` abajo y `2023-11` arriba), formando una "U" cuya pendiente lineal auxiliar es casi cero.

### 2.23. Estadístico de Durbin–Watson ($d$)
* **(a) Explicación intuitiva desde cero:** Prueba clásica que detecta si los residuos consecutivos de un modelo ($e_t$ y $e_{t-1}$) están correlacionados entre sí. Su escala va de $0$ a $4$: valores cercanos a **$2.0$** indican independencia perfecta; valores cercanos a **$0$** indican fuerte autocorrelación positiva (o un sesgo sistemático del mismo signo todos los meses); valores cercanos a **$4$** indican alternancia negativa.
* **(b) Fórmula matemática:**
  $$d = \frac{\sum_{t=2}^{N} (e_t - e_{t-1})^2}{\sum_{t=1}^{N} e_t^2} \approx 2(1 - r_1)$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 4 Tabla 3.1):**
  * `Cancha_Directa`: **$d = 0.285$** (error sistemático unidireccional grave en los 24 meses).
  * `Baseline_1B` y `Res_MLP_2_2`: **$d = 2.014$** (independencia prácticamente perfecta).
  * `RNA_MLP_2`: **$d = 1.958$** (independencia prácticamente perfecta). Todos los modelos calibrados se ubican en la zona limpia $d \in [1.94, 2.38]$.

### 2.24. Test Pareado Exacto de Wilcoxon (`method='exact'`)
* **(a) Explicación intuitiva desde cero:** Es la alternativa no paramétrica de la prueba $t$ pareada cuando la muestra es pequeña ($N = 24$) o los errores cuadráticos no son normales. Compara mes a mes los errores cuadráticos de dos modelos ($e_{A,k}^2$ vs. $e_{B,k}^2$), ordena la magnitud de las diferencias no nulas por rango y verifica si un modelo vence al otro de forma sistemática y no por azar. Usar `method='exact'` calcula la probabilidad exacta combinatoria en vez de una aproximación normal asintótica.
* **(b) Fórmula matemática:**
  $$D_k = e_{A,k}^2 - e_{B,k}^2, \qquad W^+ = \sum_{D_k > 0} \text{Rango}(|D_k|), \qquad W^- = \sum_{D_k < 0} \text{Rango}(|D_k|)$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 4 Tabla 3.1):**
  * **Contra `Cancha_Directa` (unilateral, $H_1: e_{\text{Cancha}}^2 > e_{\text{Modelo}}^2$):** `Baseline_1B` y `Res_MLP_2_2` logran **$p = 0.000010$** ($W = 284.0$); **`RNA_MLP_2`** logra **$p = 0.000186$** ($W = 267.0$). Ambos confirman superioridad estadística con $> 99.98\%$ de confianza.
  * **Contra `Baseline_1B` (bilateral, $H_1: e_{\text{B1B}}^2 \neq e_{\text{Modelo}}^2$):** **`RNA_MLP_2`** obtiene **$p = 0.988813$** (sin diferencia estadísticamente significativa en mediana cuadrática) **venciendo a `Baseline_1B` en $14/24$ meses ($58.3\%$)**; `Res_MLP_2` obtiene $p = 0.250000$ con **$0/24$ ($21$ empates)** (perdiendo levemente en solo 3 meses: $2 \times 0.5^3 = 0.25$, Auditoría `B4-03`).

### 2.25. Bootstrap No Paramétrico ($B = 10{,}000$)
* **(a) Explicación intuitiva desde cero:** Técnica de remuestreo computacional creada por Bradley Efron que permite estimar intervalos de confianza sin asumir ninguna fórmula paramétrica ni normalidad. Consiste en "clonar" el experimento $B = 10{,}000$ veces sorteando $N = 24$ meses **con reemplazo** a partir de las predicciones OOS reales, recalcular el RMSE en cada réplica y tomar los percentiles $2.5\%$ y $97.5\%$ como intervalo de confianza al $95\%$.
* **(b) Fórmula matemática (método de percentiles):**
  $$\text{RMSE}^{*(b)} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(y_{k_i^{(b)}} - \hat{y}_{k_i^{(b)}}\right)^2}, \quad b = 1, \dots, 10{,}000, \qquad \text{IC}_{95\%} = \left[Q_{0.025}(\text{RMSE}^*),\; Q_{0.975}(\text{RMSE}^*)\right]$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo ($B = 10{,}000$, semilla `42`):**
  * `Cancha_Directa`: **$[0.3153,\; 0.4169]\,\%\text{Cu}$**.
  * `Baseline_1B`: **$[0.1061,\; 0.1785]\,\%\text{Cu}$** | `Res_MLP_2_2`: **$[0.1065,\; 0.1786]\,\%\text{Cu}$**.
  * `RNA_MLP_2`: **$[0.1154,\; 0.2384]\,\%\text{Cu}$**.
  * **Lectura defensiva:** El intervalo de `Cancha_Directa` **jamás se solapa** con los intervalos de `Baseline_1B`, `Res_MLP_2_2` ni `RNA_MLP_2`, otorgando una demostración gráfica e inferencial contundente de la reducción de incertidumbre.

---

## Sección 3: Protocolo Experimental y Validación en Muestras Pequeñas

### 3.1. Fuga de Datos (*Data Leakage*) y Principio de Cero Fuga (*Zero Data Leakage*)
* **(a) Explicación intuitiva desde cero:** Ocurre cuando información del conjunto de prueba (datos "futuros" que el modelo supuestamente no conoce) se filtra de manera directa o indirecta durante el entrenamiento o el preprocesamiento. Ejemplos clásicos de fuga silenciosa: calcular la media y desviación estándar de `StandardScaler` sobre los $24$ meses completos antes de dividir en entrenamiento/prueba, restar el sesgo promedio histórico de los $24$ meses, o elegir hiperparámetros mirando los resultados de prueba. La fuga infla artificialmente las métricas y produce modelos que fracasan en faena real.
* **(b) Fórmula matemática (separación estricta en cada pliegue $k$):**
  $$\boldsymbol{\theta}_{\text{pre}}^{(-k)} = \mathcal{T}\!\left(\mathbf{X}_{\text{train}}^{(-k)}\right), \qquad \mathbf{X}_{\text{test, sc}}^{(k)} = \tau\!\left(\mathbf{X}_{\text{test}}^{(k)};\; \boldsymbol{\theta}_{\text{pre}}^{(-k)}\right), \qquad \mathcal{D}_{\text{train}}^{(-k)} \cap \mathcal{D}_{\text{test}}^{(k)} = \varnothing$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Toda la Metodología v4.0 (Bloques 2, 3, 4 y `Código unificado.ipynb`) garantiza **cero fuga**: en cada uno de los $24$ pliegues LOOCV, tanto el escalador (`StandardScaler`/`RobustScaler`), el escalador de $y$ (`sc_y`), el sesgo medio de `Baseline 1B` ($\bar{e}_{\text{train},-k} \in [0.3287, 0.3526]\%\text{Cu}$) como las grillas internas de `RidgeCV`/`LassoCV`/`ElasticNetCV`/`SVR` se ajustan **exclusivamente sobre los $23$ meses de entrenamiento**. Esta disciplina explica por qué el error real LOOCV de `RNA_Delgado_Keras` sube de $7.38\%$ (*in-sample* con memorización total) a **$11.64\%$** fuera de muestra.

### 3.2. Tubería de Procesamiento (`Pipeline` de `scikit-learn`)
* **(a) Explicación intuitiva desde cero:** Es una "cápsula hermética" de ingeniería de software que encadena secuencialmente los pasos de preprocesamiento (ej. escalado) y el algoritmo predictivo como si fueran un solo objeto indivisible. Al entrenar el `Pipeline` dentro de un bucle de validación cruzada, garantiza automáticamente que el escalador aprenda únicamente con `.fit()` en el entrenamiento y aplique `.transform()` sobre la prueba sin contaminar jamás los datos ciegos.
* **(b) Fórmula matemática (composición funcional encapsulada):**
  $$\hat{y}_k = \big(f_{\boldsymbol{\theta}^{(-k)}} \circ \tau_{\boldsymbol{\mu}^{(-k)}, \boldsymbol{\sigma}^{(-k)}}\big)(\mathbf{x}_k)$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** En la Celda 7 de `Código unificado.ipynb`, los cinco *baselines* paramétricos/kernel (`OLS`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, `SVR_RBF`) se construyen explícitamente mediante `Pipeline([('scaler', scaler_cls()), ('model', ...)])`, blindando la validación cruzada anidada (`GridSearchCV` y `cv=5`) contra cualquier fuga de escala.

### 3.3. Escalado Z-Score (`StandardScaler`) vs. Escalado Robusto (`RobustScaler`)
* **(a) Explicación intuitiva desde cero:**
  * **`StandardScaler` (Z-score):** Resta la media y divide por la desviación estándar de cada predictor, dejando ambas entradas centradas en $0$ con varianza $1$. Es fundamental para que variables con rangos grandes no aplasten los gradientes ni distorsionen penalizaciones $L_1/L_2$ o distancias de kernels.
  * **`RobustScaler`:** En lugar de media y desviación estándar (sensibles a *outliers*), resta la **mediana** y divide por el **rango intercuartílico ($\text{IQR} = Q_3 - Q_1$)**, evitando que un mes extremo comprima en exceso el cuerpo central de los datos.
* **(b) Fórmula matemática:**
  $$Z_{i,j}^{\text{Std}} = \frac{X_{i,j} - \mu_{j,-k}}{\sigma_{j,-k}}, \qquad Z_{i,j}^{\text{Rob}} = \frac{X_{i,j} - \tilde{x}_{j,-k}}{Q_{3,j,-k} - Q_{1,j,-k}}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Como `Cancha_CuT` tiene **$61.64\times$** más varianza que `Cancha_CuS` ($61.62\times$ exacto), escalar es obligatorio. Además, ante la curtosis extrema de `CuS` ($+5.8621$), el Bloque 2 contrastó ambos escaladores sin fuga: `OLS` es invariante ($11.85\%$ vs. $11.85\%$), `RidgeCV` varía solo $-0.09\%$ ($11.78\%$ vs. $11.69\%$), `LassoCV` $+0.05\%$ ($12.30\%$ vs. $12.35\%$) y `ElasticNetCV` $+0.06\%$ ($12.36\%$ vs. $12.42\%$). Al ser $|\Delta\text{NRMSE}| \le 0.09\%$, se certificó `StandardScaler` como estándar seguro para el Bloque 3.

### 3.4. Validación Cruzada Dejando Uno Fuera (*Leave-One-Out Cross-Validation*, LOOCV)
* **(a) Explicación intuitiva desde cero:** Es el protocolo de validación cruzada más exigente y eficiente en información cuando dispones de una muestra pequeña ($N = 24$ balances mensuales). Funciona así: se realizan $24$ experimentos independientes; en el experimento $k$, se esconde el mes $k$ por completo, se entrena el modelo desde cero con los otros $23$ meses y se predice a ciegas el mes escondido. Al terminar, se reúnen las $24$ predicciones ciegas y se miden las métricas de desempeño real. A diferencia de un corte único `train_test_split` 80/20 (donde el resultado cambia dramáticamente según qué $5$ meses cayeron por azar en prueba), LOOCV es determinista, aprovecha el $95.8\%$ de los datos en cada ajuste y evalúa el $100\%$ de la historia operacional fuera de muestra.
* **(b) Fórmula matemática:**
  $$\mathcal{D}_{\text{train}}^{(-k)} = \{(\mathbf{x}_i, y_i) \mid i \neq k\}, \quad \mathcal{D}_{\text{test}}^{(k)} = \{(\mathbf{x}_k, y_k)\}, \qquad \text{MSE}_{\text{LOOCV}} = \frac{1}{N} \sum_{k=1}^{N} \left(y_k - \hat{y}_k^{(-k)}\right)^2$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Habilitado formalmente en el Bloque 1 porque `Sesgo_Canje` es ruido blanco ($r_1 = -0.0866 \in [-0.4001, +0.4001]$). Sobre los $N = 24$ pliegues LOOCV ($N_{\text{train}} = 23$) se ejecuta el ranking maestro de los **18 modelos** del Bloque 4: empate `1=` en **$10.07\%$** (`Res_MLP_2_2`, `Baseline_1B`, `Res_MLP_4`, `Res_MLP_3`), `RNA_Delgado_Keras` en **$11.64\%$**, `RNA_MLP_2` en **$12.28\%$** y `Cancha_Directa` en **$25.67\%$**.

### 3.5. Validación Cruzada Anidada (*Nested Cross-Validation*)
* **(a) Explicación intuitiva desde cero:** Consiste en poner "un bucle de validación cruzada dentro de otro". El **bucle exterior** (en tu tesis, LOOCV de $24$ pliegues) sirve únicamente para medir el error honesto fuera de muestra. El **bucle interior** (`cv=5` ejecutado solo sobre los $23$ meses de entrenamiento) sirve exclusivamente para escoger los hiperparámetros óptimos ($\alpha$, $l_1\text{-ratio}$, $C$, $\epsilon$, $\gamma$) sin mirar nunca el mes de prueba $k$.
* **(b) Fórmula matemática (doble nivel de partición):**
  $$\lambda_k^* = \arg\min_{\lambda \in \Lambda} \frac{1}{M_{\text{int}}} \sum_{m=1}^{M_{\text{int}}} \mathcal{L}\!\left(\mathcal{D}_{\text{val}}^{(k,m)};\; \hat{f}_{\lambda}\big(\mathcal{D}_{\text{train}}^{(-k) \setminus m}\big)\right), \qquad \hat{y}_k = \hat{f}_{\lambda_k^*}\!\left(\mathbf{x}_k\right)$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Implementada en `RidgeCV`, `LassoCV`, `ElasticNetCV` y `GridSearchCV(SVR)` con `cv=5` (`CV_INTERNO = min(5, max(2, N - 2))`) dentro de cada pliegue exterior LOOCV. En las redes neuronales del Bloque 3, para evitar la inestabilidad de doble búsqueda en $N = 23$, se evaluó una grilla predefinida *a priori* de activaciones $\{\text{relu}, \text{tanh}\}$ y $\alpha \in \{0.001, 1.0, 2.0, 5.0\}$, auditada adicionalmente contra `Baseline 1B` y en el *Holdout* 19/5 (Auditoría `B3-04` / `A-02`).

### 3.6. Validación Prospectiva Cronológica (*Holdout* 19/5)
* **(a) Explicación intuitiva desde cero:** Simula el despliegue real en producción mirando estrictamente hacia el futuro: el modelo se entrena únicamente con el pasado histórico (los primeros $19$ meses, $\approx 80\%$) y se congela para predecir a ciegas los últimos $5$ meses futuros ($\approx 20\%$) sin volver a reentrenar. Mientras LOOCV mide el desempeño promedio global incluyendo eventos anómalos de cualquier fecha, el *Holdout* 19/5 mide cómo responde el sistema en una campaña futura regular.
* **(b) Fórmula matemática:**
  $$\mathcal{D}_{\text{train}} = \{1, \dots, 19\}\;(\text{Ene-2023 a Jul-2024}), \qquad \mathcal{D}_{\text{test}} = \{20, \dots, 24\}\;(\text{Ago-2024 a Dic-2024})$$
  $$\text{NRMSE}_{\text{Holdout}} = \frac{\sqrt{\frac{1}{5}\sum_{t=20}^{24} (y_t - \hat{y}_t)^2}}{\frac{1}{5}\sum_{t=20}^{24} y_t} \times 100\%$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 3 Sec. 6.1):** Sobre la ventana prospectiva simétrica ($n = 5$, leyes `Canje_CuT` entre $1.06\%$ y $1.48\%$):
  * **`RNA_MLP_2` (1° lugar):** $\text{MSE} = 0.007417\,(\%\text{Cu})^2$, $\text{RMSE} = 0.0861\%\text{Cu}$, **$\text{NRMSE} = 6.56\%$**, **$R^2 = +0.6549$** (perfora holgadamente la referencia de $8.63\%$).
  * **`RidgeCV`:** $7.57\%$ ($R^2 = +0.5406$) | **`OLS`:** $7.76\%$ ($R^2 = +0.5180$).
  * **`Baseline 1B` y `Res_MLP` (empate):** $8.96\%$ ($R^2 = +0.3578$).
  * **`Cancha_Directa`:** $27.70\%$ ($R^2 = -5.1463$).
  * Por rigor científico (alta varianza muestral con $n = 5$), este resultado prospectivo complementa —pero no reemplaza— al ranking global LOOCV ($N = 24$). En `Código unificado.ipynb`, el corte se generaliza dinámicamente como $80\%/20\%$ (`round(N * 0.20)`).

### 3.7. Validación Cruzada con Zona de Amortiguamiento (*Buffered-LOOCV*)
* **(a) Explicación intuitiva desde cero:** Técnica originada en geoestadística y ecología espacial para eliminar cualquier sospecha de fuga por continuidad espacial o temporal. Cuando se evalúa el mes $k$, no solo se saca el mes $k$ del entrenamiento, sino también se retiran sus vecinos inmediatos anterior ($k-1$) y posterior ($k+1$), creando una "zona tampón" (*buffer zone*) de aislamiento.
* **(b) Fórmula matemática:**
  $$\mathcal{D}_{\text{train}}^{\text{buf}(-k)} = \{(\mathbf{x}_i, y_i) \mid i \notin \{k-1, k, k+1\}\}, \qquad N_{\text{train}}^{\text{buf}} \in \{21, 22\}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloques 2 y 3):** Ejecutado como blindaje ante la autocorrelación lag-1 de `Cancha_CuT` ($r_1 = +0.4467$):
  * **`Baseline 1B` y `Res_MLP_2_2/3/4`:** pasan de $10.07\%$ a **$10.04\%$** ($\Delta = -0.03\%$, $R^2 = +0.6524$, invarianza perfecta).
  * **`LassoCV`:** $12.30\% \to 12.34\%$ ($\Delta = +0.04\%$, cero fuga temporal).
  * **`OLS`:** mejora de $11.85\%$ a $11.07\%$ (porque al evaluar `2023-10` se retira el apalancamiento de `2023-11` y viceversa).
  * **`SVR_RBF`:** colapsa de $16.68\%$ a **$20.97\%$** ($R^2 = -0.5152$) por pérdida local de soporte.
  * **`RNA_MLP_2`:** sube de $12.28\%$ a **$14.53\%$** ($\Delta = +2.25\%$, coincidencia exacta entre `BLOQUE_3` y `Código unificado.ipynb` tras alinear el layout C-contiguous en *Buffered-LOOCV*) por un hecho geoestadístico objetivo: al evaluar `2023-11` ($2.27\%$) se remueve automáticamente su único vecino rico `2023-10` ($1.79\%$), truncando el entrenamiento en un máximo de $1.57\%\text{Cu}$ y forzando extrapolación pura.

### 3.8. Compromiso Sesgo–Varianza (*Bias–Variance Trade-off*)
* **(a) Explicación intuitiva desde cero:** Es la ley fundamental del aprendizaje automático que describe la tensión entre dos fuentes de error en predicciones futuras:
  * **Sesgo de modelo (*Bias*):** Error por usar un modelo demasiado simple/rígido que no puede capturar la curvatura real del fenómeno.
  * **Varianza de estimación (*Variance*):** Error por usar un modelo demasiado flexible/complejo que cambia bruscamente cada vez que varía un dato de entrenamiento, memorizando el ruido.
  En muestras pequeñas ($N = 24$), reducir un poco el sesgo agregando muchas neuronas dispara exponencialmente la varianza fuera de muestra.
* **(b) Fórmula matemática (error cuadrático esperado en un punto $\mathbf{x}_0$):**
  $$\mathbb{E}\!\left[\big(y_0 - \hat{f}(\mathbf{x}_0)\big)^2\right] = \underbrace{\Big(\mathbb{E}[\hat{f}(\mathbf{x}_0)] - f(\mathbf{x}_0)\Big)^2}_{\text{Sesgo}^2} + \underbrace{\text{Var}\!\left(\hat{f}(\mathbf{x}_0)\right)}_{\text{Varianza}} + \underbrace{\sigma_\varepsilon^2}_{\text{Ruido Irreducible}}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Tu tesis ilustra este compromiso de libro de texto: al subir complejidad desde `Baseline 1B` ($P_{\text{eff}}=1$, $\text{NRMSE}=10.07\%$) hacia redes directas más grandes como `RNA_MLP_2` ($P=9$, $12.28\%$), `RNA_Delgado_MLP` ($P=37$, $12.61\%$) y `RNA_MLP_3_2` ($P=20$, $13.48\%$), la varianza de estimación en $N_{\text{train}}=23$ supera la ganancia en sesgo. Incluso en `Residual-MLP`, bajar regularización de $\alpha=2.0$ ($10.07\%$) a $\alpha=0.5$ ($10.73\%$) y $\alpha=0.1$ ($11.94\%$) aumenta el error OOS por explosión de varianza.

### 3.9. Sobreajuste (*Overfitting*) vs. Subajuste (*Underfitting*)
* **(a) Explicación intuitiva desde cero:**
  * **Subajuste (*Underfitting*):** El modelo es tan simple que ni siquiera logra describir los datos con los que entrenó (alto error tanto *in-sample* como *out-of-sample*), como `Cancha_Directa` ($25.67\%$).
  * **Sobreajuste (*Overfitting*):** El modelo se "aprende de memoria" los datos de entrenamiento (error *in-sample* muy bajo), pero falla al predecir meses nuevos (error *out-of-sample* alto) porque confundió fluctuaciones aleatorias con leyes físicas.
* **(b) Fórmula matemática (brecha de generalización):**
  $$\Delta_{\text{gen}} = \text{RMSE}_{\text{OOS (LOOCV)}} - \text{RMSE}_{\text{in-sample}} \gg 0 \implies \text{Indicio de sobreajuste}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * En `RNA_Delgado_Keras`: *in-sample* sobre `Canje_CuT` logra $\text{MAPE}=7.38\%$ ($\text{NRMSE}=9.19\%$), pero en LOOCV sube a $\text{NRMSE}=11.64\%$ ($\text{MAPE}=8.74\%$).
  * En **`RNA_MLP_2`** (Auditoría `C-05`, transparencia *in-sample*): sobre los $N=24$ completos alcanza $\text{RMSE}_{\text{in-sample}} = 0.0966\%\text{Cu}$ (**$\text{NRMSE} = 6.75\%$**) frente a $\text{RMSE}_{\text{LOOCV}} = 0.1758\%\text{Cu}$ (**$\text{NRMSE} = 12.28\%$**), brecha concentrada en el pulso único de Nov-2023 y controlada mediante parsimonia ($P=9$) y regularización.

### 3.10. Principio de Occam / Parsimonia
* **(a) Explicación intuitiva desde cero:** Principio científico clásico (*Navaja de Occam*): entre dos modelos con capacidad explicativa comparable, **siempre se debe preferir el más simple y con menos parámetros libres**. En minería con bases mensuales cortas, la parsimonia no es estética: es el mecanismo principal para que el modelo sobreviva al cambiar de frente de explotación.
* **(b) Fórmula matemática (criterio conceptual de complejidad penalizada tipo AIC/BIC):**
  $$\text{Costo Total} = \text{Error de Ajuste} + \lambda \cdot P_{\text{eff}}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Queda demostrado empíricamente en las dos familias de la tesis: (1) en presupuesto global, **`Baseline 1B` / `Res_MLP_2_2`** ($P_{\text{eff}} = 1$) lidera con $10.07\%$; (2) entre las redes neuronales directas no lineales, la topología más pequeña posible —**`RNA_MLP_2`** con solo $2$ neuronas ocultas y $P = 9$ parámetros— supera consistentemente a las redes profundas `(2, 2)` ($13.00\%$), `(3, 2)` ($13.48\%$) y `(4, 4)` ($12.61\%$).

### 3.11. Ratio Muestras por Parámetro ($N_{\text{train}} / P$)
* **(a) Explicación intuitiva desde cero:** Mide cuántas observaciones de entrenamiento respaldan cada peso o parámetro que el modelo intenta aprender. Si $N_{\text{train}} / P < 1$, hay más incógnitas que ecuaciones (sistema subdeterminado o sobreparametrizado), abriendo la puerta al sobreajuste inmediato si no existe regularización fuerte.
* **(b) Fórmula matemática:**
  $$\text{Ratio}_{\text{parsimonia}} = \frac{N_{\text{train}}}{P_{\text{eff}}}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 3 Tabla 2.4, $N_{\text{train}} = 23$):**
  * **`RNA_MLP_2`** y **`Res_MLP_2`** ($P = 9$): **$23 / 9 = 2.56$** (holgura parsimoniosa segura).
  * **`RNA_MLP_3`** ($P = 13$): $1.77$ | **`RNA_MLP_2_2`** ($P = 15$): $1.53$ | **`RNA_MLP_4`** ($P = 17$): $1.35$ | **`RNA_MLP_3_2`** ($P = 20$): $1.15$ (frontera).
  * **`RNA_Delgado_MLP`** `(4, 4)` ($P = 37$): **$23 / 37 = 0.62 < 1$** (sobreparametrizado, depende de $\alpha = 2.0$).
  * **`RNA_Delgado_Keras`** `(4, 4, 4)`: $P = 57$ nominales, pero $P_{\text{eff}} = 3 \implies 23 / 3 = \mathbf{7.67}$.
  * **`Res_MLP_2_2/3/4`** bajo $\alpha = 2.0$: $P = 15, 13, 17$ nominales, pero $P_{\text{eff}} \approx 1 \implies 23 / 1 = \mathbf{23.0}$.

---

## Sección 4: Modelos Predictivos, Hiperparámetros, Optimizadores y Funciones de Activación

### 4.1. Baseline 1A: Cancha Directa (Sin Intervención)
* **(a) Explicación intuitiva desde cero:** No es un algoritmo entrenado, sino la práctica operativa tradicional de terreno: asumir ingenuamente que la ley medida en la cancha de acopio será exactamente la ley que pagará ENAMI en canje. Sirve como "punto cero" para cuantificar cuánto dinero y precisión se pierde sin ciencia de datos.
* **(b) Fórmula matemática:**
  $$\hat{y}_k^{\text{1A}} = \text{Cancha\_CuT}_k$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Ranking `18` de 18):** $\text{MSE} = 0.134971\,(\%\text{Cu})^2$, $\text{RMSE} = 0.3674\%\text{Cu}$, $\text{MAE} = 0.3404\%\text{Cu}$, **$\text{NRMSE} = 25.67\%$**, **$R^2 = -1.2709$**, $\text{MAPE} = 24.90\%$, Pearson $r = +0.8347$, Sesgo $= +0.3404\%\text{Cu}$ (sobreestima $24/24$ meses), Durbin–Watson $d = 0.285$. En el *Holdout* 19/5 alcanza $\text{NRMSE} = 27.70\%$ y $R^2 = -5.1463$.

### 4.2. Baseline 1B: Corrección de Sesgo Constante en LOOCV
* **(a) Explicación intuitiva desde cero:** Es el modelo de calibración más simple, transparente y honesto que un ingeniero puede construir: calcula cuánto ha sobreestimado en promedio la cancha durante los $23$ meses de entrenamiento y le resta exactamente esa constante fija a la ley de cancha del mes nuevo, manteniendo intacta una relación física $1:1$ (pendiente unitaria).
* **(b) Fórmula matemática:**
  $$\bar{e}_{\text{train}, -k} = \frac{1}{N - 1} \sum_{i \neq k} \left(\text{Cancha\_CuT}_i - \text{Canje\_CuT}_i\right), \qquad \hat{y}_k^{\text{1B}} = \text{Cancha\_CuT}_k - \bar{e}_{\text{train}, -k}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Empate técnico `1=` de 18 — Pilar Presupuestario):** Con $\bar{e}_{\text{train},-k} \in [0.3287, 0.3526]\%\text{Cu}$ (constante en producción sobre $N=24$: $+0.340417\%\text{Cu}$), logra $\text{MSE} = 0.020782\,(\%\text{Cu})^2$, $\text{RMSE} = 0.1442\%\text{Cu}$, $\text{MAE} = 0.1178\%\text{Cu}$, **$\text{NRMSE} = 10.07\%$**, **$R^2 = +0.6503$**, $\text{MAPE} = 8.12\%$, $r = +0.8209$, Breusch–Pagan $p = 0.4075$, Durbin–Watson $d = 2.014$, Shapiro $p = 0.9696$, Wilcoxon vs. Cancha $p = 0.000010$. En *Holdout* 19/5 marca $8.96\%$ y en *Buffered-LOOCV* $10.04\%$.

### 4.3. Regresión Lineal Múltiple (`OLS` / MCO — Mínimos Cuadrados Ordinarios)
* **(a) Explicación intuitiva desde cero:** Ajusta el plano recto único que minimiza la suma de los cuadrados de las distancias verticales entre las leyes reales de canje y el plano definido por `Cancha_CuT` y `Cancha_CuS`. No aplica ninguna penalización sobre el tamaño de los coeficientes.
* **(b) Fórmula matemática:**
  $$\hat{y}_k^{\text{OLS}} = \hat{\beta}_0 + \hat{\beta}_1 Z_{1,k} + \hat{\beta}_2 Z_{2,k}, \qquad \hat{\boldsymbol{\beta}} = (\mathbf{Z}^T \mathbf{Z})^{-1}\mathbf{Z}^T \mathbf{y}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Ranking `8` de 18):** $\text{MSE} = 0.028789\,(\%\text{Cu})^2$, $\text{RMSE} = 0.1697\%\text{Cu}$, $\text{MAE} = 0.1318\%\text{Cu}$, **$\text{NRMSE} = 11.85\%$**, **$R^2 = +0.5156$**, Sesgo $= +0.0038\%\text{Cu}$. Coeficientes estandarizados medios: $\bar{\beta}_{\text{CuT}} = +0.1780$, $\bar{\beta}_{\text{CuS}} = +0.0372$. **Por qué pierde frente a `Baseline 1B`:** estima una pendiente achatada en escala física ($\approx 0.74 < 1.0$) y sufre **heterocedasticidad severa** (Breusch–Pagan $p = 0.0016$), concentrando el $40.5\%$ de su error cuadrático en `2023-10` ($+0.3284\%$) y `2023-11` ($-0.4145\%$).

### 4.4. Regularización $L_1$ (`Lasso`) vs. $L_2$ (`Ridge`) vs. `ElasticNet`
* **(a) Explicación intuitiva desde cero:** Son extensiones de `OLS` que añaden una "multa" al tamaño de los coeficientes $\beta_j$ para frenar el sobreajuste y amortiguar el apalancamiento de datos extremos:
  * **Ridge ($L_2$):** Penaliza la suma de los **cuadrados** de los coeficientes. Encoge todos los pesos suavemente hacia cero sin anularlos exactamente; ideal cuando hay correlación moderada entre predictores.
  * **Lasso ($L_1$):** Penaliza la suma de los **valores absolutos**. Por su geometría en rombo (vértices en los ejes), es capaz de llevar exactamente a **$0.0000$** los coeficientes de variables irrelevantes, actuando como selector automático de variables.
  * **ElasticNet ($L_1 + L_2$):** Combina ambas penalizaciones mediante una proporción mezcladora $\rho$ (`l1_ratio`).
* **(b) Fórmula matemática:**
  $$\hat{\boldsymbol{\beta}}^{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2\right\}$$
  $$\hat{\boldsymbol{\beta}}^{\text{Lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |\beta_j|\right\}$$
  $$\hat{\boldsymbol{\beta}}^{\text{EN}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \alpha\left[\rho\sum_{j=1}^{p}|\beta_j| + \frac{1-\rho}{2}\sum_{j=1}^{p}\beta_j^2\right]\right\}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 2 y Bloque 4):**
  * **`RidgeCV`** (Ranking `7`): $\alpha \in [3.2745, 17.4753]$ (media $11.07$), $\text{NRMSE} = 11.78\%$, $R^2 = +0.5221$, $\bar{\beta}_{\text{CuT}} = +0.1146$, $\bar{\beta}_{\text{CuS}} = +0.0537$; falla normalidad ($p = 0.0316$) y homocedasticidad ($p = 0.0228$).
  * **`LassoCV`** (Ranking `10`): $\alpha \in [0.0001, 0.0351]$ (media $0.0185$), $\text{NRMSE} = 12.30\%$, $R^2 = +0.4789$, Breusch–Pagan $p = 0.0029$. **Hito experimental clave:** retuvo $\beta_{\text{CuS}} > 0$ en **$24/24$ pliegues ($100.0\%$)** con rango $[+0.0124, +0.0627]$ (media $+0.0261$), confirmando señal marginal positiva persistente de `Cancha_CuS`.
  * **`ElasticNetCV`** (Ranking `11`): `l1_ratio` alterna entre $0.10$ y $0.99$, $\text{NRMSE} = 12.36\%$, $R^2 = +0.4735$.

### 4.5. Máquina de Vectores de Soporte para Regresión (`SVR` con Kernel RBF) e Hiperparámetros ($C, \epsilon, \gamma$)
* **(a) Explicación intuitiva desde cero:** En lugar de minimizar todos los errores cuadráticos, `SVR` construye un "tubo de tolerancia" de ancho $\pm \epsilon$ alrededor de una función curva flexible y solo penaliza los puntos que se salen del tubo (los **vectores de soporte**). El **Kernel RBF (Base Radial Gaussiana)** proyecta los datos a un espacio de dimensión infinita midiendo similitud local en forma de campanas gaussianas centradas en los puntos de entrenamiento.
  * **$C$ (Costo de regularización):** Controla cuán fuerte se castigan los puntos fuera del tubo; $C$ alto ajusta agresivamente (riesgo de sobreajuste), $C$ bajo suaviza más.
  * **$\epsilon$ (Tubo $\epsilon$-insensible):** Margen de error gratuito que el modelo ignora sin penalizar.
  * **$\gamma$ (Ancho del kernel gaussiano):** Define el radio de influencia de cada observación; $\gamma$ alto crea campanas muy estrechas y locales.
* **(b) Fórmula matemática:**
  $$K(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right), \qquad \min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}(\xi_i + \xi_i^*) \quad \text{s.a. } |y_i - f(\mathbf{x}_i)| \le \epsilon + \xi_i^{(*)}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Ranking `17` de 18):** Sintonizado por `GridSearchCV(cv=5)` (seleccionando preferentemente $C \in [1, 50]$, $\epsilon = 0.10$, $\gamma \in \{\text{'scale'}, 0.10\}$), obtiene $\text{MSE} = 0.056986\,(\%\text{Cu})^2$, $\text{RMSE} = 0.2387\%\text{Cu}$, **$\text{NRMSE} = 16.68\%$**, **$R^2 = +0.0412$**. Al ser un aproximador estrictamente local, decae hacia la media en regiones de baja densidad de datos: en `2023-11` (`Canje` $= 2.27\%$) predice solo $1.6132\%$, y en *Buffered-LOOCV* colapsa a **$20.97\%$** ($R^2 = -0.5152$). Esto demuestra por qué un MLP suave es muy superior a `SVR` en $N = 24$.

### 4.6. Perceptrón Multicapa (`MLP` — *Multilayer Perceptron*)
* **(a) Explicación intuitiva desde cero:** Es la arquitectura clásica de Red Neuronal Artificial hacia adelante (*feedforward*). Consiste en neuronas matemáticas organizadas en capas secuenciales: una **capa de entrada** que recibe los predictores (`Cancha_CuT`, `Cancha_CuS`), una o más **capas ocultas** donde cada neurona combina linealmente las entradas y aplica una función de activación no lineal, y una **capa de salida** que entrega la ley de canje estimada. Gracias al **Teorema de Aproximación Universal**, incluso una sola capa oculta con activación no lineal puede aproximar cualquier superficie continua curva en 2D.
* **(b) Fórmula matemática (MLP de 1 capa oculta, como `RNA_MLP_2`):**
  $$\hat{y} = \mathbf{w}_{\text{out}}^T \, \phi\!\left(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1\right) + b_{\text{out}}$$
  donde $\mathbf{z} \in \mathbb{R}^2$ es el vector estandarizado, $\mathbf{W}_1 \in \mathbb{R}^{h \times 2}$, $\mathbf{b}_1 \in \mathbb{R}^h$, $\mathbf{w}_{\text{out}} \in \mathbb{R}^h$, $b_{\text{out}} \in \mathbb{R}$ y $\phi(\cdot)$ es la función de activación.
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Se evaluaron $11$ arquitecturas neuronales: $6$ `Direct-MLP` en `scikit-learn` (`(2,)`, `(3,)`, `(4,)`, `(2, 2)`, `(3, 2)`, `(4, 4)`), $4$ `Residual-MLP` y $1$ réplica en Keras `(4, 4, 4)`. La red directa más parsimoniosa, **`RNA_MLP_2`** (`(2,)`, `tanh`, $\alpha = 0.001$), se consolidó como **Campeón Geometalúrgico / XAI** ($\text{NRMSE}_{\text{LOOCV}} = 12.28\%$, $\text{NRMSE}_{\text{Holdout}} = 6.56\%$).

### 4.7. Capas Ocultas (*Hidden Layers*) y Topología
* **(a) Explicación intuitiva desde cero:** Son las capas intermedias situadas entre las entradas y la salida que nunca se observan directamente en la tabla de datos. Cada neurona oculta aprende a detectar una combinación o "régimen" geometalúrgico (por ejemplo, frente sulfurado rico vs. frente oxidado con alto `CuS`). Agregar más capas (profundidad) o más neuronas por capa (anchura) aumenta la flexibilidad, pero multiplica rápidamente los parámetros a estimar.
* **(b) Fórmula matemática (propagación capa por capa para $L$ capas ocultas):**
  $$\mathbf{a}^{(0)} = \mathbf{z}, \qquad \mathbf{a}^{(l)} = \phi\!\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right) \;\;(l = 1, \dots, L), \qquad \hat{y} = \mathbf{W}^{(L+1)} \mathbf{a}^{(L)} + b^{(L+1)}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** La notación de tupla en `hidden_layer_sizes` indica neuronas por capa oculta: `(2,)` significa $1$ capa oculta con $2$ neuronas (`RNA_MLP_2`, $9$ parámetros, $\text{NRMSE} = 12.28\%$); `(2, 2)` significa $2$ capas ocultas de $2$ neuronas cada una (`RNA_MLP_2_2`, $15$ parámetros, $13.00\%$); `(4, 4, 4)` significa $3$ capas ocultas de $4$ neuronas (`RNA_Delgado_Keras`, $57$ parámetros nominales). Las arquitecturas de $1$ sola capa oculta superaron consistentemente a las de $2$ capas en `Direct-MLP`.

### 4.8. Pesos Sinápticos ($\mathbf{W}$) y Sesgos / Interceptos ($\mathbf{b}$)
* **(a) Explicación intuitiva desde cero:**
  * **Pesos sinápticos ($w_{ij}$):** Son los números multiplicadores que determinan con cuánta fuerza influye cada entrada (o neurona anterior) sobre la neurona siguiente, análogos a las pendientes $\beta_j$ en regresión lineal.
  * **Sesgos o interceptos ($b_j$):** Son constantes aditivas que desplazan el umbral de activación de cada neurona (permitiendo que la curva se mueva hacia la izquierda/derecha o arriba/abajo sin pasar forzosamente por el origen).
* **(b) Fórmula matemática (conteo exacto de parámetros nominales totales):**
  $$P_{\text{nominal}} = \sum_{l=1}^{L+1} \left(d_{l-1} \cdot d_l + d_l\right)$$
  donde $d_0 = 2$ (entradas), $d_{L+1} = 1$ (salida) y $d_l$ es el número de neuronas de la capa $l$.
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * `RNA_MLP_2` `(2,)`: $(2\cdot 2 + 2) + (2\cdot 1 + 1) = 6 + 3 = \mathbf{9}$ parámetros.
  * `RNA_MLP_3` `(3,)`: $(2\cdot 3 + 3) + (3\cdot 1 + 1) = 9 + 4 = \mathbf{13}$ parámetros.
  * `RNA_MLP_2_2` `(2, 2)`: $(2\cdot 2 + 2) + (2\cdot 2 + 2) + (2\cdot 1 + 1) = 6 + 6 + 3 = \mathbf{15}$ parámetros.
  * `RNA_Delgado_MLP` `(4, 4)`: $(2\cdot 4 + 4) + (4\cdot 4 + 4) + (4\cdot 1 + 1) = 12 + 20 + 5 = \mathbf{37}$ parámetros.
  * `RNA_Delgado_Keras` `(4, 4, 4)`: $12 + 20 + 20 + 5 = \mathbf{57}$ parámetros.

### 4.9. Parámetros Nominales ($P$) vs. Parámetros Efectivos ($P_{\text{eff}}$)
* **(a) Explicación intuitiva desde cero:**
  * **Parámetros nominales ($P$):** Es el conteo físico de todos los pesos y sesgos que existen en la memoria del computador al definir la red.
  * **Parámetros efectivos ($P_{\text{eff}}$):** Es la cantidad real de grados de libertad independientes que el modelo está utilizando después de considerar su álgebra (por ejemplo, activaciones lineales redundantes) o la contracción impuesta por una regularización $L_2$ muy fuerte.
* **(b) Fórmula matemática (grados de libertad efectivos bajo regularización $L_2$ en aproximación lineal):**
  $$P_{\text{eff}} = \text{tr}\!\left(\mathbf{H}_\alpha\right) = \sum_{j} \frac{\sigma_j^2}{\sigma_j^2 + \alpha}$$
  donde $\sigma_j$ son los valores singulares de la matriz de diseño y $\alpha$ la penalización $L_2$.
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Auditorías `A-03` y `B3-03`):** Tres casos contrastantes magistrales:
  1. **`RNA_Delgado_Keras`:** $P = 57$ nominales $\longrightarrow$ **$P_{\text{eff}} = 3$** (colapso algebraico exacto por ausencia de activación no lineal).
  2. **`Res_MLP_2_2`, `Res_MLP_3`, `Res_MLP_4` ($\alpha = 2.0$):** $P = 15, 13, 17$ nominales $\longrightarrow$ **$P_{\text{eff}} \approx 1$** porque los pesos sinápticos se contraen a $\|\mathbf{W}\| \approx 0.0005$, dejando activo solo el sesgo escalar $\bar{e}_{\text{train}}$ (explica su empate con `Baseline 1B`).
  3. **`RNA_MLP_2` ($\alpha = 0.001$):** $P = 9$ nominales $\longrightarrow$ **pesos plenamente activos** (validado por SHAP y Permutación).

### 4.10. Colapso Lineal Algebraico de Keras (`57` $\to$ `3`)
* **(a) Explicación intuitiva desde cero:** Si apilas $3$, $10$ o $100$ capas densas una detrás de otra **sin colocar una función de activación no lineal** (`relu`, `tanh`) entre ellas, multiplicar una transformación lineal por otra transformación lineal siempre produce... ¡otra transformación lineal simple! Toda la aparente "profundidad" se derrumba algebraicamente a un único plano recto, exactamente igual que una regresión lineal ordinaria (`OLS`).
* **(b) Fórmula matemática (demostración exacta de la arquitectura `(4, 4, 4)` de Keras):**
  $$\hat{y} = \mathbf{W}_4\Big(\mathbf{W}_3\big(\mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2\big) + \mathbf{b}_3\Big) + b_4 = \mathbf{W}_{\text{eq}} \mathbf{x} + b_{\text{eq}}$$
  $$\mathbf{W}_{\text{eq}} = \mathbf{W}_4 \mathbf{W}_3 \mathbf{W}_2 \mathbf{W}_1 \in \mathbb{R}^{1\times 2}, \qquad b_{\text{eq}} = \mathbf{W}_4\mathbf{W}_3\mathbf{W}_2\mathbf{b}_1 + \mathbf{W}_4\mathbf{W}_3\mathbf{b}_2 + \mathbf{W}_4\mathbf{b}_3 + b_4 \in \mathbb{R}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Celda 9 de `Código unificado.ipynb` y Bloque 3 Sec. 3):**
  * La red preliminar en Keras del Dr. Delgado (`Dense(4)-Dense(4)-Dense(4)-Dense(1)` sin parámetro `activation`) posee **$57$ parámetros nominales** ($12 + 20 + 20 + 5$) que colapsan exactamente a **$P_{\text{eff}} = 3$**: en la demostración en vivo de Celda 9, $\hat{y} = -0.0575 + 0.7916\,\text{Cancha\_CuT} + 0.5815\,\text{Cancha\_CuS}$, con diferencia máxima respecto a la red de 4 capas de apenas **$3.12 \times 10^{-7}\,\%\text{Cu}$** (precisión de máquina `float32`).
  * En LOOCV difiere de `OLS` en promedio solo $0.0023\%\text{Cu}$ ($\text{NRMSE} = 11.64\%$ vs. $11.85\%$ de `OLS`) y hereda exactamente la misma **heterocedasticidad** (Breusch–Pagan $p = 0.0040$). Su cifra histórica preliminar ($\approx 8.63\%$) corresponde a un **MAPE *in-sample*** ($7.97\%$ sobre `PSJ_CuT` / $7.38\%$ sobre `Canje_CuT`).

### 4.11. Funciones de Activación: `Linear` (Identidad) vs. `ReLU` vs. `Tanh`
* **(a) Explicación intuitiva desde cero:** La función de activación es el "interruptor curvo" dentro de cada neurona que decide cómo transformar la suma ponderada $z$ antes de pasarla a la siguiente capa. Sin ella, la red es incapaz de doblar el plano para capturar relaciones no lineales:
  * **`Linear` (Identidad):** No hace nada; deja pasar el número intacto. Solo sirve en la **capa de salida** de problemas de regresión (para predecir leyes continuas sin tope), jamás en capas ocultas (provoca colapso lineal).
  * **`ReLU` (*Rectified Linear Unit*):** Si el valor es negativo lo apaga a $0$; si es positivo lo deja pasar linealmente. Crea superficies continuas compuestas por "facetas planas" unidas en bisagras (*piecewise linear*). No satura en valores positivos grandes, pero puede sufrir el problema de *"dying ReLU"* (neuronas permanentemente apagadas en $0$ con gradiente nulo).
  * **`Tanh` (Tangente Hiperbólica):** Comprime cualquier entrada en una curva suave en forma de "S" centrada en cero entre $-1$ y $+1$. Permite superficies lisas y diferenciables en todo punto (excelentes para curvas geometalúrgicas suaves en muestras pequeñas), pero **satura asintóticamente en los extremos** ($|z| \gg 2$): cuando llega un dato fuera del rango entrenado, su derivada cae a casi $0$ y la salida se aplana.
* **(b) Fórmula matemática (función, rango y derivada):**
  * **Linear:** $g(z) = z$, $\quad \text{Rango: } (-\infty, +\infty)$, $\quad g'(z) = 1$.
  * **ReLU:** $g(z) = \max(0, z)$, $\quad \text{Rango: } [0, +\infty)$, $\quad g'(z) = \begin{cases} 1, & z > 0 \\ 0, & z < 0 \end{cases}$.
  * **Tanh:** $g(z) = \tanh(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$, $\quad \text{Rango: } (-1, +1)$, $\quad g'(z) = 1 - \tanh^2(z) \in (0, 1]$.
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * `Linear` en `RNA_Delgado_Keras` causó el colapso $57 \to 3$ ($\text{NRMSE} = 11.64\%$).
  * `Tanh` en **`RNA_MLP_2`** entregó la mejor curva suave en régimen regular (**Holdout 19/5 $= 6.56\%$**, vence en $14/24$ meses y acierta el pulso oxidado de Oct-2023 con error $0.0632\%$), **pero muestra exactamente su limitación teórica en Nov-2023**: al estar `Cancha_CuT` $= 2.47\%$ fuera del rango de los otros 23 meses, `tanh` satura y predice $1.7467\%\text{Cu}$ frente a $2.27\%\text{Cu}$ real (error $-0.5233\%$).
  * `ReLU` en `Residual-MLP` (`Res_MLP_2_2/3/4`) junto a la conexión residual evitó la saturación en Nov-2023 (prediciendo $2.1235\%\text{Cu}$, error $-0.1465\%$). *(Ver Anexo 1 para la Tabla Comparativa Rápida completa).*

### 4.12. Optimizadores: `L-BFGS` (Cuasi-Newton) vs. `Adam` vs. `SGD`
* **(a) Explicación intuitiva desde cero:** El optimizador es el "navegante" que busca el valle más bajo de la función de error ajustando iterativamente los pesos $\mathbf{W}$ y sesgos $\mathbf{b}$:
  * **`SGD` (*Stochastic Gradient Descent*):** Mira únicamente la pendiente local (primer orden) y da pasos de tamaño fijo $\eta$ (*learning rate*) cuesta abajo. Es simple, pero en valles estrechos oscila y depende fuertemente de elegir bien $\eta$.
  * **`Adam` (*Adaptive Moment Estimation*):** Algoritmo de primer orden que combina inercia (*momentum*, media móvil del gradiente) con paso adaptativo por parámetro (media móvil del gradiente al cuadrado). Es el rey del *Deep Learning* masivo con millones de filas y mini-lotes, pero en tablas diminutas ($N = 23$) requiere muchas épocas, puede oscilar si $\eta$ es demasiado alto y presenta micro-variaciones en `float32`.
  * **`L-BFGS` (*Limited-memory Broyden–Fletcher–Goldfarb–Shanno*):** Algoritmo **cuasi-Newton de segundo orden**. No solo mira la pendiente, sino que aproxima la **curvatura** (matriz Hessiana inversa) de la superficie de error usando toda la muestra a la vez (*full-batch*) y determina analíticamente la dirección y longitud de paso óptimas en cada iteración. En datasets pequeños ($N < 1{,}000$), converge a mínimos de mucha mayor precisión, más rápido y sin necesidad de calibrar *learning rate*.
* **(b) Fórmula matemática:**
  * **SGD:** $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t)$.
  * **Adam:** $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $\; v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, $\; \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \dfrac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$.
  * **L-BFGS:** $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t \mathbf{H}_t^{-1} \nabla \mathcal{L}(\boldsymbol{\theta}_t)$, donde $\mathbf{H}_t^{-1}$ es la aproximación de memoria limitada de la inversa del Hessiano actualizada con los pares $(\mathbf{s}_k, \mathbf{y}_k)$ y $\alpha_t$ se halla por búsqueda lineal (*line search*).
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * Todas las 10 redes MLP de `scikit-learn` emplean **`solver='lbfgs'`** (`max_iter=3000`, `tol=1e-5`), decisión de diseño explícita porque $N_{\text{train}} = 23$ cabe íntegro en memoria y `L-BFGS` encuentra óptimos exactos reproducibles en `float64` sin ruido de mini-lotes (y sin requerir *early stopping* interno, Auditoría `B2-03`).
  * En la réplica de Keras del Dr. Delgado (`RNA_Delgado_Keras`), el cuaderno original usaba `Adam(0.1)` en $1{,}000$ épocas *in-sample*; al llevarlo a LOOCV sobre $N_{\text{train}} = 23$ se ajustó a **`Adam(0.05)`** en $800$ épocas (Auditoría `B3-03`) para evitar oscilaciones del gradiente en los pliegues sin extremos, preservando intacta la equivalencia lineal ($P_{\text{eff}} = 3$). *(Ver Anexo 2 para la Tabla Comparativa Rápida completa).*

### 4.13. Penalización *Weight Decay* ($\alpha$ — Regularización $L_2$ en RNA)
* **(a) Explicación intuitiva desde cero:** Es el "freno de mano" que controla cuánto pueden crecer los pesos sinápticos de una red neuronal. Suma a la función de error cuadrático una penalización proporcional a $\alpha$ multiplicado por la suma de todos los pesos al cuadrado. Con $\alpha$ pequeño, la red tiene libertad para doblar su superficie; con $\alpha$ grande, los pesos se achican hacia cero suavizando o aplanando la red para impedir sobreajuste.
* **(b) Fórmula matemática (función objetivo de `MLPRegressor` en `scikit-learn`):**
  $$\mathcal{L}_{\text{reg}}(\mathbf{W}, \mathbf{b}) = \frac{1}{2N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (y_i - \hat{y}_i)^2 + \frac{\alpha}{2N_{\text{train}}} \|\mathbf{W}\|_F^2$$
  donde $\|\mathbf{W}\|_F^2 = \sum_{l, i, j} w_{ij}^{(l)\,2}$ es la norma de Frobenius al cuadrado (los sesgos $\mathbf{b}$ no se penalizan).
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** Se exploró $\alpha \in \{0.001, 1.0, 2.0, 5.0\}$:
  * `RNA_MLP_2`: $\alpha = 0.001$ permite que las $2$ neuronas `tanh` capturen la curvatura de `CuS` (ablación $-1.53\%$, Holdout $6.56\%$).
  * Redes directas más grandes (`RNA_MLP_3/4`, `RNA_MLP_2_2/3_2`, `RNA_Delgado_MLP`): requieren $\alpha = 2.0$ a $5.0$ para evitar explotar en $N_{\text{train}} = 23$.
  * Familia `Residual-MLP`: bajo **$\alpha = 2.0$**, los pesos residuales se contraen a $\|\mathbf{W}\| \approx 0.0005$ convergiendo exactamente a `Baseline 1B` ($10.07\%$); si se afloja $\alpha$ a $1.0$ (`Res_MLP_2`), el NRMSE sube a $10.26\%$, con $\alpha = 0.5$ sube a $10.73\%$ y con $\alpha = 0.1$ a $11.94\%$.

### 4.14. Escalado de la Variable Objetivo (`scale_y`)
* **(a) Explicación intuitiva desde cero:** Así como escalamos las entradas $X$, en redes neuronales a veces también estandarizamos temporalmente la variable de salida $y$ (media $0$, desviación $1$) durante el entrenamiento y luego revertimos la transformación (`inverse_transform`) sobre la predicción para volver a $\% \text{Cu}$. Esto sirve para que la magnitud de los gradientes de error y el efecto de una misma penalización $\alpha$ sean proporcionales a la varianza del objetivo, especialmente en redes `ReLU` con varias neuronas.
* **(b) Fórmula matemática (estrictamente dentro del pliegue de entrenamiento $-k$):**
  $$y^{\text{sc}}_i = \frac{y_i - \mu_{y,-k}}{\sigma_{y,-k}}, \qquad \hat{y}_k = \hat{y}_k^{\text{sc}} \cdot \sigma_{y,-k} + \mu_{y,-k}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (`CONFIGS_MLP` en Celda 10 y `LOG_HIPERPARAMETROS_RNA.csv`):**
  * **`scale_y = True`:** Aplicado en las redes directas más anchas/profundas (`RNA_Delgado_MLP`, `RNA_MLP_3`, `RNA_MLP_4`, `RNA_MLP_2_2`, `RNA_MLP_3_2`) para estabilizar la escala de regularización $\alpha \in \{2.0, 5.0\}$.
  * **`scale_y = False`:** Mantenido en **`RNA_MLP_2`** (donde la salida lineal directa sobre $2$ neuronas `tanh` trabaja limpiamente en el rango físico $[1.06, 2.27]\%\text{Cu}$ con $\alpha = 0.001$) y en **toda la familia `Residual-MLP`** (donde el objetivo ya es el residuo centrado y pequeño $e_i \approx 0.34\%\text{Cu}$).

### 4.15. Ensamble Multi-Semilla (*Multi-Seed Ensemble*)
* **(a) Explicación intuitiva desde cero:** Como los pesos iniciales de una red neuronal se sortean aleatoriamente antes de empezar a optimizar y la función de error de una MLP tiene múltiples valles locales (*no convexidad*), entrenar una sola red con una sola semilla puede dar suerte o mala suerte. El **ensamble multi-semilla** entrena $M = 10$ redes idénticas arrancando desde $10$ puntos iniciales distintos y promedia sus predicciones, cancelando la varianza aleatoria de inicialización y garantizando estabilidad robusta.
* **(b) Fórmula matemática:**
  $$\hat{y}_k^{\text{ensamble}} = \frac{1}{M} \sum_{m=1}^{M} \hat{f}^{(s_m)}(\mathbf{x}_k), \qquad M = 10$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:** En el Bloque 3, Bloque 4 y Celda 10 de `Código unificado.ipynb`, cada una de las $10$ arquitecturas MLP de `scikit-learn` promedia exactamente las **$10$ semillas deterministas** `SEMILLAS = [42, 101, 2023, 7, 13, 99, 123, 456, 789, 999]`. En producción (`BLOQUE_4/modelo_ganador_rna_mlp2.joblib`), se serializan los $10$ estimadores entrenados junto al `StandardScaler`.

### 4.16. Aprendizaje Residual (`Residual-MLP` / *Skip-Connection*) y Contracción hacia `Baseline 1B`
* **(a) Explicación intuitiva desde cero:** Inspirada en las redes residuales (*ResNets*), en lugar de obligar a la red neuronal a reinventar desde cero la ley de canje completa, le damos una "autopista física directa" (*skip-connection*) con la ley de cancha $X_1$ y le pedimos a la red únicamente que aprenda a predecir el **descuento o sesgo de conciliación** $\hat{e}_{\text{MLP}}$. Así, incluso ante un mes nunca antes visto de ley extrema, la red hereda la proporcionalidad física $1:1$ y jamás colapsa ni satura hacia la media.
* **(b) Fórmula matemática y propiedad asintótica:**
  $$\hat{y}_k = \text{Cancha\_CuT}_k - \hat{e}_{\text{MLP}}\!\left(\text{Cancha\_CuT}_k,\; \text{Cancha\_CuS}_k\right)$$
  donde la red entrena sobre $e_i = \text{Cancha\_CuT}_i - \text{Canje\_CuT}_i$. Bajo penalización $L_2$ creciente:
  $$\lim_{\alpha \to \infty} \|\mathbf{W}\| \to 0 \implies \hat{e}_{\text{MLP}}(\mathbf{x}_k) \to \bar{e}_{\text{train},-k} \implies \lim_{\alpha \to \infty} \hat{y}_k = \text{Cancha\_CuT}_k - \bar{e}_{\text{train},-k} \equiv \text{Baseline 1B}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Auditoría `A-01`):**
  * Con $\alpha = 2.0$, `Res_MLP_2_2` ($\text{MSE} = 0.020781$), `Res_MLP_3` y `Res_MLP_4` ($\text{MSE} = 0.020782$) contraen sus pesos residuales a $\|\mathbf{W}\| \approx 0.0005$ ($P_{\text{eff}} \approx 1$), empatando en el primer lugar `1=` con `Baseline 1B` ($\text{NRMSE} = 10.07\%$, $R^2 = +0.6504$, SHAP de `CuS` $= 0.0000$).
  * **Valor científico de defensa:** Lejos de ser un fallo, demuestra rigurosamente que la arquitectura residual resuelve la extrapolación de Nov-2023 ($2.1235\%$ vs. $1.7467\%$ de `RNA_MLP_2`) y que en el histórico global de $24$ meses el límite de mínima varianza coincide con el sesgo estacionario, fundamentando la **Estrategia Dual** del Bloque 4.

### 4.17. Estudio de Ablación de Entradas (`2D` vs. `1D`)
* **(a) Explicación intuitiva desde cero:** En ciencia experimental, una *ablación* consiste en quitar deliberadamente un componente del sistema para medir exactamente cuánto empeora el desempeño sin él. Aquí se reentrenan las redes quitándoles `Cancha_CuS` (dejándolas en 1D con solo `Cancha_CuT`) y comparando el error LOOCV resultante frente al modelo completo 2D $[\text{CuT}, \text{CuS}]$.
* **(b) Fórmula matemática:**
  $$\Delta\text{NRMSE}_{\text{CuS}} = \text{NRMSE}_{\text{2D [CuT, CuS]}} - \text{NRMSE}_{\text{1D [CuT]}} \qquad (\text{negativo} \implies \text{CuS mejora la precisión})$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 3 Tabla 5):**
  * **`RNA_MLP_2`:** $12.28\%$ (2D, $R^2=+0.4803$) vs. $13.81\%$ (1D, $R^2=+0.3427$) $\implies$ **$\Delta\text{NRMSE} = -1.53\%$** (mejora crítica de $+13.76$ puntos de $R^2$, validando la recomendación mineralógica del Dr. Delgado).
  * **`RNA_MLP_2_2`:** $-0.44\%$ ($13.00\%$ vs. $13.44\%$) | **`RNA_MLP_3_2`:** $-0.36\%$ ($13.48\%$ vs. $13.84\%$) | **`RNA_Delgado_MLP`:** $-0.17\%$ ($12.61\%$ vs. $12.78\%$).
  * **`Res_MLP_2`:** $-0.03\%$ ($10.26\%$ vs. $10.29\%$) | **`Res_MLP_2_2`:** $0.00\%$ ($10.07\%$ vs. $10.07\%$, coherente con la contracción $L_2$ de sus pesos).

---

## Sección 5: Métricas de Evaluación Predictiva

### 5.1. Error Cuadrático Medio ($\text{MSE}$, en $(\%\text{Cu})^2$)
* **(a) Explicación intuitiva desde cero:** Promedia los errores de predicción elevados al cuadrado. Al elevar al cuadrado, elimina signos negativos y **castiga cuadráticamente los errores grandes** (un desvío el doble de grande pesa $4$ veces más). Es la función de pérdida interna que minimizan `OLS`, `Ridge`, `MLPRegressor` y Keras, y el criterio primario para ordenar el *benchmarking* de los $18$ modelos.
* **(b) Fórmula matemática:**
  $$\text{MSE} = \frac{1}{N} \sum_{k=1}^{N} (y_k - \hat{y}_k)^2$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 4 Tabla 2.2, ordenamiento maestro):**
  * **Empate técnico cuádruple (`1=`, brecha $< 10^{-5}\,(\%\text{Cu})^2$):** `Res_MLP_2_2` (**$0.020781\,(\%\text{Cu})^2$**), `Baseline_1B`, `Res_MLP_4` y `Res_MLP_3` (**$0.020782\,(\%\text{Cu})^2$**).
  * `Res_MLP_2`: $0.021551$ | `RNA_Delgado_Keras`: $0.027735$ | `RidgeCV`: $0.028403$ | `OLS`: $0.028789$.
  * **`RNA_MLP_2`:** **$0.030890\,(\%\text{Cu})^2$** en LOOCV y **$0.007417\,(\%\text{Cu})^2$** en *Holdout* 19/5.
  * `SVR_RBF`: $0.056986$ | **`Cancha_Directa`:** **$0.134971\,(\%\text{Cu})^2$**.

### 5.2. Raíz del Error Cuadrático Medio ($\text{RMSE}$, en $\%\text{Cu}$)
* **(a) Explicación intuitiva desde cero:** Es simplemente la raíz cuadrada del $\text{MSE}$, lo que devuelve el error a las **mismas unidades físicas que la ley del mineral ($\%\text{Cu}$)** para que un ingeniero en minas pueda interpretarlo directamente. Mantiene la fuerte sensibilidad ante meses con errores grandes (como Nov-2023).
* **(b) Fórmula matemática:**
  $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{k=1}^{N} (y_k - \hat{y}_k)^2}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * `Cancha_Directa`: **$0.3674\%\text{Cu}$** (IC Bootstrap $95\%$: $[0.3153, 0.4169]$).
  * `Baseline_1B` / `Res_MLP_2_2/3/4`: **$0.1442\%\text{Cu}$** (reducción de $-60.8\%$ en RMSE frente a cancha sin calibrar).
  * `RNA_Delgado_Keras`: $0.1665\%\text{Cu}$ | `RidgeCV`: $0.1685\%\text{Cu}$ | `OLS`: $0.1697\%\text{Cu}$.
  * **`RNA_MLP_2`:** **$0.1758\%\text{Cu}$** (LOOCV), **$0.0861\%\text{Cu}$** (*Holdout* 19/5) y **$0.0966\%\text{Cu}$** (*in-sample* $N=24$).

### 5.3. Error Absoluto Medio ($\text{MAE}$, en $\%\text{Cu}$)
* **(a) Explicación intuitiva desde cero:** Promedia el valor absoluto de los errores mes a mes sin elevarlos al cuadrado. Representa directamente el desvío promedio típico en puntos porcentuales de cobre y es mucho más resistente a un único *outlier* extremo que el $\text{RMSE}$. Cuando $\text{RMSE} \gg \text{MAE}$, sabes de inmediato que unos pocos meses atípicos concentran la mayor parte del error cuadrático.
* **(b) Fórmula matemática:**
  $$\text{MAE} = \frac{1}{N} \sum_{k=1}^{N} |y_k - \hat{y}_k|$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * `Cancha_Directa`: **$0.3404\%\text{Cu}$** (coincide exactamente con el sesgo porque los $24/24$ errores tienen el mismo signo).
  * `Baseline_1B` / `Res_MLP_2_2/3/4`: **$0.1178\%\text{Cu}$** (`0.11775%` en `Res_MLP_2_2`).
  * **`RNA_MLP_2`:** **$0.1318\%\text{Cu}$** (LOOCV, idéntico a `OLS`) y **$0.0750\%\text{Cu}$** (*Holdout* 19/5). Además, en **mediana del error absoluto**, `RNA_MLP_2` ($0.0801\%\text{Cu}$) supera claramente a `Baseline_1B` ($0.1044\%\text{Cu}$), confirmando que su mayor RMSE se concentra exclusivamente en Nov-2023.

### 5.4. Error Cuadrático Medio Normalizado ($\text{NRMSE}\%$)
* **(a) Explicación intuitiva desde cero:** Expresa el $\text{RMSE}$ como porcentaje de la ley promedio real $\bar{y}$. Es el indicador adimensional estrella de tu tesis porque permite comparar errores relativos entre distintas minas, campañas o metales, y concilia las cifras porcentuales de tus reportes con la referencia histórica del profesor guía.
* **(b) Fórmula matemática:**
  $$\text{NRMSE}\;(\%) = \frac{\text{RMSE}}{\bar{y}} \times 100\%, \qquad \bar{y} = 1.43125\%\text{Cu}\;\text{(en Mantos Cobrizo }N=24\text{)}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * `Cancha_Directa`: **$25.67\%$** (LOOCV) / $27.70\%$ (*Holdout* 19/5).
  * `Baseline_1B` / `Res_MLP_2_2/3/4` (Empate `1=`): **$10.07\%$** (LOOCV) / $8.96\%$ (*Holdout* 19/5) / $10.04\%$ (*Buffered-LOOCV*).
  * `RNA_Delgado_Keras`: **$11.64\%$** | `RidgeCV`: $11.78\%$ | `OLS`: $11.85\%$.
  * **`RNA_MLP_2`:** **$12.28\%$** (LOOCV) / **$6.56\%$** (*Holdout* 19/5) / $6.75\%$ (*in-sample*) / $14.53\%$ (*Buffered-LOOCV*).

### 5.5. MECM Histórico (Reconciliación con el $26\%$ del Dr. José Delgado)
* **(a) Explicación intuitiva desde cero:** En la formulación inicial de la memoria y diapositivas preliminares (marzo 2025), el Dr. José Delgado Vega resumió el problema industrial como un *Margen/Medida de Error Cuadrático Medio relativo* ($\text{MECM}$) cercano al **$26\%$** en la ley de cancha, acompañado de un referente preliminar de red neuronal cercano al **$8.63\%$**. Para dar rigor científico y eliminar toda ambigüedad terminológica, el Bloque 0 y el Bloque 3 reconciliaron exactamente a qué fórmulas estandarizadas corresponden ambas cifras.
* **(b) Fórmula matemática (equivalencia exacta):**
  $$\text{MECM}_{\text{histórico}} \equiv \text{NRMSE}_{\text{Cancha Directa}} = \frac{\text{RMSE}}{\bar{y}} \times 100\%$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 0 Sec. 4.2 y Auditoría `A-04`):**
  * **Línea base histórica ($\approx 26\%$):** Corresponde exactamente al $\text{NRMSE}$ de `Cancha_Directa`: **$25.67\%$** sobre `Canje_CuT` ($\text{MAPE} = 24.90\%$) y **$25.98\%$** sobre `PSJ_CuT` ($\text{MAPE} = 25.05\%$).
  * **Referente preliminar RNA ($8.63\%$):** Corresponde al orden de magnitud del **MAPE *in-sample*** de la red lineal Keras original ($7.97\%$ en `PSJ_CuT` / $7.38\%$ en `Canje_CuT`), cuyo error ciego real en LOOCV es $11.64\%$, mientras que tu red parsimoniosa **`RNA_MLP_2`** logra perforar ese $8.63\%$ en régimen regular prospectivo (*Holdout* 19/5 $= \mathbf{6.56\%}$).

### 5.6. Error Porcentual Absoluto Medio ($\text{MAPE}\%$)
* **(a) Explicación intuitiva desde cero:** Calcula el error porcentual de cada mes respecto a la ley real de ese mismo mes y promedia todos los porcentajes. A diferencia del $\text{NRMSE}$ (que divide por la media global $\bar{y}$), el $\text{MAPE}$ pondera más fuerte los errores en meses de **ley baja** (dividir un error de $0.15\%$ por $1.06\%$ pesa más que dividirlo por $2.27\%$).
* **(b) Fórmula matemática:**
  $$\text{MAPE}\;(\%) = \frac{1}{N} \sum_{k=1}^{N} \left|\frac{y_k - \hat{y}_k}{y_k}\right| \times 100\%$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Auditoría `A-04`):**
  * En LOOCV ($N=24$): `Cancha_Directa` **$24.90\%$**, `Baseline_1B` / `Res_MLP_2_2/3/4` **$8.12\%$**, `Res_MLP_2` $8.22\%$, `RNA_Delgado_Keras` $8.74\%$, `OLS` $8.82\%$, `RNA_MLP_2_2` $8.92\%$, `RidgeCV` $8.93\%$, **`RNA_MLP_2` $8.95\%$**, `SVR_RBF` $12.11\%$.
  * **Distinción crucial (A-04):** En la réplica *in-sample* de Keras, siempre diferencia $\text{MAPE}$ de $\text{NRMSE}$: sobre `PSJ_CuT` es `MAPE = 7.97%` (`NRMSE = 9.91%`–`9.92%`); sobre `Canje_CuT` es `MAPE = 7.38%` (`7.37%` en vivo) vs. `NRMSE = 9.19%`.

### 5.7. Coeficiente de Determinación ($R^2$ Score) y Por Qué Puede Ser Negativo ($-1.2709$)
* **(a) Explicación intuitiva desde cero:** Mide qué fracción de la variabilidad total de la ley real de canje logra explicar tu modelo en comparación con el "modelo tonto" que siempre predice la ley promedio histórica $\bar{y} = 1.4313\%\text{Cu}$.
  * $R^2 = 1.0$: predicción perfecta ($\hat{y}_k = y_k$).
  * $R^2 = 0.0$: el modelo es igual de malo que predecir siempre el promedio constante $\bar{y}$.
  * **$R^2 < 0$ (¡SÍ puede ser negativo fuera de muestra o sin intercepto!):** Significa que el error cuadrático del modelo ($\text{SSE}$) es **mayor** que la varianza total de los datos ($\text{SST}$); es decir, **usar la ley de cancha sin corregir es matemáticamente peor que ignorar el laboratorio de mina y apostar todos los meses al promedio histórico de ENAMI**. (Solo en OLS *in-sample* con intercepto $R^2$ coincide con $r^2 \in [0, 1]$).
* **(b) Fórmula matemática:**
  $$R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum_{k=1}^{N} (y_k - \hat{y}_k)^2}{\sum_{k=1}^{N} (y_k - \bar{y})^2}, \qquad \text{con } \text{SST} = 1.42646\,(\%\text{Cu})^2$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * **`Cancha_Directa`:** $\text{SSE} = 24 \times 0.134971 = 3.2393\,(\%\text{Cu})^2 > \text{SST} = 1.42646 \implies R^2 = 1 - \frac{3.2393}{1.42646} = \mathbf{-1.2709}$ (y **$-5.1463$** en *Holdout* 19/5).
  * **`Res_MLP_2_2`:** **$+0.6504$** | **`Baseline_1B` / `Res_MLP_3/4`:** **$+0.6503$**.
  * **`RNA_MLP_2`:** **$+0.4803$** en LOOCV y **$+0.6549$** en *Holdout* 19/5 (mejor $R^2$ prospectivo de toda la tesis).

### 5.8. Varianza Explicada (`explained_variance_score`), Diferencia con $R^2$ y la Paradoja $0.6789$ vs. $0.6503$ por el Factor $(24/23)^2$
* **(a) Explicación intuitiva desde cero:** Es prima hermana del $R^2$, con una diferencia vital: mientras $R^2$ usa el error cuadrático total ($\text{MSE} = \bar{e}^2 + \sigma^2_e$), la **Varianza Explicada** solo usa la **varianza del error ($\sigma^2_e$)** e **ignora completamente cualquier sesgo constante** ($\bar{e}$). Por eso, si un modelo es insesgado ($\bar{e} = 0$), $\text{VarExp} = R^2$; si tiene sesgo ($\bar{e} \neq 0$), $\text{VarExp} > R^2$.
* **(b) Fórmula matemática y demostración exacta del factor $(24/23)^2$:**
  $$\text{VarExp} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)} = R^2 + \frac{\text{Sesgo}^2}{\sigma_y^2}$$
  **¿Por qué `Cancha_Directa` tiene $\text{VarExp} = 0.6789$ mientras `Baseline_1B` en LOOCV tiene $0.6503$?**
  En `Baseline_1B`, el residuo fuera de muestra en el pliegue $k$ es exactamente:
  $$e_k^{\text{B1B}} = y_k - \left(X_{1,k} - \frac{1}{N-1}\sum_{i \neq k} e_i\right) = -\frac{N}{N - 1}(e_k - \bar{e}) \implies \text{Var}(e^{\text{B1B}}) = \left(\frac{N}{N - 1}\right)^2 \text{Var}(e_{\text{Cancha}})$$
  Para $N = 24$, el factor de inflación teórico por validación ciega LOOCV es exactamente:
  $$\left(\frac{24}{23}\right)^2 = 1.088847 \implies 1 - 1.088847 \times (1 - 0.678859) = 1 - 1.088847 \times 0.321141 = \mathbf{0.6503}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Auditorías `C-04` y `B3-02`):**
  * `Cancha_Directa` aparenta $\text{VarExp} = \mathbf{0.6789}$ solo porque descuenta el sesgo medio *in-sample* ($+0.3404\%$, usando los $24$ meses a la vez); al hacer la corrección honestamente fuera de muestra en LOOCV (`Baseline_1B`), la penalización algebraica exacta $(24/23)^2 = 1.0888$ traslada la varianza explicada real a **$\mathbf{0.6503}$** (coincidiendo exactamente con su $R^2 = +0.6503$).
  * En **`RNA_MLP_2`** (unificado en `B3-02`): $R^2 = 0.480288$ y $\text{Sesgo} = -0.0123625\%$, luego $\text{VarExp} = 0.480288 + \frac{(-0.0123625)^2}{0.059436} = \mathbf{0.4829}$.

### 5.9. Sesgo Medio de Predicción (*Mean Bias*)
* **(a) Explicación intuitiva desde cero:** Indica si el modelo tiende a equivocarse sistemáticamente por encima (sobreestimación, signo positivo) o por debajo (subestimación, signo negativo) de la ley real liquidada. Un modelo bien calibrado debe tener un sesgo medio muy cercano a $0.0000\%\text{Cu}$ para que los desvíos mensuales se compensen financieramente a lo largo del año.
* **(b) Fórmula matemática (convención oficial de Bloque 4 y `Código unificado.ipynb`):**
  $$\text{Sesgo} = \frac{1}{N} \sum_{k=1}^{N} (\hat{y}_k - y_k)$$
  *(Positivo: el modelo sobreestima a ENAMI; Negativo: el modelo subestima a ENAMI).*
* **(c) Aplicación y valor exacto en Mantos Cobrizo:**
  * `Cancha_Directa`: **$+0.3404\%\text{Cu}$** (sobreestimación severa en $24/24$ meses).
  * `Res_MLP_2_2` / `Baseline_1B` / `Res_MLP_3/4`: **$\approx 0.0000\%\text{Cu}$** (`+9.25e-18` y `-4.17e-6`, calibración insesgada perfecta).
  * `RNA_MLP_2`: **$-0.0124\%\text{Cu}$** (ligeramente conservador) | `OLS`: $+0.0038\%\text{Cu}$ | `RidgeCV`: $+0.0108\%\text{Cu}$ | `RNA_Delgado_Keras`: $+0.0022\%\text{Cu}$.

### 5.10. Paradoja $r = +0.8347$ vs. $R^2 = -1.2709$
* **(a) Explicación intuitiva desde cero:** Es una de las preguntas trampa más frecuentes en defensas de tesis de minería y geoestadística: *"¿Cómo puede `Cancha_Directa` tener la correlación de Pearson más alta de los 18 modelos ($r = +0.8347$) y al mismo tiempo el peor coeficiente de determinación ($R^2 = -1.2709$)?"*. La respuesta es simple: la correlación de Pearson solo pregunta si dos curvas **suben y bajan al mismo ritmo** (es completamente ciega a si una está desplazada $+0.34\%\text{Cu}$ arriba de la otra). En cambio, $R^2$ exige que las predicciones caigan **encima de la recta de identidad física $y = \hat{y}$**.
* **(b) Fórmula matemática (invarianza afín de Pearson frente a sensibilidad al sesgo de $R^2$):**
  $$r(y,\; a + b\hat{y}) = r(y,\; \hat{y}) \quad \forall\, a \in \mathbb{R},\; b > 0 \qquad \text{pero} \qquad R^2 = 1 - \frac{\bar{e}^2 + \sigma_e^2}{\sigma_y^2}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Auditoría `B2-02` y `B2-04`):** `Cancha_Directa` mantiene una excelente sincronía direccional con `Canje_CuT` ($r = +0.8347$, menor varianza residual $s_e^2 = 0.019917$), pero su traslación constante de $+0.3404\%\text{Cu}$ inyecta $\bar{e}^2 = 0.115884\,(\%\text{Cu})^2$ ($85.86\%$ del MSE), hundiendo $R^2$ a **$-1.2709$**. Por eso nunca debe evaluarse un modelo de conciliación minera usando únicamente $r$ de Pearson.

---

## Sección 6: Explicabilidad XAI, Geometría de Dominio e Ingeniería de Software

### 6.1. Modelos de Caja Negra (*Black-Box*) vs. Inteligencia Artificial Explicable (XAI)
* **(a) Explicación intuitiva desde cero:** Un modelo de **caja negra** entrega una predicción numérica sin revelar cómo pesó sus variables internas (lo cual genera desconfianza en ingenieros de planta y comisiones académicas). La **Inteligencia Artificial Explicable (*eXplainable AI*, XAI)** abre esa caja negra mediante herramientas matemáticas auditables para responder *qué variable domina*, *en qué dirección empuja la predicción* y *si la física aprendida respeta las leyes de la Ingeniería en Minas* (Pierre Gy).
* **(b) Fórmula matemática (descomposición aditiva explicable):**
  $$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{p} \phi_j(\mathbf{x})$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Estrategia Dual del Bloque 4):** La Auditoría `A-01` demostró que aplicar XAI sobre `Res_MLP_2_2` produce importancia $0.0000$ para `CuS` (porque $\alpha = 2.0$ apagó los pesos residuales hacia `Baseline 1B`). Por ello, la **Estrategia Dual** reserva `Baseline 1B / Res_MLP_2_2` ($10.07\%$) para **presupuesto conservador**, y selecciona a **`RNA_MLP_2`** ($9$ parámetros activos, Holdout $6.56\%$, vence en $14/24$ meses) como **Campeón XAI** para auditar el rol físico de `Cancha_CuS` en el control de acopio y *blending*.

### 6.2. Importancia por Permutación (*Permutation Importance*)
* **(a) Explicación intuitiva desde cero:** Método XAI agnóstico al modelo que mide cuánto depende realmente una red entrenada de cada variable. Funciona así: tomas la columna `Cancha_CuS`, mezclas aleatoriamente sus $24$ filas rompiendo su vínculo con la realidad (dejando `Cancha_CuT` intacto) y mides cuánto empeora el $\text{RMSE}$ del modelo. Si el error empeora mucho, la variable es crucial; si no cambia nada, el modelo la ignoraba. Repetirlo $1{,}000$ veces entrega una media y desviación estándar sumamente estables.
* **(b) Fórmula matemática:**
  $$I_j = \frac{1}{R} \sum_{r=1}^{R} \Big(\text{RMSE}\big(y,\; \hat{f}(\mathbf{X}^{\text{perm}(j, r)})\big) - \text{RMSE}\big(y,\; \hat{f}(\mathbf{X})\big)\Big), \qquad R = 1{,}000$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (`XAI_IMPORTANCIA_VARIABLES.csv`, `n_repeats=1000`):**
  * **`RNA_MLP_2` (Campeón XAI — ambos activos):**
    * `Cancha_CuT`: **$+0.1256 \pm 0.0298\,\%\text{Cu}$** de aumento en RMSE.
    * `Cancha_CuS`: **$+0.1003 \pm 0.0216\,\%\text{Cu}$** de aumento en RMSE (prueba contundente de peso sináptico activo).
  * **`Res_MLP_2_2` (Residual contraído $\alpha = 2.0$):**
    * `Cancha_CuT`: $+0.1974 \pm 0.0387\,\%\text{Cu}$ (conexión física $1:1$ preservada).
    * `Cancha_CuS`: **$-0.0000 \pm 0.0000\,\%\text{Cu}$** (apagado por contracción $L_2$).

### 6.3. Valores SHAP (*SHapley Additive exPlanations*) y `KernelExplainer`
* **(a) Explicación intuitiva desde cero:** Basados en el Premio Nobel de Economía Lloyd Shapley (teoría de juegos cooperativos), los **valores SHAP** reparten con total justicia matemática cuánto aportó cada variable (`Cancha_CuT` y `Cancha_CuS`) para mover la predicción de un lote específico desde el valor base promedio $\phi_0$ hasta la predicción final $\hat{y}_i$. **`KernelExplainer`** es el algoritmo universal de la librería `shap` que calcula estos valores de manera exacta sobre cualquier función caja negra (como un ensamble de 10 MLPs) usando un conjunto de referencia de fondo (*background dataset*).
* **(b) Fórmula matemática (valor de Shapley exacto para el predictor $j$):**
  $$\phi_j(\mathbf{x}) = \sum_{S \subseteq \{1,\dots,p\}\setminus\{j\}} \frac{|S|!\,(p - |S| - 1)!}{p!} \Big[f_{S \cup \{j\}}(\mathbf{x}) - f_S(\mathbf{x})\Big], \qquad \text{Importancia Global} = \mathbb{E}\!\left[|\phi_j|\right] = \frac{1}{N}\sum_{i=1}^{N}|\phi_{i,j}|$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (`shap.KernelExplainer`, fondo de $10$ muestras con `random_state=42`):**
  * **`RNA_MLP_2`:** $\mathbb{E}[|\phi_{\text{CuT}}|] = \mathbf{0.0922\%\text{Cu}}$ y $\mathbb{E}[|\phi_{\text{CuS}}|] = \mathbf{0.0701\%\text{Cu}}$ (cuantifica un aporte marginal real del cobre soluble equivalente al $76\%$ del aporte marginal de `CuT` en la superficie 2D).
  * **`Res_MLP_2_2`:** $\mathbb{E}[|\phi_{\text{CuT}}|] = 0.1977\%\text{Cu}$ y $\mathbb{E}[|\phi_{\text{CuS}}|] = \mathbf{0.0000\%\text{Cu}}$ (verificación formal de la Auditoría `A-01`).

### 6.4. Superficie de Respuesta 2D (*2D Response Surface* / Mapa de Contornos)
* **(a) Explicación intuitiva desde cero:** Como tu modelo solo tiene $2$ variables de entrada (`Cancha_CuT` en el eje horizontal y `Cancha_CuS` en el eje vertical), es posible evaluar la red neuronal sobre toda una grilla densa de pares posibles y dibujar un "mapa topográfico" de curvas de nivel (*isolíneas*) donde el color representa la ley `Canje_CuT` estimada. Esto transforma completamente a `RNA_MLP_2` en una **caja de cristal** legible a simple vista por un jefe de turno o geólogo de mina.
* **(b) Fórmula matemática (evaluación sobre grilla cartesiana $90 \times 90$):**
  $$Z_{r,c} = \hat{f}_{\text{RNA\_MLP\_2}}\!\left(x_{1,c},\; x_{2,r}\right), \qquad (r, c) \in \{1, \dots, 90\}^2$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Bloque 4 Figura 3 y Celda 14):** Se construye sobre `RNA_MLP_2` reentrenada en $N = 24$, documentando con total transparencia su ajuste *in-sample* ($\text{RMSE}_{\text{in-sample}} = 0.0966\%\text{Cu}$, $\text{NRMSE} = 6.75\%$ frente a $\text{LOOCV} = 12.28\%$, Auditoría `C-05`). Permite planificar mezclas (*blending*) de frentes sulfurados y oxidados visualizando en qué combinaciones se protege mejor la ley de canje.

### 6.5. Envolvente Convexa (*Convex Hull*)
* **(a) Explicación intuitiva desde cero:** Imagina que clavas un alfiler en un tablero por cada uno de los $24$ meses históricos en el plano `(Cancha_CuT, Cancha_CuS)` y luego estiras una banda elástica alrededor de todos los alfileres y la sueltas: el polígono cerrado que forma la banda elástica es la **Envolvente Convexa (*Convex Hull*)**. Todo lote nuevo que caiga **dentro** del polígono está respaldado por interpolación de datos históricos reales; todo lote que caiga **fuera** entra en zona de **extrapolación riesgosa** sin respaldo empírico.
* **(b) Fórmula matemática (conjunto de todas las combinaciones convexas de los $N$ datos):**
  $$\text{Conv}(\mathcal{X}) = \left\{ \sum_{i=1}^{N} \lambda_i \mathbf{x}_i \;\middle|\; \lambda_i \ge 0,\; \sum_{i=1}^{N} \lambda_i = 1 \right\}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Auditorías `C-01` y `B4-02`):** Delimitada mediante `scipy.spatial.ConvexHull(X_2d)` en la Figura 3 del Bloque 4 para evitar interpretaciones infundadas en esquinas vacías (por ejemplo, $\text{CuT} \le 1.80\%$ con $\text{CuS} > 0.16\%$, donde caía el antiguo ejemplo `[1.55, 0.18]` corregido en `B4-02`). El lote certificado oficial **`[Cancha_CuT = 1.75%, Cancha_CuS = 0.14%]`** cae estrictamente **dentro del *Convex Hull*** (interpolación segura), proyectando **`1.4076 %Cu`** en `RNA_MLP_2` y **`1.4096 %Cu`** en `Res_MLP_2_2` y `Baseline 1B`.

### 6.6. Triangulación de Delaunay y Consulta de Pertenencia (`find_simplex`)
* **(a) Explicación intuitiva desde cero:** Divide el interior del *Convex Hull* en una red de triángulos (*símplices* en 2D) que conectan los puntos históricos sin superponerse y maximizando los ángulos mínimos. En producción, cuando un ingeniero ingresa la ley de un nuevo camión o lote al simulador, el algoritmo busca en microsegundos en qué triángulo cae ese punto: si encuentra un triángulo ($\ge 0$), el lote está **dentro del dominio seguro**; si devuelve `-1`, dispara inmediatamente una **alerta operativa de extrapolación**.
* **(b) Fórmula matemática (coordenadas baricéntricas en un triángulo $\triangle(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3)$):**
  $$\mathbf{x}_{\text{nuevo}} \in \text{Conv}(\mathcal{X}) \iff \exists\,\triangle \in \mathcal{T}_{\text{Delaunay}}(\mathcal{X}) \text{ tal que } \mathbf{x}_{\text{nuevo}} = \sum_{m=1}^{3} \lambda_m \mathbf{v}_m,\;\; \lambda_m \ge 0,\;\; \sum_{m=1}^{3} \lambda_m = 1$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Celda 15 de `Código unificado.ipynb`):** Implementada en vivo en la función `simular_lote_faena(valor_x1, valor_x2)` mediante `Delaunay(X_2d).find_simplex(x_nuevo)[0] >= 0`, brindando un semáforo geométrico automático tanto para Mantos Cobrizo como para cualquier dataset nuevo cargado en el cuaderno universal.

### 6.7. Serialización Portable (`.joblib`) y Diccionarios Nativos sin Dependencia `__main__`
* **(a) Explicación intuitiva desde cero:** *Serializar* significa guardar en disco un modelo ya entrenado (con todos sus pesos aprendidos y parámetros del escalador) para cargarlo después en otro programa (como el simulador económico del Bloque 5 o una app web en Streamlit del Bloque 6) sin volver a entrenar. Un error clásico de software es serializar una clase personalizada definida dentro de un notebook (`__main__.MiClase`), lo que hace fallar la carga en otro archivo `.py`. La solución profesional consiste en guardar **diccionarios puros de Python** conteniendo solo objetos $100\%$ nativos de `scikit-learn` (`StandardScaler` + lista de `MLPRegressor`) y exponer la lógica en un módulo importable dedicado (`modelos_rna.py`).
* **(b) Fórmula matemática (estructura de artefacto serializado):**
  $$\mathcal{A}_{\text{joblib}} = \big\{\text{'nombre'},\; \text{'scaler'}: \tau_{\boldsymbol{\mu},\boldsymbol{\sigma}},\; \text{'estimadores'}: [\hat{f}^{(s_1)}, \dots, \hat{f}^{(s_{10})}]\big\}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Auditoría `C-03`, Bloque 4 Sec. 6):** Se exportaron tres artefactos portables en `./BLOQUE_4/`: `modelo_ganador_rna_mlp2.joblib`, `modelo_ganador_residual_mlp.joblib` y `modelo_baseline_1b.joblib` (sesgo $+0.340417\%\text{Cu}$), consumibles desde cualquier entorno mediante `BLOQUE_4/modelos_rna.py` (`predecir_rna_mlp2`, `predecir_residual_mlp`, `predecir_baseline_1b` y `PredictorWrapper`).

### 6.8. Determinismo Numérico: `float64` (BLAS/Fortran) vs. `float32` (`oneDNN` / CUDA)
* **(a) Explicación intuitiva desde cero:** Los computadores representan los números reales con distinta precisión y orden de memoria:
  * **`float64` (doble precisión, 64 bits, $\sim 16$ dígitos significativos) + orden Fortran (`np.asfortranarray`):** Usado por `scikit-learn` y el optimizador `L-BFGS` sobre rutinas algebraicas BLAS/LAPACK. Alinear la matriz por columnas (*Fortran-contiguous*) hace que el orden de las sumas de punto flotante sea idéntico en Windows local y en Linux (Google Colab), logrando reproducibilidad exacta bit a bit.
  * **`float32` (precisión simple, 32 bits, $\sim 7$ dígitos significativos) + aceleradores `oneDNN`/CUDA:** Usado por defecto en TensorFlow/Keras para máxima velocidad en GPUs. Como la aritmética paralela de `float32` no es estrictamente asociativa ($(a+b)+c \neq a+(b+c)$), ejecutar el mismo código en dos CPUs/GPUs diferentes produce micro-variaciones en el tercer o cuarto decimal.
* **(b) Fórmula matemática (épsilon de máquina):**
  $$\varepsilon_{\text{mach}}(\text{float64}) = 2^{-52} \approx 2.22 \times 10^{-16} \qquad \text{vs.} \qquad \varepsilon_{\text{mach}}(\text{float32}) = 2^{-23} \approx 1.19 \times 10^{-7}$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Celdas 7, 9 y 10 de `Código unificado.ipynb`):**
  * Las $10$ redes `MLPRegressor` + $7$ baselines en `float64` con `np.asfortranarray` reproducen exactamente al cuarto/sexto decimal las cifras selladas (`10.07%`, `12.28%`, `25.67%`) en cualquier computador o Google Colab.
  * En `RNA_Delgado_Keras` (`float32`), la corrida en vivo con `@tf.function` arroja `11.82%` frente al vector certificado de tesis (`11.64%`), diferencia de solo $0.18\%$ totalmente explicada por aritmética `float32` CPU/oneDNN; por ello Celda 9 muestra ambos resultados transparentemente y usa el vector en vivo para cualquier base nueva.

### 6.9. Compilación en Grafo (`@tf.function`) y Diferenciación Automática (`tf.GradientTape`)
* **(a) Explicación intuitiva desde cero:**
  * **`tf.GradientTape()`:** Es una "cinta grabadora" matemática de TensorFlow que registra todas las operaciones realizadas al calcular la predicción y el error cuadrático, y luego reproduce la cinta hacia atrás (regla de la cadena / *backpropagation*) para obtener automáticamente los gradientes exactos $\nabla_{\boldsymbol{\theta}} \mathcal{L}$.
  * **`@tf.function`:** Traduce un bucle de Python (que sería lento si se interpreta línea por línea) a un **grafo computacional compilado en C++ nativo**, eliminando la sobrecarga del intérprete Python en cada época de entrenamiento.
* **(b) Fórmula matemática (paso de diferenciación automática y actualización):**
  $$\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \left(\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (y_i - \hat{y}_i)^2\right), \qquad \boldsymbol{\theta}_{t+1} \leftarrow \text{Adam}\!\left(\boldsymbol{\theta}_t,\; \mathbf{g}_t,\; \eta\right)$$
* **(c) Aplicación y valor exacto en Mantos Cobrizo (Celda 9 de `Código unificado.ipynb`):** Permite entrenar los $24$ pliegues LOOCV de la réplica Keras del Dr. Delgado ($24 \times 800 = 19{,}200$ épocas de `Adam(0.05)`) más los $2$ ajustes *in-sample* ($1{,}000$ épocas de `Adam(0.1)`) en **$\sim 20$ segundos en Google Colab**, reduciendo más de $30\times$ el tiempo respecto al método estándar `.fit()` de Keras (que tardaba $> 10$ minutos).

---

## Anexo 1: Tabla Comparativa Rápida de Funciones de Activación

| Función de Activación | Fórmula $g(z)$ | Rango de Salida | Derivada $g'(z)$ | Ventajas Principales | Desventajas / Riesgos Críticos | Rol y Resultado Exacto en Mantos Cobrizo |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **`Linear` (Identidad)** | $z$ | $(-\infty, +\infty)$ | $1$ | Necesaria en la **capa de salida** de regresión para entregar leyes continuas sin cota artificial. | En **capas ocultas** destruye la no linealidad y provoca **colapso lineal algebraico** equivalente a `OLS`. | Causó el colapso **$57 \to P_{\text{eff}} = 3$** en `RNA_Delgado_Keras` ($\text{NRMSE} = 11.64\%$, heterocedástica $p = 0.0040$). |
| **`ReLU`** (*Rectified Linear Unit*) | $\max(0, z)$ | $[0, +\infty)$ | $\begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$ | Sin saturación para $z > 0$; cómputo muy rápido; gradiente constante $1$ en zona activa; ideal con *skip-connections*. | Quiebres no suaves (*piecewise linear*); riesgo de neuronas muertas (*dying ReLU*, $g'(z) = 0$ si $z < 0$). | Usada en `Res_MLP_2_2/3/4` (**$10.07\%$**, empate `1=`, sin saturación en Nov-2023) y en `RNA_MLP_3/4/2_2` ($12.39\%$–$13.00\%$). |
| **`Tanh`** (Tangente Hiperbólica) | $\dfrac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, +1)$ | $1 - \tanh^2(z)$ | Centrada en cero; infinitamente diferenciable ($C^\infty$); genera **superficies 2D suaves** ideales en muestras pequeñas. | **Saturación asintótica** en extremos ($|z| \gg 2 \implies g'(z) \to 0$), aplanando extrapolaciones fuera del rango entrenado. | Usada en **`RNA_MLP_2`** (**Campeón XAI**: **$6.56\%$** Holdout 19/5, vence $14/24$ meses, ablación `CuS` $-1.53\%$; satura solo en Nov-2023 prediciendo $1.7467\%$ vs. $2.27\%$). |
| **`Sigmoide` (Referencia contrastiva)** | $\dfrac{1}{1 + e^{-z}}$ | $(0, 1)$ | $g(z)(1 - g(z))$ | Útil para probabilidades en clasificación binaria. | No centrada en cero; gradiente máximo de solo $0.25$; saturación doble más severa que `tanh`. | **Descartada** en capas ocultas de regresión por inferioridad frente a `tanh` y `ReLU`. |

---

## Anexo 2: Tabla Comparativa Rápida de Optimizadores

| Optimizador | Orden / Mecánica Matemática | Hiperparámetros Clave | Ventajas en Muestras Pequeñas ($N_{\text{train}} = 23$) | Desventajas / Riesgos | Rol y Resultado Exacto en Mantos Cobrizo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`L-BFGS`** (*Limited-memory BFGS*) | **Cuasi-Newton (2.º orden aproximado):** estima $\mathbf{H}_t^{-1}$ (curvatura) + búsqueda lineal *full-batch*. | `max_iter` ($3{,}000$), `tol` ($10^{-5}$), `alpha` ($L_2$). *(No requiere learning rate manual).* | Convergencia exacta, ultrarrápida y **determinista en `float64`** sin ruido estocástico de mini-lotes; óptimo para $N < 1{,}000$. | No escala a millones de filas (requiere evaluar toda la muestra en cada iteración); no utiliza *early stopping* por épocas estocásticas. | Motor oficial de las **10 arquitecturas MLP** (`Res_MLP_2_2` **$10.07\%$**, `RNA_MLP_2` **$12.28\%$** LOOCV / **$6.56\%$** Holdout) promediando $10$ semillas fijas. |
| **`Adam`** (*Adaptive Moment Estimation*) | **1.er orden adaptativo:** combina inercia ($m_t$, $\beta_1 = 0.9$) y escala adaptativa por peso ($v_t$, $\beta_2 = 0.999$). | `learning_rate` ($\eta$), `epochs`, $\beta_1, \beta_2, \varepsilon$. | Muy flexible para *Deep Learning* a gran escala y mini-lotes en GPU/TensorFlow. | Sensible a $\eta$ en $N = 23$ (`0.1` oscila en LOOCV); lento en bucles de épocas sin `@tf.function`; micro-variaciones en `float32`. | Usado en la réplica **`RNA_Delgado_Keras`**: `Adam(0.1)` ($1{,}000$ épocas *in-sample* original) y **`Adam(0.05)`** ($800$ épocas en LOOCV, **$11.64\%$**) para evitar oscilaciones. |
| **`SGD`** (*Stochastic Gradient Descent*) | **1.er orden puro:** paso fijo en dirección opuesta al gradiente $-\eta \nabla \mathcal{L}$ (con o sin *momentum*). | `learning_rate` ($\eta$), `momentum`, `batch_size`. | Sencillo y con buena generalización en *big data* cuando se calendariza bien $\eta$. | Muy sensible a la escala de variables y a $\eta$; converge lento en valles estrechos y sufre alta varianza en muestras mineras pequeñas. | **Descartado** para el entrenamiento final frente a la superioridad numérica de `L-BFGS` en $N_{\text{train}} = 23$. |

---

## Anexo 3: Hoja de Referencia Rápida — Cifras Oficiales Auditadas (Metodología v4.0)

| Categoría | Indicador / Hallazgo Certificado | Valor Exacto Sellado |
| :--- | :--- | :--- |
| **Datos y Tonelajes (Bloque 0)** | Tamaño muestral / Cancha Ton / Romana Ton / Brecha media | $N = 24$ meses \| $286{,}704\text{ t}$ \| $295{,}884.67\text{ t}$ ($12{,}328.53\text{ t/mes}$) \| **$+0.3404\%\text{Cu}$** ($24/24$ meses) |
| **Tarifas ENAMI (Bloque 0)** | Maquila implícita certificada ($100\%$ USD, Auditoría `B0-01`) | **$34.66$** (Ene-23) \| **$35.65$** (Feb-23 a Ene-24) \| **$39.60\text{ USD/tms}$** (Feb-24 a Dic-24) |
| **EDA y Escala (Bloque 1)** | Ratio varianzas `CuT`/`CuS` \| Descomposición MSE \| VIF 2D | **$61.64\times$** ($61.62\times$ exacto) \| **$85.86\%$** sesgo$^2$ + **$14.14\%$** var. \| **$\text{VIF} = 1.8045$** ($>4{,}000$ con `CuIn`) |
| **Normalidad, Outliers y ACF** | Shapiro `Sesgo_Canje` \| Mahalanobis $>5.9915$ \| ACF Lag-1 | **$W = 0.9852, p = 0.9695$** \| `2023-10` ($14.2481$) y `2023-11` ($8.4700$) \| `Sesgo` $r_1 = -0.0866$, `CuT` $r_1 = +0.4467$ |
| **Baselines y Ablación (B2–B3)** | `LassoCV` retención `CuS` \| Keras $57 \to 3$ \| Ablación `RNA_MLP_2` | **$24/24$ ($100\%$)** \| LOOCV **$11.64\%$** (*in-sample* MAPE $7.97\%$ PSJ / $7.38\%$ Canje) \| **$-1.53\%$** NRMSE ($13.81\% \to 12.28\%$) |
| **Ranking 18 Modelos (Bloque 4)** | Empate `1=` cuádruple \| `RNA_MLP_2` \| `Cancha_Directa` | **`1=`:** **$10.07\%$** ($\text{MSE}=0.020781/82$, $R^2=+0.6504/03$) \| **`RNA_MLP_2`:** **$12.28\%$** ($R^2=+0.4803$) \| **`Cancha`:** **$25.67\%$** ($R^2=-1.2709$) |
| **Holdout 19/5 y Buffered** | `RNA_MLP_2` vs. `Baseline 1B` en *Holdout* 19/5 \| *Buffered* | **Holdout:** `RNA_MLP_2` **$6.56\%$** ($R^2=+0.6549$) vs. `B1B` **$8.96\%$** \| **Buffered:** `Res_MLP` **$10.04\%$**, `RNA_MLP_2` **$14.53\%$** |
| **Inferencia Residuos (Bloque 4)** | Breusch–Pagan \| Durbin–Watson \| Wilcoxon vs. Cancha / B1B | **BP:** `B1B` $0.4075$, `RNA_MLP_2` $0.0822$ vs. `OLS` $0.0016$ \| **DW:** $2.014$ / $1.958$ \| **Wilcoxon:** $p \le 0.000186$; `RNA_MLP_2` vence **$14/24$** ($58.3\%$) |
| **XAI y Simulador (Bloque 4)** | Permutación \| SHAP `RNA_MLP_2` \| Lote Convex Hull `[1.75, 0.14]` | **Perm:** CuT $+0.1256$, CuS $+0.1003$ \| **SHAP:** CuT $0.0922$, CuS $0.0701$ (`Res_MLP` CuS $0.0000$) \| **Simulador:** **`1.4076 %Cu`** (`RNA_MLP_2`) / **`1.4096 %Cu`** (`B1B`/`Res`) |

---
*Glosario Técnico-Académico Maestro — Departamento de Ingeniería en Minas, Universidad de Antofagasta (Metodología v4.0).*
