# INFORME TÉCNICO ENTREGABLE Y GUÍA MAESTRA DE ANÁLISIS Y DEFENSA ORAL
## Síntesis Integral de los Bloques 0 al 4 y Plataforma Universal `Código unificado.ipynb`

---

<div align="center">

### **UNIVERSIDAD DE ANTOFAGASTA**
**Facultad de Ingeniería — Departamento de Ingeniería en Minas**

**TRABAJO DE TITULACIÓN PARA OPTAR AL TÍTULO DE INGENIERO CIVIL EN MINAS**

# *"Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas"*

</div>

| Campo Institucional | Especificación Oficial Certificada |
| :--- | :--- |
| **Tesista / Autor** | **Felipe Andrés Guerra Maturana** |
| **Profesor Guía** | **Dr. José Delgado Vega** (*Doctor en Geología, Especialista en Geoestadística y Explotaciones Mineras, École des Mines de Paris*) |
| **Institución** | Universidad de Antofagasta — Facultad de Ingeniería, Departamento de Ingeniería en Minas |
| **Caso de Estudio** | Faena Mantos Cobrizo (Distrito Minero Los Cóndores) $\to$ ENAMI (Planta San José) |
| **Alcance Consolidado** | **Bloque 0** (ETL y Tarifas) · **Bloque 1** (EDA y Econometría) · **Bloque 2** (7 *Baselines* LOOCV) · **Bloque 3** (11 Redes Neuronales) · **Bloque 4** (Benchmarking 18 Modelos, Inferencia y XAI) |
| **Plataforma de Código** | `Código/Código unificado.ipynb` (Modo Universal Google Colab / Jupyter Local) + `Código/traductor.md` |
| **Estándar Metodológico** | **Metodología v4.0** (Línea Temporal Estricta, *Zero Data Leakage*, Inferencia Exacta y Auditoría Experto-Crítica RCI Sellada) |
| **Estado de Certificación** | **Bloques 0 al 4 Auditados, Reconciliados y Sellados** (Habilitado para Bloques 5 y 6) |

---

## Guía de Uso de este Documento

Este informe técnico cumple un **doble propósito estratégico** para Felipe Andrés Guerra Maturana:

1. **Informe Técnico Entregable Consolidado (Bloques 0 al 4):** Reúne, sintetiza y audita en un solo documento de rigor académico la totalidad de los fundamentos físicos, protocolos experimentales, formulaciones matemáticas, tablas maestras, pruebas de hipótesis y artefactos serializados desarrollados en `BLOQUE_0/` a `BLOQUE_4/` y unificados en `Código/Código unificado.ipynb`. Todas las cifras coinciden al cuarto y sexto decimal con las tablas oficiales selladas bajo Metodología v4.0.
2. **Guía Maestra de Estudio y Defensa Oral (de Básico a Avanzado):** Explica el **porqué** de cada decisión —desde conceptos básicos (¿qué es el MSE?, ¿por qué $R^2$ puede ser negativo?) hasta fundamentos avanzados (contracción $L_2$ en *Residual-MLP*, colapso matricial en Keras, optimización cuasi-Newton `L-BFGS` frente a `Adam`, heterocedasticidad de Breusch-Pagan, Teoría de Muestreo de Pierre Gy y *Convex Hull* con triangulación de Delaunay)— culminando con un banco de **18 preguntas críticas reales de defensa oral** orientadas a responder con solvencia al **Dr. José Delgado Vega** y a la comisión evaluadora.

---

## Resumen Ejecutivo Integral (Bloques 0 al 4)

Durante el bienio **enero 2023 – diciembre 2024**, la faena **Mantos Cobrizo** despachó **$295{,}884.67\text{ t}$** (peso oficial en báscula puente `Romana_Ton`, frente a **$286{,}704\text{ t}$** estimadas en `Cancha_Ton`) hacia **Planta San José de ENAMI**, consolidados en **$N = 24$ balances mensuales oficiales**. El problema industrial central que origina esta investigación es una **brecha de conciliación Mina–Planta persistente**: en el **$100\%$ de los meses ($24/24$)**, la ley de cobre total muestreada en cancha de acopio (`Cancha_CuT`) sobreestimó sistemáticamente a la ley oficial de liquidación comercial (`Canje_CuT`), con un sesgo medio de **$+0.3404\%\text{Cu}$** (rango entre $+0.06\%\text{Cu}$ y $+0.61\%\text{Cu}$), un error cuadrático medio de **$\text{MSE} = 0.134971\,(\%\text{Cu})^2$**, un error relativo **$\text{NRMSE} = 25.67\%$** (reconciliando con exactitud la cifra histórica de $\text{MECM} \approx 26\%$ planteada por el profesor guía) y un coeficiente de determinación fuertemente negativo de **$R^2 = -1.2709$**.

A lo largo de cinco módulos secuenciales sin dependencias circulares (Bloques 0 al 4), la investigación construyó una respuesta científica, auditable y reproducible cuyos hallazgos troncales se resumen en la siguiente matriz ejecutiva:

```
+=======================================================================================================+
|                  SÍNTESIS EJECUTIVA INTEGRAL — BLOQUES 0 AL 4 (METODOLOGÍA v4.0)                      |
+=======================================================================================================+
| BLOQUE 0 — AUDITORÍA ETL Y MARCO COMERCIAL ENAMI                                                      |
|  • Integridad: N = 24 meses (0 nulos, 0 duplicados), leyes homologadas en %Cu a 2 decimales.          |
|  • Tonelajes: Cancha_Ton = 286,704 t | Romana_Ton = 295,884.67 t (12,328.53 t/mes). Exclusivos B5.    |
|  • Tarifas ENAMI Sulfuros: 24 circulares homologadas 100% USD/tms; maquila implícita exacta en        |
|    34.66 USD/tms (ene-23), 35.65 USD/tms (feb-23 a ene-24) y 39.60 USD/tms (feb-24 a dic-24).         |
|  • Física (Pierre Gy): Canje_CuT se acuerda en recepción (3 sobres) ANTES de molienda y flotación.    |
|    La brecha nace en cancha por friabilidad diferencial y segregación de finos de óxidos (CuS).       |
+-------------------------------------------------------------------------------------------------------+
| BLOQUE 1 — DIAGNÓSTICO ESTADÍSTICO, ECONOMETRÍA Y SELECCIÓN DE VARIABLES                              |
|  • Descomposición MSE (0.1350): 85.86% sesgo sistemático (0.1159) + 14.14% varianza empírica (0.0191).|
|  • Escalas: s²(CuT) / s²(CuS) = 61.62x (61.64x sobre s² redondeado a 6 dec.) -> StandardScaler oblig. |
|  • Normalidad: Leyes brutas NO normales (p < 0.05), pero Sesgo_Canje SÍ es normal (W=0.9852, p=0.9695)|
|  • Outliers (Mahalanobis D² > 5.9915): Oct-23 (14.2481) y Nov-23 (8.4700) RETENIDOS (lotes reales).   |
|  • Apalancamiento (N=24 vs N=22): Pearson CuT cae 0.8347 -> 0.6181; CuS cae 0.6406 -> 0.3688 (p=0.09).|
|  • ACF (K=6): Sesgo es ruido blanco (r1 = -0.0866, valida LOOCV); CuT inercia r1=+0.4467 (Buffered).  |
|  • VIF: [CuT, CuS] = 1.8045 (< 5.0 óptimo) vs > 4,000 al incluir CuIn -> espacio sellado en 2D.       |
+-------------------------------------------------------------------------------------------------------+
| BLOQUES 2 Y 3 — MODELADO CIEGO LOOCV (7 BASELINES + 11 REDES NEURONALES = 18 MODELOS)                 |
|  • Protocolo: LOOCV (N_train=23) + Pipeline (cero fuga) + Nested CV (K=5) + ensamble 10 semillas.     |
|  • LassoCV (L1): Retiene Cancha_CuS en 24/24 pliegues (100%, rango [+0.0124, +0.0627]).               |
|  • Réplica Dr. Delgado Keras (4,4,4): 57 params nominales colapsan linealmente a P_eff = 3; su ~8.63% |
|    histórico era MAPE in-sample (7.97% PSJ / 7.38% Canje) y sube a 11.64% NRMSE en LOOCV ciego.       |
|  • Ablación 2D vs 1D: Incluir CuS mejora a RNA_MLP_2 en -1.53% NRMSE (13.81% -> 12.28%; R² +0.1376).  |
|  • Holdout Prospectivo 19/5: RNA_MLP_2 logra 6.56% NRMSE, venciendo a RidgeCV (7.57%), OLS (7.76%)    |
|    y Baseline 1B / Res_MLP (8.96%).                                                                   |
+-------------------------------------------------------------------------------------------------------+
| BLOQUE 4 — BENCHMARKING UNIFICADO, INFERENCIA EXACTA, ESTRATEGIA DUAL Y XAI                           |
|  • Empate Cuádruple 1= (Mínima Varianza): Res_MLP_2_2, Baseline_1B, Res_MLP_4, Res_MLP_3 (10.07%).    |
|  • Diagnóstico Econométrico: TODOS los modelos lineales fallan Breusch-Pagan (heterocedásticos,       |
|    p <= 0.0228); Baseline_1B (p=0.4075) y RNA_MLP_2 (p=0.0822) preservan homocedasticidad y DW ~ 2.0. |
|  • Wilcoxon exacto vs Cancha: p = 0.000010 (B1B/Res_MLP_2_2) y p = 0.000186 (RNA_MLP_2).              |
|  • RNA_MLP_2 vs Baseline_1B: Vence en 14/24 meses (58.3%), menor error mediano (0.0801 vs 0.1044%Cu). |
|  • Estrategia Dual: Campeón Presupuestario (Baseline 1B / Res_MLP_2_2, P_eff=1) + Campeón XAI         |
|    (RNA_MLP_2, 9 params, SHAP CuT=0.0922 / CuS=0.0701, Permutación +0.1256 / +0.1003, Convex Hull).   |
|  • Código Universal y Simulador: Lote [1.75, 0.14] (dentro Convex Hull) -> RNA_MLP_2 = 1.4076 %Cu.    |
+=======================================================================================================+
```

### Entorno Computacional Certificado (Reproducibilidad Estándar UA)

Para asegurar determinismo numérico exacto en álgebra lineal (`float64` con alineamiento `np.asfortranarray`) tanto en ejecución local como en Google Colab, todo el ecosistema de software y hardware quedó documentado desde el Bloque 0 y verificado en la Celda 1 de `Código/Código unificado.ipynb`:

| Componente | Especificación Local Certificada | Rol en el Flujo Científico |
| :--- | :--- | :--- |
| **Sistema Operativo / Hardware** | Windows 11 (Build 10.0.26200-SP0), AMD64, 7.34 GB RAM | Entorno de desarrollo, auditoría y ejecución determinista. |
| **Intérprete y Núcleo Numérico** | `Python 3.12.6`, `numpy 2.4.0`, `pandas 2.3.3`, `openpyxl 3.1.5` | Ingesta ETL, homologación decimal y álgebra matricial. |
| **Estadística y Econometría** | `scipy 1.17.1`, `statsmodels 0.15.0` | Shapiro-Wilk, Wilcoxon exacto, Convex Hull/Delaunay, VIF, ACF, Breusch-Pagan, Durbin-Watson. |
| **Machine Learning y Deep Learning** | `scikit-learn 1.8.0`, `TensorFlow / Keras 2.21.0`, `joblib 1.5.3` | Pipelines sin fuga, Baselines, MLP `L-BFGS`, réplica Keras y serialización nativa `.joblib`. |
| **Explicabilidad e Interfaz** | `shap 0.54.1`, `matplotlib 3.10.8`, `seaborn 0.13.2`, `plotly 7.1.0`, `streamlit 1.67.0` | Valores Shapley exactos, figuras a 200 DPI y preparación para despliegue del Bloque 6. |

---

# CAPÍTULO 1
## Fundamento del Problema, Auditoría de Datos (Bloque 0) y Diagnóstico Estadístico (Bloque 1)

### 1.1. Cadena Operacional Mantos Cobrizo $\to$ ENAMI Planta San José y Auditoría ETL

La faena **Mantos Cobrizo** (ubicada en el Distrito Minero Los Cóndores) explota un yacimiento de cobre con **zonación supergena**, donde coexisten sulfuros primarios y de enriquecimiento secundario (calcopirita, bornita, calcocina) junto con horizontes de alteración oxidada (crisocola, malaquita). El mineral extraído se acopia en la **cancha de mina**, donde se muestrea para estimar las leyes internas (`Cancha_CuT`, `Cancha_CuS`, `Cancha_CuIn`), y posteriormente es transportado por camiones hacia la **Planta San José de ENAMI** para su compra como mineral de concentración (sulfuros).

A partir del registro histórico maestro `MATERIAL_TESIS/Data Cobrizo.xlsx`, el Bloque 0 certificó el dataset auditado `BLOQUE_0/DATASET_AUDITADO_N24.csv` con **$N = 24$ balances mensuales continuos** (enero 2023 a diciembre 2024):

1. **Integridad absoluta:** $0$ valores faltantes y $0$ registros duplicados. Todas las leyes fueron homologadas desde fracción decimal (p. ej. `0.0216`) a **puntos porcentuales ($\%\text{Cu}$)** con formato uniforme a exactamente dos decimales (`2.16`).
2. **Tonelajes acumulados y regla de exclusión en modelado:**
   * **Tonelaje acopiado en faena (`Cancha_Ton`):** **$286{,}704\text{ t}$** (valor entero estimado en acopio).
   * **Tonelaje oficial de báscula puente (`Romana_Ton`):** **$295{,}884.67\text{ t}$** (calculado dinámicamente mediante `df['Romana_Ton'].sum()`, equivalente a un despacho promedio de **$12{,}328.53\text{ t/mes}$**).
   * **Regla metodológica estricta:** En los **Bloques 1 al 4**, los tonelajes **NO ingresan** como predictores de la ley de canje; todo el EDA y los 18 algoritmos operan estrictamente sobre el espacio de leyes $X = [\text{Cancha\_CuT}, \text{Cancha\_CuS}] \in \mathbb{R}^{24 \times 2} \to y = \text{Canje\_CuT} \in \mathbb{R}^{24 \times 1}$. El tonelaje `Romana_Ton` se reserva exclusivamente para el **Bloque 5** (valorización comercial en USD/tms) y cumple un rol probatorio clave ante la comisión evaluadora: demostrar que cada uno de los $N = 24$ puntos representa un **balance industrial mensual de $\sim 12.3\text{ kt}$** y no una muestra puntual de laboratorio.
3. **Auditoría de balance químico ($\text{CuT} = \text{CuS} + \text{CuIn}$, Hallazgo RCI B0-02):**
   * En **Cancha**, la conservación de masa se satisface con exactitud: $\max |\text{Cancha\_CuT} - (\text{Cancha\_CuS} + \text{Cancha\_CuIn})| = 0.01\%\text{Cu}$, atribuible íntegramente al redondeo analítico a dos decimales.
   * En **Canje**, durante 2024 se observó una discrepancia de hasta $0.20\%\text{Cu}$ donde `Canje_CuS + Canje_CuIn` supera sistemáticamente a `Canje_CuT` entre $+0.06\%$ y $+0.20\%\text{Cu}$. La Auditoría RCI (B0-02) descartó que se tratara de redondeo y dictaminó que corresponde a una **inconsistencia de registro de origen en las columnas auxiliares (`Canje_CuS`, `Canje_CuIn`) del Excel histórico de faena**. El modelado y la valorización emplean exclusivamente la ley oficial certificada de liquidación **`Canje_CuT`**.

#### Tabla 1.1: Síntesis de los 24 Balances Mensuales Auditados (`BLOQUE_0/DATASET_AUDITADO_N24.csv`)

| Periodo | Cancha Ton ($\text{t}$) | Romana Ton ($\text{t}$) | Cancha CuT ($\%$) | Cancha CuS ($\%$) | Cancha CuIn ($\%$) | PSJ CuT ($\%$) | Canje CuT ($\%$) | Sesgo Cancha–Canje ($\%\text{Cu}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2023-01** | 11,559 | 12,017.45 | 2.16 | 0.17 | 1.99 | 1.41 | **1.55** | **+0.61** *(Máx.)* |
| **2023-02** | 9,972 | 10,109.95 | 1.76 | 0.14 | 1.62 | 1.33 | **1.41** | +0.35 |
| **2023-03** | 12,765 | 12,533.33 | 1.73 | 0.16 | 1.57 | 1.55 | **1.49** | +0.24 |
| **2023-04** | 10,530 | 11,066.41 | 1.66 | 0.14 | 1.52 | 1.38 | **1.38** | +0.28 |
| **2023-05** | 11,167 | 12,564.00 | 1.80 | 0.13 | 1.68 | 1.26 | **1.24** | +0.56 |
| **2023-06** | 12,393 | 13,127.71 | 1.77 | 0.13 | 1.64 | 1.43 | **1.34** | +0.43 |
| **2023-07** | 11,349 | 12,044.92 | 1.75 | 0.13 | 1.63 | 1.41 | **1.37** | +0.38 |
| **2023-08** | 11,208 | 11,989.45 | 1.74 | 0.12 | 1.62 | 1.54 | **1.52** | +0.22 |
| **2023-09** | 11,804 | 12,012.89 | 2.01 | 0.12 | 1.89 | 1.66 | **1.55** | +0.46 |
| **2023-10** | 10,716 | 12,704.63 | **2.21** | **0.26** | 1.95 | 1.77 | **1.79** | +0.42 *(Outlier óxidos)* |
| **2023-11** | 11,170 | 12,235.36 | **2.47** | **0.20** | 2.27 | 2.26 | **2.27** | +0.20 *(Outlier rico)* |
| **2023-12** | 11,194 | 12,216.20 | 1.72 | 0.12 | 1.60 | 1.37 | **1.35** | +0.37 |
| **2024-01** | 8,538 | 8,561.70 | 1.49 | 0.12 | 1.37 | 1.37 | **1.41** | +0.08 |
| **2024-02** | 7,102 | 7,108.08 | 1.61 | 0.14 | 1.47 | 1.37 | **1.37** | +0.24 |
| **2024-03** | 10,040 | 10,043.00 | 1.56 | 0.15 | 1.41 | 1.19 | **1.19** | +0.37 |
| **2024-04** | 11,374 | 11,379.57 | 1.62 | 0.14 | 1.48 | 1.15 | **1.15** | +0.47 |
| **2024-05** | 11,342 | 11,344.76 | 1.58 | 0.13 | 1.45 | 1.14 | **1.14** | +0.44 |
| **2024-06** | 12,751 | 12,751.34 | 1.76 | 0.15 | 1.61 | 1.68 | **1.70** | **+0.06** *(Mín.)* |
| **2024-07** | 12,468 | 12,463.09 | 1.84 | 0.16 | 1.68 | 1.57 | **1.57** | +0.27 |
| **2024-08** | 13,169 | 13,167.57 | 1.95 | 0.15 | 1.80 | 1.41 | **1.43** | +0.52 |
| **2024-09** | 14,172 | 14,177.97 | 1.64 | 0.16 | 1.48 | 1.46 | **1.48** | +0.16 |
| **2024-10** | 16,226 | 16,559.48 | 1.66 | 0.14 | 1.51 | 1.27 | **1.27** | +0.39 |
| **2024-11** | 17,142 | 17,147.09 | 1.62 | 0.15 | 1.47 | 1.31 | **1.32** | +0.30 |
| **2024-12** | 16,553 | 16,558.72 | 1.41 | 0.12 | 1.29 | 1.04 | **1.06** | +0.35 |
| **Total / Media** | **286,704 t** | **295,884.67 t** | **1.7717** | **0.1471** | **1.6250** | **1.4304** | **1.4313** | **+0.3404 (24/24 > 0)** |

### 1.2. Marco Tarifario Oficial ENAMI Sulfuros (Homologación 100% USD y Maquila Implícita)

Las 24 circulares comerciales mensuales de ENAMI para **Minerales de Concentración (Sulfuros — tarifa de flotación)** fueron estructuradas en `BLOQUE_0/TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.csv`. La ecuación oficial de valorización neta por tonelada métrica seca (tms) es:

$$\text{Tarifa Unitaria (USD/tms)} = \text{Tarifa Base}_{2.5\%} - \text{Escala}_{1\%} \times (2.5 - \text{Ley}_{\text{Canje}}) - \text{Cargos Reglamentarios}$$

$$\text{Valor Lote (USD)} = \text{Romana\_Ton (tms)} \times \text{Tarifa Unitaria (USD/tms)}$$

* **Nota comercial (Auditoría B0-03):** El término `Cargos Reglamentarios` integra el descuento del **Fondo de Estabilización de Precios** (`Flot_Fondo_Estab_USD_tms_1pct`), el cual opera como una retención por punto porcentual de ley cuando el precio internacional LME supera el precio de sustentación vigente (ej. febrero–mayo 2023 y mayo–diciembre 2024).
* **Auditoría de Homologación Monetaria (Hallazgo Crítico B0-01 resuelto):** En las circulares PDF originales de ENAMI, **22 de los 24 meses** figuraban expresados en miles de pesos chilenos (`MCLP/tms`), mientras que únicamente **abril de 2023** (`141.290 USD/tms`) y **mayo de 2024** (`152.814 USD/tms`) se emitieron directamente en dólares (`USD/tms`). Al aplicar la conversión exacta $\text{Valor}_{\text{USD}} = (\text{Valor}_{\text{MCLP}} \times 1{,}000) / \text{TC}$ a los 22 meses en CLP, la consistencia financiera queda certificada matemáticamente mediante la identidad de la **Maquila Implícita de Tratamiento**:

$$\text{Maquila Implícita (USD/tms)} = 2.5 \times \text{Escala}_{1\%} - \text{Tarifa Base}_{2.5\%}$$

Al evaluar esta identidad sobre toda la tabla homologada, los 24 meses convergen sin una sola anomalía a exactamente los tres escalones oficiales de maquila vigentes en el bienio:

| Vigencia del Escalón Tarifario | Meses Cubiertos | Maquila Implícita Certificada | Rango Precio LME (¢USD/lb) | Rango Tarifa Base 2.5% (USD/tms) |
| :--- | :---: | :---: | :---: | :---: |
| **Enero 2023** | 1 mes | **34.66 USD/tms** | 379.530 | 134.406 |
| **Febrero 2023 – Enero 2024** | 12 meses | **35.65 USD/tms** | 360.137 – 408.220 | 121.684 – 144.880 |
| **Febrero 2024 – Diciembre 2024** | 11 meses | **39.60 USD/tms** | 376.969 – 459.447 | 127.177 – 166.963 |

### 1.3. Brecha Histórica Operacional y Física Correcta según Pierre Gy (Protocolo de 3 Sobres)

El diagnóstico del estado actual sin calibración (`Cancha_Directa`: $\hat{y}_k = \text{Cancha\_CuT}_k$) arroja:
* **Sesgo sistemático medio:** $+0.3404\%\text{Cu}$ (sobreestimación en 24 de 24 meses).
* **Error Cuadrático Medio y Raíz:** $\text{MSE} = 0.134971\,(\%\text{Cu})^2$, $\text{RMSE} = 0.3674\%\text{Cu}$, $\text{MAE} = 0.3404\%\text{Cu}$.
* **Error Relativo (Reconciliación con el Dr. Delgado):** $\text{NRMSE} = \frac{0.3674}{1.43125}\times 100\% = \mathbf{25.67\%}$ sobre `Canje_CuT` ($25.98\%$ sobre `PSJ_CuT`, con $\text{MAPE} = 24.90\%$ en Canje y $25.05\%$ en PSJ), demostrando que el $\text{MECM} \approx 26\%$ de la lámina inicial del profesor guía corresponde al NRMSE histórico.
* **Ajuste respecto a la identidad $y = \hat{y}$:** **$R^2 = -1.2709$** (desempeño sustancialmente peor que predecir siempre el promedio histórico $\bar{y} = 1.4313\%\text{Cu}$).

> **⚠️ PRECISIÓN METALÚRGICA CRÍTICA (AUDITORÍAS C-01 Y B3-01) — POR QUÉ `Canje_CuT` NO DEPENDE DE FLOTACIÓN:**  
> En la relación comercial entre un productor minero y ENAMI (Planta San José), **`Canje_CuT` es la ley de cabeza química acordada en recepción mediante el protocolo reglamentario de tres sobres** (sobre del productor, sobre de ENAMI y sobre de dirimencia o arbitraje externo si la discrepancia excede la tolerancia reglamentaria). Este ensayo se ejecuta **en la etapa de recepción y muestreo mecánico ANTES de que el lote ingrese a molienda y flotación**.  
> En consecuencia, es un **error metalúrgico grave** atribuir la brecha entre `Cancha_CuT` y `Canje_CuT` a cinéticas lentas de flotación, acción de reactivos colectores (xantatos) o pérdidas en relaves de celdas concentradoras.

**¿Cuál es entonces el mecanismo físico-mineralógico real que vincula `Cancha_CuS` con la brecha hacia `Canje_CuT`?**  
La respuesta reside en la **Teoría de Muestreo de Materiales Particulados de Pierre Gy (1998)**, a través de la interacción entre **Heterogeneidad Constitucional** y **Heterogeneidad de Distribución**:

1. **Friabilidad diferencial y menor dureza de Mohs:** Los minerales oxidados supergenos cuantificados por `Cancha_CuS` —principalmente **crisocola** (dureza Mohs $2.0\text{–}4.0$, frágil y terrosa) y **malaquita** (dureza Mohs $3.5\text{–}4.0$)— son mecánicamente mucho más blandos y friables que los sulfuros competentes encajados en roca huésped (calcopirita, bornita) y la ganga silícea.
2. **Pulverización preferencial y segregación gravitacional de finos:** Durante la tronadura, el carguío, el chancado secundario y la formación del **cono de acopio en cancha de mina**, las especies oxidadas se pulverizan en mayor proporción que la roca competente, generando una fracción fina enriquecida que sufre **segregación gravitacional** y pérdidas eólicas o de manipulación al cargar camiones.
3. **Asimetría de equipos de muestreo (Cancha vs. Torre ENAMI):** El muestreo manual o semimecanizado en superficie/periferia del acopio de faena tiende a sobrerrepresentar la fracción fina enriquecida (introduciendo **Error de Delimitación DE**, **Error de Extracción EE** y **Error de Segregación GE** en la taxonomía de Pierre Gy), mientras que en Planta San José el lote descargado pasa por un circuito de reducción granulométrica controlada y una **torre de muestreo mecánico automático** diseñada para cortar el flujo completo con equiprobabilidad. Por eso, `Cancha_CuS` actúa como un **marcador geoquímico de la susceptibilidad a la friabilidad y segregación** del frente explotado.

### 1.4. Diagnóstico Estadístico y Econométrico del Bloque 1 ($N = 24$)

#### A. Descomposición Exacta Sesgo-Varianza y Formulación de la Hipótesis Predictiva

El error cuadrático medio histórico ($\text{MSE} = 0.134971\,(\%\text{Cu})^2$) admite una descomposición algebraica exacta empleando varianza poblacional ($\text{ddof} = 0$):

$$\text{MSE}_{\text{hist}} = \bar{e}^2 + \sigma^2_{\text{error}} = (0.340417)^2 + 0.019087 = 0.115884 + 0.019087 = \mathbf{0.134971\,(\%\text{Cu})^2}$$

| Componente del Error Histórico | Magnitud en $(\%\text{Cu})^2$ | Porcentaje del MSE | Desviación Equivalente | Interpretación Operacional y de Investigación |
| :--- | :---: | :---: | :---: | :--- |
| **Sesgo Sistemático al Cuadrado ($\bar{e}^2$)** | `0.115884` | **$85.86\%$** | $\bar{e} = +0.3404\%\text{Cu}$ | Descalibración estructural constante entre muestreo de cancha y torre ENAMI. |
| **Varianza de la Discrepancia ($\sigma^2_{\text{error}}$)** | `0.019087` | **$14.14\%$** | $\sigma_e = 0.1382\%\text{Cu}$ ($s_e = 0.1411$) | Combina **señal geometalúrgica explicable** ($\text{CuT}$, $\text{CuS}$) + **ruido analítico irreducible** ($\pm 0.04\text{ a }\pm 0.06\%\text{Cu}$). |
| **Error Cuadrático Medio Total (MSE)** | **`0.134971`** | **$100.00\%$** | $\text{RMSE} = 0.3674\%\text{Cu}$ | **NRMSE = $25.67\%$** ($R^2 = -1.2709$). |

**La trampa del piso trivial ($9.66\%$):** Si simplemente se restara la constante fija histórica *in-sample* ($\hat{y} = \text{Cancha\_CuT} - 0.3404$), el error bajaría inmediatamente de $25.67\%$ a $\text{NRMSE}_{\text{trivial}} = \frac{0.1382}{1.43125}\times 100\% = \mathbf{9.66\%}$ (sin ninguna red neuronal!). En validación ciega LOOCV ($N_{\text{train}}=23$), esa corrección empírica define el **`Baseline_1B` ($\text{NRMSE} = 10.07\%$)**, que constituye la **verdadera valla científica** del estudio. Sobre esta base se formuló la hipótesis de trabajo: evaluar con rigor si una red neuronal parsimoniosa extrae estructura geometalúrgica no lineal del $14.14\%$ de varianza residual para aproximarse al umbral referencial histórico del profesor guía ($\approx 8.63\%$).

#### B. Descriptivos, Contraste de Varianzas ($61.6\times$) y Normalidad (Shapiro-Wilk)

#### Tabla 1.2: Estadísticos Descriptivos, Dispersión y Normalidad (`BLOQUE_1/TABLA_ESTADISTICOS_EDA_N24.csv`)

| Variable ($N = 24$) | Media ($\mu$) | Mediana ($\tilde{x}$) | Desv. Muestral ($s$) | Var. Muestral ($s^2$) | CV ($\%$) | Asimetría ($g_1$) | Curtosis ($g_2$) | Shapiro-Wilk ($W$) | $p$-valor | Normalidad ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cancha_CuT ($\%$)** | 1.7717 | 1.7350 | 0.2411 | **0.058128** | 13.61 | +1.2682 | +1.4426 | 0.8871 | 0.0116 | **RECHAZADA** |
| **Cancha_CuS ($\%$)** | 0.1471 | 0.1400 | 0.0307 | **0.000943** | **20.88** | +2.2772 | **+5.8621** | 0.7472 | 0.00003 | **RECHAZADA** |
| **Cancha_CuIn ($\%$)** | 1.6250 | 1.6050 | 0.2221 | 0.049322 | 13.67 | +1.1747 | +1.2916 | 0.9048 | 0.0272 | **RECHAZADA** |
| **PSJ_CuT ($\%$)** | 1.4304 | 1.3950 | 0.2496 | 0.062282 | 17.45 | +1.4598 | +3.2695 | 0.8871 | 0.0116 | **RECHAZADA** |
| **Canje_CuT ($\%$)** | 1.4313 | 1.3950 | 0.2490 | 0.062020 | 17.40 | +1.5414 | +3.5543 | 0.8787 | 0.0078 | **RECHAZADA** |
| **Sesgo_Canje ($\%$)** | **+0.3404** | +0.3600 | 0.1411 | **0.019917** | 41.46 | -0.1652 | -0.4804 | **0.9852** | **0.9695** | **ACEPTADA** |
| **Ratio_Solubilidad ($\%$)** | 8.3047 | 8.3308 | 1.1609 | 1.347680 | 13.98 | +0.7824 | +1.7998 | **0.9463** | **0.2251** | **ACEPTADA** |

* **Por qué `StandardScaler` es obligatorio (Ratio de Varianzas $61.6\times$):**
  $$\text{Ratio de Varianzas} = \frac{s^2_{\text{Cancha\_CuT}}}{s^2_{\text{Cancha\_CuS}}} = \frac{0.05812754}{0.00094330} = \mathbf{61.62\times} \quad \left(\frac{0.058128}{0.000943} = 61.64\times \text{ con } s^2 \text{ a 6 dec.}\right)$$
  Sin estandarización, `Cancha_CuT` domina los gradientes y las normas $L_2$ en más de 61 veces respecto a `Cancha_CuS`, anulando en la práctica la capacidad de aprendizaje de la segunda neurona pese a que `Cancha_CuS` es proporcionalmente **más volátil** ($\text{CV} = 20.88\%$ vs. $13.61\%$). Dado que la curtosis de `CuS` es fuertemente leptocúrtica ($+5.8621$, impulsada por octubre 2023 a $+3.68\sigma$), en el Bloque 2 se incluyó además un test formal frente a `RobustScaler` (mediana e IQR).
* **Rigor Econométrico (Normalidad y Teorema de Gauss-Markov):**
  Aunque las leyes brutas rechazan normalidad por asimetría geológica positiva ($p < 0.05$), el **Teorema de Gauss-Markov** no exige normalidad marginal en las variables explicativas $X$ ni en $Y$; sus supuestos recaen sobre los **residuos condicionales** $\varepsilon \mid X$, los cuales en el sesgo empírico de cancha aceptan normalidad con máxima holgura (**$W = 0.9852, p = 0.9695$**, unificado exactamente con `Cancha_Directa` en Bloque 4 tras Auditoría B1-01). No obstante, las colas pesadas en $X$ advierten sobre **alto apalancamiento (*leverage*)**, justificando la regularización $L_2$, los contrastes no paramétricos (Spearman, Wilcoxon) y los modelos no lineales.

#### C. Matriz de Covarianza, Apalancamiento ($N=24$ vs. $N=22$) y Outliers de Mahalanobis

La matriz de covarianza muestral ($\text{ddof}=1$, en $(\%\text{Cu})^2$) satisface exactamente la identidad algebraica de la varianza de la diferencia:

$$\text{Var}(\text{CuT}) + \text{Var}(\text{Canje}) - 2\,\text{Cov}(\text{CuT}, \text{Canje}) = 0.058128 + 0.062020 - 2(0.050115) = \mathbf{0.019918} \approx 0.019917 = \text{Var}(\text{Sesgo})$$

Para auditar la geometría bivariada de $X = [\text{Cancha\_CuT}, \text{Cancha\_CuS}]$, se calculó la **Distancia de Mahalanobis** $D_M^2 = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$ comparada frente al umbral diagnóstico heurístico $\chi^2_{2, 0.95} = 5.9915$:

* **2023-10 ($\text{CuT} = 2.21\%$, $\text{CuS} = 0.26\%$, $\text{Canje} = 1.79\%$):** **$D_M^2 = 14.2481$** ($p = 0.0008$). Frente de enriquecimiento supergeno con pulso anómalo de óxidos (crisocola/malaquita).
* **2023-11 ($\text{CuT} = 2.47\%$, $\text{CuS} = 0.20\%$, $\text{Canje} = 2.27\%$):** **$D_M^2 = 8.4700$** ($p = 0.0145$). Frente competente de altísima ley sulfurada (bornita/calcocina).
* *Siguientes observaciones más alejadas (dentro de la elipse al $95\%$):* `2023-09` ($D_M^2 = 5.2671, p = 0.0718$) y `2023-01` ($D_M^2 = 2.7900, p = 0.2478$).
* **Política Geológica de Retención:** Ambos meses atípicos **se retienen íntegramente** ($N = 24$), porque representan lotes industriales reales facturados por ENAMI; eliminarlos sesgaría la herramienta hacia la esterilidad. Sin embargo, realizar un **análisis diagnóstico temporal** sin esos dos meses ($N = 22$) reveló un fenómeno crucial de apalancamiento:

| Predictor | Pearson $r$ ($N=24$) | Spearman $\rho$ ($N=24$) | Pearson $r$ ($N=22$, diag.) | $p$-valor ($N=22$) | Spearman $\rho$ ($N=22$, diag.) | Info. Mutua (MI) | Lectura Geometalúrgica |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cancha_CuT** | **+0.8347** | +0.7116 | **+0.6181** | 0.0022 | **+0.6253** | 0.2156 | En $N=24$, Oct y Nov inflan Pearson por sobre Spearman; al aislarlos ($N=22$), Spearman supera a Pearson. |
| **Cancha_CuS** | **+0.6406** | +0.5024 | **+0.3688** | **0.0912** | **+0.3520** ($p=0.1082$) | 0.1437 | En régimen regular ($N=22$), su correlación lineal no es significativa al $5\%$; su valor reside en modular interacciones no lineales en zonas oxidadas. |

*(Paradoja $r = +0.8347$ vs. $R^2 = -1.2709$: La correlación de Pearson mide sincronía lineal y es matemáticamente invariante ante traslaciones constantes, ignorando el desplazamiento de $+0.3404\%\text{Cu}$; en cambio, $R^2$ penaliza la distancia cuadrática exacta respecto a la identidad $y = \hat{y}$.)*

#### D. Autocorrelación Serial (ACF, Rezagos 1–6) y Multicolinealidad (VIF)

1. **ACF Acotada (Regla Box-Jenkins $K \le N/4 = 6$, bandas de Bartlett $\pm 1.96/\sqrt{24} = \pm 0.4001$):**
   * **`Sesgo_Canje` es ruido blanco puro en los 6 rezagos:** $r_1 = \mathbf{-0.0866}$, $r_2 = -0.2729$, $r_3 = -0.0625$, $r_4 = +0.0967$, $r_5 = +0.2610$, $r_6 = -0.1036$ (todos estrictamente dentro de $[-0.4001, +0.4001]$). Esto **valida formalmente el uso de LOOCV**, pues el error no presenta memoria temporal.
   * **Inercia de `Cancha_CuT` ($r_1 = +0.4467 > 0.4001$):** Refleja continuidad geológica espacial entre frentes contiguos de explotación y motiva incorporar la prueba complementaria **`Buffered-LOOCV`** en los Bloques 2 y 3.
   * **Nota técnica sobre Lag 6 en `Canje_CuT` ($r_6 = -0.4276$, Auditoría B1-04):** Excede marginalmente la banda por la transición operacional entre el pulso rico de fines de 2023 y los frentes de ley media-baja seis meses después (mediados de 2024), sin contagiar al error (`Sesgo_Canje` $r_6 = -0.1036$).
2. **Factor de Inflación de Varianza (VIF con intercepto obligatorio `sm.add_constant`):**
   * **Espacio Seleccionado 2D $[\text{Cancha\_CuT}, \text{Cancha\_CuS}]$:** **$\text{VIF} = 1.8045 < 5.0$** para ambas entradas ($\frac{1}{1 - 0.6677^2} = 1.8045$). Colinealidad baja, condicionamiento numérico óptimo.
   * **Espacio Rechazado 3D $[\text{CuT}, \text{CuS}, \text{CuIn}]$:** El VIF explota a **$4{,}754.0$** (`CuT`), **$83.2$** (`CuS`) y **$4{,}004.1$** (`CuIn`) por la identidad química exacta $\text{CuT} = \text{CuS} + \text{CuIn}$ (incluso el par $[\text{CuT}, \text{CuIn}]$ produce $\text{VIF} = 86.85$ al presentar $r = 0.994$). Por tanto, el espacio de entrada queda **blindado en exactamente 2 predictores**.

---

# CAPÍTULO 2
## Ingeniería del Modelado: Protocolo LOOCV, Baselines, Arquitecturas RNA y Decisiones de Hiperparámetros (Bloques 2 y 3)

### 2.1. Protocolo Experimental: LOOCV, `Pipeline` (Cero Fuga) y Validación Cruzada Anidada ($K=5$)

En minería a escala de balances mensuales, un tamaño muestral de **$N = 24$** (dos años completos de producción) impone restricciones estadísticas estrictas que prohíben prácticas ingenuas de *Big Data*:

1. **Por qué LOOCV ($N_{\text{train}} = 23$, $N_{\text{test}} = 1$, 24 iteraciones) y no un `train_test_split` 80/20 único:**  
   Una partición aleatoria única 80/20 dejaría solo 5 observaciones en prueba. Si por azar octubre o noviembre de 2023 caen en el conjunto de prueba, el error cambia drásticamente (alta varianza de partición); además, reduciría en $20\%$ la información geológica disponible para entrenar. En **Leave-One-Out Cross-Validation (LOOCV)**, cada mes $k \in \{1, \dots, 24\}$ se oculta por completo mientras el modelo se entrena con los $23$ meses restantes ($\mathcal{D}_{\text{train}}^{(-k)}$), obteniendo un vector completo de **24 predicciones 100% ciegas fuera de muestra** con máximo aprovechamiento de los datos ($95.8\%$ por pliegue) y sesgo de estimación mínimo.
2. **Encapsulamiento en `sklearn.pipeline.Pipeline` (*Zero Data Leakage*):**  
   Si se aplicara `StandardScaler().fit_transform(X)` antes de dividir los pliegues, la media y desviación estándar del mes de prueba $k$ se filtrarían silenciosamente al entrenamiento (*data leakage*). Al encapsular el escalador y el estimador dentro de `Pipeline`, los parámetros $(\mu_{j,-k}, \sigma_{j,-k})$ se computan **exclusivamente sobre los 23 meses de entrenamiento** y luego se proyectan sobre $X_k$ mediante `.transform()`.
3. **Validación Cruzada Anidada (*Nested CV*, $K=5$):**  
   En los modelos con búsqueda automática de hiperparámetros (`RidgeCV`, `LassoCV`, `ElasticNetCV` y `GridSearchCV` de `SVR_RBF`), la selección de $\alpha$, $l_1\text{-ratio}$, $C$, $\epsilon$ y $\gamma$ ocurre mediante un bucle interno de 5 pliegues **exclusivamente dentro de los 23 registros de entrenamiento**, sin ver jamás el mes de prueba exterior $k$.

### 2.2. Análisis Pedagógico de los 7 Modelos de Referencia (*Baselines*, Bloque 2)

Antes de construir redes neuronales, el Bloque 2 estableció 7 referencias ordenadas en complejidad creciente, evaluadas bajo idéntico LOOCV:

1. **`Cancha_Directa` (Línea Base 1A):** $\hat{y}_k = \text{Cancha\_CuT}_k$. Sin calibración. Registra $\text{NRMSE} = 25.67\%$, $\text{RMSE} = 0.3674\%\text{Cu}$, $R^2 = -1.2709$. Es una **"falsa valla"** fácil de batir porque el $85.86\%$ de su error es un desplazamiento constante.
2. **`Baseline_1B` (Corrección de Sesgo Constante LOOCV):** En cada pliegue calcula $\bar{e}_{\text{train}, -k} = \frac{1}{23}\sum_{i\ne k}(\text{CuT}_i - \text{Canje}_i) \in [0.3287, 0.3526]\%\text{Cu}$ y predice $\hat{y}_k = \text{Cancha\_CuT}_k - \bar{e}_{\text{train}, -k}$. Logra **$\text{NRMSE} = 10.07\%$**, $\text{MSE} = 0.020782\,(\%\text{Cu})^2$, $\text{RMSE} = 0.1442\%\text{Cu}$, $\text{MAE} = 0.1178\%\text{Cu}$, $R^2 = +0.6503$.
   * **¿Por qué un modelo de 1 parámetro vence a `OLS` ($11.85\%$) y `RidgeCV` ($11.78\%$)?** Porque impone una **pendiente física rígida $1:1$** respecto a `Cancha_CuT`. En contraste, `OLS` estima una pendiente achatada ($\approx 0.74 < 1.0$) arrastrada por los meses de leyes medias/bajas con sesgo alto; como resultado, el **$40.48\%$** del error cuadrático total de `OLS` se concentra en solo dos meses: **octubre 2023** (donde el plano lineal extrapola sobre el pico de `CuS = 0.26%` y sobreestima en `2.1184%` vs. real `1.79%`, error $+0.3284\%\text{Cu}$) y **noviembre 2023** (donde su pendiente baja subestima el frente rico en `1.8555%` vs. real `2.27%`, error $-0.4145\%\text{Cu}$, frente a `2.1235%` de `Baseline_1B`).
3. **`OLS` (Mínimos Cuadrados Ordinarios):** Hiperplano lineal sin penalización sobre $[Z_{\text{CuT}}, Z_{\text{CuS}}]$. Alcanza $\text{NRMSE} = 11.85\%$, $R^2 = +0.5156$, con coeficientes estandarizados medios exactos en `LOG_HIPERPARAMETROS_BASELINES.csv` de $\bar{\beta}_{\text{CuT}} = +0.1780$ y $\bar{\beta}_{\text{CuS}} = +0.0372$.
4. **`RidgeCV` (Regularización $L_2$ de Tikhonov):** Minimiza $\sum(y_i - \hat{y}_i)^2 + \alpha\|\boldsymbol{\beta}\|_2^2$ con validación interna `cv=5` sobre una grilla logarítmica de 100 valores ($\alpha \in [10^{-3}, 10^3]$; seleccionado en el rango exacto $[3.2745, 17.4753]$, media $11.07$, concentrado mayormente en $[5.72, 17.48]$). Contrae suavemente los pesos frente al apalancamiento de 2023 ($\bar{\beta}_{\text{CuT}} = +0.1146$, $\bar{\beta}_{\text{CuS}} = +0.0537$), mejorando levemente a OLS hasta **$\text{NRMSE} = 11.78\%$** ($R^2 = +0.5221$).
5. **`LassoCV` (Regularización $L_1$, Selector de Variables):** Minimiza $\frac{1}{2N_{\text{tr}}}\sum(y_i - \hat{y}_i)^2 + \alpha\|\boldsymbol{\beta}\|_1$ (`cv=5`, $\alpha \in [0.0001, 0.0351]$, media `0.0185`). Alcanza $\text{NRMSE} = 12.30\%$ ($R^2 = +0.4789$).
   * **Hallazgo empírico clave:** Por la geometría en forma de rombo (con vértices sobre los ejes) de la restricción $L_1$, Lasso suele anular exactamente a cero cualquier predictor ruidoso o redundante. Sin embargo, en **24 de los 24 pliegues LOOCV ($100.0\%$)**, `LassoCV` retuvo un coeficiente **estrictamente positivo** para `Cancha_CuS` ($\beta_{\text{CuS}} \in [+0.0124, +0.0627]$, media $+0.0261$). Aunque una relación lineal 2D no basta para superar a `Baseline_1B`, esta persistencia del $100\%$ demostró que `Cancha_CuS` transporta señal geoquímica real que justificaba evaluar su interacción no lineal en el Bloque 3.
6. **`ElasticNetCV` (Penalización Mixta $L_1 + L_2$):** Combina ambas normas con $l_1\text{-ratio} \in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]$ y `cv=5`, logrando $\text{NRMSE} = 12.36\%$ ($R^2 = +0.4735$). Su alternancia entre `l1_ratio = 0.10` y `0.99` obedece al particionamiento secuencial `cv=5` interno sobre 23 observaciones, sin afectar la estabilidad fuera de muestra.
7. **`SVR_RBF` (Máquina de Vectores de Soporte con Kernel Gaussiano):** Emplea $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$ con margen $\epsilon$-insensible y `GridSearchCV(cv=5)` sobre $C \in \{1, 10, 50\}$, $\epsilon \in \{0.05, 0.1\}$ y $\gamma \in \{\text{'scale'}, 0.1, 1.0\}$. **Fracasa con $\text{NRMSE} = 16.68\%$ ($R^2 = +0.0412$)** y colapsa en `Buffered-LOOCV` a **$20.97\%$ ($R^2 = -0.5152$)**.
   * **¿Por qué fracasa `SVR_RBF`?** Porque el kernel gaussiano es un **aproximador estrictamente local** que depende de la densidad de vectores de soporte cercanos. Cuando el punto de prueba cae en una región aislada de baja densidad —como **noviembre 2023** ($\text{Canje} = 2.27\%$, donde predice apenas `1.6132%`, error $-0.6568\%\text{Cu}$) o **septiembre 2023** ($\text{Canje} = 1.55\%$, donde predice `1.0603%`)—, la función de decisión decae asintóticamente hacia el nivel medio del entrenamiento.

#### Verificaciones de Sensibilidad del Bloque 2 (Escalado y Blindaje Geoestadístico)
* **`StandardScaler` vs. `RobustScaler`:** Variaciones inferiores o iguales a **$0.09\%$ en NRMSE** en toda la batería (`OLS` exactamente invariante $\Delta = 0.0000$; `RidgeCV` $11.78\% \to 11.69\%$; `LassoCV` $12.30\% \to 12.35\%$; `ElasticNetCV` $12.36\% \to 12.42\%$; `SVR_RBF` $16.68\% \to 16.74\%$). Queda certificado que `StandardScaler` es completamente seguro pese a la curtosis de `CuS`.
* **`Buffered-LOOCV` de Baselines (excluyendo $t-1$ y $t+1$):** `Baseline_1B` ($10.07\% \to 10.04\%$), `LassoCV` ($12.30\% \to 12.34\%$, $\Delta = +0.04\%$) y `ElasticNetCV` ($12.36\% \to 12.04\%$) muestran total estabilidad; `OLS` incluso mejora ($11.85\% \to 11.07\%$) porque separar octubre y noviembre de 2023 impide que un pulso apalanque el plano sobre el otro. Esto descarta que el modelo explote memoria temporal espuria de $r_1 = +0.4467$.

### 2.3. Ingeniería de las 11 Redes Neuronales Artificiales (Bloque 3): El PORQUÉ de Cada Hiperparámetro

En el Bloque 3 se diseñó una matriz sistemática de **11 arquitecturas neuronales** agrupadas en tres familias: (i) la réplica histórica del profesor guía en TensorFlow/Keras (`RNA_Delgado_Keras`), (ii) seis Perceptrones Multicapa Directos (`Direct-MLP`) y (iii) cuatro Redes Residuales con conexión física directa (`Residual-MLP`).

#### Tabla 2.1: Ficha Completa de Arquitectura e Hiperparámetros de las 11 Redes Neuronales (`Código unificado.ipynb` y `BLOQUE_3/LOG_HIPERPARAMETROS_RNA.csv`)

| Modelo Neuronal | Familia | Capas Ocultas | $P$ Nominal | $P_{\text{eff}}$ | Ratio $N_{\text{tr}}/P_{\text{eff}}$ | Activación | Optimizador | $\alpha$ ($L_2$) | `scale_y` | Semillas | Régimen de Capacidad |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`RNA_Delgado_Keras`** | Directa (Keras) | `(4, 4, 4)` | **57** | **3** | 7.67 | `linear` | `Adam(0.05)` | `0.0` | `False` | 1/pliegue | Colapso lineal algebraico |
| **`RNA_Delgado_MLP`** | Directa (MLP) | `(4, 4)` | 37 | 37 | **0.62** | `relu` | `L-BFGS` | `2.0` | `True` | 10 | Sobreparametrizado ($P > N_{\text{tr}}$) |
| **`RNA_MLP_2`** | Directa (MLP) | `(2,)` | **9** | **9** | **2.56** | **`tanh`** | `L-BFGS` | **`0.001`** | **`False`** | 10 | **Parsimonioso óptimo (pesos activos)** |
| **`RNA_MLP_3`** | Directa (MLP) | `(3,)` | 13 | 13 | 1.77 | `relu` | `L-BFGS` | `5.0` | `True` | 10 | Balanceado (alta contracción $L_2$) |
| **`RNA_MLP_4`** | Directa (MLP) | `(4,)` | 17 | 17 | 1.35 | `relu` | `L-BFGS` | `5.0` | `True` | 10 | Ajustado (alta contracción $L_2$) |
| **`RNA_MLP_2_2`** | Directa (MLP) | `(2, 2)` | 15 | 15 | 1.53 | `relu` | `L-BFGS` | `2.0` | `True` | 10 | Profundidad compacta |
| **`RNA_MLP_3_2`** | Directa (MLP) | `(3, 2)` | 20 | 20 | 1.15 | `tanh` | `L-BFGS` | `2.0` | `True` | 10 | Frontera de sobreajuste |
| **`Res_MLP_2`** | Residual-MLP | `(2,)` | 9 | $\sim 1\text{–}9$ | $\ge 2.56$ | `relu` | `L-BFGS` | `1.0` | `False` | 10 | Contracción residual parcial |
| **`Res_MLP_2_2`** | Residual-MLP | `(2, 2)` | 15 | **$\approx 1$** | $\approx 23$ | `relu` | `L-BFGS` | `2.0` | `False` | 10 | Colapso asintótico seguro a B1B |
| **`Res_MLP_3`** | Residual-MLP | `(3,)` | 13 | **$\approx 1$** | $\approx 23$ | `relu` | `L-BFGS` | `2.0` | `False` | 10 | Colapso asintótico seguro a B1B |
| **`Res_MLP_4`** | Residual-MLP | `(4,)` | 17 | **$\approx 1$** | $\approx 23$ | `relu` | `L-BFGS` | `2.0` | `False` | 10 | Colapso asintótico seguro a B1B |

A continuación se desglosa el fundamento matemático y práctico de cada componente de diseño:

#### 1. Topologías y Principio de Parsimonia ($N_{\text{train}} / P$)
En una red multicapa completamente conectada con $d_0 = 2$ entradas, capas ocultas $(h_1, \dots, h_L)$ y salida escalar $d_{L+1} = 1$, el número total de parámetros libres (pesos sinápticos + sesgos) es:

$$P = \sum_{l=1}^{L+1} (d_{l-1} \cdot d_l + d_l)$$

* Así, `(2,)` tiene exactamente $(2\cdot 2 + 2) + (2\cdot 1 + 1) = \mathbf{9}$ parámetros ($N_{\text{train}}/P = 23/9 = \mathbf{2.56}$ muestras por parámetro); `(3,)` tiene $13$ ($1.77$); `(2, 2)` tiene $15$ ($1.53$); `(4,)` tiene $17$ ($1.35$); `(3, 2)` tiene $20$ ($1.15$); `(4, 4)` tiene $37$ ($0.62$, régimen sobreparametrizado $P > N_{\text{train}}$); y la red Keras `(4, 4, 4)` tiene $(2\cdot 4 + 4) + (4\cdot 4 + 4) + (4\cdot 4 + 4) + (4\cdot 1 + 1) = 12 + 20 + 20 + 5 = \mathbf{57}$ parámetros nominales (Auditoría A-03).
* **Ley empírica verificada (Navaja de Occam):** Dentro de las redes directas no lineales, la arquitectura más compacta (`RNA_MLP_2`, solo 9 parámetros) obtuvo el mejor error LOOCV (`12.28%`) y el mejor Holdout (`6.56%`), derrotando sistemáticamente a las redes de dos capas ocultas `(2, 2)` (`13.00%`), `(3, 2)` (`13.48%`) y `(4, 4)` (`12.61%`). Con $23$ datos de entrenamiento, añadir profundidad crea grados de libertad innecesarios que memorizan ruido.

#### 2. Funciones de Activación: `linear` vs. `relu` vs. `tanh`
* **Activación `linear` ($g(z) = z$) — El Colapso Algebraico de Keras:** Cuando una capa `Dense` en Keras no especifica `activation=...`, aplica la función identidad. Componer cuatro transformaciones afines consecutivas equivale a multiplicar sus matrices:
  $$\hat{y} = \mathbf{W}_4\Big(\mathbf{W}_3\big(\mathbf{W}_2(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2\big) + \mathbf{b}_3\Big) + b_4 = \mathbf{W}_{\text{eq}}\mathbf{x} + b_{\text{eq}}$$
  donde $\mathbf{W}_{\text{eq}} = \mathbf{W}_4\mathbf{W}_3\mathbf{W}_2\mathbf{W}_1 \in \mathbb{R}^{1 \times 2}$ y $b_{\text{eq}} \in \mathbb{R}$. En la Celda 9 de `Código unificado.ipynb`, la comprobación matricial en vivo entrega la ecuación colapsada exacta $\hat{y} = -0.0575 + 0.7916\,\text{Cancha\_CuT} + 0.5815\,\text{Cancha\_CuS}$, cuya diferencia máxima frente a la propagación completa de 4 capas y 57 pesos es **$3.12 \times 10^{-7}\%\text{Cu}$** (error de máquina `float32`). Por tanto, $57$ parámetros nominales colapsan matemáticamente a **$P_{\text{eff}} = 3$** (idéntica en forma funcional a `OLS`).
* **Activación `relu` ($\phi(z) = \max(0, z)$):** Produce superficies **lineales a trozos** (*piecewise linear*) con derivada exacta $1$ para $z > 0$ y $0$ para $z \le 0$. No satura en el semieje positivo, lo que evita amortiguamiento de gradientes y permite una contracción muy limpia bajo penalización $L_2$ intensa ($\alpha \in \{1.0, 2.0, 5.0\}$). Por ello resultó la activación más estable para las redes directas anchas/profundas y para toda la familia `Residual-MLP`.
* **Activación `tanh` ($\phi(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$):** Es infinitamente diferenciable ($C^\infty$) y centrada en cero (armonizada con las entradas Z-score de media $0$). Con solo 2 neuronas (`RNA_MLP_2`), permite modelar **curvaturas geometalúrgicas suaves** en el dominio operativo regular (logrando `6.56%` en Holdout 19/5 y dominando el pulso oxidado de octubre 2023 con error `0.0632%Cu`). Su contrapartida matemática es la **saturación asintótica**: cuando una observación cae muy lejos del centro de entrenamiento (noviembre 2023, `Cancha_CuT = 2.47%Cu`, $+2.9\sigma$), $\tanh(z) \to \pm 1$, aplanando la respuesta en `1.7467%` frente al valor real `2.27%`.

#### 3. Optimizadores: Por qué `L-BFGS` supera a `SGD`/`Adam` en $N_{\text{train}} = 23$
* **`L-BFGS` (*Limited-memory Broyden–Fletcher–Goldfarb–Shanno*):** Es un optimizador **cuasi-Newton de segundo orden** que opera en precisión doble (`float64`) sobre el lote completo (*full-batch*, los 23 datos a la vez, con `max_iter=3000`, `tol=1e-5`). En lugar de dar pasos ciegos siguiendo únicamente la pendiente local (primer orden), `L-BFGS` construye una aproximación de la **inversa de la matriz Hessiana** (curvatura de la función de pérdida) y ejecuta búsqueda lineal (*line search*) en cada iteración. En $N_{\text{train}} = 23$, esto elimina el ruido estocástico de los *mini-batches*, prescinde de sintonizar manualmente una tasa de aprendizaje y converge de forma determinista y ultrarrápida.
* **`Adam` en la réplica Keras (`0.1` $\to$ `0.05`):** `Adam` (*Adaptive Moment Estimation*) es un optimizador estocástico de primer orden diseñado para millones de registros en GPU (`float32`). En el cuaderno original del Dr. Delgado se usó `Adam(learning_rate=0.1)` durante 1,000 épocas *in-sample*. En la replicación LOOCV sobre $N_{\text{train}} = 23$ (Auditoría B3-03), se ajustó a **`Adam(0.05)`** durante 800 épocas para amortiguar las oscilaciones del gradiente en pasos tan grandes y asegurar convergencia suave preservando intacta la equivalencia lineal ($P_{\text{eff}} = 3$).

#### 4. Regularización $L_2$ ($\alpha$), Escalado de Salida (`scale_y`) y Ensamble de 10 Semillas
* **Regularización $L_2$ (*Weight Decay*, $\alpha \in \{0.001, 1.0, 2.0, 5.0\}$):** En `MLPRegressor` de `scikit-learn`, la función objetivo penalizada es $\mathcal{L}(\theta) = \frac{1}{2N_{\text{tr}}}\sum_{i=1}^{N_{\text{tr}}} (y_i - \hat{y}_i)^2 + \frac{\alpha}{2N_{\text{tr}}}\|\mathbf{W}\|_F^2$ (norma de Frobenius al cuadrado escalada por $N_{\text{tr}}$).
  * En **`RNA_MLP_2` ($\alpha = 0.001$):** Como la topología ya está fuertemente restringida por estructura (solo 2 neuronas, 9 parámetros), una penalización suave permite conservar pesos activos para modelar la curvatura de `Cancha_CuS` sin colapsar a un plano rígido.
  * En **`RNA_MLP_3` y `RNA_MLP_4` ($\alpha = 5.0$)** y **`RNA_MLP_2_2`, `RNA_MLP_3_2`, `RNA_Delgado_MLP` ($\alpha = 2.0$):** Al crecer el número de parámetros ($13\text{–}37$), se requiere regularización fuerte para impedir memorización de los 23 registros.
  * En **`Residual-MLP` ($\alpha = 1.0$ y $2.0$):** Al relajar la penalización sobre el residuo, el error empeora monotónicamente ($\alpha = 0.5 \to 10.73\%$; $\alpha = 0.1 \to 11.94\%$), mientras que $\alpha = 2.0$ contrae $\|\mathbf{W}\| \approx 0.0005$ y estabiliza el NRMSE en el óptimo global de **$10.07\%$**.
* **Escalado de la variable objetivo (`scale_y = True` vs. `False`):**
  * **`scale_y = True`** (`RNA_MLP_3`, `RNA_MLP_4`, `RNA_MLP_2_2`, `RNA_MLP_3_2`, `RNA_Delgado_MLP`): Escala $y_{\text{tr}}$ a media $0$ y varianza $1$ *dentro de cada pliegue LOOCV* (invirtiendo la transformación al predecir). Esto armoniza el orden de magnitud entre el MSE normalizado ($\sim O(1)$) y la penalización fuerte $L_2$ ($\alpha \in \{2.0, 5.0\}$), evitando que la regularización arrastre artificialmente la salida hacia cero.
  * **`scale_y = False`** (`RNA_MLP_2`, `RNA_Delgado_Keras` y los cuatro `Residual-MLP`): En `RNA_MLP_2`, al ser $\alpha = 0.001$, la capa de salida lineal aprende directamente la escala física de `Canje_CuT`; en los `Residual-MLP`, mantener el residuo $e_i = \text{CuT}_i - \text{Canje}_i$ en unidades físicas directas permite que el término de sesgo de la neurona de salida $b_{\text{out}}$ converja limpiamente a $\bar{e}_{\text{train}, -k}$ cuando $\|\mathbf{W}\| \to 0$.
* **Ensamble determinista de 10 semillas fijas:** `SEMILLAS = [42, 101, 2023, 7, 13, 99, 123, 456, 789, 999]`. Como la superficie de error de una RNA no lineal es no convexa, una sola inicialización aleatoria podría quedar atrapada en un mínimo local desfavorable. Promediar las predicciones de 10 redes entrenadas desde distintas inicializaciones fijas en cada pliegue ($24 \times 10 = 240$ entrenamientos por arquitectura) elimina la varianza estocástica y asegura reproducibilidad exacta.

#### 5. Arquitectura `Residual-MLP` (*Skip-Connection*) y Propiedad Asintótica de Seguridad
Para superar el aplanamiento de pendiente en leyes extremas, la red residual introduce una **conexión física directa (*skip-connection*)**:

$$\hat{y}_k = \text{Cancha\_CuT}_k - \hat{e}_{\text{MLP}}(\text{Cancha\_CuT}_k,\; \text{Cancha\_CuS}_k)$$

donde la red entrena internamente contra el sesgo $e_i = \text{Cancha\_CuT}_i - \text{Canje\_CuT}_i$. Bajo regularización $L_2$ creciente ($\alpha = 2.0$), los pesos sinápticos se contraen a cero ($\|\mathbf{W}\| \to 0$), reduciendo los grados de libertad estructurales ($15, 13, 17$) a **$P_{\text{eff}} \approx 1$** (el intercepto $b_{\text{out}} \to \bar{e}_{\text{train}, -k}$):

$$\lim_{\alpha \to \infty} \hat{y}_k = \text{Cancha\_CuT}_k - \bar{e}_{\text{train}, -k} \equiv \text{Baseline\_1B} \quad (\text{NRMSE} = 10.07\%)$$

Esta propiedad demuestra que la red residual jamás extrapolará catastróficamente en frentes extremos como noviembre 2023 (`2.1235%` frente al real `2.27%`).

#### 6. Reconciliación Histórica con el Dr. José Delgado Vega ($8.63\%$ *In-Sample* vs. $11.64\%$ LOOCV)
Uno de los aportes metodológicos más valiosos de esta tesis es explicar con total precisión y respeto académico por qué la red exploratoria original del Dr. Delgado (`Dense(4)-Dense(4)-Dense(4)-Dense(1)` en Google Colab, marzo 2025) reportaba un error cercano al **$8.63\%$**, mientras que en la evaluación rigurosa de esta tesis marca **$11.64\%$**:

| Escenario de Evaluación de `RNA_Delgado_Keras` | Variable Objetivo | Régimen de Evaluación | RMSE ($\%\text{Cu}$) | **MAPE ($\%$)** | **NRMSE ($\%$)** | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Referencia Preliminar Dr. Delgado (Marzo 2025)** | `PSJ_CuT` | *In-sample* (sobre los 24 datos) | $\sim 0.12\text{–}0.14$ | **$\approx 8.63\%$** | — | — |
| **Réplica Exacta *In-Sample* (`Código unificado.ipynb`)** | `PSJ_CuT` | *In-sample* ($N=24$, `Adam(0.1)`) | `0.1418` | **`7.97%`** | `9.91%` (`9.92%` vivo) | — |
| **Réplica Exacta *In-Sample* sobre Canje** | `Canje_CuT` | *In-sample* ($N=24$, `Adam(0.1)`) | `0.1315` | **`7.38%`** (`7.37%` vivo) | `9.19%` | — |
| **Evaluación Ciega Certificada (LOOCV, Bloque 3 y 4)** | `Canje_CuT` | **Fuera de muestra LOOCV ($N=24$)** | **`0.1665`** | **`8.74%`** | **`11.64%`** | **`+0.5334`** |

*(Auditoría A-04)*: Quedan reconciliadas dos diferencias simultáneas: (1) el $8.63\%$ original correspondía a un **MAPE *in-sample*** (evaluar sobre los mismos datos usados para entrenar), y (2) al migrar a **validación ciega LOOCV** medida por **NRMSE**, la penalización natural por generalización eleva el error a **$11.64\%$** ($\text{MAPE}_{\text{LOOCV}} = 8.74\%$). Además, debido al colapso lineal ($P_{\text{eff}} = 3$), sus 24 predicciones difieren en promedio solo **$0.0023\%\text{Cu}$** de `OLS` (`11.85%`).

---

# CAPÍTULO 3
## Análisis Comparativo de los 18 Modelos, Robustez e Inferencia Estadística (Bloque 4)

### 3.1. Benchmarking Unificado de los 18 Modelos (LOOCV, $N = 24$)

En el Bloque 4 se integraron de forma imparcial los **7 baselines** (`BLOQUE_2/PREDICCIONES_OOS_BASELINES.csv`) y las **11 redes neuronales** (`BLOQUE_3/PREDICCIONES_OOS_RNA.csv`) bajo una misma batería de **9 métricas econométricas y geoestadísticas** calculadas sobre el vector de 24 predicciones estrictamente ciegas ($\bar{y} = 1.43125\%\text{Cu}$, $\text{SST} = \sum(y_k - \bar{y})^2 = 1.42646\,(\%\text{Cu})^2$):

$$\text{MSE} = \frac{1}{N}\sum_{k=1}^N (y_k - \hat{y}_k)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}, \qquad \text{NRMSE} = \frac{\text{RMSE}}{\bar{y}}\times 100\%$$
$$\text{MAE} = \frac{1}{N}\sum_{k=1}^N |y_k - \hat{y}_k|, \qquad \text{MAPE} = \frac{1}{N}\sum_{k=1}^N \left|\frac{y_k - \hat{y}_k}{y_k}\right|\times 100\%$$
$$R^2 = 1 - \frac{\sum (y_k - \hat{y}_k)^2}{\sum (y_k - \bar{y})^2}, \qquad \text{VarExp} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}, \qquad \text{Sesgo} = \frac{1}{N}\sum_{k=1}^N (\hat{y}_k - y_k)$$

#### Tabla 3.1: Ranking Unificado Completo de los 18 Modelos (`BLOQUE_4/TABLA_UNIFICADA_BENCHMARKING_18_MODELOS.csv`)

| Ranking | Modelo | Familia Algorítmica | Espacio de Entradas | MSE ($(\%\text{Cu})^2$) | RMSE ($\%\text{Cu}$) | MAE ($\%\text{Cu}$) | NRMSE ($\%$) | $R^2$ Score | MAPE ($\%$) | Pearson $r$ | Sesgo ($\%\text{Cu}$) | Var. Explicada |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1=** | **`Res_MLP_2_2`** | Residual-MLP `(2, 2)` | 2D $[\text{CuT}, \text{CuS}]$ | **0.020781** | **0.1442** | **0.1178**\* | **10.07%** | **+0.6504** | **8.12%** | **0.8209** | $+0.0000$ | **0.6504** |
| **1=** | **`Baseline_1B`** | Corrección Sesgo Constante | 1D $[\text{Cancha}]$ | **0.020782** | **0.1442** | **0.1178** | **10.07%** | **+0.6503** | **8.12%** | **0.8209** | $-0.0000$ | **0.6503** |
| **1=** | **`Res_MLP_4`** | Residual-MLP `(4,)` | 2D $[\text{CuT}, \text{CuS}]$ | **0.020782** | **0.1442** | **0.1178** | **10.07%** | **+0.6503** | **8.12%** | **0.8209** | $-0.0000$ | **0.6503** |
| **1=** | **`Res_MLP_3`** | Residual-MLP `(3,)` | 2D $[\text{CuT}, \text{CuS}]$ | **0.020782** | **0.1442** | **0.1178** | **10.07%** | **+0.6503** | **8.12%** | **0.8209** | $-0.0000$ | **0.6503** |
| **5** | `Res_MLP_2` | Residual-MLP `(2,)` | 2D $[\text{CuT}, \text{CuS}]$ | 0.021551 | 0.1468 | 0.1200 | 10.26% | +0.6374 | 8.22% | 0.8103 | $-0.0022$ | 0.6375 |
| **6** | `RNA_Delgado_Keras` | Directa (Keras Colapsada) | 2D $[\text{CuT}, \text{CuS}]$ | 0.027735 | 0.1665 | 0.1302 | 11.64% | +0.5334 | 8.74% | 0.7410 | $+0.0022$ | 0.5334 |
| **7** | `RidgeCV` | Lineal Regularizado ($L_2$) | 2D $[\text{CuT}, \text{CuS}]$ | 0.028403 | 0.1685 | 0.1301 | 11.78% | +0.5221 | 8.93% | 0.7248 | $+0.0108$ | 0.5241 |
| **8** | `OLS` | Regresión Lineal Clásica | 2D $[\text{CuT}, \text{CuS}]$ | 0.028789 | 0.1697 | 0.1318 | 11.85% | +0.5156 | 8.82% | 0.7346 | $+0.0038$ | 0.5159 |
| **9** | **`RNA_MLP_2`** | Directa MLP `(2,)` | 2D $[\text{CuT}, \text{CuS}]$ | **0.030890** | **0.1758** | **0.1318** | **12.28%** | **+0.4803** | **8.95%** | **0.6954** | $-0.0124$ | **0.4829** |
| **10** | `LassoCV` | Lineal Regularizado ($L_1$) | 2D $[\text{CuT}, \text{CuS}]$ | 0.030971 | 0.1760 | 0.1373 | 12.30% | +0.4789 | 9.23% | 0.7047 | $+0.0087$ | 0.4802 |
| **11** | `ElasticNetCV` | Lineal Regularizado ($L_1+L_2$) | 2D $[\text{CuT}, \text{CuS}]$ | 0.031294 | 0.1769 | 0.1369 | 12.36% | +0.4735 | 9.21% | 0.6992 | $+0.0085$ | 0.4747 |
| **12** | `RNA_MLP_4` | Directa MLP `(4,)` | 2D $[\text{CuT}, \text{CuS}]$ | 0.031446 | 0.1773 | 0.1340 | 12.39% | +0.4709 | 9.15% | 0.6867 | $+0.0060$ | 0.4715 |
| **13** | `RNA_MLP_3` | Directa MLP `(3,)` | 2D $[\text{CuT}, \text{CuS}]$ | 0.031951 | 0.1787 | 0.1350 | 12.49% | +0.4624 | 9.23% | 0.6809 | $+0.0068$ | 0.4632 |
| **14** | `RNA_Delgado_MLP` | Directa MLP `(4, 4)` | 2D $[\text{CuT}, \text{CuS}]$ | 0.032592 | 0.1805 | 0.1378 | 12.61% | +0.4516 | 9.33% | 0.6778 | $+0.0001$ | 0.4516 |
| **15** | `RNA_MLP_2_2` | Directa MLP `(2, 2)` | 2D $[\text{CuT}, \text{CuS}]$ | 0.034635 | 0.1861 | 0.1300 | 13.00% | +0.4173 | 8.92% | 0.6683 | $+0.0009$ | 0.4173 |
| **16** | `RNA_MLP_3_2` | Directa MLP `(3, 2)` | 2D $[\text{CuT}, \text{CuS}]$ | 0.037222 | 0.1929 | 0.1398 | 13.48% | +0.3737 | 9.53% | 0.6122 | $-0.0044$ | 0.3741 |
| **17** | `SVR_RBF` | Kernel No Lineal (RBF) | 2D $[\text{CuT}, \text{CuS}]$ | 0.056986 | 0.2387 | 0.1796 | 16.68% | +0.0412 | 12.11% | 0.4546 | $-0.0110$ | 0.0432 |
| **18** | **`Cancha_Directa`** | **Línea Base Operacional** | 1D $[\text{Cancha}]$ | **0.134971** | **0.3674** | **0.3404** | **25.67%** | **-1.2709** | **24.90%** | **0.8347** | **+0.3404** | **0.6789** |

*\*Nota aritmética y econométrica (Auditorías C-04 y B4-01):*
1. **Empate técnico cuádruple (`1=`):** `Res_MLP_2_2` ($\text{MSE} = 0.020781$, $\text{MAE exacto} = 0.11775 \to 0.1178$ en redondeo estándar / `0.1177` en precisión `float64` de Python/pandas por representación binaria $0.1177499999999999 < 0.11775$), `Baseline_1B`, `Res_MLP_4` y `Res_MLP_3` ($\text{MSE} = 0.020782$) difieren en menos de $10^{-6}\,(\%\text{Cu})^2$. Se oficializa el **empate técnico cuádruple `1=`**, quedando `Res_MLP_2` en el puesto `5`.
2. **Por qué `Cancha_Directa` muestra `Var. Explicada = 0.6789` superior a `0.6503`:** La fórmula de `explained_variance_score` descuenta internamente la media del error *in-sample* ($+0.3404\%\text{Cu}$). En cambio, en `Baseline_1B`, estimar el sesgo en LOOCV con $N_{\text{train}}=23$ induce exactamente el factor de penalización fuera de muestra $(24/23)^2 = 1.0888$, llevando la varianza residual de $0.019087$ a $0.019087 \times (24/23)^2 = 0.020782$ y la varianza explicada real a $1 - 0.020782/0.059436 = 0.6503$.

### 3.2. Análisis Comparativo entre las Cinco Familias Algorítmicas

1. **Familia Empírica (`Cancha_Directa` vs. `Baseline_1B`):** Muestra el contraste extremo del estudio. Eliminar la sobreestimación sistemática de cancha mediante una sola constante estimada en LOOCV derrumba el MSE en **$84.6\%$** (de `0.134971` a `0.020782`) y el NRMSE de **`25.67%`** a **`10.07%`**.
2. **Familia Residual (`Residual-MLP`):** Ocupa los primeros 5 puestos del ranking global (`10.07%` a `10.26%`). La conexión física $\hat{y} = \text{CuT} - \hat{e}_{\text{MLP}}$ hereda la inmunidad frente a extremos de la pendiente $1:1$; bajo $\alpha = 2.0$, los pesos se apagan ($\|\mathbf{W}\| \approx 0.0005$) confirmando que sobre la serie global con noviembre 2023 el mínimo de varianza coincide con el sesgo estacionario ($P_{\text{eff}} \approx 1$).
3. **Familia Lineal (`RNA_Delgado_Keras`, `RidgeCV`, `OLS`, `LassoCV`, `ElasticNetCV`):** Ocupa los puestos 6 al 11 (`11.64%` a `12.36%`). Aunque logran buena correlación lineal, su hiperplano rígido **falla en los dos pulsos de 2023** (sobreestima octubre por la pendiente sobre `CuS` y subestima noviembre por pendiente achatada sobre `CuT`), lo que se traduce en **heterocedasticidad estadísticamente significativa** en las pruebas de residuos.
4. **Familia Directa No Lineal (`Direct-MLP`):** Ocupa puestos 9 y 12 al 16 (`12.28%` a `13.48%`). Destaca **`RNA_MLP_2`** (`12.28%`), que con solo 9 parámetros supera a `LassoCV` (`12.30%`), `ElasticNetCV` (`12.36%`) y a todas las redes más grandes, domina la validación prospectiva regular (`6.56%`) y preserva **homocedasticidad**. Las redes profundas de dos capas (`(2, 2)`, `(3, 2)`) se degradan hasta `13.00%–13.48%` y pierden normalidad de residuos.
5. **Familia Kernel Local (`SVR_RBF`):** Puesto 17 (`16.68%`, $R^2 = +0.0412$). Incapaz de generalizar hacia los bordes del espacio de leyes debido al decaimiento local del kernel gaussiano en muestras pequeñas.

### 3.3. Pruebas de Robustez: Ablación, Holdout Prospectivo 19/5 y *Buffered-LOOCV*

#### A. Estudio de Ablación de Entradas (2D $[\text{CuT}, \text{CuS}]$ vs. 1D $[\text{CuT}]$)

#### Tabla 3.2: Impacto Neto de `Cancha_CuS` (`BLOQUE_3/SENSIBILIDAD_Y_ABLACION_RNA.xlsx`, Hoja `Ablacion_Entradas`)

| Arquitectura | Topología | Entradas | Parám. | MSE ($(\%\text{Cu})^2$) | RMSE ($\%\text{Cu}$) | MAE ($\%\text{Cu}$) | NRMSE ($\%$) | $R^2$ Score | $\Delta\text{NRMSE}$ (2D–1D) | Lectura Geometalúrgica |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`RNA_MLP_2`** | `(2,)` | **2D [CuT, CuS]** | **9** | **0.030890** | **0.1758** | **0.1318** | **12.28%** | **+0.4803** | **$-1.53\%$** | **Aporte crítico: $+13.76$ pts en $R^2$** |
| `RNA_MLP_2` | `(2,)` | 1D [CuT] | 7 | 0.039068 | 0.1977 | 0.1514 | 13.81% | +0.3427 | — | Sin `CuS` no distingue frentes oxidados |
| **`RNA_MLP_2_2`** | `(2, 2)` | **2D [CuT, CuS]** | 15 | **0.034635** | **0.1861** | **0.1300** | **13.00%** | **+0.4173** | **$-0.44\%$** | Favorable |
| `RNA_MLP_2_2` | `(2, 2)` | 1D [CuT] | 13 | 0.037011 | 0.1924 | 0.1365 | 13.44% | +0.3773 | — | — |
| **`RNA_MLP_3_2`** | `(3, 2)` | **2D [CuT, CuS]** | 20 | **0.037222** | **0.1929** | **0.1398** | **13.48%** | **+0.3737** | **$-0.36\%$** | Favorable |
| `RNA_MLP_3_2` | `(3, 2)` | 1D [CuT] | 17 | 0.039217 | 0.1980 | 0.1407 | 13.84% | +0.3402 | — | — |
| **`RNA_Delgado_MLP`** | `(4, 4)` | **2D [CuT, CuS]** | 37 | **0.032592** | **0.1805** | **0.1378** | **12.61%** | **+0.4516** | **$-0.17\%$** | Favorable |
| `RNA_Delgado_MLP` | `(4, 4)` | 1D [CuT] | 33 | 0.033478 | 0.1830 | 0.1355 | 12.78% | +0.4367 | — | — |
| `RNA_MLP_3` | `(3,)` | 2D / 1D | 13 / 10 | 0.031951 / 0.031839 | 0.1787 / 0.1784 | 0.1350 / 0.1341 | 12.49% / 12.47% | +0.4624 / +0.4643 | $+0.02\%$ | Neutro ($\alpha = 5.0$ aplasta pesos) |
| `RNA_MLP_4` | `(4,)` | 2D / 1D | 17 / 13 | 0.031446 / 0.031462 | 0.1773 / 0.1774 | 0.1340 / 0.1338 | 12.39% / 12.39% | +0.4709 / +0.4707 | $0.00\%$ | Neutro ($\alpha = 5.0$ aplasta pesos) |
| **`Res_MLP_2`** | `(2,)` | **2D / 1D** | 9 / 7 | 0.021551 / 0.021688 | 0.1468 / 0.1473 | 0.1200 / 0.1203 | 10.26% / 10.29% | +0.6374 / +0.6351 | **$-0.03\%$** | Favorable leve ($\alpha = 1.0$) |
| `Res_MLP_2_2` | `(2, 2)` | 2D / 1D | 15 / 13 | 0.020781 / 0.020783 | 0.1442 / 0.1442 | 0.1178 / 0.1178 | 10.07% / 10.07% | +0.6504 / +0.6503 | $0.00\%$ | Colapso asintótico ($P_{\text{eff}} \approx 1$) |

**Lectura física:** En `RNA_MLP_2`, suprimir `Cancha_CuS` deteriora el error en **$+1.53\%$ absolutos** (de `12.28%` a `13.81%`) y desploma $R^2$ de `+0.4803` a `+0.3427`. Esto valida empíricamente la directriz del Dr. Delgado y la Teoría de Pierre Gy: sin conocer la fracción de óxidos friables, la red no logra diferenciar frentes sulfurados competentes de frentes meteorizados.

#### B. Validación Prospectiva Simétrica (*Holdout* Cronológico 19/5)

Se entrenó exclusivamente con los primeros **19 meses** (`2023-01` a `2024-07`) y se predijo a ciegas la ventana futura intacta de **5 meses** (`2024-08` a `2024-12`), donde la faena operó en régimen productivo regular (`Canje_CuT` entre `1.06%` y `1.48%`):

#### Tabla 3.3: Resultados del *Holdout* Cronológico 19/5 Simétrico (`BLOQUE_3/SENSIBILIDAD_Y_ABLACION_RNA.xlsx`)

| Modelo Evaluado | Familia | MSE Holdout ($(\%\text{Cu})^2$) | RMSE Holdout ($\%\text{Cu}$) | MAE Holdout ($\%\text{Cu}$) | **NRMSE Holdout ($\%$)** | **$R^2$ Holdout** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RNA_MLP_2`** | Directa-MLP | **0.007417** | **0.0861** | **0.0750** | **6.56%** | **+0.6549** |
| `RidgeCV` | Baseline Lineal | 0.009875 | 0.0994 | 0.0880 | 7.57% | +0.5406 |
| `OLS` | Baseline Lineal | 0.010362 | 0.1018 | 0.0859 | 7.76% | +0.5180 |
| `Res_MLP_4` / `Res_MLP_2_2` / `Res_MLP_3` | Residual-MLP | 0.013804 | 0.1175 | 0.0921 | 8.96% | +0.3578 |
| **`Baseline_1B` (Sesgo Constante)** | Baseline Empírico | 0.013804 | 0.1175 | 0.0921 | **8.96%** | +0.3578 |
| `Res_MLP_2` | Residual-MLP | 0.013805 | 0.1175 | 0.0921 | 8.96% | +0.3578 |
| `RNA_Delgado_MLP` | Directa-MLP | 0.016412 | 0.1281 | 0.1121 | 9.76% | +0.2365 |
| `RNA_MLP_4` | Directa-MLP | 0.017881 | 0.1337 | 0.1174 | 10.19% | +0.1682 |
| `RNA_MLP_3` | Directa-MLP | 0.018916 | 0.1375 | 0.1194 | 10.48% | +0.1200 |
| `RNA_MLP_2_2` | Directa-MLP | 0.019501 | 0.1396 | 0.1222 | 10.64% | +0.0928 |
| `RNA_MLP_3_2` | Directa-MLP | 0.020265 | 0.1424 | 0.1226 | 10.85% | +0.0573 |
| `Cancha_Directa` | Acopio sin Corregir | 0.132120 | 0.3635 | 0.3440 | 27.70% | -5.1463 |

**Interpretación estratégica:** En condiciones de operación normal sin pulsos extremos, **`RNA_MLP_2` perfora ampliamente la referencia del $8.63\%$ logrando `6.56%`** ($\text{RMSE} = 0.0861\%\text{Cu}$, casi en la frontera de la tolerancia analítica de laboratorio), superando a `RidgeCV` (`7.57%`), `OLS` (`7.76%`) y al bloque `Baseline_1B / Res_MLP` (`8.96%`). Con rigor científico, se declara explícitamente que al evaluarse sobre $n = 5$ meses esta prueba posee mayor varianza muestral y **se supedita al LOOCV global ($N = 24$)**, funcionando como evidencia prospectiva complementaria.

#### C. Sensibilidad Geoestadística (*Buffered-LOOCV*) y Adyacencia Octubre–Noviembre 2023

Al excluir del entrenamiento los registros inmediatamente adyacentes $(k-1, k+1)$ para neutralizar la inercia de lag-1 de `Cancha_CuT` ($r_1 = +0.4467$):
* **Familia `Residual-MLP` y `Baseline_1B`:** Totalmente inmunes (`Res_MLP_2_2`, `Res_MLP_3`, `Res_MLP_4` y `Baseline_1B` pasan de `10.07%` a **`10.04%`**, $\Delta = -0.03\%$; `Res_MLP_2` pasa de `10.26%` a **`10.21%`**, $\Delta = -0.05\%$).
* **Redes Directas (`Direct-MLP`):** El NRMSE sube entre $+1.04\%$ y $+2.28\%$ (`RNA_MLP_4` $\to 13.43\%$, `RNA_MLP_3` $\to 13.60\%$, `RNA_Delgado_MLP` $\to 14.41\%$, `RNA_MLP_2` $\to 14.53\%$ coincidente exactamente entre `BLOQUE_3/SENSIBILIDAD_Y_ABLACION_RNA.xlsx` y `Código/Código unificado.ipynb`, `RNA_MLP_3_2` $\to 15.19\%$, `RNA_MLP_2_2` $\to 15.28\%$).
* **Causa geoestadística exacta:** Los únicos dos pulsos de alta ley en los 24 meses son cronológicamente consecutivos: **octubre 2023** ($t=10$, `Canje = 1.79%`) y **noviembre 2023** ($t=11$, `Canje = 2.27%`). Al activar el *buffer* $(t-1, t+1)$ cuando se predice noviembre 2023, se expulsa automáticamente a octubre 2023 del entrenamiento; la red directa queda entrenada sobre un dominio truncado cuyo máximo es apenas **`1.57%Cu`** y debe extrapolar ciegamente hasta **`2.27%Cu`**. Esto no indica memorización temporal, sino truncamiento físico del soporte muestral en el borde superior.

### 3.4. Diagnóstico Econométrico de Residuos e Inferencia No Paramétrica ($N = 24$)

Sobre los 24 residuos ciegos $e_k = y_k - \hat{y}_k$ de cada modelo se ejecutó la batería completa de pruebas exactas en `BLOQUE_4/DIAGNOSTICO_RESIDUOS_Y_PRUEBAS_HIPOTESIS.csv`:

#### Tabla 3.4: Diagnóstico Econométrico e Inferencia de los 10 Modelos Principales (Extracto de los 18 Modelos Auditados)

| Modelo Representativo | RMSE ($\%\text{Cu}$) | IC Bootstrap 95% ($B=10{,}000$) | Shapiro $W$ | Shapiro $p$ | Durbin-Watson $d$ | Breusch-Pagan $p$ | Homocedasticidad | Wilcoxon vs. Cancha $p$ | Wilcoxon vs. B1B $p$ | Meses Supera a B1B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Cancha_Directa`** | 0.3674 | $[0.3153, 0.4169]$ | 0.9852 | 0.9695 | **0.285** | 0.1709 | Homocedástica | 1.000000 | 0.000020 | $3/24$ ($12.5\%$) |
| **`Baseline_1B`** | 0.1442 | $[0.1061, 0.1785]$ | 0.9852 | 0.9696 | **2.014** | **0.4075** | **Homocedástica** | **0.000010** | 1.000000 | Empate (`1=`) |
| **`Res_MLP_2_2`** | 0.1442 | $[0.1065, 0.1786]$ | 0.9852 | 0.9696 | **2.014** | **0.4075** | **Homocedástica** | **0.000010** | 1.000000 | Empate (`1=`) |
| `Res_MLP_2` | 0.1468 | $[0.1103, 0.1807]$ | 0.9828 | 0.9411 | 2.037 | 0.2140 | Homocedástica | 0.000012 | 0.250000 | **$0/24$ ($21$ empates)** |
| `RNA_Delgado_Keras` | 0.1665 | $[0.1182, 0.2131]$ | 0.9705 | 0.6796 | 2.271 | **0.0040** | **Heterocedástica** | 0.000075 | 0.545670 | $15/24$ ($62.5\%$) |
| `RidgeCV` | 0.1685 | $[0.1150, 0.2255]$ | 0.9078 | **0.0316** | 2.092 | **0.0228** | **Heterocedástica** | 0.000103 | 0.855336 | $14/24$ ($58.3\%$) |
| `OLS` | 0.1697 | $[0.1188, 0.2159]$ | 0.9793 | 0.8819 | 2.310 | **0.0016** | **Heterocedástica** | 0.000088 | 0.545670 | $15/24$ ($62.5\%$) |
| **`RNA_MLP_2`** | 0.1758 | $[0.1154, 0.2384]$ | 0.9307 | **0.1012** | **1.958** | **0.0822** | **Homocedástica** | **0.000186** | **0.988813** | **$14/24$ ($58.3\%$)** |
| `LassoCV` | 0.1760 | $[0.1205, 0.2310]$ | 0.9671 | 0.5963 | 2.364 | **0.0029** | **Heterocedástica** | 0.000161 | 0.789838 | $14/24$ ($58.3\%$) |
| `SVR_RBF` | 0.2387 | $[0.1579, 0.3147]$ | 0.9296 | 0.0956 | 2.148 | 0.7243\* | Homocedástica\* | 0.001407 | 0.083920 | $9/24$ ($37.5\%$) |

*(En el CSV completo de los 18 modelos, `ElasticNetCV` también falla Breusch-Pagan con $p = 0.0035$, al igual que `RNA_Delgado_MLP` $p = 0.0181$, `RNA_MLP_3` $p = 0.0325$ y `RNA_MLP_4` $p = 0.0384$; mientras `RNA_MLP_2_2` $p = 0.0024$ y `RNA_MLP_3_2` $p = 0.0130$ fallan Shapiro-Wilk).*

#### Cuatro Hallazgos Econométricos Decisivos para la Defensa Oral:

1. **Colapso Econométrico de TODOS los Modelos Lineales (Breusch-Pagan exacto sobre $\hat{y}$):**  
   `OLS` ($p = 0.0016$), `LassoCV` ($p = 0.0029$), `ElasticNetCV` ($p = 0.0035$), `RNA_Delgado_Keras` ($p = 0.0040$) y `RidgeCV` ($p = 0.0228$, que además rechaza normalidad con Shapiro $p = 0.0316$) violan el supuesto de **homocedasticidad** ($p < 0.05$) porque la varianza de sus errores explota conforme sube la ley predicha. Por el contrario, **`Baseline_1B` / `Res_MLP_2_2` ($p = 0.4075$, $d = 2.014$, Shapiro $p = 0.9696$)** y **`RNA_MLP_2` ($p = 0.0822$, $d = 1.958$, Shapiro $p = 0.1012$)** satisfacen simultáneamente **normalidad, independencia serial y homocedasticidad**.
2. **La trampa geométrica de la "U" en `SVR_RBF` ($p = 0.7243$):**  
   ¿Por qué `SVR_RBF` marca $p = 0.7243$ en Breusch-Pagan pese a ser el peor modelo entrenado? Porque Breusch-Pagan contrasta una dependencia **lineal** entre los residuos cuadrados $e_k^2$ y la predicción $\hat{y}_k$. En `SVR_RBF`, los errores catastróficos ocurren simétricamente en **ambos extremos** del rango predicho —en el extremo inferior (`2023-09`, $\hat{y} = 1.06\%$, real $1.55\%$) y en el extremo superior (`2023-11`, $\hat{y} = 1.61\%$, real $2.27\%$)—, formando una **parábola en forma de "U"** cuya pendiente lineal media es casi cero ($p = 0.7243$), aunque su varianza de error sea casi tres veces superior a `Baseline_1B`.
3. **Superioridad Estadística Irrefutable frente a `Cancha_Directa` (Wilcoxon Exacto y Bootstrap 95%):**  
   * El contraste unilateral exacto de Wilcoxon (`method='exact'`) arroja **$p = 0.000010$** para `Baseline_1B` / `Res_MLP_2_2` y **$p = 0.000186$** para `RNA_MLP_2` (confianza superior al **$99.98\%$**). *(Nótese que en el contraste bilateral `Cancha_Directa` vs. `Baseline_1B`, $p = 0.000020 = 2 \times 0.000010$).*
   * Los intervalos **Bootstrap al 95%** ($B = 10{,}000$) son completamente disjuntos respecto a `Cancha_Directa` ($[0.3153, 0.4169]\%\text{Cu}$): `Baseline_1B` $[0.1061, 0.1785]$, `Res_MLP_2_2` $[0.1065, 0.1786]$ y `RNA_MLP_2` $[0.1154, 0.2384]$. Asimismo, `Cancha_Directa` exhibe un Durbin-Watson de **$d = 0.285$** (autocorrelación positiva severa causada por arrastrar el sesgo $+0.3404\%\text{Cu}$ todos los meses).
4. **Lectura Fina del Contraste Pareado frente a `Baseline_1B` (`RNA_MLP_2` y `Res_MLP_2`):**  
   * En `RNA_MLP_2` vs. `Baseline_1B` ($p = 0.988813$ bilateral, sin diferencia estadísticamente significativa en el balance global), **`RNA_MLP_2` vence a `Baseline_1B` en $14$ de los $24$ meses ($58.3\%$)**, alcanza un **error absoluto mediano $23.3\%$ menor** (**$0.0801\%\text{Cu}$** frente a **$0.1044\%\text{Cu}$**) y domina el pulso oxidado de **octubre 2023** ($|e| = 0.0632\%\text{Cu}$ frente a $0.0830\%$ de `Baseline_1B` y $0.3284\%$ de `OLS`). El único motivo por el que su RMSE global es mayor radica en **noviembre 2023**, mes que concentra por sí solo el **$36.94\%$** de todo el error cuadrático acumulado (SSE) de `RNA_MLP_2`; de hecho, sobre los otros 23 meses ($N=23$), `RNA_MLP_2` registra $\text{RMSE} = 0.1426\%\text{Cu}$ ($\text{NRMSE} = 10.22\%$), superando a `Baseline_1B` ($\text{RMSE} = 0.1441\%\text{Cu}$, $\text{NRMSE} = 10.33\%$).
   * En **`Res_MLP_2`**, la etiqueta auditada es exactamente **`0/24 (21 empates)`** (Auditoría B4-03): empata numéricamente con `Baseline_1B` en 21 meses y pierde marginalmente en los 3 meses restantes (dos diferencias $\le 0.0001$ y noviembre 2023), generando el valor exacto $p = 2 \times (0.5)^3 = 0.250000$.

---

# CAPÍTULO 4
## ¿Cuál es el Mejor Modelo y Por Qué? (Estrategia Dual, Explicabilidad XAI y Plataforma Universal)

### 4.1. La Respuesta Contundente y Matizada: Estrategia Dual de Modelo Campeón

Ante la pregunta inevitable de la comisión —*"En definitiva, ¿cuál de los 18 modelos es el mejor?"*—, la evidencia científica del Bloque 4 demuestra que elegir un único ganador absoluto sería una simplificación incompleta porque existen **dos objetivos operacionales distintos** en una faena minera: (1) minimizar la varianza global del flujo de caja bajo incertidumbre de extremos, y (2) comprender y gestionar la variabilidad mineralógica mes a mes en cancha. Por ello, la tesis formaliza la **Estrategia Dual de Modelo Campeón**:

| Criterio de Evaluación | Pilar 1: Campeón Presupuestario / Mínima Varianza (`Baseline_1B` / `Res_MLP_2_2`) | Pilar 2: Campeón Geometalúrgico / Explicabilidad XAI (`RNA_MLP_2`) |
| :--- | :--- | :--- |
| **Ranking LOOCV ($N=24$)** | **Empate Cuádruple `1=`** ($\text{NRMSE} = 10.07\%$, $R^2 = +0.6504$, $\text{MSE} = 0.020781$) | **Puesto 9 global / #1 en Direct-MLP** ($\text{NRMSE} = 12.28\%$, $R^2 = +0.4803$, $\text{MSE} = 0.030890$) |
| **Desempeño Prospectivo (*Holdout* 19/5)** | `8.96%` ($R^2 = +0.3578$) | **`6.56%`** ($R^2 = +0.6549$, vence a todos los modelos) |
| **Precisión Mes a Mes vs. B1B** | Referencia de mínima varianza | **Vence en $14/24$ meses ($58.3\%$)**; error mediano **`0.0801` vs. `0.1044%Cu`** |
| **Desempeño en Pulsos de 2023** | **Inmune en Nov-2023** (`2.1235%` vs. real `2.27%`) | **Vence en Oct-2023 oxidado** (`1.8532%` vs. real `1.79%`, $|e|=0.0632\%$) |
| **Complejidad Efectiva** | $P_{\text{eff}} = 1$ (pendiente física fija $1:1$ + sesgo medio $+0.3404\%\text{Cu}$) | $P = 9$ ($N_{\text{train}}/P = 2.56$, 2 neuronas `tanh`, pesos activos) |
| **Diagnóstico Econométrico** | Normal ($p=0.9696$), Homocedástico ($p=0.4075$), DW $= 2.014$ | Normal ($p=0.1012$), Homocedástico ($p=0.0822$), DW $= 1.958$ |
| **Sensibilidad a `Cancha_CuS`** | Nula (`SHAP CuS = 0.0000`, pesos contraídos por $L_2$) | **Activa** (Ablación **$-1.53\%$**, `SHAP CuS = 0.0701%Cu`) |
| **Uso Industrial Recomendado** | **Planificación presupuestaria conservadora**, proyección de ingresos ENAMI y resguardo ante leyes extremas fuera del *Convex Hull*. | **Gestión operativa en mina**, control de calidad en cancha, *blending* preventivo de óxidos y gatillador técnico del **Sobre 2**. |

### 4.2. Explicabilidad Geometalúrgica Completa (XAI): Permutación, SHAP y Envolvente Convexa

Para abrir la "caja negra" de las redes neuronales y demostrar el hallazgo RCI A-01, se aplicaron tres herramientas complementarias sobre los ensambles reentrenados en $N = 24$ (`BLOQUE_4/XAI_IMPORTANCIA_VARIABLES.csv` y Celda 14 de `Código unificado.ipynb`):

#### Tabla 4.1: Cuantificación XAI — Importancia por Permutación ($1{,}000$ rep.) y Valores SHAP Exactos (`KernelExplainer`)

| Modelo Analizado | Variable Predictora | Permutación $\Delta\text{RMSE}$ ($\%\text{Cu}$) | Desv. Est. Permutación | Magnitud Media SHAP $E[\|\phi\|]$ ($\%\text{Cu}$) | Contribución Relativa SHAP | Estado Sináptico Auditado |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`RNA_MLP_2` (Campeón XAI)** | **`Cancha_CuT`** | **+0.1256** | 0.0298 | **0.0922** | $56.8\%$ | **Activo** |
| **`RNA_MLP_2` (Campeón XAI)** | **`Cancha_CuS`** | **+0.1003** | 0.0216 | **0.0701** | **$43.2\%$** | **Activo (modulación no lineal)** |
| `Res_MLP_2_2` (Residual $L_2=2.0$) | `Cancha_CuT` | +0.1974 | 0.0387 | 0.1977 | $100.0\%$ | Conexión Física $1:1$ Preservada |
| `Res_MLP_2_2` (Residual $L_2=2.0$) | `Cancha_CuS` | $-0.0000$ | 0.0000 | **0.0000** | **$0.0\%$** | **Apagado (Contracción $L_2$)** |

* **Lectura de Permutación y SHAP en `RNA_MLP_2`:** Al permutar aleatoriamente `Cancha_CuT`, el RMSE empeora en **$+0.1256\%\text{Cu}$**; al permutar `Cancha_CuS`, el RMSE empeora en **$+0.1003\%\text{Cu}$**. De forma consistente, los valores de Shapley (`shap.KernelExplainer` con fondo de 10 observaciones) asignan un impacto medio absoluto de **$0.0922\%\text{Cu}$** a `Cancha_CuT` y **$0.0701\%\text{Cu}$** a `Cancha_CuS`. Esto certifica que ambas entradas cooperan activamente en las 2 neuronas `tanh` para modular la ley de canje proyectada.
* **Confirmación forense en `Res_MLP_2_2`:** Tanto Permutación (`-0.0000`) como SHAP (`0.0000`) prueban de manera irrefutable que la penalización $\alpha = 2.0$ apagó los pesos sinápticos residuales sobre `CuS`, dejando operativa únicamente la conexión directa $1:1$ de `Cancha_CuT` (`SHAP = 0.1977 %Cu`).
* **Superficie de Respuesta 2D y Envolvente Convexa (*Convex Hull* $N = 24$):** Una red neuronal evaluada sobre una grilla rectangular 2D inevitablemente dibuja predicciones en **esquinas vacías sin datos históricos** (por ejemplo, combinaciones no observadas con $\text{CuT} \le 1.80\%$ y $\text{CuS} > 0.16\%$). Para blindar la interpretación física (Auditoría C-01), el mapa de isolíneas traza explícitamente la **Envolvente Convexa (`scipy.spatial.ConvexHull`)** de los 24 meses reales; además, en producción, la **Triangulación de Delaunay** (`Delaunay(X_2d).find_simplex(x_nuevo)[0] >= 0`) alerta automáticamente al ingeniero si un lote futuro cae dentro (*interpolación segura*) o fuera (*zona de extrapolación*, donde conviene respaldarse en `Baseline_1B` / `Res_MLP_2_2`).
* **Transparencia *In-Sample* vs. LOOCV (Auditoría C-05):** Reentrenada sobre los $N = 24$ meses completos para producción y análisis XAI, `RNA_MLP_2` exhibe **$\text{RMSE}_{\text{in-sample}} = 0.0966\%\text{Cu}$ ($\text{NRMSE} = 6.75\%$)** frente a su desempeño ciego **$\text{RMSE}_{\text{LOOCV}} = 0.1758\%\text{Cu}$ ($\text{NRMSE} = 12.28\%$)**. Documentar abiertamente esta brecha previene cualquier acusación de sobreoptimismo: la red se ajusta con alta fidelidad al régimen regular mientras mantiene una brecha honesta de generalización explicada por el apalancamiento del pulso extremo de noviembre 2023.

### 4.3. Arquitectura Universal de `Código/Código unificado.ipynb` y Serialización Portable

Todo el flujo de los Bloques 0 al 4 quedó consolidado en una única plataforma ejecutable (**`Código/Código unificado.ipynb`**, documentada línea por línea en **`Código/traductor.md`**) diseñada con arquitectura **100% universal compatible con Google Colab y Jupyter Local**:

1. **Ingesta Agnóstica de Bases de Datos (Celda 3):** Permite cargar cualquier archivo `.csv` o `.xlsx` en formato tabular plano de 1 fila de encabezados (p. ej. `DATASET_AUDITADO_N24.csv`, `DATASET.csv`, `DATASET2.csv`, futuras campañas **2025+** o nuevas faenas como **Mina Tigresa** una vez completados sus balances reales —dado que `MATERIAL_TESIS/Data Tigresa.xlsx` hoy solo contiene datos en `PLAN` [`Ton` y `Plan %CuT`, 33 filas] con las columnas `CANCHA`/`Romana`/`PSJ`/`Canje` vacías en `NaN`) con autodetección de delimitadores (`sep=None, engine='python'`), conversión de comas decimales estilo Excel en español (`"2,16"` $\to$ `2.16`), homologación automática de unidades fraccionarias (`median < 0.25` $\to \times 100$), derivación automática por balance de masas si falta `Cancha_CuS` o `Cancha_CuIn`, y adaptación dinámica de $N$, rezagos ACF ($K = \min(12, \lfloor N/4 \rfloor)$), validación interna (`CV_INTERNO`), partición *Holdout* cronológica ($80\%/20\%$, que para $N=24$ reproduce exactamente `19/5`) y detección automática de empates técnicos ($\Delta\text{MSE} < 10^{-5}$).
2. **Serialización Portable sin Dependencia `__main__` (Auditoría C-03):** En `BLOQUE_4/` los modelos se exportan como diccionarios 100% nativos de `scikit-learn` (`modelo_ganador_rna_mlp2.joblib`, `modelo_ganador_residual_mlp.joblib`, `modelo_baseline_1b.joblib`), consumibles directamente desde el módulo desacoplado **`BLOQUE_4/modelos_rna.py`** sin riesgo de errores de *pickling* al desplegarlos en los Bloques 5 y 6.
3. **Simulador Operacional de Lotes Certificado (Celda 15 y Auditoría B4-02):** Al evaluar el lote de prueba oficial **`[Cancha_CuT = 1.75 %Cu, Cancha_CuS = 0.14 %Cu]`**, el motor ejecuta la verificación geométrica de Delaunay y entrega simultáneamente ambas recomendaciones de la Estrategia Dual:
   * **Verificación de dominio:** `DENTRO del Convex Hull histórico (Interpolación segura)`
   * **Proyección Campeón XAI (`RNA_MLP_2`):** **`1.4076 %Cu`**
   * **Proyección Campeón Residual (`Res_MLP_2_2`):** **`1.4096 %Cu`**
   * **Proyección Presupuestaria (`Baseline_1B`):** **`1.4096 %Cu`**

---

# CAPÍTULO 5
## Guía Maestra de Defensa Oral (18 Preguntas Críticas Reales: de Básico a Avanzado)

Esta sección prepara a Felipe Andrés Guerra Maturana para responder con rigor matemático, criterio minero-metalúrgico y claridad ejecutiva ante cualquier cuestionamiento del **Dr. José Delgado Vega** o de la comisión examinadora.

---

### NIVEL 1 — Fundamentos Estadísticos, Datos y Métricas de Evaluación

#### Pregunta 1: *"¿Qué es exactamente el MSE, qué diferencia tiene con RMSE, MAE y NRMSE, y por qué tu Cancha Directa tiene un $R^2$ negativo ($-1.2709$) si su correlación de Pearson es tan alta ($+0.8347$)?"*
* **Qué evalúa el tribunal:** Si comprendes de verdad las métricas que reportas o solo copiaste salidas de Python.
* **Respuesta técnica precisa:**
  * El **MSE** ($\frac{1}{N}\sum(y_k-\hat{y}_k)^2$, en $(\%\text{Cu})^2$) promedia los errores al cuadrado y **castiga fuertemente los errores grandes**; su raíz cuadrada es el **RMSE** (en $\% \text{Cu}$, unidades físicas directas); el **NRMSE** ($\frac{\text{RMSE}}{\bar{y}}\times 100\%$) expresa ese error en porcentaje respecto a la ley media (`1.4313%Cu`), permitiendo comparar con el $26\%$ histórico; y el **MAE** promedia errores absolutos sin amplificar outliers.
  * La aparente paradoja entre **$r = +0.8347$** y **$R^2 = -1.2709$** tiene una explicación algebraica exacta: **Pearson $r$ es invariante ante traslaciones constantes**, solo mide si `Cancha` y `Canje` suben y bajan juntos, ignorando que `Cancha` está desplazada $+0.3404\%\text{Cu}$ hacia arriba todos los meses. En cambio, **$R^2 = 1 - \frac{\text{SSE}}{\text{SST}}$** mide el error cuadrático real contra la identidad $y = \hat{y}$. Como el error cuadrático de `Cancha_Directa` ($\text{MSE} = 0.1350$) es más del doble de la varianza total de `Canje_CuT` ($\sigma_y^2 = 0.0594$), el cociente supera $1$ y **$R^2$ resulta $-1.2709$** —indicando que usar la cancha sin calibrar es peor que predecir siempre el promedio histórico.
* **Frase de cierre:** *"Una correlación alta significa que existe señal útil, pero un $R^2$ negativo prueba que la señal está severamente descalibrada por sesgo sistemático."*

#### Pregunta 2: *"Solo tienes $N = 24$ observaciones. ¿No es una muestra demasiado pequeña para hacer Machine Learning o Redes Neuronales?"*
* **Qué evalúa el tribunal:** Rigor estadístico y capacidad de defender la escala industrial del dataset.
* **Respuesta técnica precisa:**
  * Primero, desde el punto de vista minero, **$N = 24$ no son 24 muestras puntuales de laboratorio**: son **24 balances metalúrgicos mensuales consolidados** que representan **$295{,}884.67\text{ toneladas}$ oficiales en romana ENAMI** ($\sim 12{,}328.5\text{ t/mes}$), es decir, dos años completos de operación comercial real de Mantos Cobrizo.
  * Segundo, desde el punto de vista estadístico, trabajar con $N = 24$ exige un **protocolo especializado para muestras pequeñas**, que es exactamente el corazón metodológico de esta tesis: (a) validación **LOOCV** sin fuga (`Pipeline`), (b) **principio de parsimonia** respetando $N_{\text{train}} / P > 2.5$ (`RNA_MLP_2` tiene solo $9$ parámetros, ratio $2.56$), (c) optimizador de segundo orden **`L-BFGS`** en `float64` con ensamble de 10 semillas en lugar de mini-batches ruidosos, (d) **pruebas exactas no paramétricas** (Wilcoxon exacto y Bootstrap $B=10{,}000$), y (e) comparación transparente frente al piso de 1 parámetro (`Baseline_1B`).
* **Frase de cierre:** *"El riesgo en $N=24$ no es usar redes neuronales, sino usar redes sobreparametrizadas sin parsimonia ni validación ciega; justamente por eso nuestra red campeona XAI tiene solo 2 neuronas ocultas."*

#### Pregunta 3: *"¿Por qué no incluiste los tonelajes (`Cancha_Ton` o `Romana_Ton`) ni `Cancha_CuIn` como entradas de la red neuronal?"*
* **Qué evalúa el tribunal:** Dominio de multicolinealidad (VIF), física de leyes y separación de alcance metodológico.
* **Respuesta técnica precisa:**
  * **`Cancha_CuIn` se excluye por colinealidad química exacta:** como $\text{CuT} = \text{CuS} + \text{CuIn}$ (verificado con discrepancia máxima de $0.01\%\text{Cu}$), agregar `CuIn` dispara el **VIF desde $1.8045$ hasta $> 4{,}000$** (`4,754.0` en `CuT` y `4,004.1` en `CuIn`), causando matrices casi singulares e inestabilidad de gradientes. Incluso reemplazar `CuS` por `CuIn` junto a `CuT` eleva el VIF a $86.85$ ($r = 0.994$). Con $[\text{CuT}, \text{CuS}]$ preservamos el $100\%$ de la información química con $\text{VIF} = 1.8045 < 5.0$.
  * **Los tonelajes se excluyen en Bloques 1–4 por principio físico-estadístico:** la ley de canje ($\%\text{Cu}$) es una **propiedad intensiva** del mineral (independiente de la masa del lote), y agregar una tercera entrada consumiría valiosos grados de libertad en $N=24$. En cambio, `Romana_Ton` ($295{,}884.67\text{ t}$) se reserva íntegramente para el **Bloque 5**, donde valoriza en USD cada tonelada seca liquidada por ENAMI.
* **Frase de cierre:** *"Dos entradas bien condicionadas ($\text{VIF} = 1.80$) contienen toda la química de cancha sin sacrificar grados de libertad."*

#### Pregunta 4: *"¿Por qué aplicaste `StandardScaler` obligatorio y cómo justificas que no se distorsionó por los valores extremos de `Cancha_CuS`?"*
* **Qué evalúa el tribunal:** Comprensión del efecto de escala sobre gradientes y regularización $L_2$.
* **Respuesta técnica precisa:**
  * En el Bloque 1 demostramos que la varianza muestral de `Cancha_CuT` ($0.058128$) es **$61.62$ veces mayor** ($61.64\times$ sobre $s^2$ a 6 dec.) que la de `Cancha_CuS` ($0.000943$). Sin escalar, cualquier modelo regularizado por norma $L_2$ o entrenado por gradientes sería dominado por `CuT`, apagando a `CuS` pese a que `CuS` tiene mayor volatilidad relativa ($\text{CV} = 20.88\%$ vs. $13.61\%$).
  * Debido a que `Cancha_CuS` presenta fuerte curtosis ($+5.8621$, por el pulso oxidado de octubre 2023), **no asumimos a ciegas que Z-score fuera inocuo**: ejecutamos en el Bloque 2 un contraste completo frente a **`RobustScaler`** (mediana e IQR). La diferencia en todos los modelos resultó menor o igual a **$0.09\%$ de NRMSE** (p. ej., `RidgeCV`: $11.78\%$ vs. $11.69\%$; `LassoCV`: $12.30\%$ vs. $12.35\%$), certificando la estabilidad absoluta de `StandardScaler` encapsulado dentro de `Pipeline`.
* **Frase de cierre:** *"Escalar no fue un paso cosmético; corrigió una asimetría de varianzas de 61.6 veces y fue auditado frente a un escalador robusto."*

#### Pregunta 5: *"Detectaste dos outliers estadísticos con Mahalanobis (octubre y noviembre de 2023). ¿Por qué no los eliminaste para obtener un modelo más limpio y con menor error?"*
* **Qué evalúa el tribunal:** Criterio ingenieril y honestidad científica (evitar *cherry-picking* de datos).
* **Respuesta técnica precisa:**
  * Aunque octubre 2023 ($D_M^2 = 14.2481$, frente oxidado con $\text{CuS} = 0.26\%$) y noviembre 2023 ($D_M^2 = 8.4700$, frente rico sulfurado con $\text{CuT} = 2.47\%, \text{Canje} = 2.27\%$) superan el umbral heurístico $\chi^2_{2, 0.95} = 5.9915$, **no son errores de digitación ni fallas de laboratorio**: son **despachos industriales reales facturados por ENAMI** que representan precisamente los meses de mayor impacto económico en el negocio minero.
  * Eliminar meses de alta ley induciría un **sesgo hacia la esterilidad** inaceptable en ingeniería de minas. En lugar de borrarlos, hicimos un análisis diagnóstico separado ($N=22$) para medir su efecto de apalancamiento (viendo que Pearson de `CuT` cae de $+0.8347$ a $+0.6181$) y gestionamos su influencia matemáticamente mediante **regularización $L_2$**, **arquitecturas residuales** y **contrastes no paramétricos**.
* **Frase de cierre:** *"En minería los meses más ricos son los que más dinero ponen en juego; un modelo que solo funciona borrando los frentes ricos no sirve en faena."*

---

### NIVEL 2 — Diseño Experimental, Validación y Modelos *Baselines*

#### Pregunta 6: *"¿Por qué no usaste un `train_test_split` 80/20 clásico y por qué afirmas que tu validación tiene 'cero fuga de datos'?"*
* **Qué evalúa el tribunal:** Dominio de validación cruzada y prevención de *data leakage*.
* **Respuesta técnica precisa:**
  * Con $N=24$, un `train_test_split` 80/20 aleatorio deja solo 5 datos de prueba y su resultado depende por completo de la suerte de qué meses cayeron en esos 5 registros. Con **LOOCV**, cada uno de los 24 meses es evaluado de manera ciega entrenando con los otros 23, obteniendo una evaluación exhaustiva y determinista sobre el 100% del historial. Además, sí incluimos una partición cronológica 80/20 (**Holdout 19/5**) como verificación prospectiva complementaria.
  * Garantizamos **cero fuga (*Zero Data Leakage*)** porque **absolutamente todo** —la media y desviación de `StandardScaler`, la media de sesgo $\bar{e}_{\text{train}, -k}$ en `Baseline_1B`, el escalado `scale_y` y la búsqueda interna de hiperparámetros (`RidgeCV`, `LassoCV`, `ElasticNetCV`, `GridSearchCV` con `cv=5`)— se ajusta dentro de `sklearn.pipeline.Pipeline` **exclusivamente sobre los 23 meses de entrenamiento**, sin consultar nunca el registro $k$ de prueba.
* **Frase de cierre:** *"Cada una de nuestras 24 predicciones se generó como si ese mes todavía no hubiera ocurrido."*

#### Pregunta 7: *"Tus variables `Cancha_CuT` tienen autocorrelación de rezago 1 ($r_1 = +0.4467$). ¿Eso no invalida LOOCV por fuga temporal entre meses vecinos?"*
* **Qué evalúa el tribunal:** Rigor geoestadístico (especialidad directa del Dr. José Delgado Vega!).
* **Respuesta técnica precisa:**
  * Hicimos dos blindajes complementarios siguiendo Box-Jenkins ($K \le N/4 = 6$, banda Bartlett $\pm 0.4001$). Primero, demostramos que el **error de conciliación (`Sesgo_Canje`) es ruido blanco puro** en todos los rezagos ($r_1 = -0.0866$, $r_6 = -0.1036$, y Durbin-Watson $d \approx 2.0$ en los modelos campeones), lo que valida formalmente la independencia de errores en LOOCV.
  * Segundo, para responder de frente a la inercia geológica de `Cancha_CuT` ($r_1 = +0.4467$), ejecutamos **`Buffered-LOOCV`**, expulsando del entrenamiento tanto $t-1$ como $t+1$. El resultado fue contundente: `Baseline_1B` y `Residual-MLP` se mantuvieron intactos (`10.07%` $\to$ `10.04%`), `LassoCV` varió apenas $+0.04\%$ (`12.30%` $\to$ `12.34%`), y el aumento de `RNA_MLP_2` (`12.28%` $\to$ `14.53%`) no se debió a memoria temporal sino a la **adyacencia física de octubre y noviembre 2023**: al evaluar noviembre 2023 (`2.27%`), el *buffer* elimina a octubre 2023 (`1.79%`), truncando el entrenamiento a un máximo de `1.57%Cu`.
* **Frase de cierre:** *"Los modelos no están memorizando el mes anterior; están aprendiendo la relación geoquímica contemporánea $[\text{CuT}_t, \text{CuS}_t] \to \text{Canje}_t$."*

#### Pregunta 8: *"¿Cómo explicas que `Baseline_1B` (restar una constante) venza en LOOCV ($10.07\%$) a `OLS` ($11.85\%$), `RidgeCV` ($11.78\%$) y `SVR_RBF` ($16.68\%$)? ¿No significa eso que el Machine Learning no aporta?"*
* **Qué evalúa el tribunal:** Capacidad de análisis crítico profundo del balance sesgo-varianza.
* **Respuesta técnica precisa:**
  * En primer lugar, la descomposición del MSE histórico probó que el **$85.86\%$** del error de cancha es sesgo constante ($\bar{e} = +0.3404\%\text{Cu}$). `Baseline_1B` corrige ese $85.86\%$ estimando solo **1 parámetro** y fijando la pendiente física exactamente en **$1.0$** ($\hat{y} = 1.0\cdot\text{CuT} - \bar{e}_{-k}$).
  * `OLS` pierde frente a `Baseline_1B` porque estima 3 parámetros con una pendiente achatada ($\approx 0.74 < 1.0$) que falla justo en los extremos: el **$40.48\%$** del error cuadrático de `OLS` viene de solo dos meses (sobreestima octubre 2023 en $+0.3284\%$ por proyectar linealmente `CuS = 0.26%`, y subestima noviembre 2023 en $-0.4145\%$). `SVR_RBF` fracasa en `16.68%` porque el kernel gaussiano decae a la media en zonas de baja densidad de soporte.
  * ¿Aporta entonces el Machine Learning? **Rotundamente sí**, y por tres pruebas empíricas verificadas: (1) `LassoCV` retuvo `Cancha_CuS` en **24/24 pliegues**, (2) `RNA_MLP_2` **vence a `Baseline_1B` en 14 de los 24 meses ($58.3\%$)**, reduce el error mediano de `0.1044%` a **`0.0801%Cu`**, gana en el pulso oxidado de octubre 2023 (`0.0632%` vs. `0.0830%`), y vence en los 23 meses sin noviembre 2023 (`10.22%` vs. `10.33%`), y (3) en **Holdout prospectivo 19/5**, `RNA_MLP_2` logra **`6.56%`** frente a **`8.96%`** de `Baseline_1B`.
* **Frase de cierre:** *"Incluir `Baseline_1B` demuestra rigor científico: nos permitió descubrir cuándo conviene la simplicidad extrema (extremos presupuestarios) y cuándo la no linealidad de la RNA supera a la constante (régimen regular y control de óxidos)."*

---

### NIVEL 3 — Redes Neuronales, Hiperparámetros y Reconciliación con el Profesor Guía

#### Pregunta 9: *"¿Por qué mi red original en Keras daba alrededor de $8.63\%$ y en tu tesis esa misma arquitectura sube a $11.64\%$?"*
* **Qué evalúa el tribunal:** Pregunta directa del Dr. José Delgado Vega; exige rigor matemático exacto y diplomacia académica impecable.
* **Respuesta técnica precisa:**
  * Replicamos exactamente su arquitectura en TensorFlow/Keras (`Dense(4)-Dense(4)-Dense(4)-Dense(1)`, $57$ parámetros nominales) y confirmamos que su cifra de marzo 2025 era completamente coherente dentro de su contexto exploratorio: cuando se entrena y evalúa **sobre los mismos 24 registros (*in-sample*)**, nuestra réplica arroja **$\text{MAPE} = 7.97\%$** ($\text{NRMSE} = 9.91\%$) sobre `PSJ_CuT` y **$\text{MAPE} = 7.38\%$** ($\text{NRMSE} = 9.19\%$) sobre `Canje_CuT`, exactamente en el orden de magnitud del $8.63\%$.
  * La diferencia al subir a **$11.64\%$** responde a tres avances metodológicos obligatorios de la tesis: (1) evaluar sobre la ley oficial de liquidación `Canje_CuT`, (2) pasar de evaluación *in-sample* a **validación estrictamente ciega LOOCV** (donde el MAPE es $8.74\%$ y el NRMSE es **$11.64\%$**), y (3) descubrir matemáticamente que al omitir funciones de activación no lineales (`relu`/`tanh`), las 4 capas densas se multiplican matricialmente ($\mathbf{W}_{\text{eq}} = \mathbf{W}_4\mathbf{W}_3\mathbf{W}_2\mathbf{W}_1$) colapsando sus $57$ pesos a solo **$P_{\text{eff}} = 3$ parámetros lineales** (error algebraico máximo $3.12 \times 10^{-7}\%\text{Cu}$, difiriendo apenas $0.0023\%\text{Cu}$ de `OLS`). Ese descubrimiento fue justamente el punto de partida que nos motivó a diseñar redes verdaderamente no lineales y parsimoniosas.
* **Frase de cierre:** *"Su modelo preliminar fue el diagnóstico correcto para abrir el camino; al auditarlo en LOOCV y descubrir su colapso lineal, supimos exactamente qué red compacta no lineal construir."*

#### Pregunta 10: *"¿Por qué utilizaste el optimizador `L-BFGS` en tus 10 redes MLP y no `Adam` o `SGD` como es habitual en Deep Learning? ¿Y por qué cambiaste `Adam(0.1)` a `Adam(0.05)` en la réplica Keras?"*
* **Qué evalúa el tribunal:** Comprensión profunda de optimización numérica aplicada al tamaño de muestra.
* **Respuesta técnica precisa:**
  * `SGD` y `Adam` son optimizadores de **primer orden** (usan solo el gradiente) pensados para conjuntos masivos donde no cabe todo el dataset en memoria y se requiere descenso estocástico por *mini-batches* en `float32`. Con solo **$N_{\text{train}} = 23$** filas, dividir en *mini-batches* introduce ruido innecesario y requiere ajustar manualmente tasa de aprendizaje y número de épocas.
  * **`L-BFGS`** es un método **cuasi-Newton de segundo orden** en precisión doble (`float64`) que estima la curvatura local (inversa aproximada del Hessiano) sobre los 23 datos completos en cada paso con búsqueda lineal exacta. En muestras pequeñas y redes de $9\text{ a }37$ parámetros, `L-BFGS` converge con precisión de $10^{-5}$ de forma mucho más rápida, estable y reproducible que `Adam`.
  * En la réplica Keras mantuvimos `Adam` para respetar fielmente el código original del profesor guía, reduciendo el paso de `0.1` a **`0.05`** (800 épocas) exclusivamente en los 24 pliegues LOOCV porque con $N_{\text{train}}=23$ una tasa de `0.1` producía oscilaciones alrededor del mínimo sin cambiar la equivalencia lineal $P_{\text{eff}} = 3$.
* **Frase de cierre:** *"En $N_{\text{train}} = 23$, `L-BFGS` aprovecha la geometría de segundo orden sobre el batch completo, algo inviable en millones de datos pero ideal para ingeniería de procesos."*

#### Pregunta 11: *"¿Por qué `RNA_MLP_2` usa activación `tanh`, $\alpha = 0.001$ y `scale_y = False`, mientras que `RNA_MLP_3` y `RNA_MLP_4` usan `relu`, $\alpha = 5.0$ y `scale_y = True`?"*
* **Qué evalúa el tribunal:** Que entiendas cada hiperparámetro y también cómo te defiendes contra acusaciones de sobreajuste de hiperparámetros (*model selection bias*).
* **Respuesta técnica precisa:**
  * Se evaluó una **grilla predefinida a priori** de activaciones (`relu`, `tanh`), regularizaciones $L_2$ ($\alpha \in \{0.001, 1.0, 2.0, 5.0\}$) y escalado del objetivo (`scale_y`), reportando por transparencia la configuración más estable de cada topología:
    * **`RNA_MLP_2` (`tanh`, $\alpha = 0.001$, `scale_y = False`):** Con solo 2 neuronas ($9$ parámetros), la red ya está regulada estructuralmente por parsimonia; `tanh` aporta curvatura suave centrada en cero ideal para modelar la modulación no lineal de `CuS`, $\alpha = 0.001$ mantiene activos sus pesos sin aplastarlos, y la capa de salida lineal absorbe directamente la escala física de `Canje_CuT`.
    * **`RNA_MLP_3/4/2_2/3_2/Delgado_MLP` ($\alpha \in \{2.0, 5.0\}$, `scale_y = True`):** Al crecer la capacidad ($13\text{ a }37$ parámetros), se requiere regularización $L_2$ intensa para frenar el sobreajuste; `relu` se contrae limpiamente sin saturación sigmoidea y escalar `y` dentro del pliegue (`scale_y = True`) alinea la escala del error MSE ($\sim O(1)$) con el término de penalización $\alpha$, evitando que el regularizador empuje la salida hacia cero.
  * Para blindarnos frente a cualquier sesgo optimista de selección en $N=24$, **no declaramos a `RNA_MLP_2` campeona solo por LOOCV**: la contrastamos contra el benchmark sin hiperparámetros `Baseline_1B` y la convalidamos en la ventana cronológica intacta **Holdout 19/5**, donde logró **`6.56%`**.
* **Frase de cierre:** *"A mayor número de neuronas necesitamos mayor regularización $L_2$ y escalado de salida; con solo 2 neuronas, la propia parsimonia actúa como regularizador natural."*

#### Pregunta 12: *"Si tu `Residual-MLP` (`Res_MLP_2_2`) termina empatando a `Baseline_1B` ($10.07\%$) porque sus pesos sinápticos se apagan a cero ($P_{\text{eff}} \approx 1$), ¿para qué sirvió implementarla?"*
* **Qué evalúa el tribunal:** Madurez investigativa y capacidad de extraer valor científico de un resultado de contracción.
* **Respuesta técnica precisa:**
  * En ciencia aplicada, demostrar que una red compleja se contrae automáticamente hacia el estimador óptimo simple es un **resultado de alto rigor**, no un desperdicio. Implementar `Residual-MLP` ($\hat{y}_k = \text{Cancha\_CuT}_k - \hat{e}_{\text{MLP}}(\text{CuT}_k, \text{CuS}_k)$) aportó tres hallazgos concretos:
    1. **Resolución del problema de extrapolación:** Probó que la *skip-connection* elimina la atenuación de pendiente que afectaba a todas las redes directas en noviembre 2023 (`2.1235%` frente a `1.7467%` de `RNA_MLP_2`).
    2. **Prueba empírica de la frontera de complejidad:** Al reducir $\alpha$ para forzar curvatura en el residuo, el error empeoró sistemáticamente ($\alpha = 1.0 \to 10.26\%$; $\alpha = 0.5 \to 10.73\%$; $\alpha = 0.1 \to 11.94\%$). Que el optimizador elija $\|\mathbf{W}\| \to 0$ en $\alpha = 2.0$ demuestra matemáticamente que en $N=24$ el piso global de mínima varianza es efectivamente el sesgo constante.
    3. **Escalabilidad futura automática:** En cuanto la faena acumule $N = 48$ o $60$ meses en futuras campañas, esa misma arquitectura residual dispondrá de suficiente soporte estadístico para "encender" sus pesos no lineales sobre el residuo sin perder su red de seguridad asintótica.
* **Frase de cierre:** *"La red residual nos dio un mecanismo a prueba de extrapolaciones catastróficas hoy, y lista para crecer en complejidad mañana cuando haya más meses de historia."*

#### Pregunta 13: *"¿Por qué entrenaste un ensamble de 10 semillas fijas por cada red en lugar de entrenar una sola red con `random_state=42`?"*
* **Qué evalúa el tribunal:** Conocimiento de no convexidad en optimización de redes neuronales.
* **Respuesta técnica precisa:**
  * A diferencia de `OLS` o `Ridge` (cuya función objetivo es estrictamente convexa y tiene un único mínimo global), la superficie de pérdida de un MLP no lineal es **no convexa** y posee múltiples mínimos locales. Entrenar con una sola semilla en $N_{\text{train}}=23$ hace que la predicción dependa de la suerte de los pesos iniciales.
  * Al promediar 10 redes entrenadas desde 10 semillas fijas declaradas (`[42, 101, 2023, 7, 13, 99, 123, 456, 789, 999]`), reducimos drásticamente la **varianza de inicialización** (*seed bagging*) y garantizamos reproducibilidad científica exacta en cualquier computador.
* **Frase de cierre:** *"El ensamble de 10 semillas transforma un entrenamiento sensible al azar inicial en un estimador estable y reproducible."*

---

### NIVEL 4 — Metalurgia (Pierre Gy), Econometría, Estrategia Dual y Escalabilidad Industrial

#### Pregunta 14: *"Un ingeniero metalurgista te pregunta: ¿Por qué dices que `Cancha_CuS` afecta a `Canje_CuT`? ¿Acaso los óxidos no se pierden recién en las celdas de flotación de Planta San José?"*
* **Qué evalúa el tribunal:** La trampa metalúrgica más peligrosa de toda la defensa (Auditorías C-01 y B3-01).
* **Respuesta técnica precisa:**
  * **Exactamente ahí radica la distinción clave que auditamos y blindamos en esta tesis:** `Canje_CuT` **NO es una ley de concentrado ni depende de la recuperación metalúrgica en celdas de flotación**. `Canje_CuT` es la **ley de cabeza química de recepción** acordada comercialmente entre Mantos Cobrizo y ENAMI mediante el **protocolo reglamentario de tres sobres** (productor, ENAMI y dirimencia) *antes* de que el lote entre a molienda y flotación.
  * El mecanismo por el cual `Cancha_CuS` explica una parte real de la brecha (`-1.53%` en ablación y `SHAP = 0.0701%Cu` en `RNA_MLP_2`) ocurre **aguas arriba, en la cancha de acopio de la mina**, conforme a la **Teoría de Muestreo de Pierre Gy**: las especies oxidadas supergenas (crisocola, malaquita) poseen menor dureza de Mohs y mayor **friabilidad** que los sulfuros competentes. Al chancar y apilar en cancha, esos óxidos se pulverizan diferencialmente y sufren **segregación gravitacional de finos**, distorsionando la representatividad del muestreo de acopio en faena respecto a la torre cortadora automática de ENAMI.
* **Frase de cierre:** *"No es un fenómeno de cinética de flotación; es un fenómeno físico de friabilidad diferencial y segregación de muestreo en cancha regido por Pierre Gy."*

#### Pregunta 15: *"En el Bloque 4 muestras que `OLS`, `RidgeCV`, `LassoCV` y `Keras` fallan Breusch-Pagan (heterocedásticos), mientras `SVR_RBF` marca $p = 0.7243$ (homocedástico). ¿Qué significa eso en términos prácticos para la mina?"*
* **Qué evalúa el tribunal:** Profundidad econométrica y lectura crítica de pruebas estadísticas.
* **Respuesta técnica precisa:**
  * **Homocedasticidad** significa que la incertidumbre (varianza del error) del modelo se mantiene pareja en todo el rango de leyes. Que **todos los modelos lineales fallen Breusch-Pagan** (`OLS` $p=0.0016$, `LassoCV` $p=0.0029$, `ElasticNetCV` $p=0.0035$, `Keras` $p=0.0040$, `RidgeCV` $p=0.0228$) significa que en los meses de mayor ley —justamente cuando cada camión vale más dólares— sus errores se agrandan de forma desproporcionada y sus intervalos de confianza dejan de ser válidos.
  * En cambio, **`Baseline_1B` / `Res_MLP_2_2` ($p = 0.4075$)** y **`RNA_MLP_2` ($p = 0.0822$)** mantienen varianza de error estable y residuos independientes ($d \approx 2.0$) y normales (Shapiro $p > 0.10$).
  * Y respecto a `SVR_RBF` ($p = 0.7243$), es un caso de manual de por qué una prueba estadística no se lee aislada: Breusch-Pagan busca una tendencia *lineal* en $e^2$ vs. $\hat{y}$. Como `SVR_RBF` comete errores enormes tanto abajo (`2023-09`) como arriba (`2023-11`), sus errores cuadrados forman una **"U"** cuya pendiente lineal promedio es casi cero, pese a tener el triple de MSE (`0.056986`).
* **Frase de cierre:** *"Los modelos lineales pierden precisión justo cuando el lote es más valioso; nuestros dos modelos campeones conservan estabilidad estadística en todo el rango."*

#### Pregunta 16: *"¿Qué es el *Convex Hull* (Envolvente Convexa) que dibujaste en la Superficie 2D y cómo evita errores en la operación diaria de la mina?"*
* **Qué evalúa el tribunal:** Aplicación práctica de geometría computacional para gestión de riesgo operacional.
* **Respuesta técnica precisa:**
  * El **Convex Hull** es el polígono convexo más pequeño que encierra a los 24 pares históricos reales `(Cancha_CuT, Cancha_CuS)`. Todo punto dentro de ese polígono representa un lote respaldado por experiencia operativa previa (**interpolación**); todo punto fuera —como la esquina vacía con $\text{CuT} \le 1.80\%$ y $\text{CuS} > 0.16\%$— es una zona donde la red jamás vio un dato real (**extrapolación**).
  * En nuestro simulador (`Código unificado.ipynb`, Celda 15), implementamos una **triangulación de Delaunay** (`Delaunay(X_2d).find_simplex(x_nuevo)[0] >= 0`) que clasifica automáticamente cada lote nuevo. Por ejemplo, el lote certificado `[1.75, 0.14]` cae estrictamente dentro del *Convex Hull*, donde `RNA_MLP_2` proyecta **`1.4076 %Cu`** con plena confianza; si mañana llega un lote de un frente anómalo fuera del polígono, el sistema despliega una alerta de extrapolación recomendando respaldarse en el pilar conservador (`Baseline_1B` / `Res_MLP_2_2`).
* **Frase de cierre:** *"El Convex Hull convierte a la red neuronal en una herramienta con conciencia explícita de sus propios límites físicos."*

#### Pregunta 17: *"En definitiva, si eres el jefe de operaciones de Mantos Cobrizo mañana lunes, ¿cuál modelo usas para despachar camiones y cuál para presentar el presupuesto anual?"*
* **Qué evalúa el tribunal:** Criterio gerencial y dominio práctico de la Estrategia Dual.
* **Respuesta técnica precisa:**
  * Uso **ambos de manera coordinada** según la decisión de negocio, tal como lo entrega nuestro simulador:
    1. **Para el presupuesto anual y flujo de caja financiero:** Uso el **Campeón de Mínima Varianza (`Baseline_1B` / `Res_MLP_2_2`)**, porque tiene el menor error cuadrático global en LOOCV (`10.07%`, $R^2 = +0.6504$), $P_{\text{eff}}=1$ y total inmunidad frente a pulsos extraordinarios de alta ley.
    2. **Para la operación semanal, *blending* en cancha y control de recepción ENAMI:** Uso el **Campeón Geometalúrgico (`RNA_MLP_2`)**, porque supera al modelo constante en **$14$ de $24$ meses ($58.3\%$)**, exhibe menor error mediano (`0.0801` vs. `0.1044%Cu`), logra **`6.56%`** en la ventana prospectiva regular 19/5, cuantifica mediante SHAP (`0.0701%Cu`) la penalización de muestreo asociada a la solubilidad (`CuS`) y permite definir un umbral objetivo para decidir cuándo solicitar la apertura del **Sobre 2** ante discrepancias con Planta San José.
* **Frase de cierre:** *"Un modelo protege el balance financiero ante eventos extremos; el otro optimiza la operación diaria y explica la geometalurgia del acopio."*

#### Pregunta 18: *"¿Qué pasa si el próximo año llegan los datos de 2025 o queremos aplicar tu metodología a otra faena como Mina Tigresa con otro formato de CSV?"*
* **Qué evalúa el tribunal:** Escalabilidad tecnológica, ingeniería de software y transferibilidad de la tesis.
* **Respuesta técnica precisa:**
  * Precisamente para evitar un código rígido atado a Mantos Cobrizo, desarrollamos **`Código/Código unificado.ipynb`** como una **plataforma universal ejecutable en Google Colab** (y Jupyter Local). Solo se requiere subir el archivo `.csv` o `.xlsx` en formato tabular plano (1 fila de encabezados con columnas reales de leyes de origen y destino; en particular, cabe precisar que el archivo actual `MATERIAL_TESIS/Data Tigresa.xlsx` solo contiene datos en `PLAN` [`Ton` y `Plan %CuT`, 33 filas] con las columnas `CANCHA`/`Romana`/`PSJ`/`Canje` vacías en `NaN`, por lo que requiere completar sus balances reales de cancha y canje en formato tabular plano antes de entrenar) en la Celda 3 (arrastrando o activando `SUBIR_DESDE_COLAB = True`) e indicar los nombres de las columnas `COL_X1`, `COL_X2` y `COL_Y`.
  * El cuaderno autodetecta el delimitador (`,` o `;`), corrige comas decimales de Excel en español, homologa automáticamente leyes fraccionarias (`0.018` $\to$ `1.80%Cu`), deriva `CuS` o `CuIn` por balance de masas si alguna falta, ajusta dinámicamente $N$, los rezagos ACF ($K \le N/4$), la validación interna, el *Holdout* 80/20, la detección de empates técnicos, las pruebas de hipótesis, el XAI y el simulador con *Convex Hull*. Además, para operación diaria sin reentrenamiento, los artefactos nativos `.joblib` se consumen en milisegundos vía `BLOQUE_4/modelos_rna.py`.
* **Frase de cierre:** *"No entregamos solo un experimento histórico sobre 24 filas; entregamos un motor metodológico y computacional reutilizable para cualquier faena proveedora de ENAMI."*

---

# CAPÍTULO 6
## Conclusiones Generales y Hoja de Ruta hacia Bloques 5 y 6

### 6.1. Conclusiones Técnicas Consolidadas (Bloques 0 al 4)

1. **Certificación Integral de Datos y Marco Comercial (Bloque 0):** Se consolidaron $N = 24$ balances mensuales ($286{,}704\text{ t}$ en cancha; $295{,}884.67\text{ t}$ en romana ENAMI) con 0 nulos/duplicados, leyes homologadas a $\% \text{Cu}$ y 24 circulares tarifarias ENAMI Sulfuros estandarizadas $100\%$ en USD/tms (maquila implícita verificada en $34.66$, $35.65$ y $39.60\text{ USD/tms}$). Se reconcilió el $26\%$ histórico como un NRMSE de **$25.67\%$** derivado de una sobreestimación sistemática de **$+0.3404\%\text{Cu}$** en los $24/24$ meses ($R^2 = -1.2709$).
2. **Fundamentación Física y Econométrica Blindada (Bloque 1):** Se estableció la base metalúrgica correcta bajo la **Teoría de Muestreo de Pierre Gy** y el protocolo reglamentario de **3 sobres ANTES de molienda y flotación** (friabilidad diferencial y segregación de finos de óxidos `CuS`). Se demostró que el $85.86\%$ del MSE histórico es sesgo y el $14.14\%$ varianza; que el ratio de varianzas de **$61.62\times$** obliga a `StandardScaler`; que el sesgo es normal ($W=0.9852, p=0.9695$) y ruido blanco ($r_1=-0.0866$, validando LOOCV); y que el espacio 2D $[\text{CuT}, \text{CuS}]$ posee $\text{VIF} = 1.8045 < 5.0$ (frente a $>4{,}000$ con `CuIn`).
3. **Modelado Ciego de 18 Arquitecturas sin Fuga (Bloques 2 y 3):** Se demostró la retención del $100\%$ ($24/24$ pliegues) de `Cancha_CuS` en `LassoCV`, el colapso algebraico de la réplica Keras del Dr. Delgado ($57 \to P_{\text{eff}} = 3$, reconciliando su $8.63\%$ *in-sample* frente a $11.64\%$ LOOCV), la superioridad de la parsimonia (`RNA_MLP_2`, 9 parámetros, `L-BFGS`, 10 semillas) y el aporte neto de `Cancha_CuS` ($-1.53\%$ en ablación).
4. **Validación Estadística, Estrategia Dual y Explicabilidad XAI (Bloque 4):** Todos los modelos calibrados destruyen estadísticamente a `Cancha_Directa` (Wilcoxon $p \le 0.000186$, IC Bootstrap disjuntos). Mientras todos los modelos lineales fracasan por heterocedasticidad en Breusch-Pagan ($p \le 0.0228$), la **Estrategia Dual** combina dos pilares homocedásticos e independientes: **`Baseline_1B` / `Res_MLP_2_2`** (empate técnico cuádruple `1=`, `10.07%` LOOCV, $P_{\text{eff}}=1$) para protección presupuestaria y **`RNA_MLP_2`** (`6.56%` Holdout 19/5, vence en `14/24` meses, menor error mediano `0.0801%Cu`, `SHAP CuS = 0.0701%Cu`, delimitada por el *Convex Hull*) para control operacional y geometalúrgico.

### 6.2. Hoja de Ruta Inmediata hacia los Bloques 5 y 6

Con los **Bloques 0 al 4 auditados, reconciliados y sellados** y los artefactos `.joblib` 100% portables listos en `BLOQUE_4/`, el cierre del Trabajo de Titulación avanza directamente hacia los dos módulos finales:

| Módulo Final | Ubicación y Entregable Clave | Alcance Técnico y Metodológico Exacto |
| :--- | :--- | :--- |
| **BLOQUE 5**<br>*Modelo Económico Dual ENAMI y Simulación de Ahorros* | `BLOQUE_5/05_modelo_economico_enami.ipynb` | Cruzar los 24 vectores ciegos OOS de ambos campeones (`Baseline_1B` / `Res_MLP_2_2` y `RNA_MLP_2`) contra `BLOQUE_0/TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.csv` (100% USD/tms, previa doble verificación visual de `ABRIL 2023.pdf`) y **`Romana_Ton = 295,884.67 t`**, reportando rigurosamente por separado:<br>• **Nivel 1 (Impacto Presupuestario en USD):** Reducción exacta de la desviación entre el ingreso esperado en mina y el ingreso real liquidado por ENAMI ($\sum |V_{\text{Cancha}} - V_{\text{Real}}| - \sum |V_{\text{Modelo}} - V_{\text{Real}}|$).<br>• **Nivel 2 (Ahorro Real de Caja en USD):** Simulación contrafactual de decisiones operativas: falsos fletes y cargos de maquila evitados en lotes bajo ley de corte económica (*cut-off*), optimización de *blending* preventivo en cancha y ahorro de costos de contraanálisis innecesarios de laboratorio (**Sobre 2**). |
| **BLOQUE 6**<br>*Transferencia Operativa, SOP y Memoria Final* | `BLOQUE_6/app.py` + `INFORME_TESIS.docx` | • **Aplicación Web Productiva (Streamlit + Plotly):** Arquitectura desacoplada de 3 capas consumiendo directamente los artefactos nativos `.joblib` vía `BLOQUE_4/modelos_rna.py` (sin reentrenamiento en inferencia), integrando verificador automático de **Convex Hull**, predicción dual instantánea y calculadora de tarifa neta ENAMI en USD.<br>• **Procedimiento Operacional Estándar (SOP):** Formalizar el rol reglamentario de la predicción RNA como criterio técnico objetivo para gatillar la apertura del **Sobre 2** (antesala legal del **Sobre 3** de dirimencia ENAMI).<br>• **Consolidación de Memoria (`INFORME_TESIS.docx`):** Traspaso íntegro de tablas certificadas, figuras a 200 DPI y narrativa coherente de principio a fin conforme a las auditorías RCI de los Bloques 0 al 5. |

---
*Informe técnico entregable y guía maestra de defensa oral elaborados y verificados conforme a la Metodología v4.0 — Departamento de Ingeniería en Minas, Universidad de Antofagasta.*
