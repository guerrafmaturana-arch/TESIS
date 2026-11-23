# INFORME TÉCNICO ENTREGABLE — BLOQUE 4
## Validación Predictiva Unificada, Selección del Modelo Ganador (*Champion Model*), Diagnóstico de Residuos y Explicabilidad Geometalúrgica (XAI)

**Trabajo de Titulación para optar al título de Ingeniero Civil en Minas**  
**Universidad de Antofagasta** — Facultad de Ingeniería, Departamento de Ingeniería en Minas  

* **Autor:** Felipe Andrés Guerra Maturana  
* **Profesor Guía:** Dr. José Delgado Vega (*Doctor en Geología, Especialista en Geoestadística y Explotaciones Mineras, École des Mines de Paris*)  
* **Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)  
* **Módulo:** Bloque 4 — Benchmarking Unificado ($18$ Modelos), Inferencia No Paramétrica y Explicabilidad XAI  
* **Ubicación del Módulo:** `./BLOQUE_4/`  
* **Versión Metodológica:** 4.0 (Línea Temporal Estricta — Cero Fuga de Información, Inferencia Robusta, Rigor Econométrico y Auditoría RCI)  
* **Fecha de Emisión:** Noviembre 2026  

---

## Resumen Ejecutivo

El presente informe consolida el cierre de la etapa predictiva de la tesis de titulación, materializando la **evaluación imparcial, exhaustiva y simultánea de los $18$ modelos matemáticos** desarrollados en el Bloque 2 (7 baselines econométricos y geoestadísticos) y en el Bloque 3 (11 redes neuronales directas y residuales). Toda la evaluación se encuentra estrictamente anclada en el bucle de validación cruzada dejando uno fuera (**LOOCV**, $N = 24$), garantizando **cero fuga de información (*zero data leakage*)** y absoluta trazabilidad auditable.

```
+----------------------------------------------------------------------------------------------------+
|                         MATRIZ RESUMEN DE HITOS Y RESULTADOS DEL BLOQUE 4                          |
+----------------------------------------------------------------------------------------------------+
| 1. Benchmarking 18 Modelos:                                                                       |
|    - Brecha Operacional Terreno (Cancha Directa): NRMSE = 25.67%, RMSE = 0.3674%Cu, R² = -1.27     |
|    - Campeón Mínima Varianza (Empate 1=): Baseline 1B / Res_MLP_2_2 (NRMSE = 10.07%, R² = +0.6504)  |
|    - Campeón Explicabilidad XAI: RNA_MLP_2 (NRMSE LOOCV = 12.28%, NRMSE Holdout = 6.56%, R² = 0.48) |
+----------------------------------------------------------------------------------------------------+
| 2. Diagnóstico de Residuos e Inferencia Estadística:                                               |
|    - Homocedasticidad (Breusch-Pagan exacto): Baseline 1B (p = 0.4075) y RNA_MLP_2 (p = 0.0822)    |
|      preservan varianza constante; modelos lineales clásicos fallan (OLS p = 0.0016, Ridge p = 0.02)|
|    - Independencia Serial: Durbin-Watson d = 1.958 (RNA_MLP_2) y d = 2.014 (Baseline 1B)           |
|    - Normalidad: Shapiro-Wilk W = 0.9307 (p = 0.101 en RNA_MLP_2) y W = 0.9852 (p = 0.970 en B1B)  |
|    - Prueba Pareada Wilcoxon vs. Cancha: p = 0.000186 (RNA_MLP_2) y p = 0.000010 (Baseline 1B)    |
|    - Contraste vs. Baseline 1B: RNA_MLP_2 lo supera en 14 de 24 meses (58.3%, p = 0.9888 bilateral) |
|    - IC Bootstrap 95% RMSE: [0.1154, 0.2384]%Cu (RNA_MLP_2) frente a [0.3153, 0.4169]%Cu (Cancha)  |
+----------------------------------------------------------------------------------------------------+
| 3. Explicabilidad Geometalúrgica (XAI) y Teoría de Pierre Gy:                                      |
|    - Mecanismo Físico: La ley de canje es acordada antes de flotación (protocolo de tres sobres).  |
|      El impacto de CuS se origina por friabilidad diferencial y segregación en acopio mina.        |
|    - Superficie 2D: Delimitada por la Envolvente Convexa (Convex Hull N=24) para evitar sobre-     |
|      extrapolación en esquinas vacías (CuT <= 1.80%, CuS > 0.16%).                                 |
|    - Permutación: Cancha_CuT (+0.1256 %Cu en RMSE) | Cancha_CuS (+0.1003 %Cu en RMSE)               |
|    - Magnitud SHAP Media E[|φ|]: CuT = 0.0922 %Cu | CuS = 0.0701 %Cu                               |
|    - Verificación RCI (A-01): Res_MLP_2_2 contrae pesos a cero (SHAP CuS = 0.0000), validando B1B  |
|    - Transparencia: RNA_MLP_2 in-sample RMSE = 0.0966%Cu (NRMSE = 6.75%) vs LOOCV = 12.28%         |
+----------------------------------------------------------------------------------------------------+
| 4. Artefactos Serializados 100% Nativos scikit-learn (.joblib):                                    |
|    - modelo_ganador_rna_mlp2.joblib (Diccionario portable: StandardScaler + 10 MLPs N=24)         |
|    - modelo_ganador_residual_mlp.joblib (Diccionario portable: Skip-Connection + 10 MLPs N=24)     |
|    - modelo_baseline_1b.joblib (Calibrador Operacional de Sesgo Constante +0.340417%Cu)           |
|    - modelos_rna.py (Módulo de inferencia desacoplado para Bloques 5 y 6, sin dependencia __main__)|
+----------------------------------------------------------------------------------------------------+
```

---

## 1. Fundamento Teórico y Metodológico de la Validación Unificada

### 1.1. Contexto Geometalúrgico y Marco de Pierre Gy
La venta de mineral sulfurado desde faena Mantos Cobrizo a la Planta San José de ENAMI involucra la transferencia comercial de $295{,}884.67\text{ t}$ métricas durante el bienio 2023–2024. De acuerdo con la **Teoría de Muestreo de Pierre Gy (1998)**, la heterogeneidad constitucional del yacimiento —caracterizada por un frente de transición supergena donde coexisten sulfuros competentes (bornita, calcopirita) y especies oxidadas de menor dureza y alta friabilidad (crisocola, malaquita, cuantificadas por la ley soluble $\text{CuS}$)— introduce un error fundamental de muestreo y segregación gravitacional de finos durante las etapas de chancado y acopio en mina.

La práctica histórica de liquidar presupuestariamente el mineral asumiendo que la ley de despacho en mina coincide con la ley de canje liquidada por la estatal (`Cancha Directa`) ha generado una sobreestimación sistemática promedio de $+0.3404\%\text{Cu}$, ocasionando pérdidas económicas por falsos fletes, cobros de maquila en lotes bajo ley de corte y desvíos presupuestarios severos.

### 1.2. Protocolo de Evaluación Ciega LOOCV ($N = 24$)
Para comparar sin sesgo los algoritmos, se consolida la partición estricta de validación cruzada dejando uno fuera (**Leave-One-Out Cross-Validation**, LOOCV). Para cada mes $k \in \{1, \dots, 24\}$:
1. Se aísla el lote mensual de prueba $(\mathbf{x}_k, y_k)$.
2. Se reentrena o calibra el modelo exclusivamente sobre los $N_{\text{train}} = 23$ meses restantes $(\mathbf{X}_{-k}, \mathbf{y}_{-k})$, ajustando transformadores de escala dentro de la iteración.
3. Se genera la predicción ciega cronológica $\hat{y}_k$.

Al concatenar los $24$ pliegues, se obtiene el vector de pronósticos fuera de muestra $\hat{\mathbf{y}}_{\text{OOS}} \in \mathbb{R}^{24}$, sobre el cual se evalúan todas las métricas e inferencias estadísticas sin riesgo de sobreajuste memorístico.

---

## 2. Benchmarking Unificado de los 18 Modelos Evaluados

### 2.1. Batería Exhaustiva de 9 Métricas de Validación
Conforme al mandato de la auditoría experta, se calcula el conjunto completo de indicadores econométricos y geoestadísticos, incluyendo de forma mandatoria el **Error Cuadrático Medio (MSE, en $(\%\text{Cu})^2$)**:

$$\text{MSE} = \frac{1}{N}\sum_{k=1}^N (y_k - \hat{y}_k)^2, \qquad \text{RMSE} = \sqrt{\text{MSE}}, \qquad \text{NRMSE} = \frac{\text{RMSE}}{\bar{y}}\times 100\%$$
$$\text{MAE} = \frac{1}{N}\sum_{k=1}^N |y_k - \hat{y}_k|, \qquad \text{MAPE} = \frac{1}{N}\sum_{k=1}^N \left|\frac{y_k - \hat{y}_k}{y_k}\right|\times 100\%$$
$$R^2 = 1 - \frac{\sum (y_k - \hat{y}_k)^2}{\sum (y_k - \bar{y})^2}, \qquad \text{VarExp} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}, \qquad \text{Sesgo} = \frac{1}{N}\sum_{k=1}^N (\hat{y}_k - y_k)$$

> **Nota Econométrica sobre la Varianza Explicada y el Empate Técnico (Auditoría C-04):**  
> 1. `Res_MLP_2_2` ($\text{MSE} = 0.020781$) y `Baseline_1B` ($\text{MSE} = 0.020782$) difieren en menos de $10^{-6}\\,(\%\text{Cu})^2$. Por rigor científico, se declara formalmente **empate técnico (`1=`)**.  
> 2. `Cancha_Directa` exhibe un valor aparente de $\text{Var. Explicada} = 0.6789$ porque la fórmula estándar resta el sesgo medio *in-sample* ($+0.3404\%\text{Cu}$). En cambio, en LOOCV estricto, la estimación del sesgo fuera de muestra sobre $N_{\text{train}} = 23$ induce la penalización teórica $(24/23)^2 = 1.0888$, trasladando la varianza explicada real a $0.6503$ en `Baseline_1B`.

### 2.2. Tabla Maestra Consolidada (Ranking General LOOCV)
A continuación se presenta la tabla maestra consolidada de los $18$ modelos evaluados sobre el dataset auditado ($N = 24$), ordenada estrictamente de menor a mayor error cuadrático ($\text{MSE}$ / $\text{RMSE}$):

| Ranking | Modelo | Familia Algorítmica | Espacio de Entradas | MSE $(\%\text{Cu})^2$ | RMSE $(\%\text{Cu})$ | MAE $(\%\text{Cu})$ | NRMSE $(\%)$ | $R^2$ Score | MAPE $(\%)$ | Pearson $r$ | Sesgo $(\%\text{Cu})$ | Var. Explicada |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1=** | **`Res_MLP_2_2`** | Residual-MLP $(2, 2)$ | 2D $[\text{CuT}, \text{CuS}]$ | **0.020781** | **0.1442** | **0.1178** | **10.07%** | **+0.6504** | **8.12%** | **0.8209** | $+0.0000$ | **0.6504** |
| **1=** | **`Baseline_1B`** | Corrección Sesgo Constante | 1D $[\text{Cancha}]$ | **0.020782** | **0.1442** | **0.1178** | **10.07%** | **+0.6503** | **8.12%** | **0.8209** | $-0.0000$ | **0.6503** |
| **3** | `Res_MLP_4` | Residual-MLP $(4,)$ | 2D $[\text{CuT}, \text{CuS}]$ | 0.020782 | 0.1442 | 0.1178 | 10.07% | +0.6503 | 8.12% | 0.8209 | $-0.0000$ | 0.6503 |
| **4** | `Res_MLP_3` | Residual-MLP $(3,)$ | 2D $[\text{CuT}, \text{CuS}]$ | 0.020782 | 0.1442 | 0.1178 | 10.07% | +0.6503 | 8.12% | 0.8209 | $-0.0000$ | 0.6503 |
| **5** | `Res_MLP_2` | Residual-MLP $(2,)$ | 2D $[\text{CuT}, \text{CuS}]$ | 0.021551 | 0.1468 | 0.1200 | 10.26% | +0.6374 | 8.22% | 0.8103 | $-0.0022$ | 0.6375 |
| **6** | `RNA_Delgado_Keras` | Directa Keras Colapsada | 2D $[\text{CuT}, \text{CuS}]$ | 0.027735 | 0.1665 | 0.1302 | 11.64% | +0.5334 | 8.74% | 0.7410 | $+0.0022$ | 0.5334 |
| **7** | `RidgeCV` | Lineal Regularizado ($L_2$) | 2D $[\text{CuT}, \text{CuS}]$ | 0.028403 | 0.1685 | 0.1301 | 11.78% | +0.5221 | 8.93% | 0.7248 | $+0.0108$ | 0.5241 |
| **8** | `OLS` | Regresión Lineal Clásica | 2D $[\text{CuT}, \text{CuS}]$ | 0.028789 | 0.1697 | 0.1318 | 11.85% | +0.5156 | 8.82% | 0.7346 | $+0.0038$ | 0.5159 |
| **9** | **`RNA_MLP_2`** | Directa MLP $(2,)$ | 2D $[\text{CuT}, \text{CuS}]$ | **0.030890** | **0.1758** | **0.1318** | **12.28%** | **+0.4803** | **8.95%** | **0.6954** | $-0.0124$ | **0.4829** |
| **10** | `LassoCV` | Lineal Regularizado ($L_1$) | 2D $[\text{CuT}, \text{CuS}]$ | 0.030971 | 0.1760 | 0.1373 | 12.30% | +0.4789 | 9.23% | 0.7047 | $+0.0087$ | 0.4802 |
| **11** | `ElasticNetCV` | Regularizado ($L_1 + L_2$) | 2D $[\text{CuT}, \text{CuS}]$ | 0.031294 | 0.1769 | 0.1369 | 12.36% | +0.4735 | 9.21% | 0.6992 | $+0.0085$ | 0.4747 |
| **12** | `RNA_MLP_4` | Directa MLP $(4,)$ | 2D $[\text{CuT}, \text{CuS}]$ | 0.031446 | 0.1773 | 0.1340 | 12.39% | +0.4709 | 9.15% | 0.6867 | $+0.0060$ | 0.4715 |
| **13** | `RNA_MLP_3` | Directa MLP $(3,)$ | 2D $[\text{CuT}, \text{CuS}]$ | 0.031951 | 0.1787 | 0.1350 | 12.49% | +0.4624 | 9.23% | 0.6809 | $+0.0068$ | 0.4632 |
| **14** | `RNA_Delgado_MLP` | Directa MLP $(4, 4)$ | 2D $[\text{CuT}, \text{CuS}]$ | 0.032592 | 0.1805 | 0.1378 | 12.61% | +0.4516 | 9.33% | 0.6778 | $+0.0001$ | 0.4516 |
| **15** | `RNA_MLP_2_2` | Directa MLP $(2, 2)$ | 2D $[\text{CuT}, \text{CuS}]$ | 0.034635 | 0.1861 | 0.1300 | 13.00% | +0.4173 | 8.92% | 0.6683 | $+0.0009$ | 0.4173 |
| **16** | `RNA_MLP_3_2` | Directa MLP $(3, 2)$ | 2D $[\text{CuT}, \text{CuS}]$ | 0.037222 | 0.1929 | 0.1398 | 13.48% | +0.3737 | 9.53% | 0.6122 | $-0.0044$ | 0.3741 |
| **17** | `SVR_RBF` | Kernel RBF No Lineal | 2D $[\text{CuT}, \text{CuS}]$ | 0.056986 | 0.2387 | 0.1796 | 16.68% | +0.0412 | 12.11% | 0.4546 | $-0.0110$ | 0.0432 |
| **18** | **`Cancha_Directa`** | **Línea Base Operacional** | 1D $[\text{Cancha}]$ | **0.134971** | **0.3674** | **0.3404** | **25.67%** | **-1.2709** | **24.90%** | **0.8347** | **+0.3404** | **0.6789** |

---

## 3. Diagnóstico Estadístico Riguroso de Residuos e Inferencia (Auditoría C-02)

### 3.1. Pruebas de Hipótesis Auditadas sobre Residuos Fuera de Muestra
Se aplicaron cuatro pruebas diagnósticas estándares sobre los residuos LOOCV $e_k = y_k - \hat{y}_k$:

1. **Prueba de Normalidad (Shapiro-Wilk):** Contrasta $H_0$: Los residuos siguen una distribución normal.
2. **Prueba de Independencia Serial (Durbin-Watson):** Evalúa autocorrelación de primer orden ($d \approx 2.0$ denota independencia; $d < 1.0$ denota correlación positiva).
3. **Prueba de Homocedasticidad (Breusch-Pagan exacto):** Regresión del residuo respecto a la estimación $\hat{y}$ (`het_breuschpagan(residuos, sm.add_constant(y_pred))`).
4. **Prueba Pareada No Paramétrica de Wilcoxon (`method='exact'`):**
   * Unilateral frente a Cancha Directa: $H_0: e_{\text{Cancha}}^2 \le e_{\text{Modelo}}^2$ vs. $H_1: e_{\text{Cancha}}^2 > e_{\text{Modelo}}^2$.
   * Bilateral frente a Baseline 1B: $H_0: e_{\text{B1B}}^2 = e_{\text{Modelo}}^2$ vs. $H_1: e_{\text{B1B}}^2 \neq e_{\text{Modelo}}^2$.
5. **Intervalo de Confianza Bootstrap ($B = 10{,}000$ iteraciones):** Estimación no paramétrica al 95% para el $\text{RMSE}$.

A continuación se presentan los resultados auditados para los **10 modelos principales** (el archivo consolidado `DIAGNOSTICO_RESIDUOS_Y_PRUEBAS_HIPOTESIS.csv` contiene los $18$ modelos):

| Modelo Representativo | RMSE $(\%\text{Cu})$ | IC Bootstrap 95% RMSE | Shapiro $W$ | Shapiro $p$ | Durbin-Watson $d$ | Breusch-Pagan $p$ | Homocedasticidad | Wilcoxon vs. Cancha $p$ | Wilcoxon vs. B1B $p$ | Meses Supera a B1B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Cancha_Directa`** | 0.3674 | $[0.3153, 0.4169]$ | 0.9852 | 0.9695 | **0.285** | 0.1709 | Homocedástica | 1.000000 | 0.000020 | $3/24$ ($12.5\%$) |
| **`Baseline_1B`** | 0.1442 | $[0.1061, 0.1785]$ | 0.9852 | 0.9696 | **2.014** | **0.4075** | **Homocedástica** | **0.000010** | 1.000000 | Empate (`1=`) |
| **`Res_MLP_2_2`** | 0.1442 | $[0.1065, 0.1786]$ | 0.9852 | 0.9696 | **2.014** | **0.4075** | **Homocedástica** | **0.000010** | 1.000000 | Empate (`1=`) |
| `Res_MLP_2` | 0.1468 | $[0.1103, 0.1807]$ | 0.9828 | 0.9411 | 2.037 | 0.2140 | Homocedástica | 0.000012 | 0.250000 | $0/24$ empates |
| `RNA_Delgado_Keras` | 0.1665 | $[0.1182, 0.2131]$ | 0.9705 | 0.6796 | 2.271 | **0.0040** | **Heterocedástica** | 0.000075 | 0.545670 | $15/24$ ($62.5\%$) |
| `RidgeCV` | 0.1685 | $[0.1150, 0.2255]$ | 0.9078 | 0.0316 | 2.092 | **0.0228** | **Heterocedástica** | 0.000103 | 0.855336 | $14/24$ ($58.3\%$) |
| `OLS` | 0.1697 | $[0.1188, 0.2159]$ | 0.9793 | 0.8819 | 2.310 | **0.0016** | **Heterocedástica** | 0.000088 | 0.545670 | $15/24$ ($62.5\%$) |
| **`RNA_MLP_2`** | 0.1758 | $[0.1154, 0.2384]$ | 0.9307 | 0.1012 | **1.958** | **0.0822** | **Homocedástica** | **0.000186** | **0.988813** | **$14/24$ ($58.3\%$)** |
| `LassoCV` | 0.1760 | $[0.1205, 0.2310]$ | 0.9671 | 0.5963 | 2.364 | **0.0029** | **Heterocedástica** | 0.000161 | 0.789838 | $14/24$ ($58.3\%$) |
| `SVR_RBF` | 0.2387 | $[0.1579, 0.3147]$ | 0.9296 | 0.0956 | 2.148 | 0.7243 | Homocedástica | 0.001407 | 0.083920 | $9/24$ ($37.5\%$) |

### 3.2. Conclusiones Estadísticas Clave para la Defensa
1. **Destrucción Estadística de la Cancha Directa:**  
   El valor $p$ de Wilcoxon exacto para `Baseline 1B` ($p = 1.0 \times 10^{-5}$) y `RNA_MLP_2` ($p = 1.86 \times 10^{-4}$) demuestra con un nivel de confianza superior al $99.98\%$ que los modelos matemáticos reducen el error de terreno de forma sistemática y no azarosa.
2. **Falla Econométrica de Modelos Lineales Clásicos (Heterocedasticidad):**  
   `OLS` ($p = 0.0016$), `RidgeCV` ($p = 0.0228$), `LassoCV` ($p = 0.0029$) y `Keras` ($p = 0.0040$) sufren **heterocedasticidad significativa ($p < 0.05$)** porque sus errores se disparan en los frentes de alta ley (octubre–noviembre 2023). En contraste, `Baseline 1B` ($p = 0.4075$) y `RNA_MLP_2` ($p = 0.0822$) preservan varianza constante, garantizando intervalos válidos.
3. **Consistencia Empírica de `RNA_MLP_2` Frente a `Baseline 1B`:**  
   En el contraste pareado bilateral ($p = 0.9888$), `RNA_MLP_2` **supera a `Baseline 1B` en $14$ de los $24$ meses ($58.3\%$)** y logra un error absoluto mediano menor ($0.0801\%\text{Cu}$ frente a $0.1044\%\text{Cu}$). Su mayor RMSE global se concentra de forma exclusiva en el mes anómalo de noviembre de 2023.

---

## 4. Selección Dual del Modelo Ganador (*Champion Model*)

Se formaliza la **Estrategia Dual de Selección de Modelo Campeón**:

```
                                  ESTRATEGIA DUAL SELECCIONADA
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [CAMPEÓN DE MÍNIMA VARIANZA]                                  [CAMPEÓN DE EXPLICABILIDAD XAI]
      Baseline 1B / Residual-MLP                                             RNA_MLP_2
  ──────────────────────────────────                            ───────────────────────────────────
  • NRMSE LOOCV: 10.07% (R² = +0.6504)                          • NRMSE LOOCV: 12.28% (R² = +0.4803)
  • NRMSE Holdout (19/5): 8.96%                                 • NRMSE Holdout (19/5): 6.56%
  • Grados de Libertad: P_eff = 1                               • Grados de Libertad: P = 9 (Ratio = 2.56)
  • Comportamiento: Inmune a Outliers                           • Comportamiento: Sensible a CuS (-1.53%)
  • Uso Operacional: Planificación Presupuestaria               • Uso Operacional: Control y Blending en Mina
```

### 4.1. Campeón de Mínima Varianza y Robustez Ante Extremos: `Baseline 1B` / `Residual-MLP`
* **Desempeño Global:** $\text{NRMSE} = 10.07\%$, $\text{RMSE} = 0.1442\%\text{Cu}$, $R^2 = +0.6504$, $\text{MSE} = 0.020781\,(\%\text{Cu})^2$ (Empate técnico `1=`).
* **Mecanismo Físico-Matemático:** La formulación residual $\hat{y}_k = \text{Cancha\_CuT}_k - \hat{e}_{\text{MLP}}(\text{CuT}, \text{CuS})$ con regularización $L_2$ ($\alpha = 2.0$) contrae los pesos sinápticos $\Vert\mathbf{W}\Vert \to 0$. Al ocurrir esto, $\hat{e}_{\text{MLP}}$ colapsa hacia la media de entrenamiento $\bar{e}_{\text{train}}$, reproduciendo de manera idéntica la corrección constante del `Baseline 1B` ($+0.3404\%\text{Cu}$).
* **Valor para la Tesis:** Demuestra que ante frentes anómalos no recurrentes (noviembre 2023 con $\text{CuT} = 2.47\%$), forzar curvatura no lineal sobre el residuo incrementa la varianza fuera de muestra ($\alpha = 0.5 \to 10.73\%$; $\alpha = 0.1 \to 11.94\%$). La contracción a la constante es una propiedad óptima de estabilidad presupuestaria.

### 4.2. Campeón de Explicabilidad Geometalúrgica No Lineal: `RNA_MLP_2`
* **Desempeño Fuera de Muestra:** $\text{NRMSE} = 12.28\%$, $\text{RMSE} = 0.1758\%\text{Cu}$, $R^2 = +0.4803$, $\text{MSE} = 0.030890\,(\%\text{Cu})^2$.
* **Desempeño Prospectivo en Régimen Regular (*Holdout* 19/5):** $\text{NRMSE} = \mathbf{6.56\%}$ ($\text{RMSE} = 0.0861\%\text{Cu}$), superando con holgura a `Baseline 1B` ($8.96\%$) y a `RidgeCV` ($7.57\%$).
* **Parsimonia:** Solo $2$ neuronas ocultas con activación `tanh` ($9$ parámetros nominales, $N_{\text{train}}/P = 23/9 = 2.56$).
* **Aporte Geometalúrgico:** Coeficientes sinápticos activos y no nulos. En el estudio de ablación, la incorporación de $\text{CuS}$ reduce el error de la red en **$-1.53\%$** de NRMSE (de $13.81\%$ en 1D a $12.28\%$ en 2D).

---

## 5. Explicabilidad e Interpretabilidad Geometalúrgica (XAI) (Auditoría C-01 y C-05)

### 5.1. Fundamentación Metalúrgica Correcta y Marco de Pierre Gy
En la comercialización con ENAMI, `Canje_CuT` representa la **ley de cabeza química acordada en recepción** mediante el protocolo reglamentario de tres sobres (productor, ENAMI y dirimencia), **antes** de que el lote ingrese a molienda y flotación en Planta San José.

Conforme a la **Teoría de Muestreo de Pierre Gy**:
1. Los minerales oxidados supergenos (crisocola, malaquita, cuantificados por `Cancha_CuS`) presentan menor dureza de Mohs y mayor friabilidad que los sulfuros competentes (bornita, calcopirita).
2. Durante el chancado secundario y el acopio en mina, los óxidos sufren **pulverización diferencial y segregación gravitacional de finos**, modificando la masa crítica de muestreo y la discrepancia sistemática observada al transferir el mineral a la torre de muestreo de ENAMI.
3. Para evitar extrapolaciones infundadas en esquinas sin datos (ej. zonas con $\text{CuT} \le 1.80\%$ y $\text{CuS} > 0.16\%$), la Figura 3 incorpora la **envolvente convexa (*Convex Hull*)** de los 24 meses históricos.

### 5.2. Transparencia *In-Sample* vs. LOOCV (Auditoría C-05)
El ensamble final `RNA_MLP_2` ajustado sobre $N = 24$ alcanza un $\text{RMSE}_{\text{in-sample}} = 0.0966\%\text{Cu}$ ($\text{NRMSE} = 6.75\%$), frente a $\text{RMSE}_{\text{LOOCV}} = 0.1758\%\text{Cu}$ ($\text{NRMSE} = 12.28\%$). Esta brecha refleja que los pesos de $\text{CuS}$ capturan fielmente la correlación en la ventana regular de producción, estando apalancados moderadamente por los pulsos extremos de octubre y noviembre de 2023.

| Modelo Analizado | Variable Predictora | Incremento en RMSE (Permutación) | Desv. Est. Permutación | Magnitud Media SHAP $E[\|\phi\|]$ | Estado Sináptico |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`RNA_MLP_2` (Campeón XAI)** | **`Cancha_CuT`** | **+0.1256 %Cu** | 0.0298 | **0.0922 %Cu** | **Activo** |
| **`RNA_MLP_2` (Campeón XAI)** | **`Cancha_CuS`** | **+0.1003 %Cu** | 0.0216 | **0.0701 %Cu** | **Activo** |
| `Res_MLP_2_2` (Residual Contraído) | `Cancha_CuT` | +0.1974 %Cu | 0.0387 | 0.1977 %Cu | Base Lineal Preservada |
| `Res_MLP_2_2` (Residual Contraído) | `Cancha_CuS` | $-0.0000$ %Cu | 0.0000 | **0.0000 %Cu** | **Apagado ($L_2$ Contracción)** |

---

## 6. Serialización Portable y Ficha Técnica de Entrega (Auditoría C-03)

Para evitar el riesgo de fallos por dependencias de clases definidas en `__main__` en los **Bloques 5 y 6**, los modelos se serializan como diccionarios con objetos $100\%$ nativos de `scikit-learn` (`StandardScaler`, listas de `MLPRegressor`). Su consumo en código externo se realiza mediante el módulo portable `BLOQUE_4/modelos_rna.py`:

```
./BLOQUE_4/
├── modelo_ganador_rna_mlp2.joblib      (Diccionario nativo scikit-learn: scaler + 10 MLPs N=24)
├── modelo_ganador_residual_mlp.joblib  (Diccionario nativo scikit-learn: Skip-Connection N=24)
├── modelo_baseline_1b.joblib           (Diccionario con constante empírica auditada: +0.340417%Cu)
├── modelos_rna.py                      (Módulo desacoplado de funciones de inferencia portable)
└── metricas_validadas_champion.csv     (Tabla resumen con las métricas certificadas de ambos campeones)
```

### Protocolo de Inferencia Rápida (Consumo en Python)
```python
import sys
import numpy as np
sys.path.append("BLOQUE_4")
import modelos_rna

# Predicción con modelo campeón RNA_MLP_2
X_lote = np.array([[1.55, 0.18]])  # [Cancha_CuT, Cancha_CuS]
canje_estimado = modelos_rna.predecir_rna_mlp2("BLOQUE_4/modelo_ganador_rna_mlp2.joblib", X_lote)[0]
print(f"Ley Canje ENAMI Proyectada: {canje_estimado:.4f} %Cu")
```

---

## 7. Visualizaciones de Publicación de Alta Resolución (200 DPI)

Se generaron e integraron cuatro figuras de estándar de publicación en `./BLOQUE_4/`:

1. **[Figura 1: Benchmarking Unificado de 18 Modelos](./fig_bloque4_benchmarking_18_modelos.png):**  
   Gráfico de barras horizontales contrastando el NRMSE LOOCV de los 18 modelos evaluados, categorizados por familias algorítmicas, destacando las líneas de referencia operacional ($25.67\%$), de la mejor red directa ($12.28\%$) y del benchmark residual ($10.07\%$).
2. **[Figura 2: Diagnóstico Cuádruple de Residuos](./fig_bloque4_diagnostico_residuos.png):**  
   Panel cuádruple de diagnóstico econométrico: (A) Dispersión de residuos OOS vs. Valor Predicho con bandas de $\pm 2\sigma$, (B) Gráfico Q-Q Normal con estadístico Shapiro-Wilk, (C) Histograma y densidad KDE de errores, y (D) Función de Autocorrelación Serial (ACF) con bandas de Bartlett al 95% y Durbin-Watson.
3. **[Figura 3: Superficie de Respuesta Geometalúrgica 2D con Convex Hull](./fig_bloque4_superficie_respuesta_2d.png):**  
   Mapa de contorno y superficie de isolíneas 2D delimitando el **dominio empírico observado (Convex Hull $N = 24$)** frente a zonas de extrapolación, modelando la segregación y friabilidad diferencial de óxidos según Pierre Gy.
4. **[Figura 4: Explicabilidad XAI (Permutación y SHAP)](./fig_bloque4_xai_shap_y_permutacion.png):**  
   Panel comparativo de interpretabilidad XAI que ilustra el aumento porcentual de RMSE bajo permutación aleatoria (Panel A) y la magnitud marginal media de los valores SHAP $E[|\phi|]$ (Panel B).

---

## 8. Conclusiones del Bloque 4 y Habilitación para el Bloque 5

1. **Cumplimiento Integral de Metodología v4.0 y Cierre de Auditoría RCI:**  
   Se subsanaron todas las observaciones de la auditoría independiente: fundamentación física de Pierre Gy (C-01), pruebas exactas de Breusch-Pagan y Wilcoxon (C-02), portabilidad nativa `.joblib` con `modelos_rna.py` (C-03), empate técnico en el primer lugar (C-04), transparencia *in-sample* (C-05) y rutas relativas (C-06).
2. **Estrategia Dual Consolidada:**  
   `Baseline 1B` / `Residual-MLP` queda establecido como el estándar de mínima varianza para presupuestos, y `RNA_MLP_2` como el motor analítico geometalúrgico para la gestión operativa en mina.
3. **Transición al Bloque 5:**  
   Con los 24 vectores de predicción OOS sincronizados y los pipelines serializados en formato nativo, el proyecto se encuentra formalmente habilitado para ingresar al **BLOQUE 5: Modelo Económico Dual ENAMI y Simulación de Ahorros Operacionales**.

---
*Informe generado y auditado bajo el estándar de la Escuela de Minería de la Universidad de Antofagasta.*
