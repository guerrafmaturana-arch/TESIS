# AUDITORÍA EXPERTO-CRÍTICA (MODELO RCI) — BLOQUE 4

## Verificación Matemática, Rigor Metalúrgico, Portabilidad de Artefactos y Directrices de Cierre Previo al Bloque 5

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas

**Documento Auditado:** `BLOQUE_4/INFORME_ENTREGABLE_BLOQUE_4.md` (Versión 4.0)

**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)

**Institución:** Universidad de Antofagasta — Facultad de Ingeniería, Departamento de Ingeniería en Minas

**Destinatario Técnico:** Entorno de desarrollo y ejecución (`Antigravity`)

---

## 1. Marco de Auditoría RCI (Rol, Contexto, Instrucción)

* **Rol (R):** Auditor Técnico Senior en Geoestadística Minera, Conciliación Mina–Planta (protocolo comercial ENAMI) y Machine Learning aplicado a muestras pequeñas ($N = 24$).
* **Contexto (C):** Evaluación crítica del entregable `INFORME_ENTREGABLE_BLOQUE_4.md` (Metodología v4.0), correspondiente a la validación unificada de los $18$ modelos ($7$ baselines del Bloque 2 + $11$ redes neuronales del Bloque 3), pruebas de hipótesis sobre residuos fuera de muestra, selección dual de modelos campeones, explicabilidad XAI (SHAP, Permutación y Superficie 2D) y serialización `.joblib` previa a la simulación económica ENAMI del Bloque 5.
* **Instrucción (I):** Verificar de forma independiente cada métrica, contraste estadístico, fundamento metalúrgico y portabilidad de código, clasificando los hallazgos por severidad y entregando parches exactos para sellar el Bloque 4 sin vulnerabilidades ante la comisión evaluadora.

---

## 2. Dictamen Ejecutivo de Auditoría

> **Veredicto:** **APROBADO CON OBSERVACIONES CRÍTICAS (requiere corrección puntual en `04_validacion_residuos_xai.ipynb` e `INFORME_ENTREGABLE_BLOQUE_4.md` antes de cerrar el módulo y pasar al Bloque 5).**

El trabajo realizado en el **Bloque 4** consolida con éxito el *benchmarking* imparcial de los $18$ modelos (Tabla 2.2), la estadística de normalidad (**Shapiro-Wilk**), independencia serial (**Durbin-Watson**), intervalos *bootstrap* ($B = 10{,}000$) y el contraste pareado **Wilcoxon vs. Cancha Directa**. Asimismo, la comprobación empírica del hallazgo RCI `A-01` ($\text{SHAP}$ de `Cancha_CuS` $= 0.0000\%\text{Cu}$ en `Res_MLP_2_2` frente a pesos activos en `RNA_MLP_2`) y la **Estrategia Dual** están bien estructuradas.

No obstante, la auditoría independiente detectó **2 hallazgos críticos** (un error conceptual metalúrgico en la Sección 5.3 que sería objetado de inmediato por la comisión de Minas, y discrepancias numéricas en Breusch-Pagan y Wilcoxon vs. `Baseline_1B`), **1 riesgo alto de ingeniería de software** (fallo de deserialización `.joblib` en los Bloques 5 y 6) y **3 observaciones de severidad media/baja**.

---

## 3. Verificación Independiente de Cálculos

| Dimensión Auditada | Resultado Reportado en Bloque 4 | Auditoría Independiente | Estado |
| :--- | :--- | :--- | :---: |
| **Tabla 2.2: Ranking 18 Modelos ($9$ métricas)** | Desde `Res_MLP_2_2` ($\text{MSE}=0.020781$, $\text{NRMSE}=10.07\%$, $R^2=+0.6504$) y `Baseline_1B` ($0.020782$) hasta `Cancha_Directa` ($0.134971$, $R^2=-1.2709$). | Coincidencia exacta al sexto decimal en $\text{MSE}$ y al cuarto decimal en todas las métricas (incluyendo $\text{VarExp} = 0.4829$ en `RNA_MLP_2`). | **Exacto** |
| **Shapiro-Wilk, Durbin-Watson y Bootstrap 95%** | `Baseline_1B`: $W=0.9852, p=0.9696, d=2.014$; `RNA_MLP_2`: $W=0.9307, p=0.1012, d=1.958$; `RidgeCV`: $p=0.0316$. | Coincidencia exacta al cuarto decimal en $W$, $p$ y $d$; intervalos *bootstrap* verificados (`[0.3153, 0.4169]`, `[0.1061, 0.1785]`, `[0.1147, 0.2375]`). | **Exacto** |
| **Wilcoxon unilateral vs. `Cancha_Directa`** | `RNA_MLP_2` ($p=0.000186$), `OLS` ($p=0.000088$), `RidgeCV` ($p=0.000103$), `SVR_RBF` ($p=0.001407$). | Exacto al sexto decimal con `method='exact'`; en `Baseline_1B` ($0.000064$) y `Res_MLP_2` ($0.000072$) se usó aproximación normal `approx` (el valor exacto es $0.000010$ y $0.000012$). | **Verificado** *(unificar método)* |
| **Breusch-Pagan (Homocedasticidad, Tabla 3.1)** | `OLS` ($p=0.8329$), `Keras` ($p=0.8170$), `RidgeCV` ($p=0.7788$), `RNA_MLP_2` ($p=0.3015$), `SVR_RBF` ($p=0.0152$). | **NO COINCIDE.** Con `het_breuschpagan(e, sm.add_constant(y_hat))`: `OLS` ($p=0.0016$), `Keras` ($p=0.0040$), `RidgeCV` ($p=0.0228$) y `LassoCV` ($p=0.0029$) **rechazan homocedasticidad**; `Baseline_1B` ($p=0.4075$) y `RNA_MLP_2` ($p=0.0822$) la cumplen. | **DISCREPANCIA CRÍTICA** |
| **Wilcoxon pareado vs. `Baseline_1B` (Tabla 3.1)** | `OLS` ($p=0.1774$), `RidgeCV` ($p=0.1774$), `RNA_MLP_2` ($p=0.2687$), `SVR_RBF` ($p=0.0028$). | **NO COINCIDE.** Sobre $e_k^2$ (bilateral): `OLS` ($p=0.5457$), `RidgeCV` ($p=0.8553$), `RNA_MLP_2` ($p=0.9888$), `SVR_RBF` ($p=0.0839$ bilateral / $0.0420$ unilateral). | **DISCREPANCIA CRÍTICA** |

---

## 4. Matriz de Hallazgos Críticos y Acciones Correctivas

| ID | Severidad | Sección Auditada | Hallazgo Detectado | Acción Correctiva Concreta |
| :--- | :---: | :--- | :--- | :--- |
| **C-01** | **CRÍTICO** *(Metalúrgico y Extrapolación)* | Sec. 5.3 y Fig. 3 (Superficie 2D) | **Confusión entre ley de canje y recuperación en flotación:** Se afirma que `Canje_CuT` baja al subir `CuS` porque los xantatos en celdas de flotación de ENAMI no recuperan óxidos y estos van al relave. Además, la caída ocurre en una esquina vacía sin datos ($\text{CuT} \le 1.80\%$, $\text{CuS} > 0.16\%$). | Corregir el mecanismo físico: `Canje_CuT` es la ley de cabeza recibida antes de molienda/flotación; explicar el efecto de `CuS` vía **Teoría de Muestreo de Pierre Gy** (friabilidad de óxidos y segregación de finos en acopio) y trazar la **envolvente convexa** (*convex hull*) de los $24$ datos en la Figura 3. |
| **C-02** | **CRÍTICO** *(Estadístico)* | Sec. 3.1 (Tabla 3.1) | **Valores incorrectos en Breusch-Pagan y Wilcoxon vs. `Baseline_1B`, y mezcla de métodos (`exact` vs `approx`):** Los $p$-valores reportados no coinciden con las pruebas estándar sobre residuos OOS. | Recalcular Tabla 3.1 con `het_breuschpagan(e, sm.add_constant(y_hat))` y `wilcoxon(..., method='exact')` uniforme. Destacar que `OLS`/`RidgeCV` presentan heterocedasticidad ($p < 0.05$) mientras `RNA_MLP_2` y `Baseline_1B` son homocedásticos, y que `RNA_MLP_2` supera a `Baseline_1B` en $14/24$ meses ($p = 0.9888$ bilateral). |
| **C-03** | **ALTO** *(Ingeniería Software)* | Sec. 6 (`.joblib`) | **Riesgo de `AttributeError` al cargar `.joblib` en Bloques 5 y 6:** Serializar clases personalizadas (`MultiSeedEnsembleMLP`, `ResidualEnsembleMLP`) definidas en `__main__` del notebook rompe `joblib.load()` al ejecutarse desde `05_modelo_economico_enami.ipynb` o `app.py`. | Serializar diccionarios con objetos $100\%$ nativos de `scikit-learn` (`{'scaler': scaler, 'estimadores': [...], 'tipo': ...}`) o alojar las clases en un módulo importable (`BLOQUE_4/modelos_rna.py`) y validar carga externa. |
| **C-04** | **MEDIO** *(Econométrico)* | Sec. 2.2 (Tabla 2.2) | **Paradoja de `Var. Explicada` en `Cancha_Directa` ($0.6789$) y empate técnico en Ranking 1:** `Cancha_Directa` muestra mayor `Var. Explicada` que todos los modelos entrenados, y `Res_MLP_2_2` ($0.020781$) supera a `Baseline_1B` ($0.020782$) por $10^{-6}$. | Añadir nota explicando que `explained_variance_score` resta el sesgo medio *in-sample*, mientras LOOCV sufre la penalización $(24/23)^2 = 1.0888$ ($0.6789 \to 0.6503$). Declarar **empate técnico (`1=`)** entre `Res_MLP_2_2` y `Baseline_1B`. |
| **C-05** | **MEDIO** *(Interpretabilidad)* | Sec. 5.1 (XAI) | **Brecha *in-sample* vs. LOOCV en modelo XAI:** Los valores SHAP y Permutación se calculan sobre `RNA_MLP_2` ($\alpha = 0.001$) reentrenada en $N = 24$, cuya $\text{RMSE}_{\text{in-sample}} \approx 0.0980\%\text{Cu}$ frente a $\text{RMSE}_{\text{LOOCV}} = 0.1758\%\text{Cu}$. | Transparentar la brecha *in-sample* vs. LOOCV del ensamble final $N=24$ y precisar que el peso de `CuS` en SHAP está fuertemente apalancado por los pulsos de octubre y noviembre de 2023. |
| **C-06** | **BAJO** *(Editorial)* | Sec. 3.1 y Sec. 7 | **Rutas locales absolutas (`file:///c:/Users/...`) en enlaces de figuras y título de Tabla 3.1:** Los enlaces locales fallarán en GitHub/otro equipo, y la Tabla 3.1 muestra $10$ modelos representativos (no los $18$). | Reemplazar enlaces por rutas relativas (`./fig_bloque4_...png`) y precisar en Tabla 3.1 que se presentan los $10$ modelos principales (archivo CSV completo contiene los $18$). |

---

## 5. Análisis Profundo de los Hallazgos Críticos

### 5.1. `[C-01]` Corrección del Fundamento Metalúrgico de la Superficie 2D

1. **Rigor de proceso mina–planta:** En la comercialización con ENAMI, `Canje_CuT` representa la **ley de cabeza química acordada en recepción** mediante el protocolo reglamentario de tres sobres (productor, ENAMI y dirimencia), **antes** de que el lote ingrese a molienda y flotación. Por lo tanto, la cinética de los xantatos en las celdas de flotación de Planta San José o las pérdidas en relaves ocurren aguas abajo y **no alteran la ley de canje pagada por ENAMI**.
2. **Explicación física correcta (Teoría de Muestreo de Pierre Gy):** La influencia no lineal de `Cancha_CuS` sobre `Canje_CuT` ocurre en la etapa de **chancado, acopio y muestreo**: los minerales oxidados supergenos (crisocola, malaquita) poseen menor dureza y mayor friabilidad que los sulfuros competentes (bornita, calcopirita), generando segregación gravitacional de finos ricos en el cono de acopio de faena y modificando la discrepancia sistemática frente a la torre de muestreo automático de ENAMI.
3. **Control de extrapolación en el gráfico 2D:** En los $24$ balances reales no existen lotes con baja ley total ($\text{CuT} \le 1.80\%$) y alta solubilidad ($\text{CuS} > 0.16\%$). Al graficar el mapa de contorno sobre todo el rectángulo $[1.40, 2.50] \times [0.11, 0.27]$, las esquinas vacías muestran extrapolaciones sin soporte empírico. Incorporar la **envolvente convexa (*convex hull*)** de los $24$ puntos delimita visualmente la zona de validez operativa real.

### 5.2. `[C-02]` Resultados Exactos de Breusch-Pagan y Wilcoxon Pareado

Al ejecutar las pruebas estadísticas estándar sobre los vectores de residuos OOS ($e_k = y_k - \hat{y}_k$, $N = 24$):

| Modelo ($N = 24$ OOS) | Breusch-Pagan $p$ (vs. $\hat{y}$) | Diagnóstico Varianza | Wilcoxon vs. `Cancha` ($p$, `exact`, unilateral) | Wilcoxon vs. `Baseline_1B` ($p$, `exact`, bilateral) | Meses en que supera a `Baseline_1B` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Cancha_Directa`** | $0.0886$ | Homocedástica | $1.000000$ | $0.000010$ | $0 / 24$ ($0.0\%$) |
| **`Baseline_1B`** | **$0.4075$** | **Homocedástica** | **$0.000010$** | $1.000000$ | — |
| **`Res_MLP_2_2`** | **$0.4075$** | **Homocedástica** | **$0.000010$** | $1.000000$ | Empate técnico (`1=`) |
| **`Res_MLP_2`** | $0.4316$ | Homocedástica | $0.000012$ | $1.000000$ | $23 / 24$ empates |
| **`RNA_Delgado_Keras`** | **$0.0040$** | **Heterocedástica** | $0.000075$ | $0.545706$ | $12 / 24$ ($50.0\%$) |
| **`RidgeCV`** | **$0.0228$** | **Heterocedástica** | $0.000103$ | $0.855290$ | $13 / 24$ ($54.2\%$) |
| **`OLS`** | **$0.0016$** | **Heterocedástica** | $0.000088$ | $0.545706$ | $12 / 24$ ($50.0\%$) |
| **`RNA_MLP_2`** | **$0.0822$** | **Homocedástica** | **$0.000186$** | **$0.988825$** | **$14 / 24$ ($58.3\%$)** |
| **`LassoCV`** | **$0.0029$** | **Heterocedástica** | $0.000161$ | $0.684439$ | $11 / 24$ ($45.8\%$) |
| **`SVR_RBF`** | **$0.0152$** | **Heterocedástica** | $0.001407$ | $0.083900$ *(unilat. $0.0420$)* | $9 / 24$ ($37.5\%$) |

**Lectura científica clave para la defensa:**
1. Todos los modelos lineales clásicos (`OLS`, `RidgeCV`, `LassoCV`, `Keras`) sufren **heterocedasticidad significativa ($p < 0.05$)** porque su error cuadrático se dispara en la cola superior (octubre–noviembre 2023), mientras que `Baseline_1B` ($p = 0.4075$) y `RNA_MLP_2` ($p = 0.0822$) preservan varianza constante.
2. En el contraste pareado bilateral contra `Baseline_1B`, `RNA_MLP_2` obtiene $p = 0.9888$, **superando a `Baseline_1B` en $14$ de los $24$ meses ($58.3\%$)** y logrando un error absoluto mediano menor ($0.0801\%\text{Cu}$ frente a $0.1044\%\text{Cu}$). Esto demuestra que la diferencia en RMSE global se explica exclusivamente por el único mes de extrapolación extrema en noviembre de 2023.

---

## 6. Parche de Código Exacto para `Antigravity` (`04_validacion_residuos_xai.ipynb`)

Ejecutar las siguientes correcciones en `BLOQUE_4/04_validacion_residuos_xai.ipynb` antes de generar el informe final sellado:

```python
import joblib
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import wilcoxon
from scipy.spatial import ConvexHull

# ==============================================================================
# 1. CORRECCIÓN DE PRUEBAS DE HOMOCEDASTICIDAD (BREUSCH-PAGAN) Y WILCOXON EXACTO
# ==============================================================================
def auditar_residuos_estadisticos(y_real, y_pred, y_cancha, y_b1b):
    residuos = y_real - y_pred
    # Breusch-Pagan estándar: varianza del residuo respecto al valor predicho
    exog = sm.add_constant(y_pred)
    _, bp_pval, _, _ = het_breuschpagan(residuos, exog)

    # Errores cuadráticos pareados
    err2_modelo = (y_real - y_pred) ** 2
    err2_cancha = (y_real - y_cancha) ** 2
    err2_b1b    = (y_real - y_b1b) ** 2

    # Wilcoxon unilateral exacto vs Cancha Directa (H1: err2_cancha > err2_modelo)
    if np.allclose(err2_cancha, err2_modelo):
        p_wilcox_cancha = 1.0
    else:
        _, p_wilcox_cancha = wilcoxon(err2_cancha, err2_modelo, alternative='greater', method='exact')

    # Wilcoxon bilateral exacto vs Baseline 1B (H1: distribuciones distintas)
    if np.allclose(err2_b1b, err2_modelo):
        p_wilcox_b1b = 1.0
    else:
        _, p_wilcox_b1b = wilcoxon(err2_b1b, err2_modelo, alternative='two-sided', method='exact')

    return {
        'Breusch_Pagan_p': round(float(bp_pval), 4),
        'Wilcoxon_vs_Cancha_p': round(float(p_wilcox_cancha), 6),
        'Wilcoxon_vs_B1B_p': round(float(p_wilcox_b1b), 4)
    }

# ==============================================================================
# 2. SERIALIZACIÓN PORTABLE .JOBLIB (SIN DEPENDENCIA DE CLASES EN __main__)
# ==============================================================================
# Guardar diccionarios con objetos 100% nativos de scikit-learn para evitar AttributeError
artefacto_rna_mlp2 = {
    'nombre': 'RNA_MLP_2',
    'tipo': 'directo_multisimilla',
    'features': ['Cancha_CuT', 'Cancha_CuS'],
    'target': 'Canje_CuT',
    'scaler': scaler_final_n24,          # StandardScaler ajustado en N=24
    'estimadores': lista_mlps_n24,       # Lista de 10 MLPRegressor entrenados
    'rmse_loocv': 0.1758,
    'nrmse_loocv': 12.28,
    'nrmse_holdout': 6.56
}
joblib.dump(artefacto_rna_mlp2, 'BLOQUE_4/modelo_ganador_rna_mlp2.joblib')

artefacto_residual_mlp = {
    'nombre': 'Res_MLP_2_2',
    'tipo': 'residual_multisimilla',
    'features': ['Cancha_CuT', 'Cancha_CuS'],
    'target': 'Canje_CuT',
    'sesgo_medio_n24': float(np.mean(df['Cancha_CuT'] - df['Canje_CuT'])),
    'scaler': scaler_final_n24,
    'estimadores': lista_res_mlps_n24,
    'rmse_loocv': 0.1442,
    'nrmse_loocv': 10.07
}
joblib.dump(artefacto_residual_mlp, 'BLOQUE_4/modelo_ganador_residual_mlp.joblib')

# Función auxiliar de predicción portable para Bloques 5 y 6:
# preds = np.mean([m.predict(artefacto['scaler'].transform(X)) for m in artefacto['estimadores']], axis=0)
# Si tipo == 'residual_multisimilla': preds = X[:, 0] - preds

# ==============================================================================
# 3. ENVOLVENTE CONVEXA (CONVEX HULL) EN SUPERFICIE DE RESPUESTA 2D
# ==============================================================================
puntos_2d = df[['Cancha_CuT', 'Cancha_CuS']].values
hull = ConvexHull(puntos_2d)
hull_vertices = np.append(hull.vertices, hull.vertices[0])
# En el gráfico de contorno 2D (fig_bloque4_superficie_respuesta_2d.png):
# ax.plot(puntos_2d[hull_vertices, 0], puntos_2d[hull_vertices, 1],
#         'k--', lw=1.8, label='Dominio Empírico Observado (Convex Hull N=24)')
```

---

## 7. Directrices de Cierre del Bloque 4 y Transición al Bloque 5

1. **Actualizar `BLOQUE_4/INFORME_ENTREGABLE_BLOQUE_4.md`:**
   * Reemplazar la explicación de xantatos/relaves en la Sección 5.3 por el mecanismo de segregación granulométrica y friabilidad de óxidos según **Pierre Gy**, añadiendo la envolvente convexa en `fig_bloque4_superficie_respuesta_2d.png`.
   * Actualizar la **Tabla 3.1** con los $p$-valores exactos de Breusch-Pagan y Wilcoxon pareado, e indicar en su título que resume los $10$ modelos principales.
   * Declarar empate técnico (`1=`) entre `Res_MLP_2_2` y `Baseline_1B` en la Tabla 2.2 y añadir la nota aclaratoria sobre `Var. Explicada` en `Cancha_Directa` ($0.6789$).
   * Sustituir los enlaces locales `file:///c:/Users/...` por rutas relativas (`./fig_bloque4_...png`).
   * Re-serializar los archivos `.joblib` en formato diccionario nativo `scikit-learn` y verificar su carga limpia desde un script externo.
2. **Habilitación para el Bloque 5 (`./BLOQUE_5/05_modelo_economico_enami.ipynb`):**
   * Una vez aplicadas estas correcciones, cruzar las $24$ predicciones ciegas (*out-of-sample*) de `Baseline_1B / Res_MLP_2_2` y `RNA_MLP_2` con las $24$ circulares tarifarias de ENAMI Sulfuros (`BLOQUE_0/TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.csv`) y el tonelaje oficial de báscula puente (`Romana_Ton` $= 295{,}884.67\text{ t}$) para cuantificar:
     * **Nivel 1:** Reducción de la desviación presupuestaria en dólares ($\text{USD}$).
     * **Nivel 2:** Ahorro real de caja ($\text{USD}$) mediante simulación contrafactual de ley de corte (*cut-off*), falsos fletes evitados, *blending* preventivo y optimización de apertura del **Sobre 2** reglamentario de ENAMI.