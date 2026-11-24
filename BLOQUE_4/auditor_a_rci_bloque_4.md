# AUDITORÍA EXPERTO-CRÍTICA (MODELO RCI) — BLOQUE 4 (CIERRE INTEGRAL)

## Verificación Final de Benchmarking, Inferencia Estadística, Explicabilidad XAI y Consistencia Global (Bloques 0–4)

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas

**Documento Auditado:** `BLOQUE_4/INFORME_ENTREGABLE_BLOQUE_4.md` (Versión 4.0 — Cierre Consolidado)

**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)

**Institución:** Universidad de Antofagasta — Facultad de Ingeniería, Departamento de Ingeniería en Minas

**Destinatario Técnico:** Revisión final previa a presentación de avance al Profesor Guía y preparación para defensa

---

## 1. Marco de Auditoría RCI (Rol, Contexto, Instrucción)

* **Rol (R):** Auditor Técnico Senior en Geoestadística Minera, Conciliación Mina–Planta (protocolo comercial ENAMI) y Machine Learning aplicado a muestras pequeñas ($N = 24$).
* **Contexto (C):** Auditoría de cierre sobre la versión definitiva del Bloque 4 (`INFORME_ENTREGABLE_BLOQUE_4.md`, Metodología v4.0), verificando la resolución completa de los hallazgos previos (`C-01` a `C-06`), la exactitud aritmética de todas las pruebas estadísticas y la coherencia transversal con los **Bloques 0, 1, 2 y 3** antes de iniciar la fase de estudio y exposición al profesor guía Dr. José Delgado Vega.
* **Instrucción (I):** Inspeccionar en profundidad cada tabla, fórmula, ejemplo de código e interpretación física del Bloque 4, identificar cualquier micro-inconsistencia remanente por severidad y entregar el veredicto consolidado del bloque y de toda la etapa predictiva (Bloques 0–4).

---

## 2. Dictamen Ejecutivo de Auditoría

> **Veredicto:** **BLOQUE 4 APROBADO Y SELLADO EN SU NÚCLEO CIENTÍFICO Y ESTADÍSTICO (0 HALLAZGOS CRÍTICOS / 0 HALLAZGOS ALTOS) — CON 4 DETALLES FINOS DE PULIDO EDITORIAL Y DE PROLIJIDAD (`B4-01` A `B4-04`).**

Todos los hallazgos críticos y altos detectados en la revisión previa fueron resueltos con rigor:
1. **Fundamentación metalúrgica corregida (`C-01`):** La Sección 5.1 establece con precisión que `Canje_CuT` es la ley de cabeza química acordada antes de molienda/flotación (protocolo de tres sobres), fundamentando el impacto de `Cancha_CuS` en la friabilidad diferencial y segregación gravitacional de finos en acopio según **Pierre Gy**, y delimitando la Superficie 2D mediante la **Envolvente Convexa (*Convex Hull*)**.
2. **Pruebas estadísticas exactas (`C-02`):** Breusch-Pagan sobre $\hat{y}$, Shapiro-Wilk, Durbin-Watson, *bootstrap* ($B = 10{,}000$) y Wilcoxon exacto (`method='exact'`) coinciden al cuarto/sexto decimal con la verificación independiente.
3. **Portabilidad de ingeniería de software (`C-03`):** Serialización mediante diccionarios $100\%$ nativos de `scikit-learn` junto al módulo desacoplado `BLOQUE_4/modelos_rna.py`.
4. **Transparencia y rigor econométrico (`C-04` a `C-06`):** Nota de `Var. Explicada` en `Cancha_Directa` ($0.6789$ vs. $0.6503$ por el factor $(24/23)^2 = 1.0888$), transparencia *in-sample* de `RNA_MLP_2` ($\text{RMSE} = 0.0966\%\text{Cu}$, $\text{NRMSE} = 6.75\%$) y rutas relativas en figuras.

La auditoría forense de esta versión final detectó únicamente **4 detalles de pulido menor** (ninguno altera cálculos ni conclusiones) que vale la pena corregir antes de entregar el documento al profesor guía o volcarlo a Word.

---

## 3. Verificación Independiente de Cálculos Clave ($N = 24$)

| Dimensión Auditada | Resultado Reportado en Bloque 4 | Verificación Independiente | Estado |
| :--- | :--- | :--- | :---: |
| **Tabla 2.2: Ranking 18 Modelos ($9$ métricas)** | Desde `Res_MLP_2_2` ($\text{MSE} = 0.020781$) hasta `Cancha_Directa` ($\text{MSE} = 0.134971$, $R^2 = -1.2709$). | Exacto al sexto decimal en $\text{MSE}$ y al cuarto decimal en todas las métricas. | **Exacto** |
| **Meses en que cada modelo supera a `Baseline_1B`** | `Cancha_Directa`: $3/24$ ($12.5\%$); `OLS` y `Keras`: $15/24$ ($62.5\%$); `RNA_MLP_2`: $14/24$ ($58.3\%$). | Exacto mes a mes (en `2023-10`, pulso oxidado, `RNA_MLP_2` logra error $0.0632\%\text{Cu}$ frente a $0.0830\%$ de `Baseline_1B` y $0.3284\%$ de `OLS`). | **Exacto** |
| **Error absoluto mediano (`RNA_MLP_2` vs. `Baseline_1B`)** | `RNA_MLP_2`: $0.0801\%\text{Cu}$ vs. `Baseline_1B`: $0.1044\%\text{Cu}$. | Exacto ($\frac{0.0755 + 0.0846}{2} = 0.08005$ vs. $\frac{0.1039 + 0.1048}{2} = 0.10435$). | **Exacto** |
| **Relación Wilcoxon unilateral vs. bilateral (`Cancha` vs. `B1B`)** | `Baseline_1B` vs. `Cancha` ($p = 0.000010$ unilateral); `Cancha` vs. `B1B` ($p = 0.000020$ bilateral). | Exacto ($2 \times 0.000010 = 0.000020$). | **Exacto** |
| **Breusch-Pagan exacto sobre $\hat{y}$** | `Baseline_1B` ($p = 0.4075$), `RNA_MLP_2` ($p = 0.0822$), `OLS` ($p = 0.0016$), `RidgeCV` ($p = 0.0228$), `SVR_RBF` ($p = 0.7243$). | Exacto (`SVR_RBF` arroja $p = 0.7243$ frente a $\hat{y}$ porque sus errores grandes forman una "U" en ambos extremos de $\hat{y}$). | **Exacto** |
| **Transparencia *in-sample* de `RNA_MLP_2`** | $\text{RMSE} = 0.0966\%\text{Cu}$, $\text{NRMSE} = 6.75\%$. | Exacto ($\frac{0.0966}{1.43125} \times 100\% = 6.749\%$). | **Exacto** |

---

## 4. Matriz de Hallazgos Menores y Acciones de Pulido

| ID | Severidad | Sección Auditada | Hallazgo Detectado | Acción Correctiva Concreta |
| :---: | :---: | :---: | :--- | :--- |
| **B4-01** | **MEDIO–BAJO** *(Consistencia interna)* | Sec. 2.2 (Tabla 2.2) | **Empate técnico incompleto en el Ranking 1:** Se asigna `1=` solo a `Res_MLP_2_2` ($\text{MSE} = 0.020781$) y `Baseline_1B` ($\text{MSE} = 0.020782$), mientras que `Res_MLP_4` ($\text{MSE} = 0.020782$) y `Res_MLP_3` ($\text{MSE} = 0.020782$) aparecen como `3` y `4` pese a tener exactamente el mismo $\text{MSE}$, $\text{RMSE}$, $\text{MAE}$ y $R^2$ que `Baseline_1B`. | Declarar **empate técnico cuádruple (`1=`)** para los cuatro modelos equivalentes (`Res_MLP_2_2`, `Baseline_1B`, `Res_MLP_4`, `Res_MLP_3`), pasando `Res_MLP_2` al puesto `5`. |
| **B4-02** | **MEDIO–BAJO** *(Coherencia dominio empírico)* | Sec. 6 (Código de inferencia) | **Ejemplo de inferencia fuera del *Convex Hull*:** El bloque de código de prueba utiliza `X_lote = np.array([[1.55, 0.18]])` ($\text{CuT} = 1.55\% \le 1.80\%$ y $\text{CuS} = 0.18\% > 0.16\%$), cayendo exactamente dentro de la esquina vacía sin datos advertida en la Sección 5.1. | Cambiar el vector de ejemplo a un lote representativo dentro de la envolvente convexa real (por ejemplo, `np.array([[1.75, 0.14]])` o `np.array([[1.85, 0.16]])`). |
| **B4-03** | **BAJO** *(Claridad tabular)* | Sec. 3.1 (Tabla 3.1) | **Etiqueta ambigua en `Res_MLP_2`:** En la columna `Meses Supera a B1B`, la fila `Res_MLP_2` dice `$0/24$ empates`, lo que puede leerse erróneamente como "cero empates", cuando en realidad significa que superó a `Baseline_1B` en $0$ meses y empató en $21$ meses (perdiendo por $\le 0.0001$ en $2$ meses y en noviembre 2023, lo que produce $p = 2 \times 0.5^3 = 0.250000$). | Reemplazar `$0/24$ empates` por `$0/24$ ($21$ empates)` para total precisión estadística. |
| **B4-04** | **BAJO** *(Sintaxis y denominación institucional)* | Sec. 2.1 y pie de página | **Dos detalles editoriales:** (a) En la nota de la Sección 2.1 aparece una doble barra en LaTeX `$10^{-6}\\,(\%\text{Cu})^2$` que rompe el espaciado matemático al renderizar; (b) el pie de página dice *"Escuela de Minería de la Universidad de Antofagasta"*, mientras el encabezado oficial usa *"Departamento de Ingeniería en Minas"*. | Corregir a `$10^{-6}\,(\%\text{Cu})^2$` y unificar el pie de página a *"Departamento de Ingeniería en Minas de la Universidad de Antofagasta"*. |

---

## 5. Claves Técnicas del Bloque 4 para Dominar en la Defensa Oral

Al presentar el Bloque 4 al Dr. José Delgado Vega, estas **cuatro respuestas de precisión** consolidarán tu dominio total de los resultados:

1. **Por qué la Estrategia Dual es superior a forzar un solo modelo:**
   * **`Baseline 1B` / `Res_MLP_2_2`** ($\text{NRMSE} = 10.07\%$, $R^2 = +0.6504$) ofrece la menor varianza global cuando existen pulsos extremos aislados como noviembre de 2023 ($2.27\%\text{Cu}$), ideal para **planificación presupuestaria conservadora**.
   * **`RNA_MLP_2`** ($\text{NRMSE}_{\text{Holdout}} = 6.56\%$, $9$ parámetros, pesos activos) vence a `Baseline 1B` en **$14$ de los $24$ meses ($58.3\%$)**, obtiene un error absoluto mediano menor ($0.0801\%\text{Cu}$ vs. $0.1044\%\text{Cu}$), vence en el pulso oxidado de octubre de 2023 ($0.0632\%$ vs. $0.0830\%$) y aporta **explicabilidad geometalúrgica (SHAP y Superficie 2D)** para decisiones de control y *blending* en mina.
2. **Por qué los modelos lineales fallan el diagnóstico econométrico:**
   * El test de Breusch-Pagan demuestra que `OLS` ($p = 0.0016$), `Keras` ($p = 0.0040$), `LassoCV` ($p = 0.0029$) y `RidgeCV` ($p = 0.0228$) son **heterocedásticos** (sus errores explotan en leyes altas), mientras que `Baseline 1B` ($p = 0.4075$) y `RNA_MLP_2` ($p = 0.0822$) preservan **homocedasticidad**, normalidad de residuos (Shapiro-Wilk $p > 0.10$) e independencia serial (Durbin-Watson $d \approx 2.0$).
3. **Por qué `SVR_RBF` marca homocedasticidad ($p = 0.7243$) pese a ser el peor modelo entrenado:**
   * Breusch-Pagan evalúa una relación *lineal* entre $e_k^2$ y $\hat{y}_k$. En `SVR_RBF`, los errores enormes ocurren tanto en el extremo inferior (`2023-09`, $\hat{y} = 1.06\%$) como en el superior (`2023-11`, $\hat{y} = 1.61\%$), formando una "U" cuya pendiente lineal es cercana a cero ($p = 0.7243$), aunque su varianza global sea casi el triple que `Baseline 1B`.
4. **Por qué `Canje_CuT` no depende de los xantatos de flotación:**
   * Porque la ley de canje se acuerda químicamente en recepción (protocolo de $3$ sobres ENAMI) **antes** de la molienda y flotación. El rol físico de `Cancha_CuS` ocurre aguas arriba en la cancha de acopio (friabilidad diferencial y segregación de finos oxidados bajo la Teoría de Muestreo de Pierre Gy).

---

## 6. Tablero Consolidado de Salud Técnica (Bloques 0 a 4)

Con esta revisión se completa la auditoría integral bloque por bloque de toda la fase de ingeniería de datos, análisis exploratorio, modelado y validación predictiva:

| Bloque | Estado de Solidez para Reunión con Profesor Guía | Único Ajuste Pendiente Crítico / Prioritario | Impacto en Resultados ML (Bloques 1–4) |
| :---: | :--- | :--- | :---: |
| **Bloque 0** | **Aprobado en datos de leyes y tonelajes** | `[B0-01]` Homologar moneda de Tabla 5.2 de tarifas ENAMI (`22` meses en `MCLP/tms` vs. `2` en `US$/tms`) **antes del Bloque 5**; precisar causa de brecha química `Canje` 2024 (`B0-02`). | **Ninguno** (Bloques 1–4 no usan tarifas) |
| **Bloque 1** | **Aprobado (solidez metodológica alta)** | `[B1-01]` Unificar Shapiro-Wilk de `Sesgo_Canje` a $W = 0.9852, p = 0.9695$ (igual a Bloque 4); `[B1-02]` redactar umbral $8.63\%$ como hipótesis. | **Ninguno** |
| **Bloque 2** | **Aprobado (exactitud aritmética 100%)** | `[B2-01/02/03]` Sumar columna `MSE` a Tabla 3.1 y sincronizar texto de Secciones 8.1–8.2 con lo efectivamente ejecutado en Bloque 3. | **Ninguno** |
| **Bloque 3** | **Aprobado (exactitud aritmética 100%)** | `[B3-01]` Corregir párrafo de cinética de flotación en Sec. 2.1 (alineándolo a Pierre Gy del Bloque 4); `[B3-02]` unificar `Var. Explicada` de $5$ redes con Bloque 4. | **Ninguno** |
| **Bloque 4** | **Aprobado y sellado** | `[B4-01 a B4-04]` Solo pulido editorial menor (empate `1=` cuádruple, lote de ejemplo dentro del *Convex Hull*, etiqueta `Res_MLP_2`, typo LaTeX y pie de página UA). | **Ninguno** |

> **Conclusión de Auditoría Global (Bloques 0–4):** Toda la arquitectura científica, matemática y estadística de tus **Bloques 0 al 4** es sólida, coherente y defendible ante el Dr. José Delgado Vega. No necesitas reentrenar ningún modelo antes de tu reunión de avance; solo ten presentes los ajustes puntuales de coherencia documental (`B0-01` antes del Bloque 5, `B3-01` en el texto del Bloque 3, y los pulidos editoriales menores). Quedas completamente listo para iniciar la etapa de estudio y dominio conceptual de toda la materia.