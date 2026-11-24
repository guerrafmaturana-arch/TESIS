# AUDITORÍA EXPERTO-CRÍTICA (MODELO RCI) — BLOQUE 2

## Verificación Matemática de Modelos de Referencia (*Baselines*), Sensibilidad LOOCV y Coherencia Transversal (Bloques 1–4)

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas

**Documento Auditado:** `BLOQUE_2/INFORME_ENTREGABLE_BLOQUE_2.md` (Versión 4.0)

**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)

**Institución:** Universidad de Antofagasta — Facultad de Ingeniería, Departamento de Ingeniería en Minas

**Destinatario Técnico:** Revisión de cierre previa a presentación al Profesor Guía y traspaso a `INFORME_TESIS.docx`

---

## 1. Marco de Auditoría RCI (Rol, Contexto, Instrucción)

* **Rol (R):** Auditor Técnico Senior en Geoestadística Minera, Econometría Aplicada a Muestras Pequeñas ($N = 24$) y Machine Learning para conciliación Mina–Planta (estándar comercial ENAMI).
* **Contexto (C):** Revisión forense bloque por bloque (`INFORME_ENTREGABLE_BLOQUE_2.md`, Metodología v4.0) para validar que el protocolo LOOCV sin fuga de datos, los $7$ modelos de referencia (*baselines*), el contraste de escaladores (`StandardScaler` vs. `RobustScaler`), la verificación geoestadística (*Buffered-LOOCV*) y las directrices hacia el Bloque 3 sean aritméticamente exactos y **plenamente consistentes con los hallazgos finales ya consolidados en los Bloques 3 y 4**.
* **Instrucción (I):** Verificar de forma independiente cada métrica de desempeño fuera de muestra, detectar cualquier contradicción narrativa o terminológica entre bloques, clasificar los hallazgos por severidad y consolidar las claves de interpretación técnica para la defensa ante el Dr. José Delgado Vega.

---

## 2. Dictamen Ejecutivo de Auditoría

> **Veredicto:** **BLOQUE 2 APROBADO AL 100% EN RIGOR NUMÉRICO Y EXPERIMENTAL — CON 5 OBSERVACIONES DE SINCRONIZACIÓN TRANSVERSAL PARA PULIDO ANTES DE LA MEMORIA FINAL.**

Los resultados cuantitativos del **Bloque 2** son matemáticamente exactos, libres de fuga de información (*zero data leakage*) y coinciden al cuarto decimal con el *benchmarking* unificado de $18$ modelos del **Bloque 4**:

1. **Verificación aritmética perfecta:** Los $7$ vectores de predicciones fuera de muestra (`Tabla 3.2`) reproducen exactamente todas las métricas de la `Tabla 3.1` y empalman sin desfase con la tabla maestra del Bloque 4.
2. **Protocolo experimental blindado:** Se confirma la corrección de la errata `GCV/LOOCV` en `RidgeCV`, la invarianza frente al escalador ($\vert\Delta\text{NRMSE}\vert \le 0.09\%$) y la ausencia de fuga temporal en `Buffered-LOOCV` ($\Delta\text{NRMSE} = +0.04\%$ en `LassoCV` y $-0.03\%$ en `Baseline 1B`).
3. **Evidencia empírica sobre `Cancha_CuS`:** `LassoCV` retuvo un coeficiente estrictamente positivo para `Cancha_CuS` en $24$ de $24$ pliegues ($100.0\%$, $\bar{\beta}_{\text{CuS}} = +0.0264$).

No obstante, al auditar este bloque mirando el **hilo conductor completo hasta el Bloque 4**, emergen **3 hallazgos de severidad media (`B2-01` a `B2-03`)** y **2 observaciones menores (`B2-04` y `B2-05`)** que debes ajustar al pasar a Word (y tener claros en tu exposición oral) para evitar contradicciones entre capítulos o preguntas trampa de la comisión.

---

## 3. Verificación Independiente de Cálculos y Consistencia con Bloque 4 ($N = 24$)

| Modelo de Referencia | $\text{RMSE}$ Bloque 2 ($\%\text{Cu}$) | $\text{NRMSE}$ Bloque 2 ($\%$) | $R^2$ Bloque 2 | $\text{MSE}$ Verificado en Bloque 4 ($(\%\text{Cu})^2$) | Consistencia Tablas 3.1 / 3.2 / Bloque 4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Línea Base 1A (`Cancha Directa`)** | $0.3674$ | $25.67\%$ | $-1.2709$ | $0.134971$ | **Exacto** |
| **`Baseline 1B` (Sesgo LOOCV)** | $0.1442$ | $10.07\%$ | $+0.6503$ | $0.020782$ | **Exacto** |
| **`OLS` (Regresión Lineal Múltiple)** | $0.1697$ | $11.85\%$ | $+0.5156$ | $0.028789$ | **Exacto** |
| **`RidgeCV` ($L_2$)** | $0.1685$ | $11.78\%$ | $+0.5221$ | $0.028403$ | **Exacto** |
| **`LassoCV` ($L_1$)** | $0.1760$ | $12.30\%$ | $+0.4789$ | $0.030971$ | **Exacto** |
| **`ElasticNetCV` ($L_1 + L_2$)** | $0.1769$ | $12.36\%$ | $+0.4735$ | $0.031294$ | **Exacto** |
| **`SVR` (Kernel RBF)** | $0.2387$ | $16.68\%$ | $+0.0412$ | $0.056986$ | **Exacto** |

---

## 4. Matriz de Hallazgos y Acciones Correctivas

| ID | Severidad | Sección Auditada | Hallazgo Detectado | Acción Correctiva Concreta |
| :---: | :---: | :--- | :--- | :--- |
| **B2-01** | **MEDIO** *(Coherencia narrativa con Bloques 3–4)* | Sec. 8.1 (Valla Técnica) | **Auto-trampa metodológica frente a los resultados reales de Bloques 3 y 4:** Se afirma categóricamente que *"para justificar la superioridad del ML/DL, la RNA debe perforar la barrera del $10.07\%$"* y se equipara la meta de $8.63\%$ con un $\text{NRMSE}$ fuera de muestra ($R^2 > 0.74$). En los Bloques 3 y 4 se demostró que $8.63\%$ era un $\text{MAPE}$ *in-sample* y que en LOOCV global ($N=24$) el piso es $10.07\%$ (empatado por `Res_MLP_2_2`), mientras `RNA_MLP_2` perfora la meta en *Holdout* 19/5 ($6.56\%$). | Reformular la Sección 8.1 presentando $10.07\%$ como el **benchmark empírico de referencia global en LOOCV** y $8.63\%$ como **referente histórico preliminar a auditar en igualdad de condiciones (LOOCV y Holdout)**, eliminando la exigencia absoluta que contradiría la Estrategia Dual del Bloque 4. |
| **B2-02** | **MEDIO** *(Homologación de métricas y trampa de $s_e^2$)* | Sec. 3.1 (Tabla 3.1) | **Ausencia de la columna `MSE` y paradoja no explicada de `Varianza Residual` ($s_e^2$):** A diferencia de los Bloques 3 y 4, la Tabla 3.1 omite `MSE` ($(\%\text{Cu})^2$), `Sesgo` y `Var. Explicada`, y reporta `Varianza Residual` $s_e^2$, donde `Cancha Directa` figura con el valor **más bajo de todos** ($0.019917$ frente a $0.021676$ de `Baseline 1B`). Sin nota explicativa, un evaluador puede malinterpretarlo. | Añadir las columnas `MSE` ($(\%\text{Cu})^2$) y `Sesgo` ($\%\text{Cu}$) para unificar el formato con los Bloques 3 y 4, y agregar una nota aclarando que $s_e^2$ descuenta el sesgo medio *in-sample* ($\bar{e} = +0.3404\%\text{Cu}$), mientras que $\text{MSE} = \bar{e}^2 + \sigma_e^2 = 0.1350\,(\%\text{Cu})^2$ penaliza la sobreestimación sistemática. |
| **B2-03** | **MEDIO** *(Desfase de planificación vs. ejecución Bloque 3)* | Sec. 8.2 y Sec. 9 | **Inconsistencias técnicas entre lo prometido para el Bloque 3 y lo efectivamente ejecutado:** (a) Nombra `03_red_neuronal_loocv.ipynb` y `LOG_ENTRENAMIENTO_RNA.csv` (los archivos reales son `03_modelado_rna_loocv.ipynb` y `LOG_HIPERPARAMETROS_RNA.csv`); (b) indica `early stopping interno` (inaplicable al optimizador `L-BFGS` usado en Bloque 3); (c) fija $\alpha \in [0.01, 1.0]$ y $N/P \ge 2.5$ (en Bloque 3 se evaluó $\alpha \in [0.001, 5.0]$ y $N_{\text{train}}/P \in [1.15, 2.56]$). | Sincronizar la Sección 8.2 y el pie de la Sección 9 con los nombres de archivo, el optimizador `L-BFGS` (controlado por regularización $L_2$ y ensamble multi-semilla en lugar de *early stopping*) y los rangos exactos efectivamente ejecutados en el Bloque 3. |
| **B2-04** | **BAJO** *(Matiz econométrico)* | Sec. 3.1 y Sec. 4.1 | **Sobredimensión verbal en `LassoCV` y ausencia de advertencia sobre `Pearson r`:** (a) Decir que Lasso confirma la *"indispensabilidad categórica"* de `Cancha_CuS` es ligeramente excesivo cuando los modelos lineales 2D aún pierden frente al 1D `Baseline 1B`; (b) en Tabla 3.1, `Cancha Directa` muestra el mayor `Pearson r` ($+0.8347$) pese a tener $R^2 = -1.2709$. | Matizar a *"confirma una señal marginal positiva persistente en los $24$ pliegues que justifica probar su interacción no lineal en el Bloque 3"*, y recordar al pie de la Tabla 3.1 que `Pearson r` es invariante al sesgo de nivel. |
| **B2-05** | **BAJO** *(Precisión menor)* | Sec. 2.2 y Sec. 8.1 | **Dos detalles de prolijidad:** (a) En Sec. 8.1 figura $85.83\%$ en lugar del cociente exacto $85.86\%$ ($\frac{0.115884}{0.134971}$); (b) en `LassoCV`, `ElasticNetCV` y `SVR`, `cv=5` interno particiona en bloques secuenciales sobre $N_{\text{train}} = 23$, agrupando octubre y noviembre de 2023 en un mismo pliegue interno. | Homologar $85.86\%$ con el Bloque 1 y tener presente que el particionamiento interno `cv=5` secuencial explica la alternancia de `l1_ratio` ($0.10$ / $0.99$) en `ElasticNetCV`, sin afectar la validez *out-of-sample*. |

---

## 5. Claves Técnicas del Bloque 2 para Dominar en la Defensa Oral

Al mostrar el avance del Bloque 2 al Dr. José Delgado Vega, es muy probable que te haga estas **tres preguntas de fondo**. Aquí tienes la respuesta matemática y metalúrgica exacta:

### 5.1. ¿Por qué `Baseline 1B` ($10.07\%$) supera a `OLS` ($11.85\%$) y `RidgeCV` ($11.78\%$)?
1. **Pendiente fija unitaria vs. atenuación de pendiente:** `Baseline 1B` conserva una relación física $1:1$ con la ley de acopio ($\hat{y} = 1.0 \cdot \text{Cancha\_CuT} - \bar{e}_{-k}$). En cambio, `OLS` estima una pendiente achatada sobre `Cancha_CuT` ($\approx 0.74 < 1.0$) arrastrada por los meses de leyes medias/bajas con sesgo alto.
2. **El $40.5\%$ del error cuadrático de `OLS` proviene de solo dos meses (`2023-10` y `2023-11`):**
   * En **octubre de 2023** ($\text{CuT} = 2.21\%$, $\text{CuS} = 0.26\%$, $\text{Canje} = 1.79\%$): el plano lineal aplica su pendiente positiva sobre el pico anómalo de cobre soluble y sobreestima gravemente ($\hat{y}_{\text{OLS}} = 2.1184\%$, error $+0.3284\%\text{Cu}$).
   * En **noviembre de 2023** ($\text{CuT} = 2.47\%$, $\text{CuS} = 0.20\%$, $\text{Canje} = 2.27\%$): la pendiente achatada de `OLS` subestima el frente sulfurado rico ($\hat{y}_{\text{OLS}} = 1.8555\%$, error $-0.4145\%\text{Cu}$), mientras `Baseline 1B` predice $2.1235\%$ (error de solo $-0.1465\%\text{Cu}$).

### 5.2. ¿Por qué `OLS` *mejora* en `Buffered-LOOCV` ($11.85\% \to 11.07\%$) mientras `SVR` colapsa ($16.68\% \to 20.97\%$)?
* **En `OLS`:** Como octubre ($t=10$) y noviembre de 2023 ($t=11$) son adyacentes, al excluir $t \pm 1$ para evaluar octubre se remueve el apalancamiento de noviembre (y viceversa), impidiendo que un pulso distorsione la pendiente del plano sobre el otro.
* **En `SVR (Kernel RBF)`:** El kernel gaussiano $K(x, x') = \exp(-\gamma \Vert x - x' \Vert^2)$ es un aproximador estrictamente local. Al perder densidad de vectores de soporte en la cola superior (noviembre 2023 y septiembre 2023), la función decae hacia el valor medio del dataset, haciendo caer $R^2$ a $-0.5152$. Esto demuestra empíricamente por qué una **MLP parsimoniosa** (aproximador global suave) es muy superior a `SVR` en $N = 24$.

### 5.3. La paradoja de `Cancha Directa`: menor $s_e^2$ ($0.019917$) y mayor $r$ ($+0.8347$), pero $R^2 = -1.2709$
* Tanto la **correlación de Pearson ($r$)** como la **varianza residual ($s_e^2$)** son **invariantes ante traslaciones constantes**: no se enteran de que la cancha sobreestima todos los meses en $+0.3404\%\text{Cu}$.
* El **$\text{MSE}$ ($0.1350\,(\%\text{Cu})^2$)** y el **$R^2$ ($-1.2709$)** miden la distancia exacta a la recta identidad $y = \hat{y}$, penalizando el sesgo al cuadrado ($\bar{e}^2 = 0.1159\,(\%\text{Cu})^2$).

---

## 6. Puntos Fuertes del Bloque 2 para Presentar al Profesor Guía

1. **Disciplina experimental irreprochable (*Zero Data Leakage*):** Todos los escaladores (`StandardScaler` / `RobustScaler`) y sintonizadores internos (`RidgeCV`, `LassoCV`, `ElasticNetCV`, `GridSearchCV`) operan dentro de `Pipeline` sobre $N_{\text{train}} = 23$.
2. **Desmitificación de la valla del $26\%$:** Al incorporar `Baseline 1B` ($10.07\%$), demuestras madurez ingenieril al no inflar el mérito de tus redes comparándolas únicamente contra la cancha sin calibrar ($25.67\%$).
3. **Respaldo objetivo para usar `StandardScaler` y LOOCV en el Bloque 3:** Tanto el contraste con `RobustScaler` ($\vert\Delta\text{NRMSE}\vert \le 0.09\%$) como `Buffered-LOOCV` blindan tus decisiones metodológicas antes de entrenar la primera neurona.

---

## 7. Instrucciones Directas de Pulido (Para Sincronización Final / Memoria)

Sin necesidad de re-entrenar ningún modelo del Bloque 2, aplicar estos cuatro ajustes documentales al consolidar el informe final en Word (`INFORME_TESIS.docx`) o al sincronizar `./BLOQUE_2/INFORME_ENTREGABLE_BLOQUE_2.md`:

1. **Añadir columnas `MSE` ($(\%\text{Cu})^2$) y `Sesgo` ($\%\text{Cu}$)** a la Tabla 3.1 e incluir una nota al pie explicando la paradoja de `s^2_e` y `Pearson r` en `Cancha Directa`.
2. **Ajustar la redacción de la Sección 8.1:** Presentar el $10.07\%$ (`Baseline 1B`) y el $8.63\%$ (referente preliminar del Dr. Delgado) como **hipótesis y referencias comparativas** (sin afirmar que la RNA *debe* batir $10.07\%$ en LOOCV global para tener valor).
3. **Sincronizar la Sección 8.2 y la Sección 9 con el Bloque 3 ejecutado:** Actualizar los nombres `03_modelado_rna_loocv.ipynb` y `LOG_HIPERPARAMETROS_RNA.csv`, reemplazar *"parada temprana"* por *"regularización $L_2$ con optimizador cuasi-Newton `L-BFGS` y ensamble multi-semilla"*, y alinear los rangos de $\alpha$ y $N_{\text{train}}/P$.
4. **Unificar `85.86%`** en la Sección 8.1 y matizar la conclusión de `LassoCV` en la Sección 4.1.