# AUDITORÍA EXPERTO-CRÍTICA (MODELO RCI) — BLOQUE 3

## Verificación Matemática de Redes Neuronales (RNA), Coherencia Transversal con el Bloque 4 y Blindaje para Defensa Oral

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas

**Documento Auditado:** `BLOQUE_3/INFORME_ENTREGABLE_BLOQUE_3.md` (Versión 4.0 — Cierre Consolidado)

**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)

**Institución:** Universidad de Antofagasta — Facultad de Ingeniería, Departamento de Ingeniería en Minas

**Destinatario Técnico:** Revisión integral previa a presentación al Profesor Guía y consolidación en `INFORME_TESIS.docx`

---

## 1. Marco de Auditoría RCI (Rol, Contexto, Instrucción)

* **Rol (R):** Auditor Técnico Senior en Geoestadística Minera, Procesos Metalúrgicos de Comercialización ENAMI y Machine Learning aplicado a muestras pequeñas ($N = 24$).
* **Contexto (C):** Revisión forense bloque por bloque (`INFORME_ENTREGABLE_BLOQUE_3.md`, Metodología v4.0) para comprobar que las $11$ arquitecturas neuronales (`Direct-MLP`, `Residual-MLP` y réplica Keras del Dr. José Delgado Vega), el estudio de ablación, el *Holdout* 19/5 y *Buffered-LOOCV* sean aritméticamente exactos y **100% coherentes con el Bloque 4 ya auditado y sellado**.
* **Instrucción (I):** Cruzar todos los indicadores del Bloque 3 contra el Bloque 4, detectar cualquier contradicción física, matemática o editorial residual entre capítulos, clasificar los hallazgos por severidad y consolidar las respuestas técnicas clave para tu reunión con el profesor guía.

---

## 2. Dictamen Ejecutivo de Auditoría

> **Veredicto:** **BLOQUE 3 APROBADO EN EJECUCIÓN EXPERIMENTAL Y ARITMÉTICA — REQUIERE 1 CORRECCIÓN CRÍTICA DE COHERENCIA METALÚRGICA CON EL BLOQUE 4 (`B3-01`) Y 3 AJUSTES MENORES DE SINCRONIZACIÓN.**

Los resultados numéricos principales de las $11$ redes neuronales son sólidos, reproducibles y coinciden exactamente con el *benchmarking* de $18$ modelos del Bloque 4 en $\text{MSE}$, $\text{RMSE}$, $\text{MAE}$, $\text{NRMSE}$, $R^2$, $\text{MAPE}$, Pearson $r$ y Sesgo. Asimismo, quedaron correctamente resueltos el conteo de $57$ parámetros nominales de `RNA_Delgado_Keras` ($P_{\text{eff}} = 3$), la distinción entre $\text{MAPE}$ y $\text{NRMSE}$ *in-sample* y la explicación geoestadística de la adyacencia octubre–noviembre de 2023 en *Buffered-LOOCV*.

Sin embargo, la auditoría transversal detectó **1 contradicción crítica frente al Bloque 4**: la Sección 2.1 del Bloque 3 aún conserva la explicación metalúrgica errónea sobre cinética de flotación y recuperación en celdas de ENAMI que fue corregida en el Bloque 4. Corregir este párrafo antes de mostrar el avance al Dr. Delgado evita una objeción inmediata de Ingeniería Metalúrgica.

---

## 3. Verificación de Consistencia Transversal (Bloque 3 vs. Bloque 4, $N = 24$)

| Modelo Neuronal | $\text{MSE}$ ($(\%\text{Cu})^2$) | $\text{RMSE}$ ($\%\text{Cu}$) | $\text{NRMSE}$ ($\%$) | $R^2$ | $\text{Sesgo}$ ($\%\text{Cu}$) | $\text{Var. Expl.}$ Bloque 3 | $\text{Var. Expl.}$ Bloque 4 (Exacta) | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Res_MLP_2_2`** | $0.020781$ | $0.1442$ | $10.07\%$ | $+0.6504$ | $+0.0000$ | $+0.6504$ | $0.6504$ | **Exacto** |
| **`Res_MLP_3`** | $0.020782$ | $0.1442$ | $10.07\%$ | $+0.6503$ | $-0.0000$ | $+0.6503$ | $0.6503$ | **Exacto** |
| **`Res_MLP_4`** | $0.020782$ | $0.1442$ | $10.07\%$ | $+0.6503$ | $-0.0000$ | $+0.6503$ | $0.6503$ | **Exacto** |
| **`Res_MLP_2`** | $0.021551$ | $0.1468$ | $10.26\%$ | $+0.6374$ | $-0.0022$ | $+0.6375$ | $0.6375$ | **Exacto** |
| **`RNA_Delgado_Keras`** | $0.027735$ | $0.1665$ | $11.64\%$ | $+0.5334$ | $+0.0022$ | $+0.5334$ | $0.5334$ | **Exacto** |
| **`RNA_MLP_2`** | $0.030890$ | $0.1758$ | $12.28\%$ | $+0.4803$ | $-0.0124$ | **$+0.4842$** | **$0.4829$** | **Microdesfase (`B3-02`)** |
| **`RNA_MLP_4`** | $0.031446$ | $0.1773$ | $12.39\%$ | $+0.4709$ | $+0.0060$ | **$+0.4720$** | **$0.4715$** | **Microdesfase (`B3-02`)** |
| **`RNA_MLP_3`** | $0.031951$ | $0.1787$ | $12.49\%$ | $+0.4624$ | $+0.0068$ | **$+0.4638$** | **$0.4632$** | **Microdesfase (`B3-02`)** |
| **`RNA_Delgado_MLP`** | $0.032592$ | $0.1805$ | $12.61\%$ | $+0.4516$ | $+0.0001$ | **$+0.4517$** | **$0.4516$** | **Microdesfase (`B3-02`)** |
| **`RNA_MLP_2_2`** | $0.034635$ | $0.1861$ | $13.00\%$ | $+0.4173$ | $+0.0009$ | $+0.4173$ | $0.4173$ | **Exacto** |
| **`RNA_MLP_3_2`** | $0.037222$ | $0.1929$ | $13.48\%$ | $+0.3737$ | $-0.0044$ | **$+0.3746$** | **$0.3741$** | **Microdesfase (`B3-02`)** |

---

## 4. Matriz de Hallazgos y Acciones Correctivas

| ID | Severidad | Sección Auditada | Hallazgo Detectado | Acción Correctiva Concreta |
| :---: | :---: | :---: | :--- | :--- |
| **B3-01** | **CRÍTICO** *(Contradicción metalúrgica con Bloque 4)* | Sec. 2.1 (Teoría de Pierre Gy) | **Confusión entre ley de canje y recuperación en flotación:** El punto 1 afirma que *"las especies oxidadas poseen cinéticas de flotación lentas y muy baja recuperación en circuitos concentradores... provocando que la ley de canje comercial resulte inferior a la ley total de acopio"*. Esto **contradice directamente la Sección 5.1 del Bloque 4**, donde se corrigió que `Canje_CuT` es la ley de cabeza química acordada en portería (protocolo de 3 sobres) *antes* de entrar a molienda y flotación. | Reemplazar el punto 1 de la Sección 2.1 por la explicación física correcta ya certificada en el Bloque 4: **friabilidad diferencial y menor dureza de Mohs de los óxidos supergenos (crisocola, malaquita)** respecto a sulfuros competentes, lo que intensifica la pulverización y segregación de finos en el cono de acopio de faena frente a la torre de muestreo mecánico de ENAMI. |
| **B3-02** | **MEDIO–BAJO** *(Aritmético entre capítulos)* | Sec. 4.1 (Tabla 4.1) | **Microdesfase en la columna `Var. Explicada` frente al Bloque 4:** En $5$ redes directas (`RNA_MLP_2`, `RNA_MLP_4`, `RNA_MLP_3`, `RNA_Delgado_MLP` y `RNA_MLP_3_2`), la `Var. Explicada` difiere entre $0.0001$ y $0.0013$ respecto al cálculo exacto de `explained_variance_score` en el Bloque 4 ($R^2 + \frac{\text{Sesgo}^2}{\sigma_y^2}$). Por ejemplo, para `RNA_MLP_2`: $0.48028 + \frac{(-0.01236)^2}{0.059436} = \mathbf{0.4829}$ (no $0.4842$). | Actualizar los $5$ valores en la Tabla 4.1 del Bloque 3 (`0.4829`, `0.4715`, `0.4632`, `0.4516`, `0.3741`) para garantizar coincidencia al cuarto decimal con la Tabla 2.2 del Bloque 4. |
| **B3-03** | **BAJO** *(Consistencia de grados de libertad y `lr` Keras)* | Sec. 2.4 y Sec. 3.1 | **Dos desajustes menores de precisión:** (a) La Tabla 2.4 lista $P_{\text{eff}} = 15, 13, 17$ para `Res_MLP_2_2`, `Res_MLP_3` y `Res_MLP_4`, mientras el Bloque 4 (Sec. 4) declara $P_{\text{eff}} = 1$ por colapso de pesos en $\alpha = 2.0$; (b) el código Keras de Sec. 3.1 muestra `Adam(0.1)` (original del Dr. Delgado), mientras la Tabla 4.1 ejecutó `Adam(0.05)`. | (a) Aclarar en Tabla 2.4 que $15, 13, 17$ son los grados de libertad *estructurales*, los cuales se contraen a $P_{\text{eff}} \approx 1$ bajo regularización $L_2$ ($\alpha = 2.0$); (b) añadir una línea en Sec. 3.1 precisando que en la réplica LOOCV se ajustó la tasa de aprendizaje a `0.05` para evitar oscilaciones en $N_{\text{train}} = 23$ sin alterar la equivalencia lineal. |
| **B3-04** | **BAJO** *(Blindaje de defensa oral)* | Sec. 1 (A-02) y Sec. 4.1 | **Lectura de $\alpha$ fijo por topología en Tabla 4.1:** Cada arquitectura presenta una combinación distinta de activación y $\alpha$ (`0.001` en `RNA_MLP_2`, `5.0` en `RNA_MLP_3/4`, `2.0` en `RNA_MLP_2_2/3_2`). Si un evaluador pregunta por qué no todas comparten el mismo $\alpha$ si fueron "predefinidas a priori", conviene tener la respuesta precisa. | Explicar con transparencia que se evaluó una grilla predefinida de activaciones $\{\text{relu}, \text{tanh}\}$ y regularizaciones $L_2$ ($\alpha$), reportando la configuración más estable por topología; y que para controlar cualquier optimismo de selección en $N = 24$, el modelo se contrasta contra `Baseline 1B` y se verifica en la ventana prospectiva intacta *Holdout* 19/5. |

---

## 5. Corrección Exacta Recomendada para el Hallazgo Crítico `[B3-01]`

Reemplazar el numeral `1.` de la **Sección 2.1** de `BLOQUE_3/INFORME_ENTREGABLE_BLOQUE_3.md` por el siguiente texto (100% alineado con el Bloque 4):

> *1. Dado que `Canje_CuT` corresponde a la ley de cabeza química acordada en recepción mediante el protocolo reglamentario de tres sobres antes de la molienda y flotación, la divergencia no proviene de pérdidas en celdas de flotación, sino del muestreo físico del acopio: los minerales oxidados supergenos (crisocola, malaquita) poseen menor dureza y mayor friabilidad que los sulfuros competentes, generando pulverización diferencial y segregación gravitacional de finos en la cancha que alteran la representatividad frente a la torre de muestreo automático de ENAMI.*

---

## 6. Claves Técnicas del Bloque 3 para Dominar en la Defensa Oral

Al presentar el Bloque 3 al Dr. José Delgado Vega, domina estas **tres preguntas centrales**:

### 6.1. *"¿Por qué mi red en Colab daba $8.63\%$ y en tu tesis da $11.64\%$?"*
* **Respuesta exacta y diplomática:** Su modelo original en Keras (`Dense(4)-Dense(4)-Dense(4)-Dense(1)`) se calibró y evaluó sobre los mismos $24$ meses (*in-sample*, $\text{MAPE} = 7.97\%$ en PSJ y $7.38\%$ en Canje, del mismo orden que el $8.63\%$ reportado en marzo 2025). Al llevar esa arquitectura a validación estrictamente ciega (**LOOCV**, prediciendo cada mes sin haberlo visto), el error real de generalización es **$\text{NRMSE} = 11.64\%$** ($\text{MAPE} = 8.74\%$). Además, al no declarar función de activación no lineal (`relu` o `tanh`) en las capas ocultas, el producto de sus matrices colapsa algebraicamente a $P_{\text{eff}} = 3$ (equivalente a `OLS`, $11.85\%$), lo que justificó explorar redes verdaderamente no lineales y parsimoniosas.

### 6.2. *"Si `Residual-MLP` empata a `Baseline 1B` ($10.07\%$) porque sus pesos se apagan, ¿para qué sirvió implementarla?"*
* **Respuesta de alto nivel científico:** Sirvió para responder dos preguntas fundamentales en $N = 24$:
  1. **Control de extrapolación:** Probó que introducir una conexión física directa (*skip-connection*) $\hat{y} = \text{CuT} - \hat{e}_{\text{MLP}}$ elimina el problema de atenuación de pendiente de las redes directas en leyes extremas como noviembre 2023 ($2.27\%\text{Cu}$).
  2. **Evidencia de frontera de complejidad:** Demostró matemáticamente que, al regularizar con $L_2$ ($\alpha = 2.0$), el propio optimizador contrae los pesos residuales hacia cero para no sobreajustar la pequeña varianza residual ($\sigma^2_e = 0.0191\,(\%\text{Cu})^2$), certificando que en el histórico global de $24$ meses el sesgo estacionario es el límite de mínima varianza.

### 6.3. *"¿Por qué `RNA_MLP_2` logra un excelente $6.56\%$ en Holdout 19/5, pero sube a $12.28\%$ en LOOCV y $14.53\%$ en Buffered-LOOCV?"*
* **Respuesta física y estadística:**
  * En los últimos $5$ meses de 2024 (*Holdout* 19/5), la faena operó en un régimen regular sin pulsos extremos (`Canje_CuT` entre $1.06\%$ y $1.48\%$), donde `RNA_MLP_2` (con solo $9$ parámetros y activación suave `tanh`) aprovechó el aporte de `Cancha_CuS` ($-1.53\%$ en ablación) y superó a `RidgeCV` ($7.57\%$), `OLS` ($7.76\%$) y `Baseline 1B` ($8.96\%$).
  * En LOOCV global, su error sube porque en **noviembre de 2023** ($2.27\%\text{Cu}$, el máximo histórico absoluto) la función `tanh` satura ligeramente al extrapolar fuera del rango de entrenamiento.
  * En *Buffered-LOOCV*, al evaluar noviembre 2023 ($t=11$) se elimina automáticamente su único vecino rico, octubre 2023 ($t=10$), dejando al modelo entrenado solo hasta $1.57\%\text{Cu}$ para extrapolar hasta $2.27\%\text{Cu}$.

---

## 7. Puntos Fuertes del Bloque 3 para Presentar al Profesor Guía

1. **Principio de Occam / Parsimonia demostrado empíricamente:** La red más pequeña (`RNA_MLP_2`, $9$ parámetros, $N_{\text{train}}/P = 2.56$) supera consistentemente a las redes profundas de $2$ capas (`(2, 2)`, `(3, 2)` y `(4, 4)`), probando que en muestras mineras pequeñas menos neuronas generalizan mejor.
2. **Validación cuantitativa de la recomendación del Dr. Delgado sobre `Cancha_CuS`:** El estudio de ablación demuestra que retirar el cobre soluble degrada a `RNA_MLP_2` en **$+1.53\%$ de NRMSE** (de $12.28\%$ a $13.81\%$) y derrumba $R^2$ de $+0.4803$ a $+0.3427$.
3. **Honestidad metodológica total:** Evaluación simétrica en *Holdout* 19/5, métricas en escala cuadrática ($\text{MSE}$) y física ($\text{RMSE}$, $\text{MAE}$), y diagnóstico abierto del colapso lineal de Keras y de contracción en `Residual-MLP`.

---

## 8. Instrucciones Directas de Pulido (Sincronización Bloque 3 ↔ Bloque 4)

Sin necesidad de re-entrenar ninguna red neuronal, aplicar estos tres ajustes documentales al actualizar `BLOQUE_3/INFORME_ENTREGABLE_BLOQUE_3.md` y al redactar la memoria en Word (`INFORME_TESIS.docx`):

1. **Corregir Sección 2.1 (punto 1):** Reemplazar la referencia a cinética/recuperación de flotación por el mecanismo de friabilidad diferencial y segregación de finos en acopio según **Pierre Gy** (sincronización exacta con Sección 5.1 del Bloque 4).
2. **Unificar columna `Var. Explicada` en Tabla 4.1:** Actualizar `RNA_MLP_2` ($0.4829$), `RNA_MLP_4` ($0.4715$), `RNA_MLP_3` ($0.4632$), `RNA_Delgado_MLP` ($0.4516$) y `RNA_MLP_3_2` ($0.3741$) para coincidir al cuarto decimal con el Bloque 4.
3. **Anotar matices en Secciones 2.4 y 3.1:** Precisar el colapso efectivo $P_{\text{eff}} \approx 1$ de `Res_MLP_2_2/3/4` bajo $\alpha = 2.0$ y la tasa `Adam(0.05)` usada en la réplica LOOCV de Keras.