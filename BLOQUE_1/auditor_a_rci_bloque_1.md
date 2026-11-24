# AUDITORÍA EXPERTO-CRÍTICA (MODELO RCI) — BLOQUE 1

## Verificación Estadística, Consistencia Transversal (Bloques 0–4) y Blindaje Metodológico del EDA

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas

**Documento Auditado:** `BLOQUE_1/INFORME_ENTREGABLE_BLOQUE_1.md` (Versión 4.0)

**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)

**Institución:** Universidad de Antofagasta — Facultad de Ingeniería, Departamento de Ingeniería en Minas

**Destinatario Técnico:** Revisión de cierre previa a presentación al Profesor Guía y traspaso a `INFORME_TESIS.docx`

---

## 1. Marco de Auditoría RCI (Rol, Contexto, Instrucción)

* **Rol (R):** Auditor Técnico Senior en Geoestadística Minera, Estadística Aplicada a Muestras Pequeñas ($N = 24$) y Ciencia de Datos Mineros bajo estándar de la Universidad de Antofagasta.
* **Contexto (C):** Revisión forense bloque por bloque (`INFORME_ENTREGABLE_BLOQUE_1.md`, Metodología v4.0) para verificar que el Análisis Exploratorio de Datos (EDA), la caracterización de varianzas/covarianzas, el diagnóstico de atípicos, la autocorrelación serial y la auditoría de multicolinealidad (VIF) sean exactos y **plenamente coherentes con los resultados ya consolidados en los Bloques 2, 3 y 4**, antes de presentar el avance al profesor guía Dr. José Delgado Vega.
* **Instrucción (I):** Auditar independientemente cada estadístico descriptivo y prueba de hipótesis contra `DATASET_AUDITADO_N24.csv`, detectar cualquier desfase transversal entre capítulos de la tesis, clasificar los hallazgos por severidad y entregar acciones concretas de pulido y defensa oral.

---

## 2. Dictamen Ejecutivo de Auditoría

> **Veredicto:** **BLOQUE 1 APROBADO (SOLIDEZ METODOLÓGICA ALTA) — CON 4 OBSERVACIONES DE CONSISTENCIA TRANSVERSAL PARA PULIDO ANTES DE LA MEMORIA FINAL.**

El núcleo científico y metodológico del **Bloque 1** es sólido, defendible y no contiene errores que invaliden el modelado de los Bloques 2 al 4. Todas las decisiones estructurales que alimentan el resto de la tesis quedan matemáticamente blindadas:

1. **Escalado obligatorio:** Contraste de varianzas exacto ($s^2_{\text{CuT}} / s^2_{\text{CuS}} = 61.64\times$) y verificación complementaria con `RobustScaler`.
2. **Espacio de entrada 2D:** Auditoría VIF depurada ($\text{VIF} = 1.8045 < 5.0$ en $[\text{CuT}, \text{CuS}]$ vs. $\text{VIF} > 4{,}000$ al incluir $\text{CuIn}$).
3. **Retención de los $24$ registros:** Detección de los dos pulsos reales de alta ley (octubre y noviembre de 2023) y análisis de sensibilidad $N = 22$ con valores exactos ($r = +0.3688, p = 0.0912$ en `Cancha_CuS`).
4. **Rigor econométrico y geoestadístico:** Formulación correcta del Teorema de Gauss-Markov y justificación de LOOCV (ruido blanco en el sesgo, $r_1 = -0.0866$) más *Buffered-LOOCV* (inercia de `Cancha_CuT`, $r_1 = +0.4467$).

No obstante, al cruzar el Bloque 1 con el dataset exacto a $2$ decimales del Bloque 0 y con los resultados finales ya certificados en los Bloques 3 y 4, se detectaron **4 hallazgos no bloqueantes (`B1-01` a `B1-04`)** —dos de severidad media y dos de severidad baja— cuya corrección evitará contradicciones entre capítulos frente a la comisión evaluadora.

---

## 3. Verificación Independiente de Cálculos ($N = 24$)

| Elemento Auditado | Resultado Reportado en Bloque 1 | Verificación Independiente (Dataset 2 dec.) | Estado |
| :--- | :--- | :--- | :---: |
| **Descriptivos de leyes (`CuT`, `CuS`, `CuIn`, `PSJ`, `Canje`)** | $\mu_{\text{CuT}} = 1.7717$, $s^2 = 0.058128$; $\mu_{\text{CuS}} = 0.1471$, $s^2 = 0.000943$; $\mu_{\text{Canje}} = 1.4313$, $s^2 = 0.062020$. | Coincidencia exacta hasta el sexto decimal; ratio de varianza $61.64\times$ verificado. | **Exacto** |
| **Matriz de covarianza $5 \times 5$ (Sec. 3.1)** | $\text{Var}(\text{Sesgo}) = 0.019917$, $\text{Cov}(\text{CuT}, \text{Canje}) = 0.050115$. | Exacto; verifica la identidad $\text{Var}(\text{CuT}) + \text{Var}(\text{Canje}) - 2\,\text{Cov} = 0.019918$. | **Exacto** |
| **Correlaciones $N = 24$ y sensibilidad $N = 22$ (Sec. 3.2)** | `CuT` ($N=22$): $r = +0.6181$ ($p=0.0022$), $\rho = +0.6253$ ($p=0.0019$); `CuS` ($N=22$): $r = +0.3688$ ($p=0.0912$), $\rho = +0.3520$ ($p=0.1082$). | Coincidencia exacta en correlaciones, $p$-valores e Información Mutua ($\text{MI}$). | **Exacto** |
| **Auditoría VIF con constante (Sec. 7)** | $[\text{CuT}, \text{CuS}]$: $\text{VIF} = 1.8045$; con $\text{CuIn}$: $> 4{,}000$ (sin fila constante). | Exacto ($\frac{1}{1 - 0.6677^2} = 1.8045$). | **Exacto** |
| **ACF rezagos 1 a 6 (Sec. 6)** | `Sesgo_Canje`: $r_1 = -0.0866$, todos los rezagos dentro de $\pm 0.4001$; `Cancha_CuT`: $r_1 = +0.4467$. | Exacto (nótese que `Canje_CuT` en Lag 6 es $-0.4276$, fuera de banda por cambio de régimen 2023–2024). | **Exacto** |
| **Shapiro-Wilk de `Sesgo_Canje` (Tabla 2.1 y Sec. 4.1)** | $W = 0.9782$, $p = 0.8491$. | **Desfase con Bloque 4:** En `DATASET_AUDITADO_N24.csv` (2 dec.) el valor exacto es **$W = 0.9852$, $p = 0.9695$** (el reportado en Bloque 4). | **MEDIO (`B1-01`)** |
| **Descriptivos de `Ratio Solubilidad` y Mahalanobis `2023-01`** | Ratio: $\mu = 8.39\%$, $\tilde{x} = 8.04\%$, $s = 1.45\%$; `2023-01`: $D_M^2 = 3.5123$. | **Residuo pre-homologación:** Sobre los datos a 2 dec., Ratio da $\mu = 8.30\%$, $\tilde{x} = 8.33\%$, $s = 1.16\%$ y `2023-01` da $D_M^2 = 2.7900$. | **BAJO (`B1-03`)** |

---

## 4. Matriz de Hallazgos y Acciones Correctivas

| ID | Severidad | Sección Auditada | Hallazgo Detectado | Acción Correctiva Concreta |
| :---: | :---: | :--- | :--- | :--- |
| **B1-01** | **MEDIO** *(Consistencia entre capítulos)* | Sec. 2.1 (Tabla 2.1) y Sec. 4.1 | **Contradicción de Shapiro-Wilk entre Bloque 1 y Bloque 4 para el mismo vector:** El Bloque 1 reporta $W = 0.9782$ ($p = 0.8491$) para `Sesgo_Canje`, mientras el Bloque 4 (Tabla 3.1, `Cancha_Directa`) reporta **$W = 0.9852$ ($p = 0.9695$)** para exactamente el mismo vector con signo invertido. | Recalcular Shapiro-Wilk de `Sesgo_Canje` directamente desde `DATASET_AUDITADO_N24.csv` (2 decimales) y unificar ambos capítulos en **$W = 0.9852$, $p = 0.9695$**. |
| **B1-02** | **MEDIO** *(Coherencia narrativa con Bloques 3–4)* | Sec. 1.2 y Sec. 2.3 | **Afirmación aseverativa desalineada con los hallazgos reales de Bloques 3–4 y referencia adelantada al Bloque 2:** Se afirma que la varianza residual *"permite a la red neuronal parsimoniosa superar la deducción fija y alcanzar el umbral del Dr. Delgado ($\text{NRMSE} \approx 8.63\%$, $R^2 > 0.74$)"*, y se cita un resultado del Bloque 2 (`Baseline 1B` LOOCV $= 10.07\%$). | Reformular el umbral del $8.63\%$ en el Bloque 1 estrictamente como **hipótesis de trabajo a someter a prueba** (ya que en los Bloques 3–4 se demuestra que $8.63\%$ era *in-sample*, el mínimo LOOCV es $10.07\%$ y la RNA bate $8.63\%$ en *Holdout* 19/5 con $6.56\%$), y retirar la cita adelantada de resultados del Bloque 2. |
| **B1-03** | **BAJO** *(Aritmético / Residuos pre-homologación)* | Sec. 2.1, Sec. 2.3 y Sec. 5 | **Tres cifras heredadas de planilla sin redondear o redondeo intermedio:** (a) `Ratio Solubilidad` reporta $\mu = 8.39\%$, $\tilde{x} = 8.04\%$, $s = 1.45\%$ (exacto sobre 2 dec.: $8.30\%$, $8.33\%$, $1.16\%$); (b) `2023-01` en Mahalanobis reporta $D_M^2 = 3.5123$ (exacto: $2.7900$, siendo `2023-09` el 3° mayor con $D_M^2 = 5.2671 < 5.9915$); (c) Descomposición $\text{MSE}$ reporta $85.83\% / 14.17\%$ (exacto sobre $0.134971$: **$85.86\% / 14.14\%$**). | Reejecutar el cálculo de `Ratio_Solubilidad`, Mahalanobis y porcentajes de descomposición sobre las columnas homologadas a $2$ decimales para que $100\%$ de la tabla cuadre con `DATASET_AUDITADO_N24.csv`. |
| **B1-04** | **BAJO** *(Defensa oral — ACF Lag 6)* | Sec. 6 (Tabla ACF) | **Significancia en Lag 6 de `Canje_CuT` ($r_6 = -0.4276$):** Aunque el sesgo es ruido blanco en todos los rezagos, `Canje_CuT` excede levemente la banda inferior de Bartlett ($\pm 0.4001$) en el rezago 6 sin comentario aclaratorio. | Añadir una nota breve indicando que $r_6 = -0.4276$ en `Canje_CuT` responde al cambio de campañas mineras entre el pulso rico de fines de 2023 y los frentes de ley media de mediados de 2024, sin afectar la independencia del sesgo ($r_6 = -0.1036$). |

---

## 5. Análisis Profundo de los Hallazgos Clave

### 5.1. `[B1-01]` Unificación de Shapiro-Wilk entre Bloque 1 y Bloque 4
El estadístico de Shapiro-Wilk ($W$) es matemáticamente invariante ante cambios de signo ($e$ vs. $-e$) y desplazamientos por constantes. Por consiguiente, el vector `Sesgo_Canje` ($= \text{Cancha\_CuT} - \text{Canje\_CuT}$) del Bloque 1 y el vector de residuos de `Cancha_Directa` ($= \text{Canje\_CuT} - \text{Cancha\_CuT}$) del Bloque 4 **deben arrojar exactamente el mismo par $(W, p)$**.

* En el Bloque 4 se verificó sobre el CSV oficial a $2$ decimales: **$W = 0.9852$, $p = 0.9695$**.
* En el Bloque 1 quedó heredado el valor previo a la homologación de decimales: $W = 0.9782$, $p = 0.8491$.
* Aunque en ambos casos la conclusión es idéntica (no se rechaza normalidad, $p \gg 0.05$), unificar la cifra en **$W = 0.9852$, $p = 0.9695$** blinda la consistencia interna de la tesis.

### 5.2. `[B1-02]` Alineación Narrativa con los Descubrimientos de los Bloques 3 y 4
En la Sección 2.3, el Bloque 1 aún redacta el umbral histórico del profesor guía ($8.63\%$) como un resultado que la red alcanzará en LOOCV (*"permite a la red neuronal parsimoniosa superar la deducción fija y alcanzar el umbral... $\text{NRMSE} \approx 8.63\%$, $R^2 > 0.74$"*).

Ahora que ya conocemos los resultados auditados de los Bloques 3 y 4, la narrativa coherente de principio a fin de la memoria debe fluir así:
1. **Bloque 1 (Hipótesis):** Se cuantifica el piso de corrección constante ($\text{NRMSE} = 9.66\%$ *in-sample*) y se formula la **hipótesis de investigación** de verificar si una red neuronal no lineal logra extraer señal geometalúrgica de ese $14.14\%$ de varianza residual para aproximarse al referente inicial de $8.63\%$.
2. **Bloques 2–4 (Contraste empírico):** Se demuestra con rigor científico que: (i) el $8.63\%$ original era una métrica *in-sample* de una arquitectura colapsada linealmente, (ii) bajo evaluación estrictamente ciega LOOCV ($N = 24$), el piso de mínima varianza se sitúa en $10.07\%$ (`Baseline 1B / Res_MLP_2_2`), y (iii) en régimen productivo regular (*Holdout* 19/5), `RNA_MLP_2` sí perfora el $8.63\%$ alcanzando **$6.56\%$**.

Presentar el Bloque 1 como **hipótesis abierta** (sin prometer en el Capítulo 3 algo que luego matizas en el Capítulo 4) le da a tu tesis un nivel académico sobresaliente.

---

## 6. Puntos Fuertes del Bloque 1 para Presentar al Profesor Guía

Al exponer el Bloque 1 ante el Dr. José Delgado Vega, destaca estos cuatro argumentos irrefutables:

1. **Por qué es obligatorio escalar (`StandardScaler`) y contrastar `RobustScaler`:** `Cancha_CuT` tiene $61.64$ veces más varianza que `Cancha_CuS`, pero en términos relativos `Cancha_CuS` es más volátil ($\text{CV} = 20.87\%$ vs. $13.61\%$) y presenta alta curtosis ($+5.862$) por el pulso oxidado de octubre de 2023.
2. **Por qué NO se eliminan octubre y noviembre de 2023:** Aunque la distancia de Mahalanobis los identifica como atípicos ($D_M^2 = 14.25$ y $8.47 > 5.99$), corresponden a despachos reales facturados por ENAMI. El análisis de sensibilidad $N = 22$ se realiza **solo con fines diagnósticos** (sin borrar datos) y revela que Pearson cae de $+0.8347$ a $+0.6181$ por apalancamiento (*leverage*), justificando regularización $L_2$ y pruebas de ablación sobre `Cancha_CuS`.
3. **Por qué se usan exactamente $2$ entradas (`CuT`, `CuS`) y se descarta `CuIn`:** El espacio $[\text{CuT}, \text{CuS}]$ presenta $\text{VIF} = 1.8045 < 5.0$. Añadir `CuIn` dispara el $\text{VIF} > 4{,}000$ (y reemplazar `CuS` por `CuIn` junto a `CuT` produce $\text{VIF} = 86.85$ por $r = 0.994$).
4. **Por qué es válido evaluar con LOOCV:** El sesgo de reconciliación es ruido blanco puro ($r_1 = -0.0866$, dentro de $\pm 0.4001$), descartando deriva temporal del error; y la inercia geológica de `Cancha_CuT` ($r_1 = +0.4467$) se audita mediante *Buffered-LOOCV* en los Bloques 2 y 3.

---

## 7. Instrucciones Directas de Pulido (Para Sincronización Final)

Cuando se consolide el texto definitivo para `INFORME_TESIS.docx` (o si deseas dejar `BLOQUE_1` sincronizado al $100\%$ con `Antigravity`):

1. **Unificar `Shapiro-Wilk` de `Sesgo_Canje`** a **$W = 0.9852$, $p = 0.9695$** (coincidente con Bloque 4).
2. **Recalcular sobre `DATASET_AUDITADO_N24.csv` (2 dec.)** la fila `Ratio Solubilidad` ($\mu = 8.30\%$, $\tilde{x} = 8.33\%$, $s = 1.16\%$), la fila `2023-01` de Mahalanobis ($D_M^2 = 2.7900$, añadiendo mención a `2023-09` con $D_M^2 = 5.2671 < 5.9915$) y los porcentajes exactos de descomposición $\text{MSE}$ ($85.86\%$ y $14.14\%$).
3. **Reformular la Sección 2.3 como hipótesis de trabajo:** Reemplazar la afirmación predictiva sobre alcanzar $8.63\%$ en LOOCV por el planteamiento de hipótesis a evaluar en los Bloques 2–4, y retirar la cifra adelantada del Bloque 2 (`10.07%`) para respetar estrictamente el orden cronológico de la Metodología v4.0.
4. **Clarificar en la Tabla 3.2** el encabezado *"Sensibilidad sin Outliers ($N=22$, solo diagnóstico)"* para que nadie interprete erróneamente que se eliminaron meses del entrenamiento.