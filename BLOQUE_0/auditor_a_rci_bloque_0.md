# AUDITORÍA EXPERTO-CRÍTICA (MODELO RCI) — BLOQUE 0
## Verificación de Datos Maestros, Balance Metalúrgico, Reconciliación Histórica y Tarifas ENAMI

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas  
**Documento Auditado:** `BLOQUE_0/INFORME_ENTREGABLE_BLOQUE_0.md` (Versión 4.0)  
**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)  
**Institución:** Universidad de Antofagasta — Facultad de Ingeniería, Departamento de Ingeniería en Minas  
**Destinatario Técnico:** Entorno de desarrollo y ejecución (`Antigravity`)  

---

## 1. Marco de Auditoría RCI (Rol, Contexto, Instrucción)

* **Rol (R):** Auditor Técnico Senior en Ingeniería de Datos Mineros (ETL), Valorización Comercial ENAMI y Geoestadística aplicada a conciliación Mina–Planta.
* **Contexto (C):** Revisión forense bloque por bloque (`INFORME_ENTREGABLE_BLOQUE_0.md`, Metodología v4.0) para certificar que las bases técnicas, aritméticas y comerciales del **Bloque 0** estén completamente sólidas antes de consolidar los Bloques 1 al 4 y presentar el avance al profesor guía Dr. José Delgado Vega.
* **Instrucción (I):** Verificar de manera independiente cada cifra del dataset de faena ($N = 24$), el balance metalúrgico, la reconciliación de la brecha operacional histórica ($\text{MECM} \approx 26\%$) y la tabla tarifaria ENAMI 2023–2024, clasificando los hallazgos por severidad y entregando acciones correctivas concretas.

---

## 2. Dictamen Ejecutivo de Auditoría

> **Veredicto:** **BASE DE LEYES Y TONELAJES APROBADA AL 100% PARA BLOQUES 1–4 / HALLAZGO CRÍTICO DETECTADO EN TABLA TARIFARIA ENAMI (SECCIÓN 5.2).**

1. **Validez íntegra para los Bloques 1 al 4:** La tabla maestra de $N = 24$ balances mensuales (`DATASET_AUDITADO_N24`), la ficha del entorno computacional, la regla de exclusión de tonelajes en el modelado predictivo y la reconciliación de la brecha operacional ($+0.3404\%\text{Cu}$, $\text{NRMSE} = 25.67\%$) son **100% exactos y auditables**. Ningún cálculo de EDA ni de Machine Learning de los Bloques 1 al 4 se ve afectado.
2. **Hallazgo crítico en la Tabla 5.2 (Tarifas ENAMI):** Se detectó una **mezcla silenciosa de monedas** en la Sección 5.2: **$22$ de los $24$ meses están expresados en miles de pesos chilenos (`MCLP/tms`, por lectura errónea del punto separador de miles en los PDFs)** y etiquetados como `US$/tms`, mientras que **solo $2$ meses (`2023-04` y `2024-05`) están realmente en `US$/tms`**. Corregir este punto antes de presentar la Sección 5 al profesor guía (y antes de ejecutar el Bloque 5) evita un error financiero grave.

---

## 3. Verificación Independiente de Cálculos

| Elemento Auditado | Valor Reportado en Bloque 0 | Verificación Independiente | Estado |
| :--- | :--- | :--- | :---: |
| **Integridad del dataset ($N = 24$)** | $0$ nulos, $0$ duplicados, leyes a $2$ decimales en $\%\text{Cu}$. | Verificado en las $24$ filas de la Tabla 3.1. | **Exacto** |
| **Tonelajes acumulados** | `Cancha_Ton` $= 286{,}704\text{ t}$; `Romana_Ton` $= 295{,}884.67\text{ t}$ (media $12{,}328.53\text{ t/mes}$). | Sumatoria exacta (`diferencia = 0.00 t`). | **Exacto** |
| **Discrepancia Cancha–Canje** | Sesgo medio $+0.3404\%\text{Cu}$, positivo en $24/24$ meses (máx. $+0.61\%\text{Cu}$ en `2023-01`). | Exacto fila por fila (`diferencia máx. = 0.0000`, mín. $+0.06\%\text{Cu}$ en `2024-06`). | **Exacto** |
| **Reconciliación "MECM = 26%"** | Canje: $\text{NRMSE} = 25.67\%$, $\text{MAPE} = 24.90\%$; PSJ: $\text{NRMSE} = 25.98\%$, $\text{MAPE} = 25.05\%$. | Exacto ($\text{RMSE}_{\text{Canje}} = 0.3674\%\text{Cu}$; $\text{RMSE}_{\text{PSJ}} = 0.3717\%\text{Cu}$). | **Exacto** |
| **Balance químico $\text{CuT} = \text{CuS} + \text{CuIn}$** | Máx. brecha $0.01\%\text{Cu}$ en Cancha y $0.20\%\text{Cu}$ en Canje. | Valores numéricos exactos, pero **explicación de "redondeo" incorrecta en Canje 2024** *(ver B0-02)*. | **Revisar texto** |
| **Tabla 5.2: Tarifas ENAMI ($24$ meses)** | Etiquetada íntegramente en $\text{US\$/tms}$. | **Mezcla $\text{MCLP/tms}$ ($22$ meses) con $\text{US\$/tms}$ ($2$ meses: `2023-04` y `2024-05`)** *(ver B0-01)*. | **CRÍTICO** |

---

## 4. Matriz de Hallazgos y Acciones Correctivas

| ID | Severidad | Sección Auditada | Hallazgo Detectado | Acción Correctiva Concreta |
| :--- | :---: | :--- | :--- | :--- |
| **B0-01** | **CRÍTICO** *(Comercial / ETL)* | Sec. 5.2 (`TARIFAS_ENAMI`) | **Mezcla de monedas `MCLP/tms` vs. `US$/tms`:** En $22$ meses el extractor `pypdf` leyó el punto separador de miles de las tarifas en pesos chilenos como decimal y las etiquetó como $\text{US\$/tms}$, mientras `2023-04` y `2024-05` están en dólares reales. | Homologar los $24$ meses a **$\text{US\$/tms}$** en `00_auditoria_datos.ipynb` convirtiendo los $22$ meses en CLP mediante $\text{Valor} \times 1{,}000 / \text{TC}$ y verificando visualmente `ABRIL 2023.pdf` y `MAYO 2024.pdf` antes del Bloque 5. |
| **B0-02** | **MEDIO** *(Metalúrgico / Editorial)* | Sec. 3.2 (Calidad de Datos) | **Explicación imprecisa del balance químico en `Canje` 2024:** Se atribuye la brecha máxima de $0.20\%\text{Cu}$ a *"redondeo analítico a dos decimales"*. Un redondeo solo explica $\pm 0.01\%\text{Cu}$; en 2024 `Canje_CuS + Canje_CuIn` supera sistemáticamente a `Canje_CuT` entre $0.06$ y $0.20\%\text{Cu}$. | Precisar que la brecha en 2024 corresponde a una **inconsistencia de origen en las columnas auxiliares de canje (`Canje_CuS`, `Canje_CuIn`)** del Excel maestro (no a redondeo), ratificando que el modelo usa únicamente el ensayo oficial `Canje_CuT`. |
| **B0-03** | **BAJO** *(Comercial)* | Sec. 5.1 (Marco ENAMI) | **Falta de vínculo explícito entre `Cargos Reglamentarios` y `Fondo Estab.`:** La ecuación de tarifa unitaria menciona `Cargos Reglamentarios` sin precisar que corresponden al cargo de estabilización listado en la Tabla 5.2. | Añadir una nota aclaratoria indicando que `Cargos Reglamentarios` integra el descuento del Fondo de Estabilización cuando el precio LME supera el umbral de sustentación. |

---

## 5. Análisis Profundo del Hallazgo Crítico `[B0-01]`

Por definición comercial de ENAMI para minerales de flotación (sulfuros) con ley base de $2.5\%\text{Cu}$, la maquila base de tratamiento implícita cumple la identidad:
$$\text{Maquila Implícita} = 2.5 \times \text{Escala}_{1\%} - \text{Tarifa Base}_{2.5\%}$$

Al evaluar esta identidad sobre la Tabla 5.2, emergen las tres maquilas oficiales en dólares vigentes durante el bienio 2023–2024 (**$34.66\text{ US\$/tms}$** en enero 2023, **$35.65\text{ US\$/tms}$** entre febrero 2023 y enero 2024, y **$39.60\text{ US\$/tms}$** entre febrero y diciembre 2024):

| Periodo | TC ($\text{\$/US\$}$) | Flot. Base Reportada | Escala Reportada | Maquila Directa ($2.5\cdot\text{Esc} - \text{Base}$) | Maquila Convertida de CLP a USD ($\times 1{,}000 / \text{TC}$) | Moneda Real del Registro |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **2023-03** | $802.93$ | $115.545$ | $57.668$ | $28.625$ | **$35.651\text{ US\$/tms}$** | **Pesos chilenos** (`$115.545 CLP/tms`) |
| **2023-04** | $789.32$ | **$141.290$** | **$70.776$** | **$35.650\text{ US\$/tms}$** | $45.165$ | **Dólares** (`141.290 US$/tms`) |
| **2023-05** | $801.61$ | $112.879$ | $56.582$ | $28.576$ | **$35.648\text{ US\$/tms}$** | **Pesos chilenos** (`$112.879 CLP/tms`) |
| **2024-04** | $982.38$ | $132.779$ | $68.673$ | $38.904$ | **$39.601\text{ US\$/tms}$** | **Pesos chilenos** (`$132.779 CLP/tms`) |
| **2024-05** | $954.36$ | **$152.814$** | **$76.966$** | **$39.601\text{ US\$/tms}$** | $41.495$ | **Dólares** (`152.814 US$/tms`) |
| **2024-06** | $917.98$ | $153.269$ | $75.848$ | $36.351$ | **$39.599\text{ US\$/tms}$** | **Pesos chilenos** (`$153.269 CLP/tms`) |

Este diagnóstico explica por qué `2023-04` aparentaba un salto artificial de $+25\%$ respecto a marzo y mayo de 2023 pese a tener prácticamente el mismo precio internacional del cobre LME ($\sim 400\text{ ¢US\$/lb}$).

---

## 6. Puntos Fuertes del Bloque 0 para Presentar al Profesor Guía

1. **Trazabilidad y reproducibilidad completas:** Ficha de hardware y software al estándar del Departamento de Ingeniería en Minas de la UA, cero nulos/duplicados y suma dinámica exacta de báscula puente ($295{,}884.67\text{ t}$, equivalente a $12{,}328.53\text{ t/mes}$).
2. **Reconciliación matemática de su cifra histórica ($26\%$):** Queda demostrado que el $26\%$ de su lámina inicial equivale al **NRMSE** ($25.98\%$ en PSJ y $25.67\%$ en Canje), con una sobreestimación sistemática de **$+0.3404\%\text{Cu}$** presente en los $24$ meses.
3. **Separación limpia de alcance:** Queda blindado que los tonelajes no ingresan a los modelos de ley (Bloques 1–4) y se reservan exclusivamente para la valorización comercial ENAMI (Bloque 5).

---

## 7. Instrucciones Directas para `Antigravity` (Parche Pre-Bloque 5)

Antes de iniciar la simulación económica del **Bloque 5** (sin necesidad de tocar `DATASET_AUDITADO_N24.csv` ni reejecutar los Bloques 1 al 4):

1. **Homologar `TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.csv/.xlsx` en `BLOQUE_0/00_auditoria_datos.ipynb`:**
   * Para todos los periodos excepto `2023-04` y `2024-05`, convertir `Flot_Base_USD_tms`, `Flot_Escala_USD_tms_1pct`, `Flot_Fondo_Estab` y `Subprod_Cu_Insol` desde miles de CLP a USD exactos mediante:
     $$\text{Valor}_{\text{USD}} = \frac{\text{Valor}_{\text{MCLP}} \times 1{,}000}{\text{Tipo\_Cambio\_CLP\_USD}}$$
   * Verificar visualmente las circulares `ABRIL 2023.pdf` y `MAYO 2024.pdf` para confirmar que sus valores ya están en dólares netos (`141.290` y `152.814 US$/tms`).
   * Regenerar `fig_evolucion_tarifas_enami.png` y actualizar la Tabla 5.2 en `BLOQUE_0/INFORME_ENTREGABLE_BLOQUE_0.md`.
2. **Ajuste editorial en `BLOQUE_0/INFORME_ENTREGABLE_BLOQUE_0.md`:**
   * Corregir en la Sección 3.2 la explicación de la brecha química de `Canje` en 2024 (`0.20%Cu` por inconsistencia de registro en columnas auxiliares del Excel original, no por redondeo).
   * Explicitar en la Sección 5.1 que `Cargos Reglamentarios` incluye el `Fondo Estab.` de la Tabla 5.2.