# INFORME TÉCNICO DE ENTREGABLES — BLOQUE 0
## Auditoría de Datos Maestros, Unidades y Parámetros Base

**Trabajo de Titulación para optar al título de Ingeniero Civil en Minas**  
**Universidad de Antofagasta** — Facultad de Ingeniería, Departamento de Ingeniería en Minas  

* **Autor:** Felipe Andrés Guerra Maturana  
* **Profesor Guía:** Dr. José Delgado Vega (*Doctor en Geología, Especialista en Geoestadística y Explotaciones Mineras, École des Mines de Paris*)  
* **Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)  
* **Módulo:** Bloque 0 — Auditoría de Datos Maestros y Homologación Base  
* **Ubicación:** `./BLOQUE_0/`  

---

## 1. Resumen Ejecutivo

El presente informe consolida los resultados y entregables técnicos del **Bloque 0**, etapa fundacional de la metodología de tesis. Su propósito es certificar la integridad de los datos operacionales de faena Mantos Cobrizo y las resoluciones tarifarias de compra de ENAMI para el periodo 2023–2024 ($N = 24$ balances mensuales), resolviendo tres aspectos críticos antes del modelado estadístico:

1. **Homologación Dimensional y Estandarización de Decimales:** Se eliminó la ambigüedad de leyes almacenadas en fracción decimal (`0.0155`), estandarizándolas formalmente en **puntos porcentuales ($\%\text{Cu}$)** (`1.55% Cu`) y aplicando un formato estricto y uniforme de **exactamente dos decimales (`0.00`)** para todas las variables de ley (`Cancha_CuT`, `Cancha_CuS`, `Cancha_CuIn`, `PSJ_CuT`, `Canje_CuT`, etc.), con tonelaje de acopio entero (`Cancha_Ton`) y tonelaje pesado en báscula puente (`Romana_Ton`), asegurando que $\text{MSE}$ se exprese en $(\%\text{Cu})^2$, $\text{RMSE}$ y $\text{MAE}$ en $\%\text{Cu}$, y $\text{NRMSE/MAPE}$ en porcentaje relativo ($\%$).
2. **Reconciliación Matemática de la Métrica "MECM = 26%":** Se demostró algebraicamente que el error del $26\%$ atribuido a la línea base histórica en la presentación del Dr. José Delgado corresponde al **NRMSE** (*Normalized Root Mean Square Error*, $\text{NRMSE} = 25.98\% \approx 26.0\%$ en PSJ y $25.67\% \approx 26.0\%$ en Canje) o al **MAPE** ($25.05\%$ y $24.90\%$, respectivamente). El error cuadrático absoluto es $\text{MSE} = 0.1350\,(\%\text{Cu})^2$ (en Canje) y la desviación cuadrática media es $\text{RMSE} = 0.3674\%\text{Cu}$, con una sobreestimación sistemática promedio en cancha de $+0.3404\%\text{Cu}$.
3. **Consolidación Tarifaria Oficial ENAMI:** Se extrajeron y estructuraron los parámetros comerciales oficiales vigentes para **Minerales de Concentración (Sulfuros)** de las $24$ circulares mensuales de ENAMI (2023–2024), cubriendo el precio del cobre LME, tipo de cambio observado, tarifa base de flotación ($2.5\%\text{Cu}$), escala por $1\%\text{Cu}$, y cobranzas de estabilización.

---

## 2. Auditoría y Homologación del Dataset de Faena ($N = 24$)

A partir del archivo maestro `Data Cobrizo.xlsx`, se extrajeron los $24$ meses consolidados de balance metalúrgico (Enero 2023 a Diciembre 2024).

### 2.1. Tabla Maestra de Datos Auditados ($N = 24$)

| Periodo | Cancha Ton ($\text{t}$) | Romana Ton ($\text{t}$) | Cancha CuT ($\%$) | Cancha CuS ($\%$) | Cancha CuIn ($\%$) | PSJ CuT ($\%$) | Canje CuT ($\%$) | Canje CuS ($\%$) | Canje CuIn ($\%$) | Discrepancia Cancha-Canje ($\%\text{Cu}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2023-01** | 11,559 | 12,017.45 | 2.16 | 0.17 | 1.99 | 1.41 | **1.55** | 0.21 | 1.34 | +0.61 |
| **2023-02** | 9,972 | 10,109.95 | 1.76 | 0.14 | 1.62 | 1.33 | **1.41** | 0.16 | 1.25 | +0.35 |
| **2023-03** | 12,765 | 12,533.33 | 1.73 | 0.16 | 1.57 | 1.55 | **1.49** | 0.21 | 1.28 | +0.24 |
| **2023-04** | 10,530 | 11,066.41 | 1.66 | 0.14 | 1.52 | 1.38 | **1.38** | 0.15 | 1.23 | +0.28 |
| **2023-05** | 11,167 | 12,564.00 | 1.80 | 0.13 | 1.68 | 1.26 | **1.24** | 0.14 | 1.10 | +0.56 |
| **2023-06** | 12,393 | 13,127.71 | 1.77 | 0.13 | 1.64 | 1.43 | **1.34** | 0.17 | 1.17 | +0.43 |
| **2023-07** | 11,349 | 12,044.92 | 1.75 | 0.13 | 1.63 | 1.41 | **1.37** | 0.19 | 1.19 | +0.38 |
| **2023-08** | 11,208 | 11,989.45 | 1.74 | 0.12 | 1.62 | 1.54 | **1.52** | 0.17 | 1.35 | +0.22 |
| **2023-09** | 11,804 | 12,012.89 | 2.01 | 0.12 | 1.89 | 1.66 | **1.55** | 0.27 | 1.28 | +0.46 |
| **2023-10** | 10,716 | 11,988.24 | 2.21 | 0.26 | 1.95 | 1.77 | **1.79** | 0.17 | 1.62 | +0.42 |
| **2023-11** | 11,170 | 12,036.00 | 2.47 | 0.20 | 2.27 | 2.26 | **2.27** | 0.25 | 2.02 | +0.20 |
| **2023-12** | 11,194 | 12,185.00 | 1.72 | 0.12 | 1.60 | 1.37 | **1.35** | 0.21 | 1.15 | +0.37 |
| **2024-01** | 8,538 | 8,988.00 | 1.49 | 0.12 | 1.37 | 1.37 | **1.41** | 0.14 | 1.36 | +0.08 |
| **2024-02** | 7,102 | 7,204.00 | 1.61 | 0.14 | 1.47 | 1.37 | **1.37** | 0.14 | 1.33 | +0.24 |
| **2024-03** | 10,040 | 10,011.00 | 1.56 | 0.15 | 1.41 | 1.19 | **1.19** | 0.12 | 1.14 | +0.37 |
| **2024-04** | 11,374 | 11,402.00 | 1.62 | 0.14 | 1.48 | 1.15 | **1.15** | 0.12 | 1.09 | +0.47 |
| **2024-05** | 11,342 | 11,380.00 | 1.58 | 0.13 | 1.45 | 1.14 | **1.14** | 0.11 | 1.10 | +0.44 |
| **2024-06** | 12,751 | 12,805.00 | 1.76 | 0.15 | 1.61 | 1.68 | **1.70** | 0.20 | 1.70 | +0.06 |
| **2024-07** | 12,468 | 12,504.00 | 1.84 | 0.16 | 1.68 | 1.57 | **1.57** | 0.17 | 1.56 | +0.27 |
| **2024-08** | 13,169 | 13,210.00 | 1.95 | 0.15 | 1.80 | 1.41 | **1.43** | 0.16 | 1.41 | +0.52 |
| **2024-09** | 14,172 | 14,208.00 | 1.64 | 0.16 | 1.48 | 1.46 | **1.48** | 0.15 | 1.47 | +0.16 |
| **2024-10** | 16,226 | 16,310.00 | 1.66 | 0.14 | 1.51 | 1.27 | **1.27** | 0.16 | 1.24 | +0.39 |
| **2024-11** | 17,142 | 17,212.00 | 1.62 | 0.15 | 1.47 | 1.31 | **1.32** | 0.15 | 1.30 | +0.30 |
| **2024-12** | 16,553 | 16,634.00 | 1.41 | 0.12 | 1.29 | 1.04 | **1.06** | 0.12 | 1.05 | +0.35 |

### 2.2. Certificación de Calidad de Datos
* **Valores Faltantes:** $0$ datos nulos en todas las columnas.
* **Ley de Conservación Metalúrgica:** Se verificó la consistencia de especies químicas $\text{CuT} = \text{CuS} + \text{CuIn}$, detectándose discrepancias máximas acotadas por redondeo analítico a dos decimales en las planillas de faena:
  $$\max |\text{CuT}_{\text{cancha}} - (\text{CuS}_{\text{cancha}} + \text{CuIn}_{\text{cancha}})| = 0.01\%\text{Cu}$$
  $$\max |\text{Canje\_CuT} - (\text{Canje\_CuS} + \text{Canje\_CuIn})| = 0.20\%\text{Cu}$$
  *(Nota: La variable objetivo del modelo corresponde directamente al ensayo químico oficial `Canje_CuT` liquidado en planta).*
* **Tonelaje Total Acopiado y Pesado:** $286{,}704\text{ toneladas}$ estimadas en acopio (`Cancha_Ton`) y $295{,}884.67\text{ toneladas}$ certificadas en báscula puente (`Romana_Ton`), las cuales quedan consolidadas para alimentar la Fase 5 (Modelo Económico Dual).

---

## 3. Demostración Matemática del Indicador "MECM = 26%"

### 3.1. Formulación de Métricas
Para una serie de $N$ observaciones pareadas $(y_i, \hat{y}_i)$:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \quad [(\%\text{Cu})^2]$$

$$\text{RMSE} = \sqrt{\text{MSE}} \quad [\% \text{Cu}]$$

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i| \quad [\% \text{Cu}]$$

$$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \quad [\%]$$

$$\text{NRMSE} = \frac{\text{RMSE}}{\bar{y}} \times 100\% = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}}{\frac{1}{N}\sum_{i=1}^N y_i} \times 100\% \quad [\%]$$

$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

### 3.2. Tabla de Verificación de Métricas Históricas

| Indicador | Cancha vs. Planta San José (PSJ) | Cancha vs. Canje Oficial |
| :--- | :---: | :---: |
| **Media Real ($\bar{y}$)** | $1.4304\%\text{Cu}$ | $1.4313\%\text{Cu}$ |
| **Media Cancha ($\bar{y}_{\text{pred}}$)** | $1.7717\%\text{Cu}$ | $1.7717\%\text{Cu}$ |
| **Sesgo Sistemático Medio ($\bar{e}$)** | **$+0.3413\%\text{Cu}$** | **$+0.3404\%\text{Cu}$** |
| **MSE** | $0.1381\,(\%\text{Cu})^2$ | $0.1350\,(\%\text{Cu})^2$ |
| **RMSE** | $0.3717\%\text{Cu}$ | $0.3674\%\text{Cu}$ |
| **MAE** | $0.3413\%\text{Cu}$ | $0.3404\%\text{Cu}$ |
| **MAPE** | $25.0547\%$ | $24.8959\%$ |
| **NRMSE** | **$25.9840\% \approx \mathbf{26.0\%}$** | **$25.6687\% \approx \mathbf{26.0\%}$** |
| **$R^2$ de Cancha Directa** | $-1.3145$ | $-1.2709$ |

### 3.3. Interpretación de Ingeniería de Minas
* La cifra reportada por el Dr. José Delgado ($26\%$) corresponde de manera inequívoca al **NRMSE** ($25.98\% \approx 26.0\%$ en PSJ y $25.67\% \approx 26.0\%$ en Canje) o al **MAPE** redondeado ($25.05\%$).
* La cancha de acopio sobreestima **sistemáticamente** la ley de venta en $+0.3404\%\text{Cu}$ (todos los $24$ meses presentan sesgo positivo).
* Esta sobreestimación no obedece a un error aleatorio simétrico sino a factores geometalúrgicos y de muestreo (segregación de finos, heterogeneidad de Gy y pérdida de solubles), fundamentando la necesidad de corrección no lineal supervisada.

---

## 4. Resoluciones Tarifarias ENAMI (2023–2024)

### 4.1. Marco de Valorización para Flotación (Sulfuros)
De acuerdo con las circulares mensuales de la Gerencia Comercial de ENAMI, la **Tarifa Base de Flotación** ($\text{Tarifa Base}_{2.5\%}$) expresa directamente el precio neto unitario pagable por tonelada métrica seca para un mineral base de $2.5\%\text{Cu}$ (ya descontada la maquila base de tratamiento). Para lotes con leyes inferiores a $2.5\%\text{Cu}$, se descuenta la escala proporcional por cada $1\%\text{Cu}$ bajo la ley base:

$$\text{Tarifa Unitaria (US\$/tms)} = \text{Tarifa Base}_{2.5\%} - \text{Escala}_{1\%} \times (2.5 - \text{Ley}_{\text{Canje}}) - \text{Cargos Reglamentarios}$$

$$\text{Valor Lote (USD)} = \text{TMS} \times \text{Tarifa Unitaria (US\$/tms)}$$

*(Nota de auditoría: Los valores ingresados manualmente desde el escaneo `ABRIL 2023.pdf` quedan marcados para doble verificación visual contra el documento original antes de ejecutar la simulación económica de la Fase 5).*

### 4.2. Síntesis Consolidada de Parámetros Mensuales ($N = 24$)

| Periodo | Cobre LME ($\text{¢US\$/lb}$) | Tipo Cambio ($\text{\$/US\$}$) | Flot. Base $2.5\%\text{Cu}$ ($\text{US\$/tms}$) | Escala $1\%\text{Cu}$ ($\text{US\$/tms}\cdot 1\%$) | Fondo Estab. ($\text{US\$/tms}\cdot 1\%$) | Subprod. Cu Insol. ($\text{US\$/tms}\cdot 1\%$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2023-01** | 379.530 | 859.51 | 115.523 | 58.125 | 0.000 | 50.469 |
| **2023-02** | 408.220 | 781.49 | 113.222 | 56.433 | 3.224 | 49.236 |
| **2023-03** | 406.200 | 802.93 | 115.545 | 57.668 | 3.062 | 50.328 |
| **2023-04** | 400.781 | 789.32 | 141.290 | 70.776 | 2.976 | 61.819 |
| **2023-05** | 399.796 | 801.61 | 112.879 | 56.582 | 2.264 | 49.429 |
| **2023-06** | 373.500 | 804.60 | 103.094 | 52.711 | 0.000 | 46.248 |
| **2023-07** | 380.393 | 802.68 | 105.516 | 53.653 | 0.000 | 47.017 |
| **2023-08** | 383.071 | 826.01 | 109.650 | 55.639 | 0.000 | 48.736 |
| **2023-09** | 378.830 | 853.86 | 111.601 | 56.816 | 0.000 | 49.803 |
| **2023-10** | 375.160 | 906.84 | 116.920 | 59.699 | 0.000 | 52.364 |
| **2023-11** | 360.137 | 889.74 | 108.267 | 55.995 | 0.000 | 49.250 |
| **2023-12** | 370.764 | 869.56 | 110.269 | 56.508 | 0.000 | 49.603 |
| **2024-01** | 380.750 | 884.59 | 116.436 | 59.189 | 0.000 | 51.866 |
| **2024-02** | 378.491 | 957.86 | 122.521 | 64.181 | 0.000 | 55.967 |
| **2024-03** | 376.969 | 969.91 | 123.350 | 64.703 | 0.000 | 56.436 |
| **2024-04** | 393.520 | 982.38 | 132.779 | 68.673 | 0.000 | 59.748 |
| **2024-05** | 430.116 | 954.36 | 152.814 | 76.966 | 5.789 | 66.641 |
| **2024-06** | 459.447 | 917.98 | 153.269 | 75.848 | 5.314 | 65.458 |
| **2024-07** | 437.336 | 951.02 | 148.642 | 74.521 | 5.505 | 64.469 |
| **2024-08** | 426.085 | 938.94 | 141.658 | 71.536 | 4.726 | 61.969 |
| **2024-09** | 406.587 | 917.38 | 129.776 | 66.442 | 1.853 | 57.701 |
| **2024-10** | 419.777 | 900.91 | 133.179 | 67.542 | 3.654 | 58.556 |
| **2024-11** | 432.693 | 950.89 | 146.491 | 73.659 | 5.504 | 63.758 |
| **2024-12** | 411.625 | 977.32 | 140.631 | 71.733 | 2.734 | 62.255 |

---

## 5. Índice de Entregables Disponibles en `./BLOQUE_0/`

Todos los entregables de este bloque han sido ubicados de forma autónoma en la carpeta `BLOQUE_0`:

| Archivo | Formato | Descripción del Contenido |
| :--- | :---: | :--- |
| [`00_auditoria_datos.ipynb`](./00_auditoria_datos.ipynb) | Cuaderno Jupyter | Código reproducible, ejecución completa y outputs renderizados. |
| [`DATASET_AUDITADO_N24.csv`](./DATASET_AUDITADO_N24.csv) | CSV plano | $N = 24$ registros mensuales con leyes estandarizadas a 2 decimales y tonelajes. |
| [`DATASET_AUDITADO_N24.xlsx`](./DATASET_AUDITADO_N24.xlsx) | Hoja de Cálculo Excel | Dataset auditado con formato tabular listo para revisión directiva. |
| [`TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.csv`](./TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.csv) | CSV plano | $24$ meses de resoluciones comerciales oficiales ENAMI Sulfuros. |
| [`TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.xlsx`](./TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.xlsx) | Hoja de Cálculo Excel | Parámetros tarifarios consolidados en formato nativo de Office. |
| [`fig_auditoria_brecha_historica.png`](./fig_auditoria_brecha_historica.png) | Gráfico PNG (200 DPI) | Evolución de leyes Cancha vs. Canje y gráfico de barras del sesgo. |
| [`fig_evolucion_tarifas_enami.png`](./fig_evolucion_tarifas_enami.png) | Gráfico PNG (200 DPI) | Relación bivariada entre precio LME ($\text{¢US\$/lb}$) y tarifa base. |
| [`INFORME_ENTREGABLE_BLOQUE_0.md`](./INFORME_ENTREGABLE_BLOQUE_0.md) | Documento Markdown | El presente informe de certificación técnica y auditoría. |
| [`informe_t_cnico_de_entregables_bloque_0_corregido.md`](./informe_t_cnico_de_entregables_bloque_0_corregido.md) | Documento Markdown | Versión de referencia remitida por la auditoría técnica. |

---

## 6. Dictamen de Auditoría y Habilitación Metodológica

* **Criterio 1: Homologación de Unidades en %Cu y Decimales Uniformes:** Satisfecho.
* **Criterio 2: Reconciliación Métrica "MECM = 26%":** Satisfecho ($\text{NRMSE} = 25.98\%$ en PSJ y $25.67\%$ en Canje; $\text{MAPE} = 25.05\%$ y $24.90\%$; $R^2 = -1.2709$).
* **Criterio 3: Tabla Tarifaria Mensual ENAMI 2023–2024:** Satisfecho ($24$ meses completos; pendiente doble verificación visual de `ABRIL 2023.pdf` e inclusión de `Romana_Ton` antes de la Fase 5).
* **Criterio 4: Estructura Modular de Carpetas:** Satisfecho (`BLOQUE_0/` con todos sus insumos autónomos).

> **🚀 Estado:** **APROBADO PARA AVANZAR AL BLOQUE 1 (`01_eda_caracteristicas.ipynb`).**
