# Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar las Pérdidas Económicas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guerrafmaturana-arch/TESIS/blob/main/C%C3%B3digo/C%C3%B3digo%20unificado.ipynb)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn Version](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![SHAP Version](https://img.shields.io/badge/SHAP-XAI-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Metodolog%C3%ADa%20v4.0%20Auditada-green.svg)

**Memoria de Título para optar al Título de Ingeniero Civil en Minas**  
**Universidad de Antofagasta** — Facultad de Ingeniería, Departamento de Ingeniería en Minas  
* **Autor:** Felipe Andrés Guerra Maturana  
* **Profesor Guía:** Dr. José Delgado Vega (*Doctor en Geología, Especialista en Geoestadística y Explotaciones Minas, École des Mines de Paris*)  

---

## 📌 1. Planteamiento del Problema y Desafío Operativo

En el modelo de negocio de la pequeña y mediana minería en Chile (con faenas de referencia como **Mantos Cobrizo**), las operaciones no disponen típicamente de una planta de procesamiento o beneficio propia. La viabilidad económica depende de la comercialización y entrega de lotes de mineral a poderes de compra estatales o terceros (como la **Empresa Nacional de Minería - ENAMI** o plantas de maquila como la **Planta San José - PSJ**).

```
   FAENA MINERA (Origen)                      PODER DE COMPRA / PLANTA (Destino)
┌───────────────────────────────┐           ┌───────────────────────────────────┐
│ Acopio en Cancha              │  Camiones │ Recepción ENAMI / Planta San José │
│ • Muestreo interno            │ ────────> │ • Muestreo mecánico (Gy)          │
│ • Determinación de leyes:     │  (Flete)  │ • División en 3 sobres de custodia│
│   Cancha_CuT, CuS, CuIn       │           │ • Ensaye químico oficial          │
│ ➔ Planificación y Blending    │           │ ➔ LEY DE CANJE (Liquidación)      │
└───────────────────────────────┘           └───────────────────────────────────┘
               ▲                                              ▲
               │               DISCREPANCIA                   │
               └──────────────────────────────────────────────┘
                         • Error Cuadrático Medio (MSE)
                         • Castigos por impurezas / maquila
                         • Pérdidas económicas por desconciliación
```

### La Brecha de Conciliación
1. **Ley de Cancha (Origen):** La mina estima internamente la calidad del lote acopiado ($CuT$, $CuS$, $CuIn$). Con estos valores se planifica el despacho, el *blending* y las proyecciones de flujo de caja.
2. **Ley de Canje (Destino):** En la planta de destino se ejecuta el protocolo oficial de recepción, pesaje en romana, determinación de humedad y muestreo según la **Teoría de Muestreo de Pierre Gy**. Sobre esta "ley de canje" se liquidan las tarifas de tratamiento (maquila), deducciones metalúrgicas y castigos.
3. **El Problema:** Existe una discrepancia no lineal sistemática entre la ley estimada en cancha y la ley de canje liquidada en planta. Operar con estimaciones de cancha introduce un **Error Cuadrático Medio (MSE)** empírico histórico elevado (~26% en la línea base), provocando:
   - Pérdidas económicas por decisiones de despacho y corte erróneas.
   - Castigos imprevistos y cobros excesivos por maquila y flete.
   - Imposibilidad de los modelos geoestadísticos o lineales clásicos para capturar fenómenos complejos de segregación y balance metalúrgico.

---

## 🎯 2. Objetivos del Proyecto

* **Objetivo General:** Desarrollar y validar un modelo predictivo basado en Redes Neuronales Artificiales (Perceptrón Multicapa - MLP) para predecir con exactitud la ley de canje de venta a partir de las leyes medidas en cancha, minimizando el error empírico histórico y cuantificando el impacto económico en la faena.
* **Objetivos Específicos:**
  1. Cuantificar el error histórico empírico entre la ley de cancha y la ley de planta.
  2. Implementar y entrenar una arquitectura de Red Neuronal Artificial en Python / Google Colab evaluando las variables fundamentales de entrada ($CuT$, $CuS$, $CuIn$).
  3. Validar matemáticamente la reducción del Error Cuadrático Medio ($MSE$) y la ganancia en el coeficiente de determinación ($R^2$) frente a la línea base.
  4. Realizar un análisis de impacto económico y comercial demostrando el beneficio financiero del modelo predictivo en la toma de decisiones previas a la venta.

---

## 🔬 3. Fundamento Metalúrgico y Modelado Predictivo

### Variables Clave de Entrada
* **$CuT$ (Cobre Total):** Contenido global de cobre en el mineral.
* **$CuS$ (Cobre Soluble):** Fracción lixiviable en ácido sulfúrico (óxidos como crisocola, atacamita, malaquita).
* **$CuIn$ (Cobre Insoluble):** Fracción refractaria a lixiviación ácida ($CuIn = CuT - CuS$), correspondiente principalmente a sulfuros de cobre (calcopirita, bornita, calcosina). Presenta una altísima correlación ($r > 0.81$) con la ley de canje definitiva.

### Formulación Matemática

$$\text{MSE}_{\text{Histórico}} = \frac{1}{n} \sum_{i=1}^{n} \left(\text{Ley\_Cancha}_i - \text{Ley\_Planta}_i\right)^2$$

$$\text{MSE}_{\text{Predictivo}} = \frac{1}{n} \sum_{i=1}^{n} \left(\hat{y}_{\text{RNA}, i} - \text{Ley\_Planta}_i\right)^2$$

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

### Arquitectura de la Red Neuronal (MLP)
* **Entradas:** 2 neuronas ($CuT$ y $CuIn$, o $CuT$ y $CuS$).
* **Preprocesamiento:** Escalado Z-score (`StandardScaler`) ajustado sin fuga de datos (*data leakage*).
* **Capas Ocultas:** Capas densas con activación ReLU e inicialización `he_uniform`.
* **Regularización:** Penalización L2 (Ridge) en pesos para prevenir el sobreajuste (*overfitting*) en muestras de tamaño acotado ($N=24$ meses).
* **Optimizador:** Adam / RMSprop con validación cruzada interna (K-Fold Cross Validation).

---

## 📂 4. Estructura del Repositorio

```text
TESIS/
├── Código/                             # ENTREGABLE UNIFICADO Y DOCUMENTACIÓN DE DEFENSA
│   ├── Código unificado.ipynb          # Cuaderno interactivo Google Colab / Local (18 modelos, LOOCV y Buffered-LOOCV)
│   ├── traductor.md                    # Explicación pedagógica exhaustiva línea por línea del código
│   ├── glosario.md                     # Enciclopedia técnica conceptual (minería, econometría, ML, XAI)
│   └── informe entregable.md           # Informe de defensa oral ejecutiva ante comisión evaluadora
├── METODOLOGÍA/                        # Marco metodológico maestro v4.0 y contratos de bloques
│   ├── METODOLOGIA_TRABAJO_TESIS.md    # Metodología de Trabajo Integral de Tesis v4.0
│   └── Metodología.md                  # Versión ejecutiva consolidada
├── BLOQUE_0/                           # ETL, Auditoría de Fuentes ENAMI y Limpieza de Datos (N=24)
├── BLOQUE_1/                           # Análisis Exploratorio (EDA), Normalidad, Mahalanobis y VIF
├── BLOQUE_2/                           # Baselines Estadísticos (OLS, Ridge, Lasso, ElasticNet, SVR, Baseline 1B)
├── BLOQUE_3/                           # Modelado de Redes Neuronales Artificiales (MLP / ResNet) en LOOCV
├── BLOQUE_4/                           # Evaluación Rigurosa, Comparativa 18 Modelos, Ablación y SHAP
├── BLOQUE_5/                           # Modelo Económico Dual, Valorización Comercial ENAMI y VAN/TIR
├── BLOQUE_6/                           # Síntesis Global y Redacción Final de la Memoria de Título
├── DATA_COBRIZO/                       # Datos operacionales históricos de faena Mantos Cobrizo
├── MATERIAL_TESIS/                     # Circulares tarifarias ENAMI 2023-2024 y literatura de soporte
├── CÓDIGO_TESIS.ipynb                  # Cuaderno de referencia histórica inicial
└── README.md                           # Documentación principal del repositorio
```

---

## 📊 5. Descripción de los Datos (`DATA_COBRIZO/DATASET.csv`)

El archivo [`DATASET.csv`](file:///c:/Users/fguer/Desktop/TESIS/DATA_COBRIZO/DATASET.csv) contiene 24 registros mensuales reales correspondientes al período **enero 2023 - diciembre 2024** de Mantos Cobrizo, estructurado con las siguientes columnas:

| Columna | Tipo | Descripción | Unidad |
| :--- | :--- | :--- | :--- |
| **`Cancha_CuT`** | Decimal | Ley de Cobre Total medida en cancha de acopio | % Cu |
| **`Cancha_CuIn`**| Decimal | Ley de Cobre Insoluble medida en cancha ($CuT - CuS$) | % Cu |
| **`Canje_CuT`**  | Decimal | Ley oficial liquidada en la planta receptora (Target) | % Cu |

---

## 🚀 6. Instrucciones de Uso y Ejecución

### Opción 1: En Google Colab (Recomendada)
1. Abre el notebook directamente haciendo clic en el botón [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guerrafmaturana-arch/TESIS/blob/main/C%C3%B3digo/C%C3%B3digo%20unificado.ipynb).
2. El cuaderno cuenta con un **cargador universal automático**: buscará el dataset en las rutas locales o abrirá una ventana de diálogo interactiva para subir cualquier archivo `.csv` o `.xlsx` (por ejemplo, `BLOQUE_0/DATASET_AUDITADO_N24.csv` o `DATA_COBRIZO/DATASET.csv`).
3. Ejecuta las celdas secuencialmente para realizar la carga, el protocolo Leave-One-Out Cross-Validation (LOOCV), el benchmark de 18 modelos, la prueba de autocorrelación Buffered-LOOCV y la interpretabilidad SHAP.

### Opción 2: En Entorno Local
1. Clona el repositorio:
   ```bash
   git clone https://github.com/guerrafmaturana-arch/TESIS.git
   cd TESIS
   ```
2. Instala las dependencias requeridas:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn shap joblib openpyxl
   ```
3. Ejecuta el notebook [`Código/Código unificado.ipynb`](file:///c:/Users/fguer/Desktop/TESIS/Código/Código%20unificado.ipynb) en Jupyter Lab, VS Code o tu entorno preferido. Para el estudio conceptual y defensa teórica, consulta [`Código/traductor.md`](file:///c:/Users/fguer/Desktop/TESIS/Código/traductor.md), [`Código/glosario.md`](file:///c:/Users/fguer/Desktop/TESIS/Código/glosario.md) y [`Código/informe entregable.md`](file:///c:/Users/fguer/Desktop/TESIS/Código/informe%20entregable.md).

---

## 📜 7. Cita y Referencia Académica

```bibtex
@mastersthesis{guerra2026prediccion,
  author       = {Felipe Andr{\'e}s Guerra Maturana},
  title        = {Predicci{\'o}n de la ley mineral de venta mediante redes neuronales artificiales para minimizar las p{\'e}rdidas econ{\'o}micas},
  school       = {Universidad de Antofagasta, Facultad de Ingenier{\'i}a, Departamento de Ingenier{\'i}a en Minas},
  year         = {2026},
  note         = {Profesor Gu{\'i}a: Dr. Jos{\'e} Delgado Vega}
}
```
