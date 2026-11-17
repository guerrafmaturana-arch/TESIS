# Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar las Pérdidas Económicas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guerrafmaturana-arch/TESIS/blob/main/C%C3%93DIGO_TESIS.ipynb)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-En%20Desarrollo-green.svg)

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
├── ANÁLISIS_DEL_PROBLEMA/              # Gráficos y análisis exploratorio del error
│   ├── grafico_evaluacion_modelo.png   # Comparativa MSE modelo vs histórico
│   └── Gráfico de calor.png            # Mapa de calor de correlaciones
├── DATA_COBRIZO/                       # Datos operacionales de faena Mantos Cobrizo
│   ├── Data Cobrizo.xlsx               # Planilla maestra mensual (2023-2024)
│   ├── DATASET.csv                     # Dataset oficial (Cancha_CuT, Cancha_CuIn, Canje_CuT)
│   ├── DATASET2.csv                    # Dataset con tonelaje y todas las leyes de cobre
│   └── Presentación_cobrizo_050325.pptx# Presentación técnica del Dr. José Delgado
├── MATERIAL_TESIS/                     # Información regulatoria y antecedentes
│   ├── Precio mx Enami 2023-2024/      # Tarifas oficiales mensuales publicadas por ENAMI
│   ├── Data Tigresa.xlsx               # Datos de yacimiento complementario
│   ├── Presentación_cobrizo_050325.pptx
│   └── Presentación_tigresa_060325.pptx
├── REDES_NEURONALES/                   # Literatura científica base (LeCun, Glorot, Goodfellow)
│   ├── Efficient BackProp (LeCun)
│   ├── Deep sparse Rectifier neural networks (Glorot)
│   └── Neural Networks and Deep Learning
├── TESIS_COMPARTIDAS/                  # Tesis y memorias de referencia académica
│   ├── Memoria_Pedro_Manriquez.pdf
│   ├── Tesis Eduardo Carneiro.pdf
│   └── Charlotte Bazaes Jimenez.docx
├── Carta_Gantt.xlsx                    # Planificación temporal de actividades y entregas
├── CÓDIGO_TESIS.ipynb                  # Notebook principal para Google Colab
├── Código_V1_Colab.ipynb               # Primera iteración del flujo de entrenamiento
├── PRUEBA1.ipynb                       # Pruebas experimentales y exploración algorítmica
├── INFORME_TESIS.docx                  # Documento formal de la Memoria de Título
└── README.md                           # Documentación general del repositorio
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
1. Abre el notebook directamente haciendo clic en el botón [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guerrafmaturana-arch/TESIS/blob/main/C%C3%93DIGO_TESIS.ipynb).
2. Sube el archivo `DATASET.csv` (ubicado en `DATA_COBRIZO/DATASET.csv`) al entorno de ejecución de Colab.
3. Ejecuta las celdas secuencialmente para realizar la carga, escalado, entrenamiento y evaluación de métricas.

### Opción 2: En Entorno Local
1. Clona el repositorio:
   ```bash
   git clone https://github.com/guerrafmaturana-arch/TESIS.git
   cd TESIS
   ```
2. Instala las dependencias requeridas:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow openpyxl
   ```
3. Ejecuta el notebook `CÓDIGO_TESIS.ipynb` en Jupyter Lab, VS Code o tu entorno preferido.

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
