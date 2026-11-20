# AUDITORÍA TÉCNICA Y GUÍA DE ACTUALIZACIÓN — BLOQUE 0
## Incorporación de Desviación Estándar, Varianza, Covarianza y Criterio de Selección de Métricas de Error

**Destinatario:** Entorno de desarrollo / Asistente de código (`Antigravity`)  
**Módulo a Actualizar:** `./BLOQUE_0/00_auditoria_datos.ipynb` y `./BLOQUE_0/INFORME_ENTREGABLE_BLOQUE_0.md`  
**Objetivo:** Incorporar el estudio previo de dispersión (desviación estándar $s$, varianza $s^2$ y $\sigma^2$) y dependencia lineal conjunta (covarianza $\text{Cov}$) sobre los $N = 24$ balances mensuales de Mantos Cobrizo **antes** de calcular y seleccionar las métricas de error histórico ($\text{Bias}$, $\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, $\text{NRMSE}$, $\text{MAPE}$ y $R^2$).

---

## 1. Fundamento Metodológico: ¿Por qué estudiar $\sigma$, $\sigma^2$ y $\text{Cov}$ antes de elegir el error?

Antes de evaluar el desempeño de un predictor o definir qué función de pérdida minimizará la Red Neuronal Artificial, es indispensable caracterizar la variabilidad intrínseca de las variables operacionales y su estructura de covariación:

1. **Comparación de Escalas y Varianzas ($s^2_{\text{CuT}}$ vs. $s^2_{\text{CuS}}$):**
   * `Cancha_CuT` ($s = 0.2411\%\text{Cu}$, $s^2 = 0.0581\,(\%\text{Cu})^2$) y `Canje_CuT` ($s = 0.2491\%\text{Cu}$, $s^2 = 0.0620\,(\%\text{Cu})^2$) tienen magnitudes y dispersiones comparables, mientras que `Cancha_CuS` ($s = 0.0307\%\text{Cu}$, $s^2 = 0.00094\,(\%\text{Cu})^2$) posee una varianza **$61.6$ veces menor**.
   * **Impacto en la elección de métricas y preprocesamiento:** Exige estandarizar las entradas (`StandardScaler`) para evitar que `CuT` domine los gradientes de la red, y obliga a utilizar errores en escala física ($\text{RMSE}$, $\text{MAE}$ en $\%\text{Cu}$) complementados con errores relativos ($\text{NRMSE}$, $\text{MAPE}$ en $\%$) para evaluar lotes de baja y alta ley de forma equitativa.

2. **Descomposición Sesgo-Varianza del Error Cuadrático Medio ($\text{MSE}$):**
   * La relación exacta entre el $\text{MSE}$ histórico, el sesgo medio ($\bar{e}$) y la varianza poblacional de la discrepancia ($\sigma^2_{\text{error}}$) es:
     $$\text{MSE} = \bar{e}^2 + \sigma^2_{\text{error}}$$
   * En Mantos Cobrizo (Cancha vs. Canje):
     $$\bar{e}^2 = (0.3404)^2 = 0.1159\,(\%\text{Cu})^2 \quad (85.83\% \text{ del MSE total})$$
     $$\sigma^2_{\text{error}} = 0.0191\,(\%\text{Cu})^2 \quad (14.17\% \text{ del MSE total})$$
     $$\text{MSE} = 0.1159 + 0.0191 = 0.1350\,(\%\text{Cu})^2$$
   * **Impacto en la elección de métricas:**
     * **Sesgo Medio ($\bar{e}$):** Es obligatorio para auditar que el modelo elimine el $85.83\%$ del error atribuible a la sobreestimación sistemática de cancha.
     * **MSE y RMSE (penalización cuadrática) vs. MAE (penalización lineal):** Dado que la discrepancia presenta desviación estándar $s_{\text{error}} = 0.1413\%\text{Cu}$ con picos extremos de hasta $+0.61\%\text{Cu}$ (Ene-2023), $\text{RMSE}$ ($0.3674\%\text{Cu}$) supera a $\text{MAE}$ ($0.3404\%\text{Cu}$). Elegir **MSE** como función de costo de entrenamiento y **RMSE** como métrica principal penaliza severamente los errores grandes que comprometen el *cut-off* comercial de ENAMI.
     * **Coeficiente $R^2$:** Al comparar el $\text{MSE}$ ($0.1350\,(\%\text{Cu})^2$) contra la varianza poblacional de la ley real de canje ($\sigma^2_{\text{Canje}} = 0.0595\,(\%\text{Cu})^2$), se obtiene $R^2 = 1 - \frac{0.1350}{0.0595} = -1.2709$, demostrando que la varianza del error de cancha supera en más del doble a la varianza natural del yacimiento.

3. **Análisis de Covarianza ($\text{Cov}$):**
   * $\text{Cov}_{\text{muestral}}(\text{Cancha\_CuT}, \text{Canje\_CuT}) = +0.0501\,(\%\text{Cu})^2$ ($r = +0.8347$): Confirma una fuerte covariación positiva entre origen y destino, validando el uso de `Cancha_CuT` como predictor principal.
   * $\text{Cov}_{\text{muestral}}(\text{Cancha\_CuT}, \text{Cancha\_CuS}) = +0.0049\,(\%\text{Cu})^2$ ($r = +0.6677$): Muestra asociación moderada sin redundancia lineal, respaldando el espacio bidimensional de entrada $[\text{CuT}_{\text{cancha}}, \text{CuS}_{\text{cancha}}]$.
   * $\text{Cov}_{\text{muestral}}(\text{Cancha\_CuT}, \text{Sesgo}) = +0.0080\,(\%\text{Cu})^2$ ($r = +0.2356$): Indica que cuando aumenta la ley reportada en cancha, tiende a incrementarse levemente la magnitud del sesgo de sobreestimación.

---

## 2. Valores Numéricos Certificados ($N = 24$)

### 2.1. Tabla de Dispersión: Media, Desviación Estándar y Varianza

| Variable ($N = 24$) | Media ($\mu$) | Desv. Est. Muestral ($s$, $\text{ddof}=1$) | Varianza Muestral ($s^2$, $\text{ddof}=1$) | Desv. Est. Poblacional ($\sigma$, $\text{ddof}=0$) | Varianza Poblacional ($\sigma^2$, $\text{ddof}=0$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cancha CuT ($\%$)** | $1.7717\%\text{Cu}$ | $0.2411\%\text{Cu}$ | $0.0581\,(\%\text{Cu})^2$ | $0.2360\%\text{Cu}$ | $0.0557\,(\%\text{Cu})^2$ |
| **Cancha CuS ($\%$)** | $0.1471\%\text{Cu}$ | $0.0307\%\text{Cu}$ | $0.0009\,(\%\text{Cu})^2$ | $0.0301\%\text{Cu}$ | $0.0009\,(\%\text{Cu})^2$ |
| **PSJ CuT ($\%$)** | $1.4304\%\text{Cu}$ | $0.2496\%\text{Cu}$ | $0.0623\,(\%\text{Cu})^2$ | $0.2443\%\text{Cu}$ | $0.0597\,(\%\text{Cu})^2$ |
| **Canje CuT ($\%$)** | $1.4313\%\text{Cu}$ | $0.2491\%\text{Cu}$ | $0.0620\,(\%\text{Cu})^2$ | $0.2438\%\text{Cu}$ | $0.0595\,(\%\text{Cu})^2$ |
| **Sesgo (Cancha $-$ Canje)** | $+0.3404\%\text{Cu}$ | $0.1413\%\text{Cu}$ | $0.0200\,(\%\text{Cu})^2$ | $0.1383\%\text{Cu}$ | $0.0191\,(\%\text{Cu})^2$ |

### 2.2. Matriz de Covarianza Muestral ($\text{ddof}=1$, en $(\%\text{Cu})^2$)

| Variable | `Cancha_CuT` | `Cancha_CuS` | `PSJ_CuT` | `Canje_CuT` | `Sesgo_Canje` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Cancha_CuT`** | **$0.0581$** | $0.0049$ | $0.0497$ | $0.0501$ | $0.0080$ |
| **`Cancha_CuS`** | $0.0049$ | **$0.0009$** | $0.0047$ | $0.0049$ | $0.00004$ |
| **`PSJ_CuT`** | $0.0497$ | $0.0047$ | **$0.0623$** | $0.0613$ | $-0.0116$ |
| **`Canje_CuT`** | $0.0501$ | $0.0049$ | $0.0613$ | **$0.0620$** | $-0.0119$ |
| **`Sesgo_Canje`** | $0.0080$ | $0.00004$ | $-0.0116$ | $-0.0119$ | **$0.0200$** |

---

## 3. Instrucciones de Modificación para `Antigravity`

1. **En `BLOQUE_0/00_auditoria_datos.ipynb`:**
   * Insertar una nueva celda de código y explicación **inmediatamente antes de la Sección 2 (Reconciliación Matemática del Indicador "MECM = 26%")** utilizando el siguiente script:

```python
# ==============================================================================
# ESTUDIO PREVIO DE DISPERSIÓN, VARIANZA, COVARIANZA Y CRITERIO DE ERROR
# ==============================================================================
df_audit['Sesgo_Canje'] = df_audit['Cancha_CuT'] - df_audit['Canje_CuT']
vars_estudio = ['Cancha_CuT', 'Cancha_CuS', 'PSJ_CuT', 'Canje_CuT', 'Sesgo_Canje']

# 1. Tabla de Media, Desviación Estándar y Varianza (Muestral y Poblacional)
resumen_disp = []
for col in vars_estudio:
    serie = df_audit[col]
    resumen_disp.append({
        'Variable': col,
        'Media (%Cu)': serie.mean(),
        'Desv_Est_Muestral_s (%Cu)': serie.std(ddof=1),
        'Varianza_Muestral_s2 ((%Cu)^2)': serie.var(ddof=1),
        'Desv_Est_Poblacional_sigma (%Cu)': serie.std(ddof=0),
        'Varianza_Poblacional_sigma2 ((%Cu)^2)': serie.var(ddof=0)
    })

df_dispersion = pd.DataFrame(resumen_disp)
print("📊 TABLA DE DISPERSIÓN Y VARIANZA (N = 24):")
display(df_dispersion.round(4))

# 2. Matriz de Covarianza Muestral (ddof=1)
cov_matrix = df_audit[vars_estudio].cov(ddof=1)
print("\n📐 MATRIZ DE COVARIANZA MUESTRAL ((%Cu)^2):")
display(cov_matrix.round(5))

# 3. Descomposición Sesgo-Varianza del MSE Histórico (Cancha vs Canje)
bias_sq = (df_audit['Sesgo_Canje'].mean()) ** 2
var_err = df_audit['Sesgo_Canje'].var(ddof=0)
mse_total = bias_sq + var_err

print("\n🎯 DESCOMPOSICIÓN SESGO-VARIANZA DEL MSE HISTÓRICO:")
print(f"   * Sesgo al cuadrado (Bias^2)   = {bias_sq:.4f} (%Cu)^2 ({bias_sq/mse_total*100:.2f}% del MSE)")
print(f"   * Varianza del error (sigma^2) = {var_err:.4f} (%Cu)^2 ({var_err/mse_total*100:.2f}% del MSE)")
print(f"   * MSE Total (Bias^2 + sigma^2) = {mse_total:.4f} (%Cu)^2")

# Exportar tablas estadísticas del Bloque 0
df_dispersion.to_csv(os.path.join(bloque_dir, 'TABLA_DISPERSION_VARIANZA_B0.csv'), index=False)
cov_matrix.to_csv(os.path.join(bloque_dir, 'MATRIZ_COVARIANZA_B0.csv'))
```

2. **En `BLOQUE_0/INFORME_ENTREGABLE_BLOQUE_0.md`:**
   * Reemplazar el archivo actual por la versión actualizada generada en este mismo turno, la cual integra la nueva **Sección 3 (Estudio de Dispersión, Varianza, Covarianza y Criterio de Selección de Métricas de Error)** antes de la evaluación de errores históricos, corrige la suma verificada de `Romana_Ton` ($295{,}543.35\text{ t}$) y limpia el índice de entregables.