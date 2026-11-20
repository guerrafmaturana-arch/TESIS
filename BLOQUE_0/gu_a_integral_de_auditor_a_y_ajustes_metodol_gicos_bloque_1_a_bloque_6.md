# GUÍA INTEGRAL DE AUDITORÍA Y AJUSTES METODOLÓGICOS
## Sincronización de Bloques 0–1 y Reglas de Ejecución para Bloques 2–6 (Metodología v4.0)

**Proyecto:** Predicción de la Ley Mineral de Venta Mediante Redes Neuronales Artificiales para Minimizar Pérdidas Económicas  
**Caso de Estudio:** Faena Mantos Cobrizo (Distrito Minero Los Cóndores) / ENAMI (Planta San José)  
**Destinatario:** Entorno de ejecución y desarrollo (`Antigravity`)  

---

## 1. Estado del Bloque 0 y Aclaración Definitiva sobre los Tonelajes

### 1.1. Dictamen de Cierre del Bloque 0 (`INFORME_ENTREGABLE_BLOQUE_0.md`)
El **Bloque 0** queda **aprobado y sellado** bajo la **Metodología v4.0**, cumpliendo con la separación estricta de etapas: prepara las fuentes crudas, documenta el entorno computacional, homologa las leyes a puntos porcentuales ($\%\text{Cu}$ a dos decimales), consolida las $24$ circulares tarifarias de ENAMI Sulfuros y plantea la brecha operacional histórica ($+0.3404\%\text{Cu}$ de sobreestimación media; $\text{NRMSE} = 25.67\%$ en Canje y $25.98\%$ en PSJ).

### 1.2. Uso Estricto de los Tonelajes (`Cancha_Ton` y `Romana_Ton`)
Se establece la siguiente regla operativa para todo el repositorio:
* **Bloques 1 al 4 (EDA, Entrenamiento y Validación Matemática):** **NO se utilizan tonelajes.** Todo el análisis estadístico y el modelado predictivo operan estrictamente sobre las leyes minerales:
  * **Entradas ($X \in \mathbb{R}^{24 \times 2}$):** `Cancha_CuT` ($\%\text{Cu}$) y `Cancha_CuS` ($\%\text{Cu}$).
  * **Salida ($y \in \mathbb{R}^{24 \times 1}$):** `Canje_CuT` ($\%\text{Cu}$).
* **Bloque 5 (Evaluación Económica) y Defensa ante Comisión:** **SÍ se conserva `Romana_Ton` ($295{,}884.67\text{ t}$)** en `DATASET_AUDITADO_N24.csv`. Dado que las tarifas oficiales de ENAMI se expresan por unidad de masa ($\text{US\$/tms}$), la valorización financiera real de cada lote requiere multiplicar la tarifa unitaria por el peso certificado en báscula puente:
  $$\text{Valor Lote (USD)} = \text{TMS} \times \text{Tarifa Unitaria (US\$/tms)}$$
  Además, el tonelaje consolidado sirve como respaldo descriptivo ante la comisión evaluadora para demostrar que los $N = 24$ registros representan balances mensuales industriales completos (promedio de $12{,}328.5\text{ t/mes}$ en romana) y no muestras puntuales de laboratorio.

### 1.3. Tres Ajustes Menores de Limpieza en el Bloque 0 (Para Traspaso a `INFORME_TESIS.docx`)
1. **Encabezado de Tabla de Entorno (Sección 2):** Cambiar *"Especificación Técnica Certificada en Faena"* por *"Especificación Técnica del Entorno de Desarrollo"*.
2. **Nota de Conciliación de Tonelajes (Sección 3.2):** Conservar únicamente la cifra oficial exacta ($295{,}884.67\text{ t}$) y omitir referencias a estimaciones de borradores anteriores ($295{,}960.67\text{ t}$ o $295{,}543.35\text{ t}$).
3. **Verificación Previa al Bloque 5:** Mantener la confirmación visual del escaneo `ABRIL 2023.pdf` (`Flot_Base_USD_tms = 141.290` y `Flot_Escala_USD_tms_1pct = 70.776`) antes de ejecutar las simulaciones financieras del Bloque 5.

---

## 2. Ajustes Obligatorios para Sincronizar el Bloque 1 (`BLOQUE_1/`) bajo Metodología v4.0

Para mantener la línea temporal estricta de la **Metodología v4.0** (donde todo el análisis estadístico del dataset limpio se concentra antes del entrenamiento), actualizar `01_eda_caracteristicas.ipynb` e `INFORME_ENTREGABLE_BLOQUE_1.md` incorporando los siguientes **5 puntos**:

### 2.1. Integrar el Estudio de Dispersión, Varianza y Covarianza (Movido desde Bloque 0)
* Incorporar en el Bloque 1 la tabla de **Desviación Estándar y Varianza** muestral ($s, s^2$ con $\text{ddof}=1$) y poblacional ($\sigma, \sigma^2$ con $\text{ddof}=0$), junto con la **Matriz de Covarianza Muestral** ($5 \times 5$, en $(\%\text{Cu})^2$) para `Cancha_CuT`, `Cancha_CuS`, `PSJ_CuT`, `Canje_CuT` y `Sesgo_Canje`.
* **Justificación explícita del escalado:** Documentar que $\text{Cancha\_CuT}$ ($s^2 = 0.0581\,(\%\text{Cu})^2$) tiene una varianza **$61.6$ veces mayor** que $\text{Cancha\_CuS}$ ($s^2 = 0.00094\,(\%\text{Cu})^2$), lo que hace obligatorio aplicar `StandardScaler` dentro de cada pliegue de validación en los Bloques 2 y 3 para equilibrar los gradientes de ambas neuronas de entrada.
* **Descomposición Sesgo-Varianza de la Brecha Histórica:** Documentar que el error cuadrático medio histórico ($\text{MSE} = 0.1350\,(\%\text{Cu})^2$) se descompone en:
  $$\text{MSE} = \bar{e}^2 + \sigma^2_{\text{error}} = 0.1159 + 0.0191 = 0.1350\,(\%\text{Cu})^2$$
  demostrando que el **$85.83\%$** del problema es sobreestimación sistemática ($\bar{e} = +0.3404\%\text{Cu}$) y el **$14.17\%$** es variabilidad residual ($s_{\text{error}} = 0.1413\%\text{Cu}$).

### 2.2. Diagnóstico de Complejidad Relativa mediante Coeficiente de Variación ($\text{CV}\%$)
* Destacar el contraste del Coeficiente de Variación ($\text{CV} = \frac{s}{\mu} \times 100\%$) entre variables: `Cancha_CuT` ($\text{CV} = 13.6\%$), `Canje_CuT` ($\text{CV} = 17.4\%$) y `Cancha_CuS` ($\text{CV} = 20.9\%$). Esto demuestra que, en términos relativos, la fracción soluble oxidada es la más errática de la operación y aporta la señal geometalúrgica necesaria para explicar las desviaciones no lineales.

### 2.3. Explicación de la Paradoja Pearson vs. Spearman y Efecto *Leverage*
* Precisar en la sección de correlaciones por qué Pearson ($r = +0.8347$) es mayor que Spearman ($\rho = +0.7116$) en `Cancha_CuT` vs. `Canje_CuT`: los dos pulsos extremos de alta ley (**Octubre 2023** y **Noviembre 2023**) ejercen un fuerte efecto de apalancamiento (*leverage*) que infla la correlación lineal. Si se excluyen esos $2$ meses ($N = 22$), Pearson cae a $r = +0.6181$ y queda por debajo de Spearman ($\rho = +0.6253$), probando que los modelos lineales clásicos quedan dominados por los extremos.
* Añadir una nota explicando por qué una correlación lineal alta ($r = +0.8347$) coexiste con un coeficiente de determinación negativo ($R^2 = -1.2709$): la correlación mide sincronía de movimiento y es invariante ante traslaciones o cambios de escala, mientras que $R^2$ penaliza la desviación directa respecto a la recta identidad $y = \hat{y}$ causada por el sesgo de $+0.3404\%\text{Cu}$.

### 2.4. Depuración Formal de la Tabla de VIF
* Eliminar la fila `Constante (VIF = 1.0000)` de la tabla visible de VIF en el informe y en la tesis, reportando únicamente las variables explicativas (`Cancha_CuT` y `Cancha_CuS`, ambos con $\text{VIF} = 1.8045 < 5.0$), manteniendo `sm.add_constant(X)` internamente en el script de Python para garantizar el cálculo centrado correcto.

### 2.5. Blindaje Geoestadístico sobre Autocorrelación (ACF)
* Mantener la demostración de que el sesgo empírico es ruido blanco puro ($r_1 = -0.0866$, dentro de bandas de Bartlett $\pm 0.4001$), lo que valida **LOOCV** como esquema principal.
* Agregar una nota metodológica indicando que, dado que la ley bruta `Cancha_CuT` presenta inercia de rezago 1 ($r_1 = +0.4467$) por continuidad geológica entre mantos contiguos, en los Bloques 2 y 3 se ejecutará una **verificación secundaria de sensibilidad mediante *Buffered-LOOCV*** (excluyendo los meses adyacentes $t-1$ y $t+1$) para blindar los resultados frente a preguntas de geoestadística.

---

## 3. Reglas de Ejecución y Contratos de Salida para Bloques 2 al 6

Para continuar el desarrollo sin errores de secuencia, fugas de información (*data leakage*) ni duplicación de análisis, `Antigravity` debe respetar estrictamente las siguientes reglas en los bloques restantes:

### 3.1. BLOQUE 2: Entrenamiento de Modelos de Referencia (`./BLOQUE_2/02_baselines_loocv.ipynb`)
* **Regla de Cero Fuga de Datos:** Encapsular `StandardScaler` y cada estimador dentro de un `sklearn.pipeline.Pipeline`, ajustando media y desviación estándar **exclusivamente sobre los $23$ datos de entrenamiento** de cada pliegue LOOCV ($k = 1, \dots, 24$).
* **Optimización Interna:** Todo ajuste de hiperparámetros ($\lambda^*$ en `RidgeCV`, `LassoCV`, `ElasticNetCV`, y grilla de $C, \epsilon, \gamma$ en `SVR`) debe realizarse mediante validación cruzada interna ($K = 5$) dentro de los $23$ datos de entrenamiento.
* **Contrato de Salida:** Exportar `PREDICCIONES_OOS_BASELINES.csv/.xlsx` (columnas: `Fecha`, `Canje_CuT`, `Pred_Cancha`, `Pred_OLS`, `Pred_Ridge`, `Pred_Lasso`, `Pred_ElasticNet`, `Pred_SVR`) y `LOG_HIPERPARAMETROS_BASELINES.csv/.xlsx`.

### 3.2. BLOQUE 3: Entrenamiento de Redes Neuronales (`./BLOQUE_3/03_modelado_rna_loocv.ipynb`)
* **Arquitecturas a Evaluar bajo el mismo LOOCV:**
  1. Réplica de la arquitectura de referencia del Dr. José Delgado Vega.
  2. Grilla parsimoniosa de topologías MLP: `(2,)` ($9$ parámetros), `(3,)` ($13$ parámetros), `(4,)` ($17$ parámetros), `(2, 2)` ($15$ parámetros) y `(3, 2)` ($20$ parámetros), respetando la relación $\text{Parámetros} < N_{\text{train}} = 23$.
* **Configuración Técnica para $N = 24$:** Utilizar `MLPRegressor(solver='lbfgs')` (optimizador cuasi-Newton determinista superior a mini-batches estocásticos en muestras pequeñas), selección interna de regularización $L_2$ ($\alpha \in \{0.001, 0.01, 0.05, 0.1, 0.5\}$), contraste de activación (`relu` vs. `tanh`) y promediado sobre múltiples semillas aleatorias fijas (ej. $10$ semillas por pliegue) para eliminar varianza de inicialización.
* **Verificaciones Secundarias:** Ejecutar validación temporal prospectiva (primeros $19$ meses entrenamiento $\to$ últimos $5$ meses prueba) y sensibilidad *Buffered-LOOCV* (excluyendo $t-1$ y $t+1$).
* **Contrato de Salida:** Exportar `PREDICCIONES_OOS_RNA.csv/.xlsx`, `LOG_HIPERPARAMETROS_RNA.csv/.xlsx` y `SENSIBILIDAD_TEMPORAL_Y_BUFFERED.csv/.xlsx`.

### 3.3. BLOQUE 4: Validación Unificada, Selección del Ganador y XAI (`./BLOQUE_4/04_validacion_residuos_xai.ipynb`)
* **Evaluación Imparcial Única:** Cargar `PREDICCIONES_OOS_BASELINES.csv` y `PREDICCIONES_OOS_RNA.csv` y calcular en una sola tabla comparativa: $\text{MSE}$, $\text{RMSE}$, $\text{MAE}$, $\text{MAPE}$, $\text{NRMSE}$, Sesgo Medio ($\bar{e}$), $R^2$ y **Varianza Explicada** (`explained_variance_score`).
* **Selección del Modelo Ganador (*Champion Model*):** Elegir el modelo con menor $\text{RMSE}_{\text{LOOCV}}$, mayor parsimonia y estabilidad frente a *outliers*.
* **Auditoría Estadística sobre los $24$ Residuos Ciegos Cronológicos:**
  * Normalidad de residuos (Shapiro-Wilk + Q-Q Plot).
  * Homocedasticidad (Breusch-Pagan + Residuos vs. Predichos).
  * Independencia serial (Durbin-Watson $d \approx 2$).
  * Prueba pareada no paramétrica de **Wilcoxon** ($p < 0.05$) sobre las diferencias de error cuadrático frente a Cancha y frente al mejor modelo lineal, más intervalo de confianza *bootstrap* ($B = 10{,}000$).
* **Explicabilidad (XAI) y Serialización:** Reentrenar el pipeline ganador sobre $N = 24$, generar **Superficie de Respuesta 2D** (mapa de contorno), **SHAP exacto (`shap.KernelExplainer`)** e **Importancia por Permutación**, y serializar `modelo_ganador_mantos_cobrizo.joblib` + `metricas_validadas_champion.csv`.

### 3.4. BLOQUE 5: Modelo Económico Dual ENAMI (`./BLOQUE_5/05_modelo_economico_enami.ipynb`)
* Cruzar las $24$ predicciones ciegas del modelo ganador con `TARIFAS_ENAMI_CONSOLIDADAS_2023_2024.csv` (tras confirmar visualmente `ABRIL 2023.pdf`) y `Romana_Ton` ($295{,}884.67\text{ t}$).
* Reportar por separado:
  * **Nivel 1:** Reducción de la desviación presupuestaria en $\text{USD}$ ($\sum |V_{\text{Cancha}} - V_{\text{Real}}| - \sum |V_{\text{RNA}} - V_{\text{Real}}|$).
  * **Nivel 2:** Ahorro real de caja en $\text{USD}$ mediante simulación contrafactual (falsos fletes y maquilas evitadas bajo *cut-off*, *blending* preventivo y ahorro de contraanálisis de laboratorio Sobre 2).

### 3.5. BLOQUE 6: Transferencia Operativa y Memoria Final (`./BLOQUE_6/`)
* Construir `app.py` en Streamlit + Plotly (arquitectura de 3 capas consumiendo `modelo_ganador_mantos_cobrizo.joblib` sin reentrenar en inferencia).
* Documentar el SOP operacional y el rol reglamentario de la RNA como gatillador del **Sobre 2** (antesala legal del Sobre 3 de arbitraje ENAMI).
* Volcar tablas, figuras a $200\text{ DPI}$ y resultados auditados en `./INFORME_TESIS.docx`.