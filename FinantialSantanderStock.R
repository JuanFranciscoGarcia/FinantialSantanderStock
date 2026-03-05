rm(list=ls())
install.packages("imputeTS")

library(zoo)
library(imputeTS)
library(forecast)
library(tseries)
library(rugarch)
library(tsoutliers)
library(lmtest)
library(devtools)
library(xts)

datos = read.table('Datos_Santander.txt', header = TRUE)
datos$Fecha = as.Date(datos$Fecha, format = "%d/%m/%y")
datos = datos[order(datos$Fecha), ]
datos_train = subset(datos, Fecha <= as.Date("2025-02-28"))

# Generamos fechas laborales
fecha_completa = seq(from = min(datos_train$Fecha), to = as.Date("2025-02-28"), by = "day")
fecha_completa = fecha_completa[!(weekdays(fecha_completa) %in% c("sábado", "domingo"))]

# Serie original con NAs
serie_zoo = zoo(datos_train$Ultimo, order.by = datos_train$Fecha)
serie_completa = merge(serie_zoo, zoo(, fecha_completa), all = TRUE)

# Guardamos índices NA
interp_indices = which(is.na(serie_completa))
fechas_interp = time(serie_completa)[interp_indices]

# Interpolación con zoo --------------------------------------------------------
serie_zoo_interp = na.approx(serie_completa)
valores_zoo = coredata(serie_zoo_interp)[interp_indices]

# Interpolación con imputeTS ---------------------------------------------------
serie_ts = ts(coredata(serie_completa), frequency = 1)
serie_ts_interp = na.interpolation(serie_ts, option = "linear")
serie_ts_interp_zoo = zoo(serie_ts_interp, order.by = time(serie_completa))
valores_ts = coredata(serie_ts_interp_zoo)[interp_indices]

# Ponemos una tolerancia numérica para interpretar si los puntos son o no iguales
umbral = 1e-6

# Si coinciden se pone en → verde
coinciden = which(abs(valores_zoo - valores_ts) < umbral)
fechas_verde = fechas_interp[coinciden]
valores_verde = valores_zoo[coinciden]

# Hacemos también diferencias
difieren = setdiff(1:length(fechas_interp), coinciden)
fechas_rojo = fechas_interp[difieren]
valores_rojo = valores_zoo[difieren]
valores_azul = valores_ts[difieren]

# Datos reales (no interpolados)
fechas_reales = time(serie_completa)[-interp_indices]
valores_reales = coredata(serie_completa)[-interp_indices]

# CONCLUSIÓN -------------------------------------------------------------------
cat("Total interpolados:", length(fechas_interp), "\n")
cat("Coinciden (verde):", length(fechas_verde), "\n")
cat("Difieren (rojo-azul):", length(fechas_rojo), "\n")

# Hacemos un gráfico de puntos y de la serie temporal para compararlo
X11()
plot(fechas_reales, valores_reales, main = "Comparación de Métodos de Interpolación", 
     ylab = "Precios al cierre", xlab = "Fecha", pch = 19, col = "black", cex = 0.7)
points(fechas_verde, valores_verde, col = "green4", pch = 19, cex = 0.7)
points(fechas_rojo, valores_rojo, col = "red", pch = 19, cex = 0.7)
points(fechas_rojo, valores_azul, col = "blue", pch = 19, cex = 0.7)
legend("topleft", legend = c("Valor original", "Interpolación igual", "Zoo distinto", "imputeTS distinto"),
       col = c("black", "green4", "red", "blue"), pch = 19, bty = "n", cex = 0.8)

X11()
plot(serie_zoo_interp, main = "Serie Temporal con Interpolaciones Comparadas", 
     ylab = "Precios al cierre", xlab = "Fecha", type = "l", col = "black")
points(fechas_verde, valores_verde, col = rgb(0, 0.5, 0, alpha = 0.7), pch = 19, cex = 0.7)  # Verde
points(fechas_rojo, valores_rojo, col = rgb(1, 0, 0, alpha = 0.7), pch = 19, cex = 0.7)      # Rojo
points(fechas_rojo, valores_azul, col = rgb(0, 0, 1, alpha = 0.7), pch = 19, cex = 0.7)      # Azul
legend("topleft", legend = c( "Interpolación igual", "Zoo distinto", "imputeTS distinto"),
       col = c( "green4", "red", "blue"), pch = 19, bty = "n", cex = 0.8)

# Seguimos con el cáculo del modelo ARIMA ======================================
Santander.ts = ts(serie_zoo_interp, start = c(1,1), frequency = 1)
print(Santander.ts)

X11()
plot(Santander.ts, main="Cotizaciones Santander (Interpoladas)", ylab="Precios al cierre", xlab="Fecha")

#La serie no es estacionaria en varianza ni en media, tiene tendencia
# Primero tomamos logs 
logSantander = log(Santander.ts)
logSantander.ts = ts(logSantander, start = c(1,1), frequency = 1)
plot(logSantander.ts, main="log Cotizaciones Santander")
#Al tomar logs la varianza es mas estable

#Tomamos una diff sobre la serie en logs para hacerla estacionaria en media
#Al tomar una diff sobre la serie en logs, obtenemos los rendimientos (returns)
returnsSAN = diff(logSantander)
plot(returnsSAN, main="rendimientos Santander")
#Ya tenemos una serie estacionaria con la que podemos trabajar

par(mfrow=c(1,2))
plot(logSantander.ts, main="log Cotizaciones Santander")
plot(returnsSAN, main="rendimientos Santander")

x11()
par(mfrow=c(3,1))
plot(Santander.ts, main="Cotizaciones Santander inicial")
plot(logSantander, main="log Cotizaciones Santander")
plot(returnsSAN, main="rendimientos Cotizaciones Santander")

#Estimamos los modelos ARIMA
x11()
par(mfrow=c(1,2))
acf(returnsSAN, main="acf returns Santander")
pacf(returnsSAN, main="pacf returns Santander")
#De acuerdo con esto, ACF y PACF limpias de los returns
#No se necesita modelo ARIMA, es un I(1)

#POSIBILIDADES
#mod0 = ARIMA(0,1,0)
#mod1 = ARIMA(1,1,0)

#Definimos el mod0 Arima (para modelos no estacionarios)
mod0 = Arima(logSantander, order = c(0,1,0), include.drift = TRUE, method = c("ML"))
#Los precios evolucionan multiplicativamente con un pequeño crecimiento esperado (drift) y una parte aleatoria
summary(mod0)
coeftest(mod0)
#Obtenemos drift = 0.0006 y s.e. = 0.0004
se = 0.00065926
drift = 0.00090548
t = 0.00065926/ 0.00090548
#t=0.7280 < 1.96
#Como no supera 1.96, no es estadísticamente significativo → no se puede afirmar que hay una tendencia real
#El componente de tendencia (drift) no tiene una evidencia estadística fuerte, 
#pero el modelo puede seguir siendo útil para predecir y explicar el comportamiento del precio.

#===========================================================================

#Diganosis del mod0
#Prueba de homocedasticidad (sabemos que no se aprobará  porque hay estructura en variabilidad)
res0 = residuals(mod0)
x11()
par(mfrow=c(1,1))
plot(res0, main = "Residuos del modelo ARIMA(0,1,0)", ylab = "Error")
abline(h = 0.0567, col = "red", lty = 2)
abline(h = -0.0567, col = "red", lty = 2)

# Prueba de normalidad con gráficos QQ (la normalidad también se verá afecatada pareciéndose a una tstudents, colas más grandes, visto además si se calcula la curtosis)
x11()
par(mfrow=c(1,1))
qqnorm(res0, main="QQ-Plot de Residuos")
qqline(res0)

# Test de normalidad Kolmogorov-Smirnov
ks.test(res0, "pnorm", mean(res0), sd(res0))

install.packages("moments")  # Solo la primera vez
library(moments)

curtosis_res0 = kurtosis(res0)
print(paste("Curtosis de los residuos:", round(curtosis_res0, 2)))

# Contraste de independencia individual
par(mfrow=c(1,2))
acf(res0, lag.max=100, main="ACF de los Residuos")
pacf(res0, lag.max=100, main="PACF de los Residuos")

# Prueba de independencia grupal Ljung-Box
x11()
tsdiag(mod0)

#===========================================================================

#Definimos el mod01
mod01 = Arima(logSantander, order = c(0,1,0), include.drift = FALSE, method = c("ML"))
#Los precios evolucionan multiplicativamente con un pequeño crecimiento esperado (drift) y una parte aleatoria
summary(mod01)

#===========================================================================

#Definimos el mod1
mod1 = Arima(logSantander, order = c(1,1,0), include.drift = TRUE, method = c("ML"))
#Los precios evolucionan multiplicativamente con un pequeño crecimiento esperado (drift) y una parte aleatoria
summary(mod1)
coeftest(mod1)

#Diganosis del mod1
#Prueba de homocedasticidad
res1 = residuals(mod1)
par(mfrow=c(1,1))
plot(res1, main = "Residuos del modelo ARIMA(1,1,0)", ylab = "Error")
abline(h = 0.0567, col = "red", lty = 2)
abline(h = -0.0567, col = "red", lty = 2)

# Prueba de normalidad con gráficos QQ
par(mfrow=c(1,1))
qqnorm(res1, main="QQ-Plot de Residuos")
qqline(res1)

# Test de normalidad Kolmogorov-Smirnov
ks.test(res1, "pnorm", mean(res1), sd(res1))

# Contraste de independencia individual
par(mfrow=c(1,2))
acf(res1, lag.max=100, main="ACF de los Residuos")
pacf(res1, lag.max=100, main="PACF de los Residuos")

# Prueba de independencia grupal Ljung-Box
tsdiag(mod1)

#prediccion pre volatilidad===========================================================================

# Escenario A ==================================================================
# Filtramos marzo 2025 como conjunto de validación
datos_valid = subset(datos, Fecha > as.Date("2025-02-28") & Fecha <= as.Date("2025-03-31"))
fecha_validacion = seq(from = as.Date("2025-03-01"), to = as.Date("2025-03-31"), by = "day")
fecha_validacion = fecha_validacion[!(weekdays(fecha_validacion) %in% c("sábado", "domingo"))]
h = length(fecha_validacion)

# PREDICCIÓN con mod0 -----------------------------------------------------------
pred_mod0 = forecast(mod0, h = h)
pred_precios_0 = exp(pred_mod0$mean)
ic_lower_0 = exp(pred_mod0$lower[,2])
ic_upper_0 = exp(pred_mod0$upper[,2])

# PREDICCIÓN con mod01 ---------------------------------------------------------
pred_mod01 = forecast(mod01, h = h)
pred_precios_01 = exp(pred_mod01$mean)
ic_lower_01 = exp(pred_mod01$lower[,2])
ic_upper_01 = exp(pred_mod01$upper[,2])

# Hacemos un gráfico comparado -------------------------------------------------
x11()
plot(fecha_validacion, datos_valid$Ultimo, type = "l", col = "black", lwd = 2,
     ylim = range(c(ic_lower_0, ic_upper_0, ic_lower_01, ic_upper_01, datos_valid$Ultimo)),
     ylab = "Precio", xlab = "Fecha", main = "Predicción mod0 vs mod01 con IC mes de Marzo completo")

# Sombra IC mod0 (azul)
polygon(c(fecha_validacion, rev(fecha_validacion)),
        c(ic_upper_0, rev(ic_lower_0)),
        col = rgb(0.2, 0.2, 1, alpha = 0.2), border = NA)

# Sombra IC mod01 (verde)
polygon(c(fecha_validacion, rev(fecha_validacion)),
        c(ic_upper_01, rev(ic_lower_01)),
        col = rgb(0.2, 0.8, 0.2, alpha = 0.2), border = NA)

# Líneas de predicción
lines(fecha_validacion, pred_precios_0, col = "blue", lwd = 2)
lines(fecha_validacion, pred_precios_01, col = "darkgreen", lwd = 2)

# Línea real
lines(datos_valid$Fecha, datos_valid$Ultimo, col = "black", lwd = 2)

# Leyenda
legend("topleft", legend = c("Real", "Pred. mod0 (con drift)", "Pred. mod01 (sin drift)", "IC mod0", "IC mod01"),
       col = c("black", "blue", "darkgreen", rgb(0.2, 0.2, 1, 0.2), rgb(0.2, 0.8, 0.2, 0.2)),
       lty = 1, lwd = c(2, 2, 2, 10, 10), pch = c(NA, NA, NA, 15, 15), pt.cex = 2, bty = "n")

# Calculamos los errores del A
# mod0
error_relativo_0 = abs(pred_precios_0 - datos_valid$Ultimo) / datos_valid$Ultimo
mape_0 = mean(error_relativo_0) * 100
mape_0_2 = mean(error_relativo_0[10:21]) * 100

# mod01
error_relativo_01 = abs(pred_precios_01 - datos_valid$Ultimo) / datos_valid$Ultimo
mape_01 = mean(error_relativo_01) * 100
mape_01_2 = mean(error_relativo_01[10:21]) * 100

cat("MAPE mod0 (con drift)[1-31 marzo]:", round(mape_0, 2), "%\n")
cat("MAPE mod01 (sin drift)[1-31 marzo]:", round(mape_01, 2), "%\n")
cat("MAPE mod0 (con drift)[16-31 marzo]:", round(mape_0_2, 2), "%\n")
cat("MAPE mod01 (sin drift)[16-31 marzo]:", round(mape_01_2, 2), "%\n")

# ESCENARIO B: PREDICCIÓN CON ENTRENAMIENTO HASTA EL 15 MARZO ==================
# 1. Nuevos datos de entrenamiento y validación
datos_train_B = subset(datos, Fecha <= as.Date("2025-03-15"))
datos_valid_B = subset(datos, Fecha > as.Date("2025-03-15") & Fecha <= as.Date("2025-03-31"))
fecha_validacion_B = seq(from = as.Date("2025-03-16"), to = as.Date("2025-03-31"), by = "day")
fecha_validacion_B = fecha_validacion_B[!(weekdays(fecha_validacion_B) %in% c("sábado", "domingo"))]
h_B = length(fecha_validacion_B)

# 2. Transformamos nueva serie interpolada para el entrenamiento extendido
serie_train_B = zoo(datos_train_B$Ultimo, order.by = datos_train_B$Fecha)
fecha_completa_B = seq(from = min(datos_train_B$Fecha), to = as.Date("2025-03-15"), by = "day")
fecha_completa_B = fecha_completa_B[!(weekdays(fecha_completa_B) %in% c("sábado", "domingo"))]
serie_completa_B = merge(serie_train_B, zoo(, fecha_completa_B), all = TRUE)
serie_interp_B = na.approx(serie_completa_B)
logSantander_B = log(ts(serie_interp_B, frequency = 1))

# 3. Estimación de modelos con más datos
mod0_B = Arima(logSantander_B, order = c(0,1,0), include.drift = TRUE, method = "ML")
mod01_B = Arima(logSantander_B, order = c(0,1,0), include.drift = FALSE, method = "ML")

# 4. Predicción
pred_mod0_B = forecast(mod0_B, h = h_B)
pred_mod01_B = forecast(mod01_B, h = h_B)

# 5. Transformación a precios
pred_precios_0_B = exp(pred_mod0_B$mean)
ic_lower_0_B = exp(pred_mod0_B$lower[,2])
ic_upper_0_B = exp(pred_mod0_B$upper[,2])

pred_precios_01_B = exp(pred_mod01_B$mean)
ic_lower_01_B = exp(pred_mod01_B$lower[,2])
ic_upper_01_B = exp(pred_mod01_B$upper[,2])

# 6. Hacemos un gráfico comparado del B
x11()
plot(fecha_validacion_B, datos_valid_B$Ultimo, type = "l", col = "black", lwd = 2,
     ylim = range(c(ic_lower_0_B, ic_upper_0_B, ic_lower_01_B, ic_upper_01_B, datos_valid_B$Ultimo)),
     ylab = "Precio", xlab = "Fecha", main = "Predicción mod0 vs mod01 (16–31 marzo)")

# Sombra IC mod0
polygon(c(fecha_validacion_B, rev(fecha_validacion_B)),
        c(ic_upper_0_B, rev(ic_lower_0_B)),
        col = rgb(0.2, 0.2, 1, alpha = 0.2), border = NA)

# Sombra IC mod01
polygon(c(fecha_validacion_B, rev(fecha_validacion_B)),
        c(ic_upper_01_B, rev(ic_lower_01_B)),
        col = rgb(0.2, 0.8, 0.2, alpha = 0.2), border = NA)

# Líneas
lines(fecha_validacion_B, pred_precios_0_B, col = "blue", lwd = 2)
lines(fecha_validacion_B, pred_precios_01_B, col = "darkgreen", lwd = 2)
lines(datos_valid_B$Fecha, datos_valid_B$Ultimo, col = "black", lwd = 2)

legend("topleft", legend = c("Real", "Pred. mod0", "Pred. mod01", "IC mod0", "IC mod01"),
       col = c("black", "blue", "darkgreen", rgb(0.2, 0.2, 1, 0.2), rgb(0.2, 0.8, 0.2, 0.2)),
       lty = 1, lwd = c(2, 2, 2, 10, 10), pch = c(NA, NA, NA, 15, 15), pt.cex = 2, bty = "n")

# 7. Calculamos los errores del B
error_relativo_0_B = abs(pred_precios_0_B - datos_valid_B$Ultimo) / datos_valid_B$Ultimo
mape_0_B = mean(error_relativo_0_B) * 100

error_relativo_01_B = abs(pred_precios_01_B - datos_valid_B$Ultimo) / datos_valid_B$Ultimo
mape_01_B = mean(error_relativo_01_B) * 100

cat("MAPE mod0 (con drift) [16–31 marzo]:", round(mape_0_B, 2), "%\n")
cat("MAPE mod01 (sin drift) [16–31 marzo]:", round(mape_01_B, 2), "%\n")


#Estructura-Volatilidad===========================================================================

returnsCENTRADOS = returnsSAN - mean(returnsSAN)  # a_t = returns centrados
VolatilidadEmpirica = returnsCENTRADOS^2

x11()
par(mfrow = c(2,1))
plot(returnsSAN, main = "Rendimientos Santander", ylab = "Returns")
plot(returnsCENTRADOS, main = "Rendimientos centrados Santander", ylab = "Returns")

x11()
par(mfrow = c(2,1))
plot(returnsCENTRADOS, main = "Rendimientos centrados Santander", ylab = "Returns")
plot(VolatilidadEmpirica, main = "Volatilidad empírica (residuos centrados²)", ylab = "Varianza aproximada")

x11()
par(mfrow = c(2,1))
acf(VolatilidadEmpirica, main="FAS de los residuos centrados²")
pacf(VolatilidadEmpirica, main="FAP de los residuos centrados²")

#dado que la fas y fap de los returns estaban "limpios". p=0, q=0
returnsCENTRADOS = ts(returnsCENTRADOS, start = c(1,1), frequency = 1)

#===========================================================================

#primero probamos ARCH1
modeloARCH1 = garch(returnsCENTRADOS, order = c(0,1))
summary(modeloARCH1)

x11()
plot(sqrt(VolatilidadEmpirica), type = "l", col = "red", lwd = 2,
     ylab = "Volatilidad (raíz cuadrada)", xlab = "Tiempo",
     main = "Volatilidad empírica vs sigma ARCH(1)")

lines(modeloARCH1$fitted.values[,1], col = "green", lwd = 2)

legend("topright", legend = c("Raíz cuadrada de volatilidad empírica", "Sigma estimada ARCH(1)"),
       col = c("red", "green"), lwd = 2, bty = "n")

x11()
plot(sqrt(VolatilidadEmpirica), col = "red", main = "Raíz cuadrada de volatilidad empírica y sigma estimada")
par(new = TRUE)
plot(modeloARCH1$fitted.values[,1], col = "green", lwd = 2)


#Diagnosis de residuos ARCH
residARCH1 = na.omit(modeloARCH1$residuals)
residARCH1_centered = residARCH1 - mean(residARCH1)
sq_residARCH1_centered = residARCH1_centered^2

x11()
par(mfrow = c(2,1))
acf(sq_residARCH1_centered, main = "ACF residuos² del modelo ARCH(1)")
pacf(sq_residARCH1_centered, main = "PACF residuos² del modelo ARCH(1)")

#===========================================================================
#probamos GARCH11
modeloGARCH11 = garch(returnsCENTRADOS, order = c(1,1))
summary(modeloGARCH11)
persistencia = coef(modeloGARCH11)[2] + coef(modeloGARCH11)[3]
print(paste("Persistencia GARCH(1,1):", round(persistencia, 4)))


#dado que en t=1 tenemos un NaN porque no hay para esa t una estimaci?n de los residuos pero s? para t=2... en adelante... 
residGARCH11 = na.omit(modeloGARCH11$residuals)

#Diagnosis de residuos GARCH
#los resid del GARCH(1,1) son los epsilon_t
X11()
par(mfrow=c(2,1))
acf(residGARCH11, main = "ACF residuos del GARCH(1,1)")
pacf(residGARCH11, main = "PACF residuos del GARCH(1,1)")

X11()
hist(residGARCH11<- na.omit(modeloGARCH11$residuals))
#se puede calcular la var muestral de los residuos del GARCH(1,1)
var(residGARCH11)

#se puede calcular la media muestral de los residuos del GARCH(1,1)
mean(residGARCH11)

#para chequear que toda la estructura en volatilidad se ha modelado y captado se pinta la fas y fap de los resid del GARCH11 al cuadrado
x11()
par(mfrow = c(2,1))
acf(residGARCH11^2, main = "ACF residuos² del GARCH(1,1)")
pacf(residGARCH11^2, main = "PACF residuos² del GARCH(1,1)")

#Evaluar si queda estructura de volatilidad sin modelar

#Comparación triple
x11()
plot(sqrt(VolatilidadEmpirica), type = "l", col = "red", lwd = 2,
     ylab = "Volatilidad (raíz cuadrada)", xlab = "Tiempo",
     main = "Volatilidad empírica vs. GARCH(1,1) vs. ARCH(1) función [Garch]")

# Añadir sigma GARCH(1,1)
lines(modeloGARCH11$fitted.values[,1], col = "blue", lwd = 2)

# Añadir sigma ARCH(1)
lines(modeloARCH1$fitted.values[,1], col = "green", lwd = 2)

# Añadir leyenda
legend("topright", legend = c("Volatilidad empírica", "Sigma GARCH(1,1)", "Sigma ARCH(1)"),
       col = c("red", "blue", "green"), lwd = 2, bty = "n")


#AHORA CON RUGARCH vs GARCH 

# 1. Crear especificación GARCH(1,1)
spec_garch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
  distribution.model = "norm"
)

# 3. Ajustar ambos modelos
fit_garch <- ugarchfit(spec = spec_garch, data = returnsCENTRADOS)

# 4. Volatilidades estimadas (sigma_t)
sigma_garch <- sigma(fit_garch)

# 5. Volatilidad empírica (raíz de residuos centrados²)
VolatilidadEmpirica <- sqrt((returnsCENTRADOS - mean(returnsCENTRADOS))^2)

# Aseguramos que ambas series tengan la misma longitud
min_len_garch <- min(length(modeloGARCH11$fitted.values[,1]), length(sigma_garch))
garch_tseries_sigma <- modeloGARCH11$fitted.values[1:min_len_garch, 1]
garch_rugarch_sigma <- as.numeric(sigma_garch)[1:min_len_garch]

# Gráfico de comparación
x11()
plot(garch_tseries_sigma, type = "l", col = "blue", lwd = 2,
     ylab = "Sigma (volatilidad)", xlab = "Tiempo",
     main = "Comparación GARCH(1,1): tseries vs rugarch")

lines(garch_rugarch_sigma, col = rgb(1, 0.8, 0, alpha = 0.5), lwd = 2)

legend("topright", legend = c("Sigma GARCH(1,1) [tseries]", "Sigma GARCH(1,1) [rugarch]"),
       col = c("blue", "orange"), lwd = 2, bty = "n")

#==============================================================================

spec_garch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
  distribution.model = "norm"
)

fit_garch <- ugarchfit(spec = spec_garch, data = returnsCENTRADOS)

# 2. Hacer forecast de h pasos
forecast_garch <- ugarchforecast(fit_garch, n.ahead = h)

# --- Paso 1: Extraer la predicción de la media (retornos esperados) y volatilidad ---
retornos_pred <- as.numeric(fitted(forecast_garch))  # Retornos predichos
volatilidad_pred <- as.numeric(sigma(forecast_garch))  # Volatilidad predicha

# --- Paso 2: Definir el último precio conocido ---
P0 <- as.numeric(tail(exp(logSantander_B), 1))  # Último precio conocido
log_P0 <- log(P0)  # Log del precio base

# --- Paso 3: Nivel de confianza para los intervalos de predicción ---
z_alpha <- qnorm(0.975)  # Para IC al 95%

# --- Paso 4: Cálculo de los intervalos de predicción en términos de log-retornos (GARCH) ---
log_returns_upper <- retornos_pred + z_alpha * volatilidad_pred
log_returns_lower <- retornos_pred - z_alpha * volatilidad_pred

# --- Paso 5: Acumulación de log-retornos y transformación a precios ---
log_price_pred <- cumsum(retornos_pred) + log_P0
log_price_upper <- cumsum(log_returns_upper) + log_P0
log_price_lower <- cumsum(log_returns_lower) + log_P0

price_pred <- exp(log_price_pred)
price_upper <- exp(log_price_upper)
price_lower <- exp(log_price_lower)

# --- Paso 6: Asumir homocedasticidad (volatilidad constante) ---
volatilidad_constante <- mean(volatilidad_pred)

log_returns_upper_hom <- retornos_pred + z_alpha * volatilidad_constante
log_returns_lower_hom <- retornos_pred - z_alpha * volatilidad_constante

log_price_pred_hom <- cumsum(retornos_pred) + log_P0
log_price_upper_hom <- cumsum(log_returns_upper_hom) + log_P0
log_price_lower_hom <- cumsum(log_returns_lower_hom) + log_P0

price_pred_hom <- exp(log_price_pred_hom)
price_upper_hom <- exp(log_price_upper_hom)
price_lower_hom <- exp(log_price_lower_hom)

# --- Paso 7: Fechas predicción ---
fechas_pred <- fecha_validacion_B[1:length(price_pred)]

# --- Paso 8: Gráfico con los intervalos de predicción (GARCH vs Homocedasticidad) ---
X11()
plot(fechas_pred, price_pred, type = "l", col = "blue", lwd = 2,
     ylim = range(c(price_lower, price_upper, price_lower_hom, price_upper_hom, datos_valid_B$Ultimo)),
     ylab = "Precio", xlab = "Fecha", main = "Predicción GARCH vs Homocedasticidad (Intervalos)")

# Intervalos GARCH (Volatilidad Variable)
lines(fechas_pred, price_upper, col = "skyblue", lty = 2, lwd = 1.5)
lines(fechas_pred, price_lower, col = "skyblue", lty = 2, lwd = 1.5)

# Intervalos Homocedasticidad (Volatilidad Constante)
lines(fechas_pred, price_upper_hom, col = "lightgreen", lty = 2, lwd = 1.5)
lines(fechas_pred, price_lower_hom, col = "lightgreen", lty = 2, lwd = 1.5)

# Mostrar precios reales
lines(datos_valid_B$Fecha, datos_valid_B$Ultimo, col = "black", lwd = 2)

# Leyenda
legend("topleft", legend = c("Predicción GARCH", "IC GARCH", "Predicción Homocedástica", "IC Homocedástica", "Precio Real"),
       col = c("blue", "skyblue", "green", "lightgreen", "black"), lty = c(1, 2, 1, 2, 1), lwd = 2, bty = "n")