library(readxl)

library(nnfor)
BEGINNING <- read_excel("Downloads/FRANCEBEGINNING1.xlsx")

label <- ts(BEGINNING$FRANCEDailyCases)
fit <- mlp(label)
print(fit)
plot(fit)
fit2 <- mlp(label, hd = c(10,5))
plot(fit2)
fit10 <- mlp(label, lags=1:24)
plot(fit10)
fit7 <- mlp(label, lags=1:24, keep=c(rep(TRUE,12), rep(FALSE,12)))
plot(fit7)
fit1 <- mlp(label, lags=1:24, sel.lag=FALSE)
print(fit1)
plot(fit1)
fit3 <- mlp(label, hd.auto.type="valid",hd.max=8)
print(fit3)
plot(fit3)
x <- ts(sin(1:120*2*pi/12),frequency=12)
mlp(x, model=fit)
mlp(x, model=fit, retrain=TRUE)
frc <- forecast(fit1,h=5)
print(frc)
plot(frc)
frc$mean
frc$all.mean 
z <- 1:(length(label)+24) # I add 24 extra observations for the forecasts
z <- cbind(z) # Convert it into a column-array
fit4 <- mlp(label,xreg=z,xreg.lags=list(0),xreg.keep=list(TRUE),
            # Add a lag0 regressor and force it to stay in the model
            difforder=0) # Do not let mlp() to remove the stochastic trend
print(fit4)
plot(fit4)
fit8 <- mlp(label,difforder=0,xreg=z,xreg.lags=list(1:12))
plot(fit8)
fit9 <- mlp(label,difforder=0,xreg=z,xreg.lags=list(1:12),xreg.keep=list(c(rep(TRUE,3),rep(FALSE,9))))
plot(fit9)
frc.diff <- forecast(fit8,h=120)
print(frc.diff)
library(tsutils)
loc <- residout(label - fit$fitted, outplot=FALSE)$location
zz <- cbind(z, 0)
zz[loc,2] <- 1
fit5 <- mlp(label,xreg=zz, xreg.lags=list(c(0:6),0),xreg.keep=list(rep(FALSE,7),TRUE))
print(fit5)
plot(fit5)
frc.reg <- forecast(fit5,xreg=zz,h=24)
print(frc.reg)
plot(frc.reg)
fit.elm <- elm(label)
print(fit.elm)
plot(fit.elm)
par(mfrow=c(2,2))
for (i in 1:4){plot(fit.elm,i)}
par(mfrow=c(1,1))
frc.elm <- forecast(fit.elm,h=120)
plot(frc.elm)
par(mfrow=c(2,2))
plot(fit1)
plot(frc)
plot(fit.elm)
plot(frc.elm)
par(mfrow=c(1,1))

