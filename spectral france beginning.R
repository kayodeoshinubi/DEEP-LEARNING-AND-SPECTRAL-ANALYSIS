library(readxl)

BEGINNING <- read_excel("Downloads/FRANCEBEGINNING1.xlsx")
t <- seq(0,556)

x <- BEGINNING$FRANCEDailyCases  # CHANGES DEPENDING ON COUNTRY
par(mfrow=c(2,1))

plot(t,x,'l')
spectrum(x)
del<- 0.1 # sampling interval
x.spec <- spectrum(x,log="no",span=10,plot=FALSE)
spx <- x.spec$freq
spy <- 2*x.spec$spec
#PICK
plot(spy~spx,xlab="frequency",ylab="spectral density",type="l")
spectrum(x)
library(ggplot2)
acf1 <- acf(BEGINNING$FRANCEDailyCases, lag.max = 7 * 20, plot = FALSE)
plot1 <- ggplot() + geom_line(aes(x = acf1$lag/7, y = acf1$acf))
plot1
# Create dataframe with different harmonics
X <- data.frame(Days=BEGINNING$Day,
                y = BEGINNING$FRANCEDailyCases,
                sin(2*pi*1*BEGINNING$Day), cos(2*pi*1* BEGINNING$Day), # sine and cos for frequency = 1
                sin(2*pi*1/7*BEGINNING$Day), cos(2*pi*1/7*BEGINNING$Day), # freq. equals 1/7 (i.e. period= 7 days)
                sin(2*pi*1/3*BEGINNING$Day), cos(2*pi*1/3*BEGINNING$Day), # freq = 1/3 (period=3 days)
                sin(2*pi*1/3.5*BEGINNING$Day), cos(2*pi*1/3.5*BEGINNING$Day), # freq=1/3.5 (period=3.5 days)
                sin(2*pi*1/6*BEGINNING$Day), cos(2*pi*1/6*BEGINNING$Day),   # freq=1/6 (period=6 days)
                sin(2*pi*1/14*BEGINNING$Day), cos(2*pi*1/14*BEGINNING$Day) # freq=1/14 (period=14 DAYS)
)
#YOU CAN CHANGE THE DAY AND 3 9 13 ETC
ggplot(data=subset(X, Days>500)) + geom_line(aes(x=Days, y=X[X$Days>500,3]))
ggplot(data=subset(X, Days>500)) + geom_line(aes(x=Days, y=X[X$Days>500,7]))
mod <- lm(y ~ . - Days, data = X)  # Regress y on everything (but days)
summary(mod)
X$resid <- residuals(mod)
X$pred <- predict(mod)
#YOU CAN CHANGE THE DAY
ggplot(data = subset(X, Days > 109)) + geom_line(aes(x = Days, y = y)) + geom_line(aes(x = Days, 
                                                                                 y = pred), color = "red")
#PICK
raw.spec <- spec.pgram(BEGINNING$TurkeyDeathsCases, taper = 0)
plot(raw.spec)

plot(raw.spec, log = "no")
# spec.df <- as.data.frame(raw.spec)
spec.df <- data.frame(freq = raw.spec$freq, spec = raw.spec$spec)
# Create a vector of periods to label on the graph, units are in days
yrs.period <- rev(c(7, 10, 14, 21, 28, 31, 62,93, 124, 178 ))
yrs.labels <- rev(c("7", "10", "14", "21", "28", "31", "62", "93","124", "178"))
yrs.freqs <- 1/yrs.period * 1/7  #Convert weekly period to weekly freq, and then to daily freq
spec.df$period <- 1/spec.df$freq
ggplot(data = subset(spec.df)) + geom_line(aes(x = freq, y = spec)) + scale_x_continuous("Period (Days)", 
                                                                                         breaks = yrs.freqs, labels = yrs.labels) + scale_y_continuous()
ggplot(data = subset(spec.df)) + geom_line(aes(x = freq, y = spec)) + scale_x_continuous("Period (Days)", 
                                                                                         breaks = yrs.freqs, labels = yrs.labels) + scale_y_log10()
ggplot(data = subset(spec.df)) + geom_line(aes(x = freq, y = spec)) + scale_x_log10("Period (Days)", 
                                                                                    breaks = yrs.freqs, labels = yrs.labels) + scale_y_log10()
plot(kernel("daniell", m = 10))  # A short moving average
k = kernel("daniell", c(9, 9, 9))
#PICK
smooth.spec <- spec.pgram(BEGINNING$TurkeyDeathsCases, kernel = k, taper = 0)
library(gridExtra)
k = kernel("daniell", c(9, 9))

smooth.spec <- spec.pgram(BEGINNING$TurkeyDeathsCases, kernel = k, taper = 0, plot = FALSE)

spec.df <- data.frame(freq = smooth.spec$freq, `0%` = smooth.spec$spec)
names(spec.df) <- c("freq", "0%")
# Add other tapers
spec.df[, "10%"] <- spec.pgram(BEGINNING$TurkeyDeathsCases, kernel = k, taper = 0.1, plot = FALSE)$spec
spec.df[, "30%"] <- spec.pgram(BEGINNING$TurkeyDeathsCases, kernel = k, taper = 0.3, plot = FALSE)$spec

spec.df <- melt(spec.df, id.vars = "freq", value.name = "spec", variable.name = "taper")
plot1 <- ggplot(data = subset(spec.df)) + geom_path(aes(x = freq, y = spec, 
                                                        color = taper)) + scale_x_continuous("Period (Days)", breaks = yrs.freqs, 
                                                                                             labels = yrs.labels) + scale_y_log10()

plot2 <- ggplot(data = subset(spec.df)) + geom_path(aes(x = freq, y = spec, 
                                                        color = taper)) + scale_x_log10("Period (Days)", breaks = yrs.freqs, labels = yrs.labels) + 
  scale_y_log10()
#PICK
grid.arrange(plot1, plot2)
install.packages("multitaper")
library(multitaper)
#dunits can be in "second", "hour", "day", "month", "year", dT can be changed to different values
#PICK
mt.spec <- spec.mtm(BEGINNING$TurkeyDeathsCases, nw = 16, k = 2 * 16 - 1, jackknife = TRUE, dtUnits = "day",Ftest = TRUE)
mt.spec <- spec.mtm(BEGINNING$TurkeyDeathsCases, nw = 16, k = 2 * 16 - 1, jackknife = TRUE,  dT = 0.03,
                    dtUnits = "day")
install.packages("dplR")
library(dplR)
wave.out <- morlet(y1 = BEGINNING$TurkeyDeathsCases, x1 = BEGINNING$Day, p2 = 9, dj = 0.1, siglvl = 0.95)

wave.out$period <- wave.out$period
levs <- quantile(wave.out$Power, c(0, 0.25, 0.5, 0.75, 0.95, 1))
wavelet.plot(wave.out, wavelet.levels = levs, crn.ylim = c(22.5, 30))
wave.avg <- data.frame(power = apply(wave.out$Power, 2, mean), period = (wave.out$period))
#PICK
plot(wave.avg$period, wave.avg$power, type = "l")

library(ts.extend)
#x2<-ts(rnorm(x))
X3<-ts(x)
x1<-as.numeric(x)
is.vector(x1)
TEST <- spectrum.test(x1)
plot(TEST)
fisher.g.test(x1)
fdr.out <- fdrtool(fisher.g.test(x1), statistic="pvalue")
sum(fdr.out$qval < 0.05) # tail area-based Fdr
sum(fdr.out$lfdr < 0.7)  # density-based local fdr
library("fdrtool")
TEST2 <- fdrtool(x1, statistic=c("normal", "correlation", "pvalue"),
        plot=TRUE, color.figure=TRUE, verbose=TRUE, 
        cutoff.method=c("fndr", "pct0", "locfdr"),
        pct0=0.75)
plot(TEST2$pval, type='l')
censored.fit(x1, cutoff, statistic=c("normal", "correlation", "pvalue", "studentt"))
cutoff<-  fndr.cutoff(x1, statistic=c("normal", "correlation", "pvalue", "studentt"))

