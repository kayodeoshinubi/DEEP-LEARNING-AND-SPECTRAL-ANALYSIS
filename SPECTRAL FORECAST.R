install.packages("Rssa")
library(Rssa)
BEGINNING <- read_excel("Downloads/FRANCEBEGINNING1.xlsx")
x <- BEGINNING$TurkeyDailyCase
s1 <- ssa(x, L=12)
plot(s1)
plot(s1, type = "vectors", idx=1:6)
plot(s1, type = "series", groups = as.list(1:6))
res1 <- reconstruct(s1, groups = list(1))
trend <- res1$F1
plot(res1, add.residuals = FALSE, plot.type = "single",col = c("black", "red"), lwd = c(1, 2))
cases<- residuals(res1)
#PICK
spec.pgram(cases, detrend = FALSE, log = "no")
#PICK
# Produce 24 forecasted values of the series using different sets of eigentriples
# as a base space for the forecast.
rfor <- rforecast(s1, groups = list(c(1,4), 1:4), len = 50, only.new=FALSE)
matplot(data.frame(c(x, rep(NA, 50)), rfor), type = "l")

par(mfrow=c(1,2))
spec.pgram(cases, detrend = FALSE, log = "no")
matplot(data.frame(c(x, rep(NA, 50)), rfor), type = "l")
par(mfrow=c(1,1))
