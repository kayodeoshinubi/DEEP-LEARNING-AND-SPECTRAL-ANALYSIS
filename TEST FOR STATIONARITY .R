library(readxl)
FRANCEBEGINNING1 <- read_excel("Downloads/FRANCEBEGINNING1.xlsx")

First_wave_deaths <- ts(FRANCEBEGINNING1$TurkeyDeathsCases) # THIS CHANGES BASE ON COUNTRY
lag.length = 25
Box.test(First_wave_deaths, lag=lag.length, type="Ljung-Box")
options(warn=-1)
library(tseries)
pp.test(First_wave_deaths)
adf.test(First_wave_deaths)
library(fBasics)
jarqueberaTest(First_wave_deaths) 
kpss.test(First_wave_deaths, null="Trend")
