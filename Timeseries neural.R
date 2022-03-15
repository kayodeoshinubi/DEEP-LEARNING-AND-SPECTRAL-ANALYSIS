library(readxl)

FRANCEBEGINNING1 <- read_excel("Downloads/FRANCEBEGINNING1.xlsx")
t <- seq(0,556)

x <- FRANCEBEGINNING1$TurkeyDeathsCases #this changes base on country



Index  <-data.frame(t, x)
install.packages("neuralnet")
require(neuralnet)
nn=neuralnet(x ~ t,data=Index, hidden=3,act.fct = "logistic",
             linear.output = FALSE)
plot(nn)
trainingRowIndex <- sample(1:nrow(Index), 0.6*nrow(Index))  # row indices for training data
train <- Index[trainingRowIndex, ]  # model training data
test  <- Index[-trainingRowIndex, ] 
Predict=compute(nn,test)
Predict$net.result
summary(nn)
print(nn)

