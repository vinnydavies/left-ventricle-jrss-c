# Set Working Directories
base_dir = "C:/Users/Vinny/work/" # set this to the folder containing the GIT repository
data_dir = "left-ventricle-jrss-c/method-localgp/Simulations/Design4D/"
code_dir = "left-ventricle-jrss-c/method-low-rank-gps/"

# Load Libraries
if (!require("mgcv")) install.packages("mgcv")
if (!require("readr")) install.packages("readr")

# Load Data
X_train <- data.frame(read_csv(paste(base_dir,data_dir,"XTrain4D.csv",sep="")))
X_test <- data.frame(read_csv(paste(base_dir,data_dir,"XTest4D.csv",sep="")))
Y_train <- as.matrix(read_csv(paste(base_dir,data_dir,"YTrain4DStd.csv",sep="")))
Y_test <- as.matrix(read_csv(paste(base_dir,data_dir,"YTest4DStd.csv",sep="")))
X_train8 <- data.frame(X_train,1/X_train)
colnames(X_train8)<-c("p1","p2","p3","p4","q1","q2","q3","q4")

# Load Functions
source(paste(base_dir,code_dir,"LR_GP_Functions.R",sep=""))

# Train Model for Output Method
model_list<-list()
for(i in 1:25){
  m = gam(Y_train[,1] ~ s(p1,p2,p3,p4,bs="gp",k=2000) + s(q1,q2,q3,q4,bs="gp",k=2000),data=X_train8)
  model_list[[i]] = m
  print(paste("trained",i,"models"))
}

# general some starting points
n_starting_points = 50
starting_points = runif(n_starting_points*4,0.1,5)
starting_points2 = matrix(runif(n_starting_points*4),ncol=4)

# Low Rank GP, Output, Mean Squared Error (MSE), one starting point
output_mse_results = LearnParam25.v(sqrt(starting_points),y=Y_test[1,],list_m=model_list,maxit=1000,error_method="MSE",Cov_y=cov(cbind(Y_train))) 

# Low Rank GP, Output, Mahalanobis Distance, one starting pointn_starting_points = 10
output_mse_results = LearnParam25.v(sqrt(starting_points),y=Y_test[1,],list_m=model_list,maxit=1000,error_method="Mahal",Cov_y=cov(cbind(Y_train))) 

# Low Rank GP, Loss, Mean Squared Error (MSE), multiple starting points
loss_mse_results<-emulate_loss(Y_test[1,],Y_train,X_train8,start_point=starting_points2,loss_method="MSE")

# Low Rank GP, Loss, Mahalanobis Distance, multiple starting points
loss_mahal_results<-emulate_loss(Y_test[1,],Y_train,X_train8,start_point=starting_points2,loss_method="Mahal")

# Results
# Reccommend running more/multiple starting points for both methods in practise
param_output_mse <- output_mse_results$opt
param_output_mahal <- output_mahal_results$opt
param_loss_mse <- get_param_loss(loss_mse_results)
param_loss_mahal <- get_param_loss(loss_mahal_results)

