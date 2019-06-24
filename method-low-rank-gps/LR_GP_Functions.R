predict_loss<-function(gam_model,theta){
  x<-matrix(c(theta^2,1/theta^2),nrow=1)
  x<-data.frame(x)
  colnames(x)<-c("p1","p2","p3","p4","q1","q2","q3","q4")
  predict(gam_model,x)
}

emulate_loss<-function(new_data,Data_Matrix,param_Matrix,start_points,loss_method,knots=2000,max_it=1000){
  # create vector of differences
  y<-t(t(Data_Matrix)-new_data)
  inv_Cov_y<-solve(cov(Data_Matrix))
  if(loss_method == "Mahal"){
    output<-diag(y%*%inv_Cov_y%*%t(y))
  }
  else if(loss_method=="MSE"){
    output<-c()
    for(i in 1:nrow(y)){
      output[i]<-log(mean(y[i,]^2))
    }
  }
  else{
    print("Error: Incorrect loss_method specified")
  }
  # fit gam
  model<-gam(output ~ s(p1,p2,p3,p4,bs="gp",k=knots) + s(q1,q2,q3,q4,bs="gp",k=knots),data=param_Matrix)
  # optimise to find solution
  
  EL_List<-list()
  
  for(i in 1:nrow(start_points)){
    sp<-start_points[i,]
    EL_List[[i]]<-optim(sp,predict_loss,gam_model=model,control = list(maxit=max_it),method="CG")
  }
  EL_List      
}

predict25.v<-function(param,y,list_m,error_method,Cov_y){
  p1<-param[1]^2
  p2<-param[2]^2
  p3<-param[3]^2
  p4<-param[4]^2
  q1<-1/p1
  q2<-1/p2
  q3<-1/p3
  q4<-1/p4
  newd<-data.frame(p1=p1,p2=p2,p3=p3,p4=p4,q1=q1,q2=q2,q3=q3,q4=q4)
  predictions<-c(predict(list_m[[1]],newd),predict(list_m[[2]],newd),predict(list_m[[3]],newd),predict(list_m[[4]],newd),predict(list_m[[5]],newd),predict(list_m[[6]],newd),predict(list_m[[7]],newd),predict(list_m[[8]],newd),predict(list_m[[9]],newd),predict(list_m[[10]],newd),predict(list_m[[11]],newd),predict(list_m[[12]],newd),predict(list_m[[13]],newd),predict(list_m[[14]],newd),predict(list_m[[15]],newd),predict(list_m[[16]],newd),predict(list_m[[17]],newd),predict(list_m[[18]],newd),predict(list_m[[19]],newd),predict(list_m[[20]],newd),predict(list_m[[21]],newd),predict(list_m[[22]],newd),predict(list_m[[23]],newd),predict(list_m[[24]],newd),predict(list_m[[25]],newd))
  if(error_method=="MSE"){
    MSE<-c(log(mean((y-predictions)^2)))
    return(MSE=MSE) # log MSE
  }
  else if(error_method=="Mahal"){
    Mahal<-log(t(y-predictions)%*%solve(Cov_y)%*%(y-predictions))
    return(Mahal=Mahal)
  }
  else{
    print("ERROR: Incorrect error_method selected")
  }
}

LearnParam25.v<-function(param0,y,list_m,maxit=500,method="CG",error_method,Cov_y){
  if(error_method=="MSE"){
    opt<-optim(param0,predict25.v,method=method,control = list(maxit=maxit),y=y,list_m=list_m,error_method="MSE",Cov_y=Cov_y)
    return(list(opt=opt))
  }
  else if(error_method=="Mahal"){
    opt<-optim(param0,predict25.v,method=method,control = list(maxit=maxit),y=y,list_m=list_m,error_method="Mahal",Cov_y=Cov_y)
    return(list(opt=opt))
  }
  else{
    print("ERROR: Incorrect error_method selected")
  }
}

get_param_loss <- function(loss_list){
  selected <- NA
  value <- Inf
  for(i in 1:length(loss_list)){
    if(loss_list[[i]]$convergence == 0){
      if(loss_list[[i]]$value < value & all(loss_list[[i]]$par^2 > 0.1) & all(loss_list[[i]]$par^2 < 5)){
        value <- loss_list[[i]]$value
        selected <- i
      }
    }
  }
  return(loss_list[[selected]]$par^2)
}