testsetgen<-function(samples,noDropFile,testFile,infoFile){
  genegroups<-20
  sampdiv<-floor(70/samples)
  generegs<-matrix(0,genegroups,samples)
  types<-c('u','d','b')
  for (i in 1:samples){
    generegs[,i]<-sample(types,genegroups,replace=TRUE)
  }
  background<-1
  new<-data.frame(matrix(0,(14400*background + 80*genegroups),(sampdiv*samples)))
  smplnm<-rep(0,samples*sampdiv)
  for (i in 1:samples){
    for (j in 1:sampdiv){
      smplnm[(j+sampdiv*(i-1))]<-paste(i,'-',j,sep='')
    }
  }
  colnames(new)<-smplnm
  genename<-rep(0,16000)
  for (i in 1:(14400*background)){
    genename[i]<-paste('backgnd',i,sep='')
  }
  for (i in 1:genegroups){
    for (j in 1:80){
      genename[(background*14400 + j +(80*(i-1)))]<-paste(i,'g-',j,sep='-')
    }
  }
  rownames(new)<-genename
  #matrix made
  for (j in 1:(sampdiv*samples)){
    new[1:(background*14400),j]<-rnorm(background*14400,mean=3,sd=1)
  } #fills background genes by column
  k=1 #gene group tracker
  for (j in ((14400*background)+1):length(new[,1])){
    for (i in 1:samples){
      if ((j-(14400*background))>(k*80)){
        k=k+1 #augment tracker when it hits a new group
      }
      if (generegs[k,i]=='d'){
        new[j,(1+(i-1)*sampdiv):(i*sampdiv)]<-rnorm(sampdiv,mean=1,sd=1)
      } #fill in row within sample within gene group
      if (generegs[k,i]=='b'){
        new[j,(1+(i-1)*sampdiv):(i*sampdiv)]<-rnorm(sampdiv,mean=3,sd=1)
      }
      if (generegs[k,i]=='u'){
        new[j,(1+(i-1)*sampdiv):(i*sampdiv)]<-rnorm(sampdiv,mean=5,sd=1)
      } #fill in row within sample within gene group
    }
  }
  #zgene<-rep(0,length(new[,1]))
  #for (i in 1:length(new[,1])){
  #  zgene[i]<-length(which(new[i,]==0))
  #} #check if all values are filled
  new[new<0]<-0 #remove values less than zero
  write.csv(new,noDropFile,row.names=TRUE,col.names = TRUE)


  newsum<-sum(colSums(new))
  newmean<-newsum/(dim(new)[1]*dim(new)[2])
  lmb=(-log(.75)/(newmean^2)) #get lambda

  p0<-exp(-lmb*(new*new)) #get p0 matrix
  check<-data.frame(matrix(runif(dim(new)[1]*dim(new)[2]),dim(new)[1],dim(new)[2]))
  thestrokes<-p0>=check
  new[which(thestrokes,arr.ind=TRUE)]<-0

  write.csv(new,testFile,row.names=TRUE,col.names = TRUE)

  info<-'INFO'
  for (i in 1:samples){
    info<-c(info,i,generegs[,i])
  }
  write(info,infoFile)
}