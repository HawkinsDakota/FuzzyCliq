#SNN portion of SNN Cliq
#Will Hackett
#11.21.16
import re,os,sys,getopt
import numpy
import math
import csv

def usage():
    state="\n SNNCliq is a clustering algorithm \n\
    usage: SNN.py -e <input_matrix> -i <edgeFile> -k <neighbors>  \n\
    -e input_matrix has sample rows and gene columns \n\
    -i is output file for use with cliq.py \n\
    -k number of k nearest neighbors, 3 as default \n\
    -d method of measuring distance (functionality in progress current euclidian"
    print(state)

#KNN list creation made with reference to machinelearningmastery.com
def FileRead(input_matrix):
    with open(input_matrix, 'rt') as csvfile:
        samples=csv.reader(csvfile,delimiter=",")
        dataset=list(samples)
        lends=len(dataset)
        lendr=len(dataset[0])
        sampname=[]
        data=[[None for x in range(lendr-1)]for y in range(lends-1)]
        for i in range(1,lends):
            a=dataset[i][0]
            sampname.append(a)
            for j in range(1,lendr):
                data[i-1][j-1]=float(dataset[i][j])
    return data

def eucDist(vec1,vec2,length):
    distance=0
    for x in range(length):
        distance += pow((vec1[x]-vec2[x]),2)
    return math.sqrt(distance)

def distMatrix(dataset):
    lends=len(dataset)
    lendr=len(dataset[0])
    dMat=[[0 for x in range(lends)] for y in range(lends)]
    for j in range(lends):
        for l in range(lends):
            dMat[j][l]=eucDist(dataset[j],dataset[l],lendr)
    return dMat

def KNNFind(dMat,k):
    lends=len(dMat)
    KList=[[0 for x in range(k)] for y in range(lends)]
    for j in range(lends):
        neighborlist=sorted(range(lends),key=lambda k: dMat[j][k])
        for l in range(k):
            KList[j][l]=neighborlist[l+1] #the first value will be itself
    return KList

def sharedK(v1,v2):
    return list(set(v1).intersection(v2))

def matches(vec,ref):
    match=[]
    for i, x in enumerate(ref):
        for y in vec:
            if (x==y):
                match.append(float(i+1))
    return match

def EdgeList(KList,k):
    lenkl=len(KList)
    edge=[[0,0,0]]
    for i in range(lenkl):
        j=i
        while (j<(lenkl-1)):
            j=j+1
            shared=sharedK(KList[i],KList[j])
            if (len(shared)>0):
                kmi=matches(shared,KList[i])
                kmj=matches(shared,KList[j])
                s=[None for x in range(len(kmi))]
                for ii in range(len(kmi)):
                    s[ii]=float(k)-.5*(kmi[ii]+kmj[ii])
                strength=max(s)
                if (strength>0):
                    row=[[int(i+1),int(j+1),strength]]
                    edge=numpy.concatenate((edge,row),axis=0)
    return edge

def main(argv):
    print("SNN Starting")
    input_matrix=None
    edgeFile=None
    k=3

    try:
        opts, args=getopt.getopt(argv, "e:i:k:h", ["input=","output=","merging=","r-quasi-cliq=","number=","help"])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(0)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
        elif opt in ('-e','--input_matrix'):
            input_matrix=arg
        elif opt in ('-i', '--edge_file'):
            edgeFile=arg
        elif opt in ('-m','--merging'):
            k=int(arg)
        else:
            print("Error in arguments: unknown option\n")
            usage()
            sys.exit(0)
    if (edgeFile is None) or (edgeFile is None):
        sys.stderr.write("Error in arguments: must specify -i -o\n")
        usage()
    if (k<1) or not (isinstance(k,int)):
        sys.stderr.write("Error: k must be positive integer.\n")
        usage()
        sys.exit(0)

    data=FileRead(input_matrix)#get table
    simMat=distMatrix(data) #similarity matrix
    KList=KNNFind(simMat,k) #Neighbor list
    Edges=EdgeList(KList,k)#edge list creation
    numpy.savetxt(edgeFile,Edges, fmt = ['%d', '%d' ,'%1.3f'])


if __name__=="__main__":
    main(sys.argv[1:])
