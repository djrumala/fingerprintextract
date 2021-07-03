import numpy as np

def minu(X):
    [M, N] = np.shape(X)
    Y=np.zeros((M,N))
    Y[:,:]=X
    mi=np.zeros((M,N))

    for i in range(1, M-1):
        for j in range(1,N-1):
            # % * * * y11 y12 y13
            # % * @ * y21 y22 y23
            # % * * * y31 y32 y33
            y11=Y[i-1,j-1]
            y12=Y[i-1,j]
            y13=Y[i-1,j+1]
            y21=Y[i,j-1]
            y22=Y[i,j]
            y23=Y[i,j+1]
            y31=Y[i+1,j-1]
            y32=Y[i+1,j]
            y33=Y[i+1,j+1]
            if y22==0:
                if y11+y12+y13+y23+y33+y31+y21+y32==7:
                    mi[i,j]=1
                    
                elif y11==0  and y12==1  and y13==0  and y21==1  and y23==1  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==1  and y23==1  and y31==0  and y32==1  and y33==0:
                    mi[i,j]=2
                elif y11==0  and y12==1  and y13==1  and y21==1  and y23==0  and y31==0  and y32==1  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==1  and y13==0  and y21==0  and y23==1  and y31==1  and y32==1  and y33==0:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==0  and y23==0  and y31==1  and y32==1  and y33==1:
                    mi[i,j]=2
                    
                elif y11==1  and y12==1  and y13==1  and y21==0  and y23==0  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==0  and y23==1  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==1  and y23==0  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==0  and y12==1  and y13==0  and y21==1  and y23==1  and y31==0  and y32==1  and y33==1:
                    mi[i,j]=2
                elif y11==0  and y12==1  and y13==1  and y21==1  and y23==1  and y31==0  and y32==1  and y33==0:
                    mi[i,j]=2
                    
                elif y11==0  and y12==1  and y13==0  and y21==1  and y23==1  and y31==1  and y32==1  and y33==0:
                    mi[i,j]=2
                elif y11==1  and y12==1  and y13==0  and y21==1  and y23==1  and y31==0  and y32==1  and y33==0:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==0  and y23==0  and y31==0  and y32==1  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==0  and y23==0  and y31==1  and y32==1  and y33==0:
                    mi[i,j]=2
                elif y11==0  and y12==1  and y13==1  and y21==0  and y23==0  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                    
                elif y11==1  and y12==1  and y13==0  and y21==0  and y23==0  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==1  and y13==0  and y21==0  and y23==1  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==1  and y23==0  and y31==0  and y32==1  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==0  and y23==1  and y31==1  and y32==1  and y33==0:
                    mi[i,j]=2
                elif y11==0  and y12==1  and y13==1  and y21==1  and y23==0  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                    
                elif y11==1  and y12==0  and y13==1  and y21==1  and y23==0  and y31==0  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==0  and y12==0  and y13==1  and y21==1  and y23==0  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==0  and y21==0  and y23==1  and y31==1  and y32==0  and y33==1:
                    mi[i,j]=2
                elif y11==1  and y12==0  and y13==1  and y21==0  and y23==1  and y31==1  and y32==0  and y33==0:
                    mi[i,j]=2

    Y=mi[:,:]
    return Y