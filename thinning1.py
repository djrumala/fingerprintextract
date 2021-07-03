import numpy as np

path = 'dilated1.png' #directory foto disimpan
X = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
X = X/255
[M, N] = np.shape(X)
Y=np.zeros((M,N,20))
Y[:,:,0]=X
# plt.imshow(Y[:,:,1], cmap='gray')
for k in range(1,20):
    Y[:,:,k]=np.copy(Y[:,:,k-1])
    plt.figure(1)
    plt.imshow(Y[:,:,k], cmap='gray')
    # plt.colorbar()
    plt.figure(2)
    plt.imshow(Y[:,:,k-1], cmap='gray')
    # plt.colorbar()
    for i in range(1, M-1):
        for j in range(1,N-1):
            # % * * * y11 y12 y13
            # % * @ * y21 y22 y23
            # % * * * y31 y32 y33
            y11=Y[i-1,j-1,k]
            y12=Y[i-1,j,k]
            y13=Y[i-1,j+1,k]
            y21=Y[i,j-1,k]
            y22=Y[i,j,k]
            y23=Y[i,j+1,k]
            y31=Y[i+1,j-1,k]
            y32=Y[i+1,j,k]
            y33=Y[i+1,j+1,k]
            # print(np.shape(y11))
            if y22==0:
                if y11==1 and y12==1 and y13==1 and y31==0 and y32==0 and y33==0:
                    Y[i,j,k]=1;
                    
                elif y11==1 and y13==0 and y21==1 and y23==0 and y31==1 and y33==0:
                    Y[i,j,k]=1 
                    
                elif y11==0 and y12==0 and y13==0 and y31==1 and y32==1 and y33==1 :
                    Y[i,j,k]=1 
                elif y11==0 and y13==1 and y21==0 and y23==1 and y31==0 and y33==1 :
                    Y[i,j,k]=1 
                elif y12==1 and y13==1 and y21==0 and y23==1 and y32==0 :
                    Y[i,j,k]=1 
                    
                elif y11==1 and y12==1 and y21==1 and y23==0 and y32==0 :
                    Y[i,j,k]=1 
                elif y12==0 and y21==1 and y23==0 and y31==1 and y21==1 :
                    Y[i,j,k]=1 
                elif y12==0 and y21==0 and y23==1 and y32==1 and y33==1 :
                    Y[i,j,k]=1 
    
                    # %extra
                elif y11==0 and y12==0 and y23==1 and y31==1 and y32==1 and y33==1 :
                    Y[i,j,k]=1 
            # else:
    if (Y[:,:,k]==Y[:,:,k-1]).all():
        print(k)
        break
Y=Y[:,:,k]

plt.figure(5)
plt.imshow(Y, cmap='gray')
plt.colorbar()