import numpy as np
import math 
import cv2 
from matplotlib import pyplot as plt
from scipy import signal
from skimage import morphology
from thin1 import thinning1
from minu import minu
#THIS IS THE MAIN PROGRAM
# A REMINDER: WE ARE GOING TO HANDLE LOADS OF DATA. To validate and to compare the output values of arrays here and the matlab version more easily,
#I have learned that we can use this one line of code using pandas, it is so great to turn array to dataframe:
# ```pd.DataFrame.from_records(aveg1)
# If you want to see the whole cells values, it is better to convert your pandas dataframe to excel table
# ```df = pd.DataFrame.from_records(In)
# ```df.to_excel('fingervalid.xlsx')

path = 'D:/JCM ELEKTRO ITS/2nd Semester/Sistem Biometrika/fingerprint.jpeg' #directory foto disimpan
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
[m,n] = np.shape(image) #image shape to see the size and grayscale or rgb type

#IMAGE ENHANCEMENT
# -----------NORMALIZATION------------
M1=np.sum(image)/(m*n) #mean of all cells
var=np.sum(np.multiply((image-M1),np.subtract(image, M1)))/(m*n) #variance of all cells

m0=150 #can be set up as you like to get the best normalization
var0=50
T = np.zeros((m,n))
for x in range(0,m):
    for y in range(0,n):
        if image[x,y]>=M1:
            T[x,y]=m0+np.sqrt(var0*(image[x,y]-M1)*(image[x,y]-M1)/var)
        else:
            T[x,y]=m0-np.sqrt(var0*(image[x,y]-M1)*(image[x,y]-M1)/var)

#Showing Original image
plt.figure(1)
plt.imshow(image, cmap='gray') 
plt.clim(0,255) #defining colorbar range, so it will appear as image  the real color
plt.colorbar()
plt.title('Original image')

#Showing Normalized image
plt.figure(2)
plt.imshow(T, cmap='gray') 
plt.clim(0,255) #defining colorbar range, to make it appear 
plt.colorbar()
plt.title('Normalized image')

#-----------SEGMENTATION--------------
w=12 #size of window
H=m/w #number of blocks to x axis
L=n/w #number of blocks to y axis
aveg1=np.zeros((int(H), int(L)))
var1=np.zeros((int(H), int(L)))
for i in range(0,int(H)):
    for j in range(0,int(L)):
        block=T[i*w:i*w+w,j*w:j*w+w] #taking cells values for each block
        aveg1[i,j]=np.sum(block)/(w*w) #mean of each block
        var1[i,j]=np.sum(np.power((block-aveg1[i,j]),2))/(w*w) #variance of each block
    
Gmean= np.sum(aveg1)/(H*L) #average of all blocks mean
Vmean= np.sum(var1)/(H*L) #average of all blocks variance

TGf = np.sum(aveg1[aveg1>Gmean]) #sum of all blocks whose mean > Gmean
NGf = (aveg1>Gmean).sum() #number of blocks whose mean > Gmean
TVf = np.sum(var1[var1>Vmean]) #sum of all blocks whose variance > Vmean
NVf = (var1>Vmean).sum() #number of blocks whose variance > Vmean
Gf=TGf/NGf #relative mean of the foreground area
Vf=TVf/NVf #relative variance of the foreground area

TGb = np.sum(aveg1[aveg1>Gf]) #sum of all blocks whose mean > Gmean
NGb = (aveg1>Gf).sum() #number of blocks whose mean > Gmean
TVb = np.sum(var1[var1<Vf]) #sum of all blocks whose variance > Vmean
NVb = (var1<Vf).sum() #number of blocks whose variance > Vmean
Gb=TGb/NGb
Vb=TVb/NVb

#Defining background and foreground
grnd=np.zeros((int(H),int(L)))
for i in range(0,int(H)):
    for j in range(0,int(L)):
        if aveg1[i,j]>Gb and var1[i,j]<Vb: #background
            grnd[i,j]=1 

for x in range(1,int(H)-1):
    for y in range(1,int(L)-1):
        if grnd[x,y]==1:
            if grnd[x-1,y]+grnd[x-1,y+1]+grnd[x,y+1]+grnd[x+1,y+1]+grnd[x+1,y]+grnd[x+1,y-1]+grnd[x,y-1]+grnd[x-1,y-1]<=4:
                grnd[x,y]=0 #removing possible background, so now its foreground

I=np.copy(T)
Icc=np.ones((m,n))
for i in range(0,int(H)):
    for j in range(0,int(L)):
        if grnd[i,j]==1: #this is the background
            I[i*w:i*w+w,j*w:j*w+w]=0
            Icc[i*w:i*w+w,j*w:j*w+w]=0

plt.figure(3)
plt.imshow(I, cmap='gray') 
plt.clim(0,255) #defining colorbar range, so it will appear normal
plt.colorbar()
plt.title('Segmented image')

# ------ORIENTATION ESTIMATION-----
Gx=np.zeros((m,n))
Gy=np.zeros((m,n))
Gsx=np.zeros((m,n))
Gsy=np.zeros((m,n))
for i in range(1,m-1):
    for j in range(1,n-1):
        Gx[i,j]=(T[i+1,j]-T[i-1,j])
        Gy[i,j]=(T[i,j+1]-T[i,j-1])
        Gsx[i,j]=Gx[i,j]**2-Gy[i,j]**2
        Gsy[i,j]=2*Gx[i,j]*Gy[i,j]

Gbx=np.zeros((int(H), int(L)))
Gby=np.zeros((int(H), int(L)))
theta2=np.zeros((int(H), int(L)))
theta3=np.zeros((int(H), int(L)))
Phix=np.zeros((int(H), int(L)))
Phiy=np.zeros((int(H), int(L)))
OT=np.zeros((int(H), int(L)))

for i in range(0,int(H)):
    for j in range(0,int(L)):
        xblock=Gsx[i*w:i*w+w,j*w:j*w+w] #taking cells values for each block
        yblock=Gsy[i*w:i*w+w,j*w:j*w+w] #taking cells values for each block
        Gbx[i,j]=np.sum(xblock) #sum of each block
        Gby[i,j]=np.sum(yblock)
        theta3[i,j]= math.atan(Gby[i,j]/Gbx[i,j])/2
        if Gbx[i,j]>0:
            theta2[i,j]= math.pi/2+math.atan(Gby[i,j]/Gbx[i,j])/2
        if Gbx[i,j]<0 and Gby[i,j]>=0:
            theta2[i,j]= math.pi/2+(math.atan(Gby[i,j]/Gbx[i,j])+math.pi)/2
        if Gbx[i,j]<0 and Gby[i,j]<0:
            theta2[i,j]= math.pi/2+(math.atan(Gby[i,j]/Gbx[i,j])-math.pi)/2
# ---------ORIENTATION FIELD FILTERING----------
        Phix[i,j]=math.cos(2*theta2[i,j])
        Phiy[i,j]=math.sin(2*theta2[i,j])

def fspecial_gauss(shape=(3,3),sigma=0.5): #Constructing gaussian kernel like fspecial in MATLAB
    m,n = [(ss-1)/2 for ss in shape]
    y,x = np.ogrid[-n:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

f=fspecial_gauss((10,10),2) #apply gaussian filter with size=10 and sigma =2
Phix = signal.correlate2d(Phix,f, mode='same')
Phiy = signal.correlate2d(Phiy,f, mode='same')

for x in range(0,int(H)):
    for y in range(0,int(L)):
        OT[x,y]=1/2*math.atan2(Phiy[x,y],Phix[x,y]) #hold the magnitude of each block

plt.figure(1)
for i in range(0,int(H)):
    for j in range(0,int(L)):
        y0 = i*w + w/2
        x0 = j*w + w/2
        y1 = y0+w/2*math.cos(OT[i,j])
        x1 = x0+w/2*math.sin(OT[i,j])
        x=[x0,x1]
        y=[y0,y1]
        plt.plot(x,y, 'r', linewidth=1)
# plt.imshow(image, cmap='gray')
plt.title('Orientation Estimated') 

#MINUTIAE EXTRACTION
#---------BINARIZATION-----------
temp = np.asarray([[1,1,1],[1,1,1],[1,1,1]])*1/9
Im=np.zeros((m,n))
for a in range (1,m-1):
    for b in range (1,n-1):
        Im[a,b]=I[a-1,b-1]*temp[0,0]+I[a-1,b]*temp[0,1]+I[a-1,b+1]*temp[0,2]+I[a,b-1]*temp[1,0]+I[a,b]*temp[1,1]+I[a,b+1]*temp[1,2]+I[a+1,b-1]*temp[2,0]+I[a+1,b]*temp[2,1]+I[a+1,b+1]*temp[2,2]
I=np.copy(Im)
for x in range(4,m-5):
    for y in range(4,m-5):
        sum1=I[x,y-4]+I[x,y-2]+I[x,y+2]+I[x,y+4]
        sum2=I[x-2,y+4]+I[x-1,y+2]+I[x+1,y-2]+I[x+2,y-4]
        sum3=I[x-2,y+2]+I[x-4,y+4]+I[x+2,y-2]+I[x+4,y-4]
        sum4=I[x-2,y+1]+I[x-4,y+2]+I[x+2,y-1]+I[x+4,y-2]
        sum5=I[x-2,y]+I[x-4,y]+I[x+2,y]+I[x+4,y]
        sum6=I[x-4,y-2]+I[x-2,y-1]+I[x+2,y+1]+I[x+4,y+2]
        sum7=I[x-4,y-4]+I[x-2,y-2]+I[x+2,y+2]+I[x+4,y+4]
        sum8=I[x-2,y-4]+I[x-1,y-2]+I[x+1,y+2]+I[x+2,y+4]
        sumi=[sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8]
        summax=max(sumi)
        summin=min(sumi)
        b=np.sum(sumi)/8
        if (summax+summin+4*I[x,y])>(3*b):
            sumf = summin
        else:
            sumf = summax;
        if sumf > b: #threshold, this is fg
            Im[x,y]=128
        else: #threshold, this is bg
            Im[x,y]=255

for i in range(0,m):
    for j in range(0,n):
        Icc[i,j]=Icc[i,j]*Im[i,j] #combined segmented background and image with treshhold
        if Icc[i,j]==128: 
            Icc[i,j]=0 #enhance the foreground
        else: 
            Icc[i,j]=1 #remove segmented background and the main background

plt.figure(4)
plt.imshow(Icc, cmap='gray')
plt.title('Binarized Image')
plt.colorbar()

Icc = Icc.astype(bool) #Array converted to bool, for the need of small objects removal
Icc = morphology.remove_small_objects(Icc, 80) #like bwareopen in matlab, to fill the hole whose area is less than 80 pixels
Icc = np.invert(Icc) #inverting each cell value
Icc = morphology.remove_small_objects(Icc, 80)
Icc = np.invert(Icc)
Icc = np.uint8(Icc)
kernel = np.ones((2,2),np.uint8)
Icc = cv2.dilate(Icc, kernel, iterations=1) 

plt.figure(5)
plt.imshow(Icc, cmap='gray')
plt.colorbar()
plt.title('Dilated image')

#-------------THINNING-----------------
Im=np.copy(Icc);
In=np.copy(Im);
for a in range(0,4):
    for i in range(1, m-1):
        for j in range(1, n-1):
            if Im[i,j]==1:
                if Im[i-1,j] + Im[i-1,j+1] +Im[i,j+1] + Im[i+1,j+1]+ Im[i+1,j] + Im[i+1,j-1] + Im[i,j-1] + Im[i-1,j-1] <=3:
                    In[i,j]=0
            if Im[i,j]==0:
                if Im[i-1,j] + Im[i-1,j+1] +Im[i,j+1] + Im[i+1,j+1]+ Im[i+1,j] + Im[i+1,j-1] + Im[i,j-1] + Im[i-1,j-1] >=7:
                    In[i,j]=1
    
    Im=In

Icc = thinning1(Icc)
plt.figure(6)
plt.imshow(Im, cmap='gray')
plt.colorbar()
plt.title('Im')

plt.figure(7)
plt.imshow(Icc, cmap='gray')
plt.colorbar()
plt.title('Image after Thinning Process')


Mi=np.zeros((m,n))
Mi=minu(Icc)

plt.figure(8)
plt.imshow(Icc, cmap='gray')
plt.colorbar()
plt.title('Minutiae Extracted Image')

# False Minutiae 
d=0
for i in range (m): 
    for j in range (n): 
        if Mi[i,j]==1 or Mi[i,j]==2:
            d=20 
        if i<d or i+d>m or j<d or j+d>n:
            Mi[i,j]=0
 
for x in range (m): 
    for y in range (n): 
        if Mi[x,y]==1 or Mi[x,y]==2:
            for i in range (m): 
                for j in range (n): 
                    if Mi[i,j]==1 or Mi[i,j]==2: 
                        if Mi[x,y]==Mi[i,j]: 
                            d=10 
                        else: 
                            d=5 
                            a=((x-i)**2+(y-j)**2)**0.5 
                            if a<d and a>0: 
                                Mi[x,y]=0
                                Mi[i,j]=0
    
for x in range (m): 
    for y in range (n): 
        if Mi[x,y]==1: 
            row=round((x-1)/12)+1 
            col=round((y-1)/12)+1 
            plt.plot(y, x, 'r.', markersize = 5)
        elif Mi[x,y]==2: 
            row = round((x - 1) / 12) + 1 
            col = round((y - 1) / 12) + 1 
            plt.plot(y, x, 'b.', markersize = 5)

plt.show()