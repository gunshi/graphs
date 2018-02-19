import numpy
from matplotlib import pyplot as plt
from PIL import Image
import time



def oneapproach():
    basepath='/home/gunshi/mapillary-all/'
    train_data=[]
    running = False
    dictdata={}
    dictdata2={}
    with open('/home/gunshi/annstest/mapillary-posval.txt', 'r') as file1:
        train_data=[]
        for row in file1:
            temprow=row.split('/', 1)[0]
            temp=temprow.split()



            if(len(temp)>0 and temp[0][0]!='/'):
                train_data.append(temp)


    #every first and second line's first word is seq 
        for i in range(0,len(train_data),7):
            seq1 = train_data[i+1][0]
            seq2 = train_data[i+2][0]
            if(seq1 not in dictdata):
                dictdata[seq1]=[]
            if(seq2 not in dictdata):
                dictdata[seq2]=[]

    #load mapillary concat file also and make seq dict

    #search in this dict
    with open('/home/gunshi/mapillaryconcat-nopano.txt', 'r') as file1:

        counter=0
        close=True
        angleclose=True
        angledefaulter=0
        defaulter=0
        seqname = ''
        for row in file1:
            if row[0]=='#':
                if((defaulter<10 and angledefaulter<10 and counter>50)): #((angleclose or close) and counter>40)
                    print('')
                    print('....................................')
                    print(seqname)
                    print('total length: '+str(counter))
                    print('result odom: '+str(close))
                    print('result angle: '+str(angleclose))
                    print('defaulter '+str(defaulter))
                    print('angledefaulter '+str(angledefaulter))
                    #time.sleep(0.5)                                                                                                        
                temp=row.split()
                seqname = temp[1]
                dictdata2[temp[1]]=[]
                assert(temp[2]=='False')
                counter = 0
                running = False
                angledefaulter=0
                defaulter=0
                close=True
                angleclose=True
            else:
                temp=row.split()
                if len(temp)==0:
                    continue
                dictdata2[seqname].append(temp)
                counter += 1
                if(running and (abs(float(temp[2])-x)<0.00013) and (abs(float(temp[3])-y)<0.00013)):
                    #close=True
                    f=2
                else:
                    close=False
                    defaulter+=1

                if(running and (abs(float(temp[1])-angle)<8)):
                    #angleclose=True
                    f=2
                else:
                    angleclose=False
                    angledefaulter+=1

                if(not running):
                    if(counter==2):
                        running=True

                x=float(temp[2])
                y=float(temp[3])
                angle=float(temp[1])


#only check near intersections
def twoapproach():


#horz vert areas
def threeapproach():







"""
f, axarr = plt.subplots(4,6) 

for i in range(868,870,7):
    if(train_data[i+4][0]!='separate'):
        continue
    print('one iter')
    concatrel=' '.join(train_data[i+4])
    imgpaths1=[]
    imgpaths2=[]
    image_datas1=[]
    image_datas2=[]
    print(train_data[i+1][0])

    for j in range(1,len(train_data[i+1]),1):
        path1=basepath+train_data[i+1][0]+'/'+train_data[i+1][j]+'.jpg'
        print(path1)
        #imgpaths1.append(path1)
        i1=Image.open(path1)
        i1=i1.resize((224,224))
        image_datas1.append(i1)
    print('............')
    print(train_data[i+2][0])
    for j in range(1,len(train_data[i+2]),1):
        path2=basepath+train_data[i+2][0]+'/'+train_data[i+2][j]+'.jpg'
        print(path2)
        #imgpaths2.append(path2)
        i2=Image.open(path2)
        i2=i2.resize((224,224))
        image_datas2.append(i2)

    im1=[]
    im2=[]

    new_im = Image.new('RGB', (1120, 896))#896

    x_offset = 0
    y_offset = 0
    for im in image_datas1[0:11]:
        new_im.paste(im, (x_offset,y_offset))
        x_offset += 224
        if(image_datas1.index(im)==5):
            x_offset=0
            y_offset=448

    x_offset = 0
    y_offset = 224

    for im in image_datas2[0:11]:
        new_im.paste(im, (x_offset,y_offset))
        x_offset += 224
        if(image_datas2.index(im)==5):
            
            x_offset=0
            y_offset=672
    new_im.show(title='yes')
    new_im.save('same'+str(i)+'.png')

"""