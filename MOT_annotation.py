import glob
import os

classes=['1','2','7']
val='MOT17-13'

data_dir='/home/waiyang/crowd_counting/Dataset/MOT17det/train'
train_data='/home/waiyang/crowd_counting/keras-yolo3/MOT_train.txt'
val_data='/home/waiyang/crowd_counting/keras-yolo3/MOT_val.txt'
MOT_sets={}

for MOTset in glob.glob(os.path.join(data_dir,"*")):
    MOT_dict={}
    MOT_name=MOTset.replace(data_dir,'').replace('/','')
    if MOT_name!=val:
        continue
    for img in glob.glob(os.path.join(MOTset,'img1','*jpg')):

        img_info={}
        img_index=img[:-4].replace(os.path.join(MOTset,'img1'),'').replace('/','').lstrip('0')

        img_info['location']=img
        img_info['bbox']=[]
        MOT_dict[img_index] = img_info

    with open (os.path.join(MOTset,'gt','gt.txt'),'r') as f:
        lines=f.readlines()
        for line in lines:
            data=line.split(',')
            if data[7] in classes:
                bbox={}
                bbox['box']=data[2:6]
                bbox['class']='0'

                MOT_dict[data[0]]['bbox'].append(bbox)
    print(MOT_dict)


    MOT_sets[MOT_name]=MOT_dict

with open(val_data,'w') as outfile:
    for k in MOT_sets:
        MOT_dict=MOT_sets[k]
        for i in MOT_dict:
            outfile.write(MOT_dict[i]['location']+' ')

            for bbox in MOT_dict[i]['bbox']:
                for b in bbox['box']:
                    outfile.write(b+',')
                outfile.write(bbox['class'])
                outfile.write(' ')
            outfile.write('\n')
