import xml.etree.ElementTree as ET
import os

#sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val')]
#sets=[('2007','test')]
#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes=["person"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id, list_file,rootpath):
    in_file = open(rootpath+'/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    contain=False
    boxes=[]


    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text), cls_id)
        boxes.append(b)
        #b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
        #     float(xmlbox.find('ymax').text))
        #bb = convert((w, h), b)
        #list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        contain=True
    return contain, boxes

#wd = getcwd()
root='/home/waiyang/crowd_counting/Dataset'
for year, image_set in sets:
    image_ids = open(root+'/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s_new.txt'%(year, image_set), 'w')
    for image_id in image_ids:

        contain, boxes=convert_annotation(year, image_id, list_file,root)
        if contain:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (root, year, image_id))
            for b in boxes:
                list_file.write(" " + ",".join([str(a) for a in b[:-1]]) + ',' + str(b[4]))
            list_file.write('\n')
    list_file.close()
os.system("cat 2007_train_new.txt 2007_val_new.txt 2012_train_new.txt 2012_val_new.txt > train_voc_ppl.txt")
#os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train_voc_ppl.txt")
