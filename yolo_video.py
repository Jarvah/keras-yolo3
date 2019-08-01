import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time

def detect_img(yolo):

    img_dir = '/home/waiyang/crowd_counting/Dataset/test_image_20190527'
    #img_dir='/home/waiyang/crowd_counting/faster_rcnn/video/images'
    #img_dir='/home/waiyang/crowd_counting/Dataset/MOT17det/train/MOT17-13/img1'
    output_dir = '/home/waiyang/crowd_counting/keras-yolo3/tiny_yolo_mot_output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    start = time.time()
    print(start)
    num = 0
    pred_times = []
    for Infolder in glob.glob(os.path.join(img_dir, '*')):
        #video_name = output_dir+'/'+Infolder.replace(img_dir,'').replace('/','')+'.avi'
        #print(video_name)

        #video = cv2.VideoWriter(video_name, 0, 1, (640, 480))
        for inInfolder in glob.glob(os.path.join(Infolder, '*')):
            for inImg in glob.glob(os.path.join(inInfolder, '*.jpg')):
    #for inImg in glob.glob(os.path.join(img_dir,'*jpg')):
                try:
                    image=Image.open(inImg)
                    #image=cv2.imread(inImg)
                    #img = np.array(image)
                    #coordinateStore1 = CoordinateStore(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    #if count == 0:

                    #    cv2.namedWindow('test draw')
                    #    cv2.setMouseCallback('test draw', coordinateStore1.line_drawing)
                    #    while (1):
                    #        cv2.imshow('test draw', coordinateStore1.img)
                    #        if cv2.waitKey(1) & 0xFF == 27:
                    #            break
                    #    cv2.destroyAllWindows()
                    #    all_points = coordinateStore1.points
                    #image=np.array(image)
                    r_image, pred_time = yolo.detect_image(image)
                    pred_times.append(pred_time)
                    #r_image.show()
                    #result = np.asarray(r_image)
                    r_image.save(output_dir + Infolder.replace(img_dir, '') + inImg[:-4].replace(inInfolder, '') + "_output.jpg")
                    #r_image.save(output_dir+inImg[:-4].replace(img_dir,'')+"_out.jpg")
                    num += 1
                    #cv2.imwrite(
                    #    output_dir + Infolder.replace(img_dir, '') + inImg[:-4].replace(inInfolder, '') + "_output.jpg",
                    #    result)
                    #video.write(result)
    end = time.time() - start
    avg_time = end / num
    print("average time: ", avg_time)
    avg_pred_time = sum(pred_times) / len(pred_times)
    print("avg_pred_time: ", avg_pred_time)
    merge_images(output_dir)
    yolo.close_session()

def merge_images(raw_fold):
    for inFolder in os.listdir(raw_fold):

        video_name = raw_fold + '/' + inFolder + '.avi'
        img_folder_name = raw_fold + '/' + inFolder
        make_video(img_folder_name, video_name)
        print(raw_fold, inFolder)

def make_video(image_folder, video_name):

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()




FLAGS = None

if __name__ == '__main__':

    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    drawing=False # true if mouse is pressed
    class CoordinateStore:
        def __init__(self,
                     img):
            self.points = []
            self.img = img

        def line_drawing(self, event, x, y, flags, param):
            global pt1_x, pt1_y, drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                pt1_x, pt1_y = x, y
                self.points.append((x, y))

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing == True:
                    cv2.line(self.img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=3)
                    pt1_x, pt1_y = x, y
                    self.points.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                cv2.line(self.img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=3)

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
