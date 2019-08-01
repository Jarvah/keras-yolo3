# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
#function to do tracking
from sort import iou,convert_bbox_to_z, convert_x_to_bbox,KalmanBoxTracker, associate_detections_to_trackers, Sort
import os

#from deep_sort import nn_matching
#from deep_sort.detection import Detection
#from deep_sort.tracker import Tracker
#from tools import generate_detections as gdet
#from shapely import geometry
#from keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolov3.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.id_color = self.generate_id_color()
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def generate_id_color(self):
        colors=[]
        for i in range(32):
            colors.append(list(np.random.random(size=3) * 256))
        return colors

    def draw_box_PIL(self, image,box):
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        label=str(int(box[4]))
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        left,top,right,bottom=box[:4]
        print(label, (left, top), (right, bottom))
        #top=box[1]
        #left=box[0]
        #bottom=box[3]-box[1]
        #right = box[2]-box[0]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        # points_for_poly = geometry.box(left, top, right, bottom)
        # if poly_ground.intersects(points_for_poly):
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=self.colors[0]) #!!noted to change for multi-class
        # draw.rectangle(
        #    [tuple(text_origin), tuple(text_origin + label_size)],
        # fill=self.colors[c])
        #draw.text(text_origin, label, fill=self.colors[int(box[4]%32)], font=font)
        return image

    def draw_box_cv2(self, frame, box):
        import cv2
        text=str(box[4])
        top, left, bottom, right = box[0:4]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(frame.shape[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(frame.shape[0], np.floor(right + 0.5).astype('int32'))
        # points_for_poly = geometry.box(left, top, right, bottom)
        cv2.rectangle(frame, (left, top), (right, bottom), self.colors[0], 1)
        color=self.id_color[int(box[4]%32)]
        print(color)
        cv2.putText(frame, text, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

    def detect_image(self, image, frame, ppl_tracker):

        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        #font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        #thickness = (image.size[0] + image.size[1]) // 500
        #poly_ground = geometry.Polygon([[p[0], p[1]] for p in points])

        dets=[]
        #_points = np.array(points)
        #_points = _points.reshape((-1, 1, 2))
        #cv2.polylines(image, [_points], True, (255, 255, 0))
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class=='person':
                box = out_boxes[i]   #[y1,x1,y2,x2]
                y1,x1,y2,x2=box
                box=[x1,y1,x2,y2] #x1,y1,x2,y2
                score = out_scores[i]
                dets.append(box)
        dets=np.array(dets)

        trackers=ppl_tracker.update(dets)
        for d in trackers:
            image=self.draw_box_PIL(image,d)

        end = timer()
        print(end - start)
        return image, end-start

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):

    import cv2


    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    avg_time=0
    prev_time = timer()
    ppl_tracker=Sort()
    while True:
        return_value, frame = vid.read()
        t1 = timer()
        image = Image.fromarray(frame)
        image, pred_time = yolo.detect_image(image,frame,ppl_tracker)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        t2 = timer() - t1
        print(t2)
        if avg_time == 0:
            avg_time = t2
        else:
            avg_time = (avg_time + t2) / 2
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

