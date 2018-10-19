import yaml
import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os


class detector(object):

    def __init__(self, params):
        self.config_params = params
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.gpu_num = int(params['gpu_num'])
        self.thresh_score = float(params['thresh_score'])
        self.thresh_iou = float(params['thresh_iou'])
        self.is_image, self.target_path = self._get_detect_target()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self._generate()

    def _get_class(self):
        with open(self.config_params['classes_path']) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        with open(self.config_params['anchors_path']) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_detect_target(self):
        image_path = self.config_params['image_path']
        video_path = self.config_params['video_path']
        if os.path.isfile(image_path):
            is_image = True
            path = image_path
        elif os.path.isfile(video_path):
            is_image = False
            path = video_path
        else:
            assert False, 'Input file not exists.'
        return is_image, path

    def _generate(self):
        model_path = self.config_params['model_path']
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            if is_tiny_version:
                self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            else:
                self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.thresh_score, iou_threshold=self.thresh_iou)
        return boxes, scores, classes

    def _detect_frame(self, image):
        start = timer()
        boxed_image = letterbox_image(image, (416, 416))
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                            feed_dict={
                                                                self.yolo_model.input: image_data,
                                                                self.input_image_shape: [image.size[1], image.size[0]],
                                                                K.learning_phase(): 0
                                                            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        end = timer()
        print('Detected after {:.2f} s'.format(end - start))
        return image

    def _close_session(self):
        self.sess.close()

    def _detect_video(self):
        import cv2
        capture = cv2.VideoCapture(self.target_path)
        if not capture.isOpened():
            raise IOError("Couldn't open webcam or video")
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = capture.read()
            image = Image.fromarray(frame)
            image = self._detect_frame(image)
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
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._close_session()

    def _detect_image(self):
        try:
            image = Image.open(self.target_path)
        except:
            assert False, "Couldn't open image"
        else:
            r_image = self._detect_frame(image)
            r_image.show()
        self._close_session()

    def detect(self):
        if self.is_image:
            self._detect_image()
        else:
            self._detect_video()

## -------------------------------------------------------------------------------
if __name__ == '__main__':
    # Read config file
    with open('./config.yml') as f:
        params = yaml.load(f)
        detector = detector(params)
        detector.detect()
