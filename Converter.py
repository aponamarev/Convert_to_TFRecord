"""
@misc{
  author = {Alexander Ponamarev},
  title = {Project Title},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{...}},
}

Class converts each data point into TFRecord format
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
from os.path import dirname, exists

class TFConvert(object):
    def __init__(self, file_path):
        assert exists(dirname(file_path)), "Dataset in TFRecord format will be save at {}\n".format(file_path)
        self.__path = file_path

        print("Dataset in TFRecord format will be save at {}\n".format(file_path))

        self.__writer = tf.python_io.TFRecordWriter(self.path)


    @property
    def path(self):
        return self.__path

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


    def convert_to_record(self, labels, bboxes, anchors):
        # check the inputs
        assert type(labels)==type(bboxes)==type(anchors)==list,\
            "Method accepts labels, bboxes, anchors as lists only. The following datatypes were provided {}, {}, {}".\
                format(type(labels).__name__, type(bboxes).__name__, type(anchors).__name__)
        assert len(labels) == len(bboxes) == len(anchors), \
            "Agrument size mismatch. Provided labels, bboxes, anchors should have the same number of elements. The following was provided {}, {}, {}". \
                format(len(labels), len(bboxes), len(anchors))

        bboxes_raw = [np.array(box).tostring() for box in bboxes]


        rec = tf.train.Example(
            features =tf.train.Features(
                feature = {
                'n_objects': self._int64_feature([len(labels)]),
                'labels': self._int64_feature(labels),
                'anchors': self._int64_feature(anchors),
                'box_raw_cxcywh': self._bytes_feature(bboxes_raw)
                }
            )
        )

        self.__writer.write(record=rec.SerializeToString())


    def read_labels_anchors_bboxes(self, filename_queue=None):

        #filename_queue = tf.train.string_input_producer([self.path])
        try:
            filename_queue = filename_queue or self.filename_queue
        except:
            self.filename_queue = tf.train.string_input_producer([self.path])
            filename_queue = self.filename_queue

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'n_objects': tf.FixedLenFeature([], tf.int64),
                'labels': tf.FixedLenFeature([], tf.int64),
                'anchors': tf.FixedLenFeature([], tf.int64),
                'box_raw_cxcywh': tf.FixedLenFeature([], tf.string)
            })

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        n_objects = tf.cast(features['n_objects'], tf.int16)
        labels = tf.cast(features['labels'], tf.int16)
        anchors = tf.cast(features['anchors'], tf.int16)
        box_raw_cxcywh = features['box_raw_cxcywh']
        decoded_bbox = tf.decode_raw(box_raw_cxcywh, tf.int64) #tf.uint8
        bbox_shape = tf.pack([n_objects, 4])
        decoded_bbox = tf.reshape(decoded_bbox, bbox_shape)
        bboxes = tf.cast(decoded_bbox, tf.int16)

        return labels, anchors, bboxes



    def close(self):
        self.__writer.close()

if __name__ == '__main__':
    from skimage import io

    converter = TFConvert(file_path='/Users/aponamaryov/GitHub/Convert_to_TFRecord/test.tfrecords')

    examples = [
        dict(l=[1], b=[[5,7,21,6]], a=[7],
             img=io.imread('/Users/aponamaryov/Downloads/coco_train_2014/images/COCO_train2014_000000000072.jpg')),
        dict(l=[1], b=[[5, 7, 21, 6]], a=[7],
             img=io.imread('/Users/aponamaryov/Downloads/coco_train_2014/images/COCO_train2014_000000000077.jpg')),
        dict(l=[1], b=[[5, 7, 21, 6]], a=[7],
             img=io.imread('/Users/aponamaryov/Downloads/coco_train_2014/images/COCO_train2014_000000000078.jpg'))
    ]

    for e in examples:

        converter.convert_to_record(labels=e['l'], bboxes=e['b'], anchors=e['a'])

    converter.close()

    labels, anchors, bboxes = converter.read_labels_anchors_bboxes()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print(labels.eval())
        print(anchors.eval())
        bboxes_result = bboxes.eval()
        print(len(bboxes_result, bboxes))
        coord.request_stop()
        coord.join(threads)





