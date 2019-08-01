import tensorflow as tf
from tensorflow.summary import FileWriter

sess = tf.Session()
tf.train.import_meta_graph("C:/Users/joh3146/Documents/dssg/air_pollution_estimation/data/ref/detection_model/yolov3_tf/yolov3.ckpt.meta")
FileWriter("__tb", sess.graph)


if __name__ == '__main__':
    print('yay')