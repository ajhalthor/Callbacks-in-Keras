import os
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

# Path to Computation graphs
LOGDIR = './graphs_4'

# start session
sess = tf.Session()

#Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 1000
EPOCHS = 10

# Layers
HL_1 = 1000
HL_2 = 500

# Other Parameters
INPUT_SIZE = 28*28
N_CLASSES = 10

# https://github.com/fchollet/keras/issues/1064
"For forward layers, activations can either be used through an Activation layer or through the activation argument"

model = Sequential()
model.add(Dense(1000, input_dim=INPUT_SIZE, activation="relu"))
#model.add(Activation(activation="relu"))
model.add(Dense(500, activation="relu"))
#model.add(Activation("relu"))
model.add(Dropout(rate=0.9))
model.add(Dense(10, activation="softmax"))

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])


# train_tensorboard = TensorBoard(log_dir="./log_dir", write_graph=True)

class LossHistory(TensorBoard):

    def __init__(self):
	    # self.writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"))
	    self.log_dir = "./log_dir"
	    self.my_writer = tf.summary.FileWriter(self.log_dir) # Created here as site-packages/keras/callback.py
	    self.step = 0
	    super().__init__(self.log_dir)

    def on_batch_begin(self, batch, logs={}):
        self.losses = []
        self.accuracies = []

    def on_epoch_begin(self, epoch, logs={}):
    	self.epoch = epoch
    	print("self.epoch = ", self.epoch)
    	super().on_epoch_begin(epoch, logs)

    def on_batch_end(self, batch, logs={}):
        """
        Called after every batch with 
        logs.items() = dict_items([('batch', 38), ('size', 1000), ('loss', 0.46829113), ('acc', 0.87)])
        """

        print("batch = ", batch)
        #self.losses.append(logs.get('loss'))
        #self.accuracies.append(logs.get('acc'))  

        # print(logs.items())

        for name, value in logs.items():

            #print("name = ", name, " val = ", value)

            if name in ['acc', 'loss']:
                summary = tf.Summary()
                summary_value = summary.value.add() #Empty
                summary_value.simple_value = value.item() # 0.87
                summary_value.tag = name #Can say if "acc", tag = "accuracy" for more defined tags on the tensorboard
                self.my_writer.add_summary(summary, self.step)
    
        self.my_writer.flush()
        self.step += 1


#cb = LossHistory() #LossHistory()
cb = TensorBoard() #LossHistory()

model.fit(
	x=mnist.train.images, 
	y=mnist.train.labels, 
	epochs=EPOCHS, 
	batch_size=BATCH_SIZE,
	verbose=1,
	callbacks=[cb])


score = model.evaluate(
	x=mnist.test.images,
	y=mnist.test.labels)


print("score = ", score)

