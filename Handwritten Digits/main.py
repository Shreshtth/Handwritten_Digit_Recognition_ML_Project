import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test)=mnist.load_data()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation='relu'))
model.add(tf.keras.layers.Dense(units=128,activation='relu'))
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model.fit(x_train,y_train,epochs=10)
model.save("handwritten.model")
loss,accuracy=model.evaluate(x_test,y_test)
print(loss)
print(accuracy)

imageno=1
while os.path.isfile(f"Testdigits/digits{imageno}.png"):
    try:
        img=cv2.imread(f"Testdigits/digits{imageno}.png")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"This digit is probably {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        imageno+=1


