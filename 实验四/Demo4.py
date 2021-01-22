import time

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

start = time.perf_counter()

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 1.将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape)
# 分出验证集
validation_images = train_images[49000:50000, :]
validation_labels = train_labels[49000:50000, :]
train_images = train_images[0:49000, :]
train_labels = train_labels[0:49000, :]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print(train_labels[0][0])


def draw_figure():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # 由于 CIFAR 的标签是 array，
        # 因此您需要额外的索引（index）。
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


# 2.构建模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 2.2.增加Dense层
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# 3.编译并训练模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(validation_images, validation_labels))

# 4.计算在测试集上的准确率
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

end = time.perf_counter()

print("运行的时间为：" + str((end - start) / 60) + "分钟")

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0, 2])
plt.legend(loc='lower right')
plt.show()
