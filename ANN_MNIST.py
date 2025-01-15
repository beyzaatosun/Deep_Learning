#%%
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure(figsize=(10,5))
for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"index: {i} , label: {y_train[i]}")
    plt.axis("off")
plt.show()


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype("float32")/255
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype("float32")/255

num_class=10
y_train = to_categorical(y_train,num_class)
y_test=to_categorical(y_test, num_class)

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(28*28,)))
model.add(Dense(256, activation="tanh"))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


#early stopping - eğer val_loss iyilesmiyorsa egitimi durdurmak
#monitor:dogrulama setindeki val kaybi izler
#patience:3 - 3 epoch boyunca val loss degismiyors erken durdurma yapalım
#restore_best_weights: en iyi modelin agirliklarini geri yükler
early_stopping= EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

#model_checkpoint: en iyi modelin agirliklarini kaydeder

checkpoint= ModelCheckpoint("ann_best_model.keras", monitor="val_loss",save_best_only=True)

history=model.fit(x_train, y_train,
          epochs=10,
          batch_size=60,#veri setini 60lı parçalar ile eğitim yapilacak
          validation_split=0.2,
          callbacks=[early_stopping,checkpoint])


test_loss, test_acc=model.evaluate(x_test,y_test)
print("test acc:", test_acc)
print("test_loss:", test_loss)

plt.figure()
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"],marker="o",label="Validation Accuracy")
plt.title("ANN Accuracy on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"],marker="o",label="Validation Loss")
plt.title("ANN Loss on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()




