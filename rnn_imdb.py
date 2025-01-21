from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, auc

import kerastuner as kt
from kerastuner.tuners import RandomSearch

max_features = 10000  # En fazla 10.000 kelime
max_len= 100 #cümle başına kelime sınırı
(x_train, y_train),(x_test, y_test)=imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print(f"train shape: {x_train.shape}, Test shape: {x_test.shape}")

def build_model(hp):
    model = Sequential()
    #embedding kelimeleri vektörlere cevirir
    model.add(Embedding(input_dim=max_features,
                        output_dim=hp.Int("embedding_output", min_value=32, max_value=128, step=32),
                        input_length=max_len))
    model.add(SimpleRNN(units=hp.Int("rnn_units", min_value=32, max_value=128,step=32)))
    model.add(Dropout(rate=hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=hp.Choice("optimizer", ["adam","rmsprop"]),
                                     loss="binary_crossentropy",
                                     metrics=["accuracy","AUC"])
    return model

tuner = RandomSearch(
    build_model, #optimiz edilecek model fonksiyonu
    objective = "val_loss" , #en düsük olan en iyisidir
    max_trials=2, #2farklı model deneyecek
    executions_per_trial=1, #her model için 1 egitim denemesi
    directory="rnn_tuner_directory",
    project_name="rnn_imdb"
    
    )

early_stopping = EarlyStopping(monitor ="val_loss", patience=3, restore_best_weights=True)
tuner.search(x_train, y_train,
             epochs=2,
             validation_split=0.2,
             callbacks=[early_stopping])


best_model = tuner.get_best_models(num_models=1)[0]
loss, accuracy, auc = best_model.evaluate(x_test,y_test)
print(f"Test loss:{loss}, test accuracy:{accuracy:.3f}, test auc: {auc:.3f}")

y_pred_prob = best_model.predict(x_test)
y_pred = (y_pred_prob>0.5).astype("int32")
print(classification_report(y_test, y_pred))
