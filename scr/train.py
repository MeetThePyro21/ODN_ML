import tensorflow as tf
from tensorflow import keras
from preprocess import n, target_size, X_train, y_train, X_val, y_val, log_odn

# входной слой (реализует операцию: input)
# 3 слоев lstm
# слой dense (реализует операцию: output)
#
# return_sequences: Следует ли возвращать последний вывод. в выходной последовательности или в полной последовательности

input_ = keras.layers.Input(shape=[None, n])
log_odn.info('создали входной слой')

layer1 = keras.layers.LSTM(n, return_sequences=True, dropout=0.2)(input_)
layer2 = keras.layers.LSTM(n, return_sequences=True, dropout=0.4)(layer1)
layer3 = keras.layers.LSTM(n, return_sequences=True, dropout=0.4)(layer2)
log_odn.info('создали скрытые слои')

dense_layer = keras.layers.Dense(n)(layer1[:, -target_size:, :])
log_odn.info('создали выходной слой')

model = keras.Model(inputs=[input_], outputs=[dense_layer])
log_odn.info('объединили в однц модель')

# функция потерь
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001, name='Adam')
log_odn.info('создали оптимайзер (learning_rate, функция потерь)')

model.compile(loss="mae", optimizer=optimizer)

history = model.fit(X_train, y_train, epochs=10, batch_size=10,
                    validation_data=(X_val, y_val))
log_odn.info('обучили модель')

model.save('D:/Работа/ML_ODN/odn_model.h5')
log_odn.info('сохранили модель')


