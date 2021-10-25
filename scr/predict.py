import pandas as pd
import numpy as np
from tensorflow import keras
from preprocess import unq_object_id, n, window_size, scaler, tensor, tensor_df_REAL, log_odn

model = keras.models.load_model('D:/Работа/ML_ODN/odn_model.h5')
log_odn.info('импортировали готовую модель')

pred_window = tensor[-window_size : tensor.shape[0], :].reshape(1, window_size, n)
log_odn.info('получили данные для предсказания')

preds = model.predict(pred_window)
log_odn.info('получили данные после применения модели')

trans_preds = scaler.inverse_transform(preds[0])
log_odn.info('вернули данные в исходный вид')

preds_values = pd.DataFrame(trans_preds)
preds_values.columns = unq_object_id
log_odn.info('предсказанные данные занесли в таблицу')

real_values = pd.DataFrame(tensor_df_REAL.reshape((1, n)))
real_values.columns = unq_object_id
log_odn.info('тестовые данные занесли в таблицу (для проверки точности модели)')

pred_real_df = pd.concat((preds_values, real_values))
pred_real_df.index = ['predicted', 'real']
pred_real_df = pred_real_df.T
log_odn.info('объединили данные в одну таблицу')

# считаем отклонение по формулам в зависимости от значений:
#     a < b = ((b-a)/a) * 100
#     a > b = ((a-b)/a) * 100
# если отсутствуют показания, то отклонение будет 0 (можно изменить)

pred_real_df['deviation'] = np.where(
    pred_real_df['predicted'] >= pred_real_df['real'], abs(100 * ((pred_real_df["real"] - pred_real_df["predicted"])/pred_real_df["real"])), np.where(
    pred_real_df['predicted'] < pred_real_df['real'], abs(100 * ((pred_real_df["predicted"] - pred_real_df["real"])/pred_real_df["predicted"])), 0))
pred_real_df["deviation"] = pred_real_df["deviation"].round()
log_odn.info('посчитали отклонение по квартилям')

print(pred_real_df[pred_real_df['deviation'] < 26])
print(pred_real_df.shape)
