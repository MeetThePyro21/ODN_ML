import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import requests

logging.basicConfig(level=logging.INFO)
log_odn = logging.getLogger('odn_info')

df = pd.DataFrame(requests.get(url='http://172.25.170.245:8200/api/PSK/GetCommonPSKData/1281').json())
log_odn.info('получили датасет')

df = df[['object_id', 'value', 'datetime']]
df = df.sort_values(["object_id", "datetime"], ascending=(True, True)).reset_index(drop = True)
df['datetime'] = pd.to_datetime(df['datetime'])
log_odn.info('обработали датасет')

unq_object_id = np.unique(df['object_id'])
log_odn.info('получили список уникальных id')

min_date = df['datetime'].min()
log_odn.info('получили минимальную дату из датасета')

max_date = df['datetime'].max()
log_odn.info('получили максимальную дату из датасета')

dates = np.arange(min_date, np.datetime64('2021-07'), dtype='datetime64[M]')

m = len(dates)
n = len(unq_object_id)

tensor_df = pd.DataFrame(np.zeros(shape=(m, n)))
tensor_df.index = dates
tensor_df.columns = unq_object_id

for i, ser in enumerate(unq_object_id):
    data = df[df['object_id'] == ser]
    index = data['datetime'].values
    exist_index = [idx for idx in index if idx in dates]
    vals = data[data['datetime'].isin(exist_index)]['value'].values
    tensor_df.loc[exist_index, ser] = vals

# real values for predict test
tensor_df_REAL = tensor_df.iloc[-1:].values
log_odn.info('получили тензор с реальными значениями за последний месяц')

# without last row
tensor_df = tensor_df[:-1]
log_odn.info('получили тензор для обучения, валидации модели')

tensor = tensor_df.to_numpy()

scaler = StandardScaler()
tensor = scaler.fit_transform(tensor)
log_odn.info('нормализовали значения с помощью StandardScaler')


# 44 по 10
# с 0 по 44
# со 1 по 45
# ...
# с 10 по 54
window_size = 10
target_size = 1
batch_size = tensor.shape[0] - window_size - target_size + 1

data_train = np.zeros(shape=(batch_size, window_size, n), dtype=np.float64)
targets = np.zeros(shape=(batch_size, target_size, n), dtype=np.float64)
for i in range(batch_size):
    window_x = tensor[i: window_size + i, :]
    window_y = tensor[window_size + i: window_size + i + target_size, :]
    data_train[i, :, :] = window_x
    targets[i, :, :] = window_y

log_odn.info('преобразовали данные для обучения')

val_share = 0.1
val_size = round(targets.shape[0] * val_share)

X_train, X_val = data_train[val_size:], data_train[:val_size]
y_train, y_val = targets[val_size:], targets[:val_size]
log_odn.info('разбили данные на тренировочные и валидационные')




