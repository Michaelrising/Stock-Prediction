from mega_model import *
from utils import *
import pandas as pd
from cv import PurgedGroupTimeSeriesSplit
import keras.backend as K
import os, gc
import datetime



#
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

data = pd.read_csv('../data/panel_zscore.csv')
data.date=pd.to_datetime(data.date)
data.symbol=data.symbol.astype('str').apply(lambda x:x.zfill(6))
data = data.drop('x_90', axis=1)
# imp = KNNImputer(n_neighbors=2, weights="uniform")
# imputed_data = imp.fit_transform(data.drop(['date', 'symbol'], axis=1))
def fillna_group(group):
    return group.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)

data = data.groupby('symbol').apply(fillna_group).reset_index(drop=True)
data = data.dropna()


train_data = data[data['date'] >= '2018-01-01']
train_data = train_data[train_data['date'] < '2021-01-01']

print(train_data.isnull().sum().sum())
# Group the data by symbol and sort within each group by date
grouped_data = train_data.groupby('symbol').apply(lambda x: x.sort_values('date'))

# Find the maximum number of dates for any symbol
# max_dates = grouped_data.groupby('symbol').size().max()
seq_len = 250

# Initialize a list to store the 3D data
train_data_dict = {}
train_mask_dict = {}

# Iterate over the grouped data and create padded 3D arrays
padding_days = []
symbols = []
for _, group in grouped_data.groupby('symbol'):
    symbols.append(group.symbol.values[0])
    group_features = group[['y'] + ['x_' + str(i) for i in range(1, 90)]].values
    if 200 < len(group_features) < seq_len:
        pad_length = seq_len - len(group_features)
        padded_features = np.pad(group_features, ((pad_length, 0), (0, 0)), mode='constant', constant_values=0)
        train_data_dict[group.symbol.values[0]] = padded_features
        padding_days.append(seq_len - len(group_features))
        group_mask = np.pad(np.ones_like(group_features), ((pad_length, 0), (0, 0)), mode='constant', constant_values=0)
    elif len(group_features) >= seq_len:
        # crop the data
        pad_length = 0
        train_data_dict[group.symbol.values[0]] = group_features[-seq_len:]
        # Create a mask for the group, with 1s for valid data points and 0s for the padded ones
        group_mask = np.ones_like(group_features[-seq_len:])
    else:
        continue
        # skip this symbol due to too many missing values
    train_mask_dict[group.symbol.values[0]] = group_mask
print('Average padding days ' + str(np.mean(padding_days)))
print("Filtered symbols " + str(len(symbols)))

# Convert the list of 3D arrays into a single numpy array

eval_data = data[data['date'] >= '2021-01-01']
print(eval_data.isnull().sum())

eval_data = eval_data[eval_data.symbol.isin(symbols)] # filter the data in train set

eval_dates = np.sort(np.unique(eval_data.date.values))

n_splits = 4
group_gap = 10
TRAIN = True

params = {
        'chunk_size': 5,
        "features": 89,
        "mid_feature": 512, # 64
        "hidden_dim": 64,
        "out_dim":1,
}
#

res_fold = '../results/mega/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
res_fold = '../results/mega/20230410-200713'
if not os.path.exists(res_fold):
    os.makedirs(res_fold)
log_fold = '../logs/mega/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(log_fold):
    os.makedirs(log_fold)


batch_size = 32
prediction_length = 10
results = pd.DataFrame(columns=['symbol', 'date', 'pred_y', 'y'])

predict_return = pd.DataFrame()

if TRAIN:
    scores = []

    for k in range(len(eval_dates) // prediction_length):
        ckp_path = res_fold + f'/Model_{k}.hdf5'
        model = MegaPredictor(**params)
        ckp = keras.callbacks.ModelCheckpoint(ckp_path, monitor='val_loss', verbose=0,
                              save_best_only=True, save_weights_only=True, mode='min')
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, mode='min',
                           baseline=None, restore_best_weights=True, verbose=0)
        tb = keras.callbacks.TensorBoard(log_dir = log_fold + '/prediction_' + str(k),
                                         histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch',
                                         embeddings_freq=0, embeddings_metadata=None) #profile_batch=2,

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005, amsgrad=True), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError(), PearsonCorrelation()])

        # find the next prediction_length dates
        pred_dates = eval_dates[k * prediction_length: (k + 1) * prediction_length]
        pred_data = eval_data[eval_data['date'].isin(pred_dates)]
        train_data = []
        train_mask = []
        val_data = []
        val_mask = []

        forward_length = []
        for symbol, data in train_data_dict.items():
            data1 = pred_data[pred_data['symbol'] == symbol].drop(['symbol', 'date'], axis=1).values
            # forward_length.append(pred_data[pred_data['symbol'] == symbol].date.values)
            train_data_dict[symbol] = np.concatenate([data, data1], axis=0)
            train_mask_dict[symbol] = np.concatenate([np.ones((prediction_length-data1.shape[0], data1.shape[1])),train_mask_dict[symbol], np.ones_like(data1)], axis=0)
            if train_data_dict[symbol].shape[0] < seq_len + prediction_length:
                pad_length = seq_len + prediction_length - train_data_dict[symbol].shape[0]
                train_data_dict[symbol] = np.pad(train_data_dict[symbol], ((pad_length, 0), (0, 0)), mode='constant', constant_values=0)
                train_mask_dict[symbol] = np.pad(train_mask_dict[symbol], ((pad_length, 0), (0, 0)), mode='constant', constant_values=0)

            train_data.append(train_data_dict[symbol][-(seq_len+ prediction_length):-prediction_length])
            val_data.append(train_data_dict[symbol][-prediction_length:])
            train_mask.append(train_mask_dict[symbol][-(seq_len+ prediction_length):-prediction_length])
            val_mask.append(train_mask_dict[symbol][-prediction_length:])

        train_data = np.array(train_data)
        val_data = np.array(val_data)
        train_x = train_data[:, :, 1:]
        train_y = train_data[:, :, 0]
        val_x = val_data[:, :, 1:]
        val_y = val_data[:, :, 0]
        t_mask = np.array(train_mask)[:, :, 0].reshape(-1, seq_len, 1)
        val_mask = np.array(val_mask)[:, :, 0].reshape(-1, prediction_length, 1)

        # model.call(train_x[:batch_size])
        # hist = pd.DataFrame(history.history)
        # score = hist['val_pred_IC'].max()
        # print(f'The {k}-th Prediction has IC:\t', score)
        # scores.append(score)
        history = model.fit(train_x[:batch_size], (train_y[:batch_size], t_mask[:batch_size]), epochs=1, batch_size=batch_size, verbose=0)

        model.load_weights(ckp_path)
        prediction = []
        pred_x = train_x
        for i in range(prediction_length):
            pred_x = np.append(pred_x, val_x[:, i, :].reshape(-1, 1, 89), axis=1)[:, -seq_len:, :]
            pred_y1 = model.predict(pred_x[:112*batch_size], batch_size)
            pred_y2 = model.call(pred_x[112*batch_size:])
            pred_y = np.concatenate((pred_y1, pred_y2), axis=0)
            prediction.append(pred_y[:, -1])
        prediction_y = np.array(prediction)
        # map the prediction data to the prediction dataframe
        predict_return = pd.concat((predict_return, pd.DataFrame(prediction_y, columns=train_data_dict.keys(), index=pred_dates)), axis=0)
        # symbol_date_info = pred_data.drop(['x_' + str(i) for i in range(1, 90)], axis=1)
        # for i, symbol in enumerate(train_data_dict.keys()):
        #     symbol_data = symbol_date_info[symbol_date_info['symbol'] == symbol]
        #     symbol_pred_data = prediction_y[:, i][-(symbol_data.shape[0]):]
        #     symbol_data['pred_y'] = symbol_pred_data
        #     results = pd.concat((results, symbol_data), axis=0)
        #




        # pred_y = model.predict(val_x, batch_size)
        # prediction = pred_data.drop(['x_'+str(i) for i in range(1, 90)], axis=1)
        #
        # for i, symbol in enumerate(train_data_dict.keys()):
        #     symbol_data = prediction[prediction['symbol'] == symbol]
        #     # the prediction data has length 10 but the true may not have length 10
        #     symbol_pred_data = pred_y[i][-symbol_data.shape[0]:]
        #     # map the prediction data to the prediction dataframe
        #     symbol_data['pred_y'] = symbol_pred_data
        #     results = pd.concat((results, symbol_data), axis=0)

        print(f"Correlation {IC(tf.convert_to_tensor(val_y.T, tf.float32), prediction_y)}")
        K.clear_session()
        del model
        rubbish = gc.collect()

        # save the prediction results
        print(f'======= Save model until now {k} =======')
        predict_return.to_csv(res_fold + '/mega_pred_results_one_day_ahead.csv')

print('Weighted Average CV Score:', np.mean(scores))


