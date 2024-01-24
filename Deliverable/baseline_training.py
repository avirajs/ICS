# %% codecell
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mil_functions import get_dws_from,moving_average,inject_anomaly,next_step_splitter,plot_anomaly_detection



preprocessed_message_df = pd.read_pickle("preprocessed.pkl")

preprocessed_message_df.columns = ('Sync_Waveform', 'Remote Terminal Address', 'MError', 'Instrumentation',
       'Service Request', 'Reserved', 'Broadcast', 'Busy', 'Subsystem Flag',
       'D Bus Control', 'Terminal Flag', 'Parity', 'Sync_Waveform.1',
       'Remote Terminal Address.1', 'T/R', 'Subaddress Mode',
       'DW-Count/Mode Word', 'Parity.1', 'Sync Waveform', 'Data', 'Parity.2',
       'Sync Waveform.1', 'Data.1', 'Parity.3', 'Sync_Waveform.2', 'Data.2', 'Parity.4')
preprocessed_message_df = preprocessed_message_df[~preprocessed_message_df.index.duplicated()]
preprocessed_message_df["Remote Terminal Address"] = preprocessed_message_df["Remote Terminal Address"].apply(lambda row: str(row))
preprocessed_message_df["Parity"] = preprocessed_message_df["Parity"].apply(lambda row: str(row))
preprocessed_message_df["T/R"] = preprocessed_message_df["T/R"].apply(lambda row: str(row))
preprocessed_message_df["DW-Count/Mode Word"] = preprocessed_message_df["DW-Count/Mode Word"].apply(lambda row: str(row))
preprocessed_message_df["Subaddress Mode"] = preprocessed_message_df["Subaddress Mode"].apply(lambda row: str(row))
preprocessed_message_df["MError"] = preprocessed_message_df["MError"].apply(lambda row: str(row))
preprocessed_message_df["Remote Terminal Address.1"] = preprocessed_message_df["Remote Terminal Address.1"].apply(lambda row: str(row))

pickle.dump(preprocessed_message_df, open("Data/preprocessed_strings.pkl", "wb"))
preprocessed_message_df = pickle.load(open("Data/preprocessed_strings.pkl", "rb"))



import itertools

#get all datawords through combination these key variables
somelists = [
   preprocessed_message_df["Remote Terminal Address"].unique(),
   preprocessed_message_df["Subaddress Mode"].unique(),
   preprocessed_message_df["T/R"].unique(),
    preprocessed_message_df["DW-Count/Mode Word"].unique()
]




#manual raw dataword extraction by key
raw_samples = []
for paras in itertools.product(*somelists):
    try:
        dw1_data = get_dws_from(preprocessed_message_df,paras[0],paras[1],paras[2],paras[3])[0]
        dw2_data = get_dws_from(preprocessed_message_df,paras[0],paras[1],paras[2],paras[3])[1]
        dw3_data = get_dws_from(preprocessed_message_df,paras[0],paras[1],paras[2],paras[3])[2]
        if np.isnan(dw1_data).any():
            continue
        raw_samples.extend([dw1_data,dw2_data,dw3_data])
    except:
        continue



# remove nans
raw_samples = list(filter(lambda x: not np.isnan(x).any(),raw_samples))

len(raw_samples)

pickle.dump(raw_samples, open("Data/raw_dws.pkl", "wb"))


raw_samples = pickle.load(open("Data/raw_dws.pkl", "rb"))

plt.title("Raw Sensor Values over time")
plt.ylabel("Bit Value")
plt.xlabel("Packet Occurrence")
plt.plot(raw_samples[8])




#normalize by maximum of 65535, or min max scale
raw_samples = list(map(lambda samples: (samples - np.min(samples))/(np.max(samples)- np.min(samples)+1), raw_samples))

plt.title("Normalized sensor values over time")
plt.ylabel("Bit Value")
plt.xlabel("Packet Occurrences")
plt.plot(raw_samples[7])


#take moving average
avg_samples = [ moving_average(samples,1000) for samples in raw_samples]

pickle.dump(avg_samples, open("Data/dw_normal_averaged.pkl", "wb"))
avg_samples = pickle.load(open("Data/dw_normal_averaged.pkl", "rb"))

plt.title("Normalized, Averaged samples over time")
plt.ylabel("Bit Value")
plt.xlabel("Packet Occurrences")
plt.plot(avg_samples[8])






from keras.layers import Input, Dropout, MaxPooling1D, UpSampling1D, BatchNormalization, RepeatVector, Flatten
from keras.layers import Dense, Conv1D, LSTM, Embedding, Add, CuDNNLSTM
from keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.optimizers import Adam



config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)




def test_cnn_stream(X_train, y_train, input_shape,e=30):

    input_window = Input(shape=input_shape)
    x = Conv1D(16, 3, activation="relu", padding="causal")(input_window) # 10 dims
    x = MaxPooling1D(2, padding="same")(x) # 5 dims
    x = Flatten()(x) # 3 dims
    x = Dropout(.5)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(y_train.shape[1])(x)
    model = Model(input_window, output)

    model.compile(optimizer=Adam(lr=0.00001), loss='mse', metrics=["mean_absolute_error"])
    model.summary()


    history = model.fit(X_train, y_train,
                    epochs=e,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.3 )

    return model, history

# choosing a sample, we should be training on our definition of baseline data
plt.title("Baseline Training using this sample")
for i in range(60):
    plt.title(i)
    plt.plot(avg_samples[i])
    plt.figure()




x_train_baseline, y_train_baseline = next_step_splitter(avg_samples[51], width=100, prediction_size=2)
x_train_baseline = np.expand_dims(x_train_baseline,axis=2)
y_train_baseline = y_train_baseline
print(x_train_baseline.shape, y_train_baseline.shape)
# %% codecell
model, history = test_cnn_stream(x_train_baseline, y_train_baseline, x_train_baseline.shape[1:],e=100)

import pickle
pickle.dump(model, open("Data/cnn_model_pred.pkl", "wb"))
model = pickle.load(open("Data/cnn_model_pred.pkl", "rb"))
