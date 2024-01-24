# https://nfstream.github.io/docs/api#nfobserver-the-packet-observation-layer
#average packet payload_size
# http://www.doiserbia.nb.rs/img/doi/1820-0214/2014/1820-02141400035A.pdf


from nfstream import NFStreamer, NFPlugin
import numpy as np
import pickle
import pandas as pd





 #   _____                _     _
 #  / ____|              | |   (_)
 # | |     ___  _ __ ___ | |__  _ _ __   ___
 # | |    / _ \| '_ ` _ \| '_ \| | '_ \ / _ \
 # | |___| (_) | | | | | | |_) | | | | |  __/
 #  \_____\___/|_| |_| |_|_.__/|_|_| |_|\___|





#combine network datas together
norm_network_data  = pickle.load(open('network_trial.pkl','rb'))
attack_network_data  = pickle.load(open('attack_network_trial.pkl','rb'))
network_data = pd.concat([norm_network_data, attack_network_data], ignore_index=True)
network_data.columns

#remove uncessary networks signals sich as pings and system loopbacks
network_data.drop(network_data[network_data.category_name.str.contains("System")].index, inplace=True)
network_data.drop(network_data[network_data.application_name.str.contains("ICMPV6")].index, inplace=True)
network_data.head()

#combine motor datas together
motor_data  = pickle.load(open('motor_trial.pkl','rb'))
attack_motor_data  = pickle.load(open('attack_motor_trial.pkl','rb'))
motor_data = pd.concat([motor_data, attack_motor_data])
motor_data.head()

#function to combine motor and network data across splices
def get_motor(network_data, motor_data):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    flow_motor_segments = {'src2dst_first_seen_ms': [], 'Gyro': [], "Accel":[]}

    for index,flow in network_data.iterrows():
        print(flow["State"], flow["Trial"], flow["src2dst_first_seen_ms"])
        flow_motor_segments['src2dst_first_seen_ms'] += [flow["src2dst_first_seen_ms"]]
        try:
            #gets the motor timestamps for the state and trial
            curr_motor_trial_data = motor_data[motor_data.State.str.contains(flow["State"]) & (motor_data.Trial.eq(flow["Trial"]))]
            motor_sample_times = curr_motor_trial_data["Motor_TIMESTAMP"][0]

            #gets the flow timstamps
            start_at  = flow["src2dst_first_seen_ms"]/1000
            end_at = flow["src2dst_last_seen_ms"]/1000

            #gets the nearest mtor timestamps
            start_idx = find_nearest(motor_sample_times, start_at)
            end_idx =  find_nearest(motor_sample_times, end_at)

            #get the corresnpoding motor data for those network flow
            gyro = np.stack(curr_motor_trial_data.iloc[:,4:7].values[0], axis=0)[:,start_idx:end_idx]
            accel = np.stack(curr_motor_trial_data.iloc[:,1:4].values[0], axis=0)[:,start_idx:end_idx]

            flow_motor_segments['Gyro'] += [[gyro]]
            flow_motor_segments['Accel'] += [[accel]]
        #some have no motor movement for correspodnign packets
        except:
            flow_motor_segments['Gyro'] += [[]]
            flow_motor_segments['Accel'] += [[]]



    return pd.DataFrame.from_dict(flow_motor_segments)

#take get the corresponding teh motor slices to match up with the network flows
motor_splices = get_motor(network_data, motor_data)
#combine on the first seen to have both in one dataframe
dataset = pd.merge(motor_splices, network_data, on='src2dst_first_seen_ms')









 #  _____
 # |  __ \
 # | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___
 # |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __|
 # | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \
 # |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/
 #                | |
 #                |_|







#drop uncessary colymn; could run moving average in preprocessing if it is better
useful_columns = np.array(['src2dst_first_seen_ms', 'Accel', 'Attack','Gyro', 'State', 'Trial','flow_timestamps','payload_sizes', 'src_ip', 'flow_payloads'])
extra_columns = np.setdiff1d(dataset.columns.values,useful_columns)

dataset.drop(columns=extra_columns, inplace=True)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

    #these have empy motor during HALT states
dataset.drop([22,23], inplace=True)



def pad_raw_payloads(string_raw):
    string_raw = np.expand_dims(eval(string_raw), axis=1)
    string_raw = np.apply_along_axis(lambda x: np.frombuffer(x, dtype=np.uint8), 1,string_raw)
    return string_raw

dataset['flow_payloads'] = dataset['flow_payloads'].apply(lambda x: pad_raw_payloads(x))

dataset['flow_timestamps'] = dataset['flow_timestamps'].apply(lambda x: moving_average(np.diff(eval(x)), 1))
dataset['payload_sizes'] = dataset['payload_sizes'].apply(lambda x: moving_average(eval(x), 1))

dataset['Gyro_X'] = dataset['Gyro'].apply(lambda x: moving_average(x[0][0,], 1))
dataset['Gyro_Y'] = dataset['Gyro'].apply(lambda x: moving_average(x[0][1,], 1))
dataset['Gyro_Z'] = dataset['Gyro'].apply(lambda x: moving_average(x[0][2,], 1))

dataset['Accel_X'] = dataset['Accel'].apply(lambda x: moving_average(x[0][0,], 1))
dataset['Accel_Y'] = dataset['Accel'].apply(lambda x: moving_average(x[0][1,], 1))
dataset['Accel_Z'] = dataset['Accel'].apply(lambda x: moving_average(x[0][2,], 1))

dataset.drop(columns=["Accel", "Gyro"], inplace=True)



#save off preprocessed dataset
import pickle
pickle.dump(dataset,open(f'preprocessed_dataset.pkl','wb'))









 #
 #  _____  _       _
 # |  __ \| |     | |
 # | |__) | | ___ | |_
 # |  ___/| |/ _ \| __|
 # | |    | | (_) | |_
 # |_|    |_|\___/ \__|



 dataset  = pickle.load(open('preprocessed_dataset.pkl','rb'))







import matplotlib.pyplot as plt

sample = dataset[dataset.State.str.contains("A4") & dataset.Attack.eq(1) ]
sample = dataset[dataset.State.str.contains("SLOW") & dataset.Trial.eq(1) ]
# test sample pltting

def plot_sample_features(sample):

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(15,1))

    fig.suptitle(f'{sample.State.values[0]} {sample.Trial.values[0]}', size=16, y=1.4)

    ax1.plot(np.absolute(moving_average(np.hstack(sample.Accel_X.values), 10000)))
    ax1.set_title('Accel_X')

    ax2.plot(np.absolute(moving_average(np.hstack(sample.Accel_Y.values), 10000)))
    ax2.set_title('Accel_Y')

    ax3.plot(np.absolute(moving_average(np.hstack(sample.Accel_Z.values), 10000)))
    ax3.set_title('Accel_Z')

    ax4.plot(np.absolute(moving_average(np.hstack(sample.Gyro_X.values), 10000)),label='X')
    ax4.plot(np.absolute(moving_average(np.hstack(sample.Gyro_Y.values), 10000)),label='Y')
    ax4.plot(np.absolute(moving_average(np.hstack(sample.Gyro_Z.values), 10000)),label='Z')
    ax4.set_title('Gyro_XYZ')
    ax4.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, borderaxespad=0.)



    ax5.plot(moving_average(np.hstack(sample.payload_sizes.values), 15000))
    ax5.set_title('payload_sizes')

    ax6.plot(moving_average(np.hstack(sample.flow_timestamps.values), 10000))
    ax6.set_title('flow_timestamps deltas')

plot_sample_features(sample)

sample.flow_payloads.values[0][:100,].shape
plt.imshow(sample.flow_payloads.values[0][:53,])



trials = np.unique(dataset.Trial.values)
states = np.unique(dataset.State.values)
trials



import warnings
warnings.filterwarnings("ignore")


for state in states[8:]:
    for trial in trials:
        # print(trial, state)
        sample = dataset[dataset.State.str.contains(str(state)) & dataset.Trial.eq(trial) ]
        try:
            plot_sample_features(sample)
        except:
            continue




for state in states[:8]:
    for trial in trials[:1]:

        # print(trial, state)
        sample = dataset[dataset.State.str.contains(str(state)) & dataset.Trial.eq(trial)]
        try:
            plot_sample_features(sample)
        except:
            continue
