import pandas as pd
import numpy as np
import pickle



 #  __  __       _
 # |  \/  |     | |
 # | \  / | ___ | |_ ___  _ __
 # | |\/| |/ _ \| __/ _ \| '__|
 # | |  | | (_) | || (_) | |
 # |_|  |_|\___/ \__\___/|_|




#section for getting motor trials

def get_arranged_motor_trials():

    states = ["FAST", "HALT", "MEDIUM", "OFF", "REVERSE", "SLOW"]
    trials = list(range(1,11))

    trial_motor_data = []

    for state in states:
        for trial in trials:
            curr_trial_motor = pd.read_csv(f'TRAIN/{state}/Trial{trial}/{state.lower()}_{trial}.csv', error_bad_lines=False)
            curr_trial_motor = curr_trial_motor.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            if(trial<7):
                trial_row_temp = pd.DataFrame(
                    {"Motor_TIMESTAMP" : [curr_trial_motor["TIMESTAMP"].values],
                    "ACCELEROMETER_X" : [curr_trial_motor["ACCELEROMETER_X"].values],
                    "ACCELEROMETER_Y" : [curr_trial_motor["ACCELEROMETER_Y"].values],
                    "ACCELEROMETER_Z" : [curr_trial_motor["ACCELEROMETER_Z"].values],
                    "GYROSCOPE_X" : [curr_trial_motor["GYROSCOPE_X"].values],
                    "GYROSCOPE_Y" : [curr_trial_motor["GYROSCOPE_Y"].values],
                    "GYROSCOPE_Z" : [curr_trial_motor["GYROSCOPE_Z"].values],
                    "State":state,
                    "Trial":trial,
                    "Attack":0
                    }
                )
            else:
                trial_row_temp = pd.DataFrame(
                    {"Motor_TIMESTAMP" : [curr_trial_motor["TIMESTAMP"].values[::10]],
                    "ACCELEROMETER_X" : [curr_trial_motor["ACCELEROMETER_X"].values[::10]],
                    "ACCELEROMETER_Y" : [curr_trial_motor["ACCELEROMETER_Y"].values[::10]],
                    "ACCELEROMETER_Z" : [curr_trial_motor["ACCELEROMETER_Z"].values[::10]],
                    "GYROSCOPE_X" : [curr_trial_motor["GYROSCOPE_X"].values[::10]],
                    "GYROSCOPE_Y" : [curr_trial_motor["GYROSCOPE_Y"].values[::10]],
                    "GYROSCOPE_Z" : [curr_trial_motor["GYROSCOPE_Z"].values[::10]],
                    "State":state,
                    "Trial":trial,
                    "Attack":0
                    }
                )


            trial_motor_data.append(trial_row_temp)


    return pd.concat(trial_motor_data)

def get_arranged_attack_motor_trials():

    states = ["A3", "A4", "A5", "A6","A7", "A8", "A9", "A10"]
    trials = list(range(1,2))

    trial_motor_data = []

    for state in states:
        for trial in trials:
            curr_trial_motor = pd.read_csv(f'TEST/{state}/Trial_{trial}/motionData.csv', error_bad_lines=False)
            curr_trial_motor = curr_trial_motor.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
            attack_log = pd.read_csv(f'TEST/{state}/Trial_{trial}/monitorLog.csv', error_bad_lines=False, header=None)
            #get attack session info only; edit do this manually in later preprocessing when combining with network flows
            # curr_trial_motor = curr_trial_motor[curr_trial_motor['TIMESTAMP'].between(attack_log.iloc[1,0], attack_log.iloc[2,0])]

            trial_row_temp = pd.DataFrame(
                {"Motor_TIMESTAMP" : [curr_trial_motor["TIMESTAMP"].values],
                "ACCELEROMETER_X" : [curr_trial_motor["ACCELEROMETER_X"].values],
                "ACCELEROMETER_Y" : [curr_trial_motor["ACCELEROMETER_Y"].values],
                "ACCELEROMETER_Z" : [curr_trial_motor["ACCELEROMETER_Z"].values],
                "GYROSCOPE_X" : [curr_trial_motor["GYROSCOPE_X"].values],
                "GYROSCOPE_Y" : [curr_trial_motor["GYROSCOPE_Y"].values],
                "GYROSCOPE_Z" : [curr_trial_motor["GYROSCOPE_Z"].values],
                "State":state,
                "Trial":trial
                # "Attack":1
                }
            )

            trial_row_temp.head()
            trial_motor_data.append(trial_row_temp)


    return pd.concat(trial_motor_data)



# view and save motor data in pickles
motor_labeled =  get_arranged_motor_trials()
attack_motor_labeled =  get_arranged_attack_motor_trials()

attack_motor_labeled.head()
motor_labeled.head()

import pickle
pickle.dump(motor_labeled,open(f'motor_trial.pkl','wb'))
pickle.dump(attack_motor_labeled,open(f'attack_motor_trial.pkl','wb'))




 #
 #  _   _      _                      _
 # | \ | |    | |                    | |
 # |  \| | ___| |___      _____  _ __| | __
 # | . ` |/ _ \ __\ \ /\ / / _ \| '__| |/ /
 # | |\  |  __/ |_ \ V  V / (_) | |  |   <
 # |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\





# section for getting network data; uses custom nfstream plugins
import warnings
warnings.filterwarnings("ignore")
from nfstream import NFStreamer, NFPlugin
import numpy as np


def get_arranged_pcap_trials():

    class flow_timestamps(NFPlugin):
        def on_init(self, pkt): # flow creation with the first packet
            if pkt.payload_size >=0 and pkt.direction==0:
                return [pkt.time]
            else:
                return 0
        def on_update(self, pkt, flow): # flow update with each packet belonging to the flow
            flow.flow_timestamps += [pkt.time]

    class payload_sizes(NFPlugin):
        def on_init(self, pkt): # flow creation with the first packet
            if pkt.raw_size >=0 and pkt.direction==0:
                return [pkt.raw_size]
            else:
                return 0
        def on_update(self, pkt, flow): # flow update with each packet belonging to the flow
            flow.payload_sizes.append(pkt.raw_size)

    class flow_payloads(NFPlugin):
        def on_init(self, pkt): # flow creation with the first packet
            if pkt.payload_size >=0 and pkt.direction==0 and b';' not in pkt.ip_packet:
                return [ pkt.ip_packet]
            else:
                return []
        def on_update(self, pkt, flow): # flow update with each packet belonging to the flow
            if(pkt.payload_size <=25 and b';' not in pkt.ip_packet):
                flow.flow_payloads.append(pkt.ip_packet)



    states = ["FAST", "HALT", "MEDIUM", "OFF", "REVERSE", "SLOW"]
    trials = list(range(1,11))

    trial_network_data = []

    for state in states:
        for trial in trials:
            print(f'{state} {trial}')
            try:
                trial_traffic = NFStreamer(f'TRAIN/{state}/Trial{trial}/{state.lower()}_{trial}.pcap',plugins=[flow_timestamps(), payload_sizes(), flow_payloads()], statistics=True).to_pandas()
                trial_traffic["State"] = state
                trial_traffic["Trial"] = trial
                trial_traffic["Attack"] = 0

                #remove traffic with only few packets

                trial_network_data.append(trial_traffic)
            except:
                print("Not working")


    return pd.concat(trial_network_data)

def get_arranged_attack_pcap_trials():


    class flow_timestamps(NFPlugin):
        def on_init(self, pkt): # flow creation with the first packet
            if pkt.payload_size >=0 and pkt.direction==0:
                return [pkt.time]
            else:
                return [0]
        def on_update(self, pkt, flow): # flow update with each packet belonging to the flow
            flow.flow_timestamps += [pkt.time]

    class payload_sizes(NFPlugin):
        def on_init(self, pkt): # flow creation with the first packet
            if pkt.raw_size >=0 and pkt.direction==0:
                return [pkt.raw_size]
            else:
                return [0]
        def on_update(self, pkt, flow): # flow update with each packet belonging to the flow
            flow.payload_sizes.append(pkt.raw_size)

    class flow_payloads(NFPlugin):
        def on_init(self, pkt): # flow creation with the first packet
            if pkt.payload_size >=0 and pkt.direction==0 and b';' not in pkt.ip_packet:
                return [ pkt.ip_packet]
            else:
                return []
        def on_update(self, pkt, flow): # flow update with each packet belonging to the flow
            if(pkt.payload_size <=25 and b';' not in pkt.ip_packet):
                flow.flow_payloads.append(pkt.ip_packet)





    states = ["A3", "A4", "A5", "A6","A7", "A8", "A9", "A10"]
    trials = list(range(1,2))

    trial_network_data = []


    for state in states:
        for trial in trials:
            print(f'{state} {trial}')
            try:
                trial_traffic = NFStreamer(source=f'TEST/{state}/Trial_{trial}/pcapData.pcap',plugins=[ flow_timestamps(), payload_sizes(), flow_payloads() ], statistics=True).to_pandas()
                trial_traffic.head()
                trial_traffic.drop(trial_traffic[trial_traffic.category_name.str.contains("System")].index, inplace=True)
                trial_traffic.drop(trial_traffic[trial_traffic.application_name.str.contains("ICMPV6")].index, inplace=True)
                trial_traffic.head()
                attack_log = pd.read_csv(f'TEST/{state}/Trial_{trial}/monitorLog.csv', error_bad_lines=False, header=None)

                #label the closest to the attack logged as an attack
                start_attack = attack_log.iloc[1,0]*1000 #start of attack in mlliseconds
                idx_nearest_flow_attacked = trial_traffic['src2dst_first_seen_ms'].sub(start_attack).abs().idxmin() #find nearest traffic flow start to attack start
                trial_traffic["Attack"] = 0
                trial_traffic["Attack"][idx_nearest_flow_attacked] = 1

                trial_traffic["State"] = state
                trial_traffic["Trial"] = trial

                trial_network_data.append(trial_traffic)
            except:
                print("Not working")


    return pd.concat(trial_network_data)


pcap_labeled = get_arranged_pcap_trials()
pcap_attack_labeled = get_arranged_attack_pcap_trials()

pcap_labeled.head()
pcap_attack_labeled.head()

import pickle
pickle.dump(pcap_labeled,open(f'network_trial.pkl','wb'))
pickle.dump(pcap_attack_labeled,open(f'attack_network_trial.pkl','wb'))
