from data_readers.credit_card_data_reader import CreditCardDataReader
from data_readers.data_reader import DataReader
from data_readers.energy_data_reader import EnergyDataReader
from data_readers.mixed_ids_data_reader import MixedIdsDataReader
from data_readers.nsl_data_reader import NslDataReader
from data_readers.unsw_data_reader import UnswDataReader
from data_readers.wind_rel_data_reader import WindEnergyDataReader

# unsw_data_reader = UnswDataReader('data/unsw/unsw_clustered_10_closest_anomalies.npy',
#                                           name='unsw_clustered_10_closest_anomaly')
#
#
# energy_data_reader = EnergyDataReader('data/energy/energy_pv_hours_short.npy')
# wind_energy_data_reader =WindEnergyDataReader('data/energy/wind_short.npy', 'wind_clustered_10')
ngids5 = MixedIdsDataReader('data/ngids5/ngids_5_repetition_messed.npy', name='www_adfa_ngids_clustered')
# www_data_reader = MixedIdsDataReader('data/www/www_6x2_short.npy',
#                                                    name='www_adfa_ngids_clustered')
# credit_card_data_reader = CreditCardDataReader('data/creditcard/creditcard_25.npy', name='creditcard_5')
# nsl_data_reader = NslDataReader('data/nsl/nsl_10_r.npy', 'nsl_10')
# three_ids_reader = MixedIdsDataReader('data/3ids/3ids_repetition_messed.npy', name='3ids2')
data_reader: DataReader = ngids5

# for t in data_reader.iterate_tasks():
#     print(t.data.shape)


for t in data_reader.load_test_tasks():
    print(t.name)
    # print(len([c for c in t.labels if c == 1]))