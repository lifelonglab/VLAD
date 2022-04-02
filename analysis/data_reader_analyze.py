from data_readers.credit_card_data_reader import CreditCardDataReader
from data_readers.data_reader import DataReader
from data_readers.energy_data_reader import EnergyDataReader
from data_readers.mixed_ids_data_reader import MixedIdsDataReader
from data_readers.nsl_data_reader import NslDataReader
from data_readers.unsw_data_reader import UnswDataReader
from data_readers.wind_rel_data_reader import WindEnergyDataReader

unsw_data_reader = UnswDataReader('data/unsw/unsw_clustered_10_closest_anomalies.npy',
                                          name='unsw_clustered_10_closest_anomaly')


energy_data_reader = EnergyDataReader('data/energy/energy_pv_hours_short.npy')
wind_energy_data_reader =WindEnergyDataReader('data/energy/wind_clustered_5.npy', 'wind_clustered_5')
mixed_ids_data_reader = MixedIdsDataReader('data/ngids/ngids_5.npy',
                                                   name='www_adfa_ngids_clustered')
www_data_reader = MixedIdsDataReader('data/www/www_6x2_short.npy',
                                                   name='www_adfa_ngids_clustered')
credit_card_data_reader = CreditCardDataReader('data/creditcard/creditcard_25.npy', name='creditcard_5')
nsl_data_reader = NslDataReader('data/nsl/nsl_10.npy', 'nsl_10')

data_reader: DataReader = wind_energy_data_reader

for t in data_reader.iterate_tasks():
    print(t.data.shape)