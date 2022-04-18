import json

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="darkgrid")

inputs = {
    # 'unsw_10':  'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_32_8_HLW_lim_1024_2000_p2_mf_5_str_5_steps_15000_.json',
    # '3ids3': 'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_8_4_HLW_lim_1024_4000_p0.75_mf_5_str_1_steps_30000_.json',
    'ngids':  'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_32_8_HLW_lim_1024_500_p1.25_mf_5_str_1_steps_30000_.json',
    'wind_5':  'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_16_4_HLW_lim_1024_250_p1.75_mf_5_str_1.5_steps_10000_.json',
    'nsl_8': 'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_32_8_HLW_lim_1024_1000_p1.25_mf_5_str_1.25_steps_30000_.json'
}


results = []
for dataset, filename in inputs.items():
    filepath = f'out/results/{dataset}_repetition_long/IncrementalBatchLearner_0/{filename}'

    with open(filepath) as f:
        data = json.load(f)
        results.extend([[dataset, i, d['memory']['concepts']] for i, d in enumerate(data['other_measurements'])])

print(results)
df = pd.DataFrame(results, columns=['dataset', 'task', 'concepts'])
print(df)

sns.lineplot(data=df, x='task', y='concepts', hue='dataset')
plt.show()