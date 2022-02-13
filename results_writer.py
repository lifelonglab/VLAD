import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from data_readers.data_reader import DataReader
from metrics.tasks_matrix.predictions_collector import CollectedResults
from models.model_base import ModelBase
from strategies.strategy import Strategy


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_results(model: Strategy, data_reader: DataReader, processed_results: Dict, collected_results: CollectedResults,
                 times: Dict, other_measurements: List[Dict]):
    metadata = {
        'name': model.name(),
        'strategy': model.strategy_name(),
        'model': model.model_name(),
        'dataset': data_reader.dataset_id(),
        'tasks': [t.name for t in data_reader.iterate_tasks()],
        'parameters': model.parameters()
    }

    output = {'metadata': metadata,
              'times': times,
              'other_measurements': other_measurements,
              'results': processed_results}

    path = Path(f'out/results/{data_reader.dataset_id()}/{model.strategy_name()}')
    path.mkdir(exist_ok=True, parents=True)

    json_str = json.dumps(output, indent=4)
    (path / Path(f'{model.name()}.json')).write_text(json_str, encoding='utf-8')

    debug_output = {'metadata': metadata, 'collected_results': collected_results}
    debug_path = (path / Path(f'debug'))
    debug_path.mkdir(parents=True, exist_ok=True)
    debug_file = (debug_path / Path(f'{model.name()}.json'))
    np.save(str(debug_file), debug_output, allow_pickle=True)
    # json_str = json.dumps(debug_output, indent=4, cls=NumpyEncoder)
    # debug_path = (path / Path(f'debug'))
    # debug_path.mkdir(parents=True, exist_ok=True)
    # (debug_path / Path(f'{model.name()}.json')).write_text(json_str, encoding='utf-8')
