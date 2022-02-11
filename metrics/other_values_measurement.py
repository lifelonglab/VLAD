from typing import Dict, List, Optional


class OtherValuesMeasurement:
    def __init__(self):
        self.values = []

    def add(self, measurement: Optional[Dict], task_name):
        if measurement is not None:
            self.values.append({'task': task_name, **measurement})

    def results(self) -> List[Dict]:
        return self.values
