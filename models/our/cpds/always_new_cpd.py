from typing import List, Dict

from models.our.cpds.cpd import CPD, ChangePoint


class AlwaysNewCPD(CPD):
    def detect_cp(self, data) -> List[ChangePoint]:
        return [ChangePoint(index=0, is_new_dist=True)]

    def name(self) -> str:
        return 'AlwaysNewCPD'

    def params(self) -> Dict:
        return {}
