from typing import List

from models.our.cpds.cpd import CPD, ChangePoint


class AlwaysNewCPD(CPD):
    def detect_cp(self, data) -> List[ChangePoint]:
        return [ChangePoint(index=0, is_new=True)]
