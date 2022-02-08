from itertools import chain


class SimpleFlatMemory:
    def __init__(self):
        self.memory = []
        self.limit_per_task = 100

    def new_data(self, data, is_new: bool, task=None):
        if is_new:
            self.memory.append(data[:self.limit_per_task])
        else:
            self.memory[-1] = (self.memory[-1] + data)[-self.limit_per_task:]

    def get_replay(self):
        return list(chain(*self.memory))
