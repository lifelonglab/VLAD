from typing import Dict

from models.model import Model
from models.our.models.ae import AE
from models.our.models.vae import VAE


class OurModelAdapter(Model):
    def __init__(self):
        self.model = VAE()
        # self.model = AE()

    def name(self):
        return f'our_model_base_vae'

    def learn(self, data) -> None:
        self.model.learn(data)

    def predict(self, data, task_name=None):
        return self.model.predict(data)

    def parameters(self) -> Dict:
        return self.model.params
