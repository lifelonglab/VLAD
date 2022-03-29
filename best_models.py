from models.our.hierarchical_lifewatch import HierarchicalLifewatchMemory
from models.our.models.vae import VAE
from models.our.models.vae_2 import VAEParams
from models.our.our import create_our_model_mixed


def _return_models(max_samples, threshold_ratio, subconcept_threshold_ratio, steps, inter, latent):
    clean_vae = lambda input_features: VAEParams(input_features, inter, latent)
    full_model = lambda input_features: create_our_model_mixed(VAEParams(input_features, inter, latent),
                                                               HierarchicalLifewatchMemory(max_samples=max_samples,
                                                                                           threshold_ratio=threshold_ratio,
                                                                                           subconcept_threshold_ratio=subconcept_threshold_ratio),
                                                               steps=steps)
    no_cpd = lambda input_features: create_our_model_mixed(VAEParams(input_features, inter, latent),
                                                           HierarchicalLifewatchMemory(max_samples=max_samples,
                                                                                       threshold_ratio=threshold_ratio,
                                                                                       subconcept_threshold_ratio=subconcept_threshold_ratio,
                                                                                       disable_cpd=True),
                                                           steps=steps)
    no_replay = lambda input_features: create_our_model_mixed(VAEParams(input_features, inter, latent),
                                                              HierarchicalLifewatchMemory(max_samples=max_samples,
                                                                                          threshold_ratio=threshold_ratio,
                                                                                          subconcept_threshold_ratio=subconcept_threshold_ratio,
                                                                                          disable_replay=True),
                                                              steps=steps)
    scaled = lambda input_features: create_our_model_mixed(VAEParams(input_features, inter, latent),
                                                           HierarchicalLifewatchMemory(max_samples=max_samples,
                                                                                       threshold_ratio=threshold_ratio,
                                                                                       subconcept_threshold_ratio=subconcept_threshold_ratio),
                                                           steps=steps, enable_scaled_pred=True)


    return [
        # clean_vae,
        full_model,
        # no_cpd,
        # no_replay,
        # scaled
    ]


def wind_rel_wind_models():
    max_samples = 9_000
    threshold_ratio = 2
    subconcept_threshold_ratio = 2
    steps = 15_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=8, latent=4)


def unsw_5_models():
    max_samples = 10_000
    threshold_ratio = 2
    subconcept_threshold_ratio = 5
    steps = 15_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=32, latent=8)


def energy_pv_models():
    max_samples = 5_000
    threshold_ratio = 1.125
    subconcept_threshold_ratio = 20
    steps = 15_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=32, latent=16)


def three_ids_models():
    max_samples = 9_000
    threshold_ratio = 1.25
    subconcept_threshold_ratio = 10
    steps = 15_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=128, latent=32)


