from models.our.hierarchical_lifewatch import HierarchicalLifewatchMemory
from models.our.models.vae import VAE
from models.our.models.vae_2 import VAEParams
from models.our.models.vae_multi import VAEMultiParams
from models.our.our import create_our_model_mixed


def _return_models(max_samples, threshold_ratio, subconcept_threshold_ratio, steps, inter, latent, max_size_ratio=5):
    clean_vae = lambda input_features: VAEParams(input_features, inter, latent)
    full_model = lambda input_features: create_our_model_mixed(VAEParams(input_features, inter, latent),
                                                               HierarchicalLifewatchMemory(max_samples=max_samples,
                                                                                           threshold_ratio=threshold_ratio,
                                                                                           subconcept_threshold_ratio=subconcept_threshold_ratio,
                                                                                           max_size_ratio=max_size_ratio),
                                                               steps=steps)
    no_cpd = lambda input_features: create_our_model_mixed(VAEParams(input_features, inter, latent),
                                                           HierarchicalLifewatchMemory(max_samples=max_samples,
                                                                                       threshold_ratio=threshold_ratio,
                                                                                       subconcept_threshold_ratio=subconcept_threshold_ratio,
                                                                                       max_size_ratio=max_size_ratio,
                                                                                       disable_cpd=True),
                                                           steps=steps)
    no_replay = lambda input_features: create_our_model_mixed(VAEParams(input_features, inter, latent),
                                                              HierarchicalLifewatchMemory(max_samples=max_samples,
                                                                                          threshold_ratio=threshold_ratio,
                                                                                          subconcept_threshold_ratio=subconcept_threshold_ratio,
                                                                                          max_size_ratio=max_size_ratio,
                                                                                          disable_replay=True),
                                                              steps=steps)
    scaled = lambda input_features: create_our_model_mixed(VAEParams(input_features, inter, latent),
                                                           HierarchicalLifewatchMemory(max_samples=max_samples,
                                                                                       threshold_ratio=threshold_ratio,
                                                                                       subconcept_threshold_ratio=subconcept_threshold_ratio,
                                                                                       max_size_ratio=max_size_ratio),
                                                           steps=steps, enable_scaled_pred=True)


    return [
        # clean_vae,
        # full_model,
        # no_cpd,
        # no_replay,
        scaled
    ]


def wind_rel_wind_models():
    # max_samples = 9_000
    # threshold_ratio = 2
    # subconcept_threshold_ratio = 2
    # max_size_ratio = 5
    # steps = 10_000
    # return _return_models(max_samples, threshold_ratio=threshold_ratio,
    #                       subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=8, latent=4, max_size_ratio=max_size_ratio)
    max_samples = 1_000
    threshold_ratio = 1.5
    subconcept_threshold_ratio = 1.5
    max_size_ratio = 5
    steps = 10_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=16, latent=4, max_size_ratio=max_size_ratio)


def unsw_5_models():
    max_samples = 3_000
    threshold_ratio = 2
    subconcept_threshold_ratio = 5
    steps = 15_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=32, latent=8)


def energy_pv_models():
    max_samples = 5_000
    threshold_ratio = 1
    subconcept_threshold_ratio = 20
    steps = 15_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=32, latent=16)


def three_ids_models():
    max_samples = 8000
    threshold_ratio = 1.25
    subconcept_threshold_ratio = 1
    steps = 10_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=48, latent=8)


def credit_card_models():
    max_samples = 12_000
    threshold_ratio = 1.5
    subconcept_threshold_ratio = 5
    steps = 15_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=32, latent=16)


def ngids_models():
    max_samples = 4000
    threshold_ratio = 1.25
    subconcept_threshold_ratio = 1
    steps = 30_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=32, latent=8)


def www_models():
    max_samples = 4000
    threshold_ratio = 1
    subconcept_threshold_ratio = 1.5
    steps = 10_000
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, steps=steps, inter=32, latent=8)


def nsl_models():
    max_samples = 7500
    threshold_ratio = 1.25
    subconcept_threshold_ratio = 1.25
    max_size_ratio = 5
    steps = 7_500
    return _return_models(max_samples, threshold_ratio=threshold_ratio,
                          subconcept_threshold_ratio=subconcept_threshold_ratio, max_size_ratio=max_size_ratio, steps=steps, inter=32, latent=8)


