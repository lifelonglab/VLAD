from models.classic.isolation_forest import IsolationForestAdapter
from models.classic.lof import LocalOutlierFactorAdapter
from models.classic.oc_svm import OneClassSVMAdapter
from models.modern.copod_adapter import COPODAdapter
from models.modern.suod_adapter import SUODAdapter


def best_unsw_competitors():
    return [
        lambda _: SUODAdapter(contamination=0.0001),
        lambda _: COPODAdapter(contamination=0.001),
        lambda _: IsolationForestAdapter(n_estimators=100, contamination=0.001),
        lambda _: LocalOutlierFactorAdapter(n_neighbors=2),
        lambda _: OneClassSVMAdapter(nu=0.1, gamma=0.01)
    ]


def best_wind_competitors():
    return [
        lambda _: SUODAdapter(contamination=0.001),
        lambda _: COPODAdapter(contamination=0.001),
        lambda _: IsolationForestAdapter(n_estimators=200, contamination=0.001),
        lambda _: LocalOutlierFactorAdapter(n_neighbors=10),
        lambda _: OneClassSVMAdapter(nu=0.01, gamma=0.01)
    ]


def best_3ids_competitors():
    return [
        lambda _: SUODAdapter(contamination=0.0001),
        lambda _: COPODAdapter(contamination=0.001),
        lambda _: IsolationForestAdapter(n_estimators=200, contamination=0.0001),
        lambda _: LocalOutlierFactorAdapter(n_neighbors=5),
        lambda _: OneClassSVMAdapter(nu=0.01, gamma=0.01),
    ]


def best_energy_competitors():
    return [
        lambda _: SUODAdapter(contamination=0.0001),
        lambda _: COPODAdapter(contamination=0.001),
        lambda _: IsolationForestAdapter(n_estimators=100, contamination=0.0001),
        lambda _: LocalOutlierFactorAdapter(n_neighbors=10),
        lambda _: OneClassSVMAdapter(nu=0.1, gamma=0.01),
    ]


def best_credit_card_competitors():
    return [

        lambda _: IsolationForestAdapter(n_estimators=200, contamination=0.001),
        lambda _: LocalOutlierFactorAdapter(n_neighbors=10),
        lambda _: OneClassSVMAdapter(nu=0.01, gamma=0.01),
        lambda _: SUODAdapter(contamination=0.00001),
        lambda _: COPODAdapter(contamination=0.001),
    ]


def best_ngids_competitors():
    return [
        lambda _: SUODAdapter(contamination=0.00001),
        lambda _: COPODAdapter(contamination=0.001),
        lambda _: IsolationForestAdapter(n_estimators=200, contamination=0.001),
        lambda _: LocalOutlierFactorAdapter(n_neighbors=10),
        lambda _: OneClassSVMAdapter(nu=0.01, gamma=0.01),
    ]