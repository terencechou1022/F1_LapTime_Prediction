"""f1lab — Object-oriented toolkit for F1 lap-performance regression.

Public API:
    BaseLapPreprocessor, WindPreprocessor, TempPreprocessor
    ModelTrainer, ModelEvaluator
    Visualizer
    FastF1Downloader, RaceDataMerger

Usage:
    from f1lab import ModelTrainer, WindPreprocessor

    preprocessor = WindPreprocessor.from_excel("data/merged/2022-2024_Azerbaijan_Grand_Prix.xlsx")
    trainer = ModelTrainer(preprocessor)
    trainer.fit()
    trainer.report()
    trainer.save("models/azerbaijan_rf.joblib")
"""
from f1lab.data import FastF1Downloader, RaceDataMerger
from f1lab.experiments import TempPreprocessor, WindPreprocessor, get_preprocessor
from f1lab.modeling import ModelEvaluator, ModelTrainer
from f1lab.preprocessing import BaseLapPreprocessor
from f1lab.strategy import ScenarioResult, UndercutScenario
from f1lab.visualization import Visualizer

__all__ = [
    "BaseLapPreprocessor",
    "WindPreprocessor",
    "TempPreprocessor",
    "get_preprocessor",
    "ModelTrainer",
    "ModelEvaluator",
    "Visualizer",
    "UndercutScenario",
    "ScenarioResult",
    "FastF1Downloader",
    "RaceDataMerger",
]
