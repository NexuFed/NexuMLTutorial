"""Typed source for NexuML-owned prepared dataset views."""

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source
from nexuml.data.exported import ExportedDataset as CoreExportedDataset


@data_source("TutorialExportedDataset")
class ExportedDataset(DataSourceDefinition):
    root: str

    def build(self) -> CoreExportedDataset:
        return CoreExportedDataset(
            self.root,
            feature_keys=["features"],
            label_keys=["class"],
        )
