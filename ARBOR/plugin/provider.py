# -*- coding: utf-8 -*-
"""
provider.py  -  ARBOR processing provider.

The provider's name() is the top-level group shown in the Processing Toolbox
("ARBOR"). Algorithms registered here appear under it, sub-grouped by each
algorithm's group() (e.g. "Shihuahuaco").
"""

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithm import ShihuahuacoDetectionAlgorithm


class ArborProvider(QgsProcessingProvider):

    def id(self) -> str:
        # Internal provider id. Changed from "forge" -> "arbor"; any saved
        # Processing models/scripts that referenced "forge:..." must be updated.
        return "arbor"

    def name(self) -> str:
        # Top-level group name in the Processing Toolbox (brand name, kept as-is
        # in both languages).
        return "ARBOR"

    def longName(self) -> str:
        return "ARBOR - Amazon tree-crown detection"

    def icon(self):
        path = os.path.join(os.path.dirname(__file__), "icons", "icon.png")
        if os.path.exists(path):
            return QIcon(path)
        return QgsProcessingProvider.icon(self)

    def loadAlgorithms(self):
        self.addAlgorithm(ShihuahuacoDetectionAlgorithm())