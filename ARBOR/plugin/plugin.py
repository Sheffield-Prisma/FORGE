# -*- coding: utf-8 -*-
"""
plugin.py  -  ARBOR plugin entry point.

Registers the ARBOR processing provider when the plugin loads and removes it on
unload. The UI ships in Spanish.
"""

from qgis.core import QgsApplication

from .provider import ArborProvider


class ArborPlugin:

    def __init__(self, iface=None):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        self.provider = ArborProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        # Called by QGIS when the plugin is enabled.
        self.initProcessing()

    def unload(self):
        # Called by QGIS when the plugin is disabled / QGIS closes.
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None