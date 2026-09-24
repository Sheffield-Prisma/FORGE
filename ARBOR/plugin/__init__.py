# -*- coding: utf-8 -*-
def classFactory(iface):
    from .plugin import ArborPlugin
    return ArborPlugin(iface)