"""Experiment constants shared by data preparation, training, and evaluation."""

KNOWN_CLASSES = ("COPAIBA", "CUMARU", "GARAPA", "MANITE", "PINHO")
UNKNOWN_CLASSES = (
    "CAUCHO",
    "CASTANHEIRA",
    "CEDRO",
    "MACARANDUBA",
    "TAUARI",
    "TAXI",
)

LABEL_MAP = {name: index for index, name in enumerate(KNOWN_CLASSES)}
UNKNOWN_LABEL = -1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

