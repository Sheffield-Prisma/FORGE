# -*- coding: utf-8 -*-
"""
algorithm.py  -  ARBOR : Shihuahuaco Tree Crown Detection (single-class)

QGIS Processing Algorithm, ONNX-only inference. No PyTorch / ultralytics / SAHI.

Dependencies (install in the OSGeo4W Shell):
    pip install numpy onnxruntime
    pip install onnxruntime-gpu        # optional, NVIDIA GPU
    pip install onnxruntime-directml   # optional, AMD/Intel GPU on Windows
(opencv is optional: with it mask->polygon is faster; without it the code
 falls back to GDAL Polygonize.)

Pipeline:
    1. Tile at 56 m ground blocks + 50% overlap, resample each to 1024 px.
    2. Single-class YOLO-seg ONNX inference (low det_conf) + per-tile box NMS.
    3. Masks -> world-coordinate polygons.
    4. Dissolve overlapping/touching polygons across tiles -> one crown each.
    5. Re-detection (centre-hit): for a stitched (multi-fragment) crown, re-sample
       a same-scale window on its bbox centre. If a fresh detection covers that
       centre the crown is confirmed and its clean outline replaces the loose
       blob; if nothing covers the centre the merge is discarded as a false
       merge. Confidence is NOT re-judged here - the fragment stage already
       validated it, so the fragment-stage confidence (best_raw) is kept.
    6. Filter by keep_conf / area / aspect ratio -> write output.

UI language: Spanish (strings are driven by the STRINGS table via tr(); an
English table is kept for reference but _lang() returns "es").
"""

import base64
import os
import threading
import queue
import numpy as np

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterDefinition,
    QgsProcessingException,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsGeometry,
    QgsPointXY,
    QgsSpatialIndex,
    QgsVectorLayer,
    QgsFillSymbol,
    QgsSingleSymbolRenderer,
    QgsPalLayerSettings,
    QgsTextFormat,
    QgsTextBufferSettings,
    QgsVectorLayerSimpleLabeling,
    QgsProcessingLayerPostProcessorInterface,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


TILE_M_DEFAULT = 56.0        # ground block size (m); must match training
OVERLAP_DEFAULT = 0.50       # 50% overlap; must match training
MIN_AREA_DEFAULT = 30.0      # minimum crown area (m^2)
MAX_ASPECT_INTERNAL = 4.0    # long/short OBB ratio above which a shape is a
                             # sliver/spike. Internal: anything filtered by it
                             # is degenerate, so it is not exposed in the UI.

# --- lone-fragment tile-cut detection (n_frag == 1 only) -------------------
# A real crown outline is organic (short, direction-varying edges). A crown that
# was clipped at a tile boundary keeps one or more long straight edges (the cut
# line). Summing the length of all "long straight runs" is robust: a clipped
# fragment has one or more long straight sides (their total is a big fraction of
# the perimeter), while an organic crown has essentially none. Applied ONLY to
# lone fragments (n_frag == 1); merged clusters are handled by re-detection.
CUT_LONGEDGE_FRAC = 0.20     # a single straight run >= this fraction of the
                             # perimeter counts as a "long" (tile-cut-like) edge
CUT_STRAIGHT_FRAC = 0.40     # if long straight runs together make up >= this
                             # fraction of the perimeter, the outline is
                             # dominated by straight cut edges -> a clipped crown
CUT_TURN_DEG = 12.0          # consecutive edges within this turn angle are one run


# ---------------------------------------------------------------------------
# i18n : English / Spanish
# ---------------------------------------------------------------------------

def _lang() -> str:
    """UI language. The plugin currently ships Spanish-only."""
    return "es"


def tr(key: str) -> str:
    """Look up a UI string in the current language."""
    lang = _lang()
    return STRINGS.get(lang, STRINGS["en"]).get(
        key, STRINGS["en"].get(key, key))


_HELP_EN = """
<div style="font-family: Segoe UI, Arial, sans-serif; line-height:1.5;">
<div style="text-align:center; margin-bottom:12px;">{LOGOS}</div>
<h3 style="color:#2c7bb6;">ARBOR &mdash; Shihuahuaco Detector</h3>
<p>Single-class segmentation of <i>Dipteryx</i> (Shihuahuaco) tree crowns in
UAV orthomosaics.
<h4>Pipeline</h4>
<ol>
<li><b>Ground-referenced tiling</b> &ndash; 56&nbsp;m blocks, 50% overlap,
resampled to 1024&nbsp;px (matches training GSD).</li>
<li><b>YOLO-seg</b> &ndash; per-tile inference with box NMS.</li>
<li><b>Dissolve</b> &ndash; overlapping detections merged across tiles into
single crowns.</li>
<li><b>Re-detection (centre-hit)</b> &ndash; each stitched crown is re-sampled
at its centre; if a fresh detection sits on that centre the crown is confirmed
and its clean outline is used, otherwise the merge is discarded.</li>
</ol>
<h4>Key parameters</h4>
<ul>
<li><b>det-conf</b> (0.25): low detection threshold so dissolve can stitch
crowns across tiles.</li>
<li><b>keep-conf</b> (0.50): final per-crown confidence filter.</li>
<li><b>min area</b> (30&nbsp;m&sup2;): drop tiny shapes.</li>
</ul>
<h4>Dependencies</h4>
pip install numpy onnxruntime<br>
pip install onnxruntime-gpu&nbsp;&nbsp;# NVIDIA GPU (optional)
</div>
"""

_HELP_ES = """
<div style="font-family: Segoe UI, Arial, sans-serif; line-height:1.5;">
<div style="text-align:center; margin-bottom:12px;">{LOGOS}</div>
<h3 style="color:#2c7bb6;">ARBOR &mdash; Detector de Shihuahuaco</h3>
<p>Segmentaci&oacute;n de una sola clase de copas de <i>Dipteryx</i>
(Shihuahuaco) en ortomosaicos de dron.
<h4>Flujo de procesamiento</h4>
<ol>
<li><b>Teselado referenciado al terreno</b> &ndash; bloques de 56&nbsp;m,
50% de solape, remuestreados a 1024&nbsp;px (coincide con el GSD de
entrenamiento).</li>
<li><b>YOLO-seg</b> &ndash; inferencia por tesela con NMS de cajas.</li>
<li><b>Disoluci&oacute;n</b> &ndash; detecciones solapadas fusionadas entre
teselas en copas &uacute;nicas.</li>
<li><b>Re-detecci&oacute;n (acierto central)</b> &ndash; cada copa ensamblada
se re-muestrea en su centro; si una nueva detecci&oacute;n cae sobre ese centro
la copa se confirma y se usa su contorno limpio; de lo contrario, la
fusi&oacute;n se descarta.</li>
</ol>
<h4>Par&aacute;metros clave</h4>
<ul>
<li><b>det-conf</b> (0.25): umbral de detecci&oacute;n bajo para que la
disoluci&oacute;n ensamble copas entre teselas.</li>
<li><b>keep-conf</b> (0.50): filtro final de confianza por copa.</li>
<li><b>&aacute;rea m&iacute;nima</b> (30&nbsp;m&sup2;): descarta formas
diminutas.</li>
</ul>
<h4>Dependencias</h4>
pip install numpy onnxruntime<br>
pip install onnxruntime-gpu&nbsp;&nbsp;# GPU NVIDIA (opcional)
</div>
"""

STRINGS = {
    "en": {
        "alg_name": "Detect Shihuahuaco Trees",
        "p_language": "Language / Idioma",
        "p_input": "Input orthomosaic (RGB)",
        "p_model": "YOLO-seg model (.onnx)",
        "p_det_conf": "Detection threshold (det-conf, low for stitching)",
        "p_keep_conf": "Keep threshold (keep-conf, high to drop clutter)",
        "p_min_area": "Minimum crown area (m\u00b2)",
        "p_redetect": "Verify stitched crowns by re-detection (centre-hit)",
        "p_tile_m": "Ground block size (m) - must match training",
        "p_overlap": "Tile overlap ratio - must match training",
        "p_center_tol": "Re-detection centre-hit tolerance (m) (Advanced)",
        "p_output": "Shihuahuaco crowns",
        "p_auto_style": "Auto-style result (outline + confidence labels)",
        "e_invalid_raster": "Invalid input raster.",
        "e_sink": "Could not create output sink.",
        "e_open_raster": "Cannot open raster: {}",
        "e_geotransform": "Raster has no valid geotransform / resolution.",
        "e_onnx": "onnxruntime is not installed. In the OSGeo4W Shell run:",
        "style_label": "Outline + confidence (auto)",
        "help_html": _HELP_EN,
    },
    "es": {
        "alg_name": "Detectar \u00e1rboles Shihuahuaco",
        "p_language": "Language / Idioma",
        "p_input": "Ortomosaico de entrada (RGB)",
        "p_model": "Modelo YOLO-seg (.onnx)",
        "p_det_conf": "Umbral de detecci\u00f3n (det-conf, bajo para el ensamblado)",
        "p_keep_conf": "Umbral de conservaci\u00f3n (keep-conf, alto para descartar ruido)",
        "p_min_area": "\u00c1rea m\u00ednima de copa (m\u00b2)",
        "p_redetect": "Verificar copas ensambladas por re-detecci\u00f3n (acierto central)",
        "p_tile_m": "Tama\u00f1o de bloque en el terreno (m) - debe coincidir con el entrenamiento",
        "p_overlap": "Proporci\u00f3n de solape entre teselas - debe coincidir con el entrenamiento",
        "p_center_tol": "Tolerancia de acierto central en la re-detecci\u00f3n (m) (Avanzado)",
        "p_output": "Copas de Shihuahuaco",
        "p_auto_style": "Aplicar estilo autom\u00e1tico (contorno + etiquetas de confianza)",
        "e_invalid_raster": "R\u00e1ster de entrada no v\u00e1lido.",
        "e_sink": "No se pudo crear la capa de salida.",
        "e_open_raster": "No se puede abrir el r\u00e1ster: {}",
        "e_geotransform": "El r\u00e1ster no tiene una geotransformaci\u00f3n / resoluci\u00f3n v\u00e1lida.",
        "e_onnx": "onnxruntime no est\u00e1 instalado. En la consola OSGeo4W ejecute:",
        "style_label": "Contorno + confianza (autom\u00e1tico)",
        "help_html": _HELP_ES,
    },
}


# ---------------------------------------------------------------------------
# Output styling: outline-only fill + confidence label, applied automatically
# when the result layer is loaded onto the canvas.

class _CrownStyler(QgsProcessingLayerPostProcessorInterface):
    """Style the output crown layer: transparent fill, coloured outline,
    and a per-crown confidence label (shown as a percentage)."""

    instance = None  # keep a reference alive so Python doesn't GC it

    def postProcessLayer(self, layer, context, feedback):
        if not isinstance(layer, QgsVectorLayer):
            return

        # Outline only (no fill) — boundary stays visible over the imagery.
        symbol = QgsFillSymbol.createSimple({
            "style": "no",                  # No Brush = transparent fill
            "outline_color": "230,30,30,255",
            "outline_width": "0.4",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        # Label each crown with its confidence as a percentage, e.g. 87%.
        settings = QgsPalLayerSettings()
        settings.fieldName = "round(\"confidence\" * 100) || '%'"
        settings.isExpression = True

        text_format = QgsTextFormat()
        text_format.setSize(9)
        text_format.setColor(QColor(20, 20, 20))

        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setSize(1.0)
        buf.setColor(QColor(255, 255, 255))
        text_format.setBuffer(buf)

        settings.setFormat(text_format)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)
        layer.triggerRepaint()

        # If the result was written to a GeoPackage on disk, embed this style
        # into the file (layer_styles table) so it is re-applied automatically
        # whenever the .gpkg is opened in QGIS later - not just this session.
        try:
            src = layer.source().lower()
            if layer.dataProvider().name() == "ogr" and ".gpkg" in src:
                layer.saveStyleToDatabase(
                    "arbor_default",
                    tr("style_label"),
                    True,        # use as default for this layer
                    "")
        except Exception:
            pass             # style still applied in-session even if embed fails

    @staticmethod
    def create() -> "QgsProcessingLayerPostProcessorInterface":
        _CrownStyler.instance = _CrownStyler()
        return _CrownStyler.instance


# ---------------------------------------------------------------------------
# Logo / help embedding
# ---------------------------------------------------------------------------

def _logo_uri(filename: str) -> str:
    """Return a base64 data URI for an image in the plugin's icons/ folder."""
    path = os.path.join(os.path.dirname(__file__), "icons", filename)
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif"}.get(ext, "image/png")
    try:
        with open(path, "rb") as fh:
            return f"data:{mime};base64," + base64.b64encode(fh.read()).decode("ascii")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------

def _load_ort_session(model_path: str, feedback):
    try:
        import onnxruntime as ort
    except ImportError:
        raise QgsProcessingException(
            tr("e_onnx") + "\n"
            "  pip install onnxruntime\n"
            "NVIDIA GPU:  pip install onnxruntime-gpu\n"
            "AMD/Intel:   pip install onnxruntime-directml"
        )
    avail = ort.get_available_providers()
    feedback.pushInfo(f"[ARBOR] Available ONNX providers: {avail}")
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4))
    order = []
    for p in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if p in avail:
            order.append(p)
    order.append("CPUExecutionProvider")
    try:
        sess = ort.InferenceSession(model_path, sess_options=so, providers=order)
    except Exception as exc:
        feedback.pushWarning(f"[ARBOR] GPU provider failed, falling back to CPU: {exc}")
        sess = ort.InferenceSession(model_path, sess_options=so,
                                    providers=["CPUExecutionProvider"])
    feedback.pushInfo(f"[ARBOR] Active providers: {sess.get_providers()}")
    return sess


# ---------------------------------------------------------------------------
# Resize (numpy bilinear, cv2 if available)
# ---------------------------------------------------------------------------

def _resize_bilinear_np(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    in_h, in_w = img.shape[:2]
    if in_w == out_w and in_h == out_h:
        return img.copy()
    ys = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    ys = np.clip(ys, 0, in_h - 1)
    xs = np.clip(xs, 0, in_w - 1)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1); x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0); wx = (xs - x0)
    if img.ndim == 3:
        wy = wy[:, None, None]; wx = wx[None, :, None]
    else:
        wy = wy[:, None]; wx = wx[None, :]
    Ia = img[np.ix_(y0, x0)]; Ib = img[np.ix_(y0, x1)]
    Ic = img[np.ix_(y1, x0)]; Id = img[np.ix_(y1, x1)]
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bot * wy


def _imresize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if HAS_CV2:
        interp = cv2.INTER_AREA if (img.shape[1] > w) else cv2.INTER_LINEAR
        return cv2.resize(img, (w, h), interpolation=interp).astype(np.float32)
    return _resize_bilinear_np(img, w, h)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def _geom_iou(a: QgsGeometry, b: QgsGeometry) -> float:
    """IoU of two polygon geometries (0 if disjoint). Reference metric only."""
    inter = a.intersection(b)
    if inter is None or inter.isEmpty():
        return 0.0
    ai = inter.area()
    uni = a.area() + b.area() - ai
    return ai / uni if uni > 1e-9 else 0.0


def _nms_per_tile(dets: list, iou_thresh: float = 0.7) -> list:
    """Greedy per-tile box NMS so we only decode masks for survivors."""
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: dets[i]["conf"], reverse=True)
    boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets]
    keep = []
    suppressed = [False] * len(dets)
    for idx_pos, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(dets[i])
        for j in order[idx_pos + 1:]:
            if not suppressed[j] and _iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed[j] = True
    return keep


def _mask_contours_px(binary: np.ndarray) -> list:
    """Binary mask -> list of exterior contour rings [(N,2), ...] in pixel
    coords (x=col, y=row). Uses cv2 if available, else GDAL Polygonize."""
    b = np.ascontiguousarray(binary.astype(np.uint8))
    if HAS_CV2:
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1, 2).astype(np.float32) for c in cnts if len(c) >= 3]
    return _polygonize_gdal(b)


def _polygonize_gdal(b: np.ndarray) -> list:
    """Fallback (no cv2): vectorize a binary mask with GDAL Polygonize."""
    from osgeo import gdal, ogr
    h, w = b.shape
    rds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    band = rds.GetRasterBand(1)
    band.WriteArray(b)
    vds = ogr.GetDriverByName("Memory").CreateDataSource("m")
    lyr = vds.CreateLayer("p", geom_type=ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("v", ogr.OFTInteger))
    gdal.Polygonize(band, band, lyr, 0, [], callback=None)  # maskBand=band -> skip 0
    rings = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        ring = geom.GetGeometryRef(0)
        if ring is None:
            continue
        pts = np.array([(ring.GetX(k), ring.GetY(k))
                        for k in range(ring.GetPointCount())], dtype=np.float32)
        if len(pts) >= 3:
            rings.append(pts)
    return rings


def _is_tile_cut_fragment(geom: QgsGeometry) -> bool:
    """Detect a lone fragment that is really a crown clipped at a tile boundary:
    its outline is dominated by long straight edges (the cut line(s)). Real
    crowns are organic (short, direction-varying edges) and have essentially no
    long straight runs, so they are not touched. Returns True -> drop it.

    Criterion: sum the lengths of all straight runs that are individually >=
    CUT_LONGEDGE_FRAC of the perimeter; if that total is >= CUT_STRAIGHT_FRAC of
    the perimeter, the outline is straight-cut-dominated -> a clipped fragment.
    """
    try:
        import math
        poly = geom
        parts = geom.asGeometryCollection()
        if parts:
            poly = max(parts, key=lambda p: p.area())
        pts = poly.asPolygon()
        if not pts or len(pts[0]) < 4:
            return False
        ring = pts[0]
        # per-edge angle + length
        seg = []
        for i in range(len(ring) - 1):
            dx = ring[i + 1].x() - ring[i].x()
            dy = ring[i + 1].y() - ring[i].y()
            L = math.hypot(dx, dy)
            if L > 1e-9:
                seg.append((math.atan2(dy, dx), L))
        if not seg:
            return False
        perim = sum(L for _, L in seg)
        if perim < 1e-6:
            return False
        # group consecutive near-collinear edges into straight runs
        runs = []
        run = seg[0][1]
        for k in range(1, len(seg)):
            da = abs(seg[k][0] - seg[k - 1][0])
            da = min(da, 2 * math.pi - da)
            if da < math.radians(CUT_TURN_DEG):
                run += seg[k][1]
            else:
                runs.append(run); run = seg[k][1]
        runs.append(run)
        # total length of individually-long straight runs
        straight_total = sum(r for r in runs if r >= CUT_LONGEDGE_FRAC * perim)
        return straight_total >= CUT_STRAIGHT_FRAC * perim
    except Exception:
        return False


def _aspect_ratio(geom: QgsGeometry) -> float:
    """Oriented-bounding-box long/short ratio. Thin slivers/spikes -> large."""
    try:
        res = geom.orientedMinimumBoundingBox()
        # PyQGIS returns (geometry, area, angle, width, height)
        width = float(res[2]); height = float(res[3])
        if len(res) >= 5:
            width = float(res[3]); height = float(res[4])
        lo, hi = min(width, height), max(width, height)
        return hi / lo if lo > 1e-6 else 999.0
    except Exception:
        return 1.0


def _clean_polygon(geom: QgsGeometry):
    """Reduce any geometry to a single, non-empty Polygon suitable for a
    Polygon sink. makeValid() / intersection() / union can return a
    GeometryCollection (a polygon plus dangling lines/points) or a
    MultiPolygon; both are rejected by a single-Polygon layer -> the feature
    is silently dropped on write. Explode to parts, keep only polygon parts,
    and return the largest by area."""
    if geom is None or geom.isEmpty():
        return None
    try:
        parts = geom.asGeometryCollection()      # [geom] for a single geometry
    except Exception:
        parts = [geom]
    polys = [p for p in parts
             if p is not None and not p.isEmpty()
             and p.type() == QgsWkbTypes.PolygonGeometry]
    if not polys:
        return None
    return max(polys, key=lambda p: p.area())


# ---------------------------------------------------------------------------
# YOLO-seg ONNX output parsing  (single-class; nc derived from shape)
# ---------------------------------------------------------------------------

def _parse_yolo_seg(outputs: list, conf_thresh: float,
                    input_w: int, input_h: int) -> list:
    """
    Parse YOLO-seg ONNX outputs.
      output0 shape (1, 4+nc+32, num_anchors): boxes + class scores + mask coeffs
      output1 shape (1, 32, mh, mw): mask prototypes
    Returns detection dicts with x1,y1,x2,y2 (model-input pixel space),
    conf, mask_coeffs, prototypes.
    """
    raw = outputs[0][0]          # (4+nc+32, num_anchors)
    prototypes = outputs[1]      # (1, 32, mh, mw)
    nc = raw.shape[0] - 4 - 32

    bbox = raw[0:4, :]
    scores = raw[4:4 + nc, :]
    coeffs = raw[4 + nc:, :]

    conf = scores.max(axis=0)
    mask = conf >= conf_thresh
    if not mask.any():
        return []

    bbox = bbox[:, mask]
    conf = conf[mask]
    coeffs = coeffs[:, mask]

    results = []
    for i in range(bbox.shape[1]):
        cx, cy, w, h = bbox[:, i]
        x1 = max(0.0, float(cx - w / 2))
        y1 = max(0.0, float(cy - h / 2))
        x2 = min(float(input_w), float(cx + w / 2))
        y2 = min(float(input_h), float(cy + h / 2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        results.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": float(conf[i]),
            "mask_coeffs": coeffs[:, i],
            "prototypes": prototypes,
        })
    return results


def _det_to_input_rings(det: dict, in_w: int, in_h: int) -> list:
    """
    Decode the mask at FULL model-input resolution, then take contours.
    Returns contour rings [(N,2), ...] in model-input pixel space.
    """
    proto = det["prototypes"][0]              # (32, mh, mw)
    _, mh, mw = proto.shape
    coeffs = det["mask_coeffs"].astype(np.float32)
    raw = (coeffs @ proto.reshape(32, -1)).reshape(mh, mw)
    raw = 1.0 / (1.0 + np.exp(-raw))                   # sigmoid (proto resolution)
    full = _imresize(raw, in_w, in_h)                  # upsample to model input

    x1i = max(0, int(det["x1"])); y1i = max(0, int(det["y1"]))
    x2i = min(in_w, int(det["x2"])); y2i = min(in_h, int(det["y2"]))
    binary = np.zeros((in_h, in_w), dtype=np.uint8)
    if x2i > x1i and y2i > y1i:
        binary[y1i:y2i, x1i:x2i] = (full[y1i:y2i, x1i:x2i] > 0.5).astype(np.uint8)
    return [r.astype(np.float32) for r in _mask_contours_px(binary)]


def _infer_window(sess, inp_name, ds, gt, inv_gt, in_w, in_h,
                  raster_w, raster_h, tile_px_x, tile_px_y,
                  center_wx, center_wy, det_conf):
    """
    Run ONE YOLO-seg inference on a fixed-ground-size window centred on a world
    coordinate, and return world-space crown polygons.

    The window keeps the SAME ground size (tile_px_x/y) as the main tiling, so
    the effective GSD matches training and the geometry/confidence are on the
    same scale as the main pass. The only difference is that the crown is now
    centred in one window instead of split across tile seams.

    Returns [(QgsGeometry world-polygon, conf), ...].
    """
    px = inv_gt[0] + center_wx * inv_gt[1] + center_wy * inv_gt[2]
    py = inv_gt[3] + center_wx * inv_gt[4] + center_wy * inv_gt[5]
    tx = int(round(px - tile_px_x / 2.0))
    ty = int(round(py - tile_px_y / 2.0))
    tx = max(0, min(tx, raster_w - tile_px_x))
    ty = max(0, min(ty, raster_h - tile_px_y))
    ntw = min(tile_px_x, raster_w - tx)
    nth = min(tile_px_y, raster_h - ty)
    if ntw <= 0 or nth <= 0:
        return []

    bands = []
    for b in range(1, 4):
        a = ds.GetRasterBand(b).ReadAsArray(tx, ty, ntw, nth)
        if a is None:
            return []
        bands.append(a)
    tile_rgb = np.stack(bands, axis=-1).astype(np.float32)

    resized = _imresize(tile_rgb, in_w, in_h)
    inp = (resized / 255.0).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    sx = ntw / float(in_w); sy = nth / float(in_h)

    outs = sess.run(None, {inp_name: inp})
    if outs[0].ndim == 4:            # ensure [dets, protos] order
        outs = [outs[1], outs[0]]
    dets = _parse_yolo_seg(outs, det_conf, in_w, in_h)
    dets = _nms_per_tile(dets, 0.7)

    results = []
    for det in dets:
        for ring in _det_to_input_rings(det, in_w, in_h):
            if len(ring) < 3:
                continue
            ptsxy = []
            for (cx, cy) in ring:
                X = tx + cx * sx; Y = ty + cy * sy
                wx = gt[0] + X * gt[1] + Y * gt[2]
                wy = gt[3] + X * gt[4] + Y * gt[5]
                ptsxy.append(QgsPointXY(wx, wy))
            g = QgsGeometry.fromPolygonXY([ptsxy])
            if g and not g.isEmpty():
                if not g.isGeosValid():
                    g = g.makeValid()
                results.append((g, float(det["conf"])))
    return results


# ---------------------------------------------------------------------------
# Processing Algorithm
# ---------------------------------------------------------------------------

class ShihuahuacoDetectionAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    YOLO_MODEL = "YOLO_MODEL"
    DET_CONF = "DET_CONF"
    KEEP_CONF = "KEEP_CONF"
    MIN_AREA = "MIN_AREA"
    TILE_M = "TILE_M"
    OVERLAP = "OVERLAP"
    CENTER_TOL_M = "CENTER_TOL_M"
    OUTPUT = "OUTPUT"

    def name(self): return "detect_shihuahuaco"
    def displayName(self): return "Shihuahuaco"
    def group(self): return ""
    def groupId(self): return ""
    def createInstance(self): return ShihuahuacoDetectionAlgorithm()

    def icon(self):
        from qgis.PyQt.QtGui import QIcon
        p = os.path.join(os.path.dirname(__file__), "icons", "icon.png")
        return QIcon(p) if os.path.exists(p) else super().icon()

    def shortHelpString(self) -> str:
        """Right-panel help text (localised). Logos are base64 data URIs from
        the icons/ subfolder."""
        # Logos: edit this list — (filename in icons/, display width in px).
        # Two extra slots are included for your own logos; point them at your
        # files (or delete any line you don't want).
        logo_entries = ""
        for fname, width in [  
            ("logo.png",      320), 
        ]:
            uri = _logo_uri(fname)
            if uri:
                logo_entries += (
                    f'<img src="{uri}" width="{width}" '
                    f'style="margin:6px 8px;">'
                )
        return tr("help_html").replace("{LOGOS}", logo_entries)

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, tr("p_input")))
        self.addParameter(QgsProcessingParameterFile(
            self.YOLO_MODEL, tr("p_model"), extension="onnx"))
        self.addParameter(QgsProcessingParameterNumber(
            self.DET_CONF, tr("p_det_conf"),
            QgsProcessingParameterNumber.Double, 0.25, minValue=0.01, maxValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.KEEP_CONF, tr("p_keep_conf"),
            QgsProcessingParameterNumber.Double, 0.50, minValue=0.01, maxValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_AREA, tr("p_min_area"),
            QgsProcessingParameterNumber.Double, MIN_AREA_DEFAULT, minValue=0.0))

        for pid, key, default, lo, hi in [
            (self.TILE_M, "p_tile_m", TILE_M_DEFAULT, 8.0, 256.0),
            (self.OVERLAP, "p_overlap", OVERLAP_DEFAULT, 0.0, 0.9),
        ]:
            p = QgsProcessingParameterNumber(
                pid, tr(key), QgsProcessingParameterNumber.Double, default,
                minValue=lo, maxValue=hi)
            p.setFlags(p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
            self.addParameter(p)

        # Tolerance (m) for the centre-hit test: how far a re-detection may sit
        # from the stitched blob's sampling centre and still count as the same
        # object (covers the L-shape case where the bbox centre lands just
        # outside a concave crown). Advanced.
        p_tol = QgsProcessingParameterNumber(
            self.CENTER_TOL_M, tr("p_center_tol"),
            QgsProcessingParameterNumber.Double, 2.0, minValue=0.0, maxValue=50.0)
        p_tol.setFlags(p_tol.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(p_tol)

        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, tr("p_output")))

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        raster = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        model_path = self.parameterAsFile(parameters, self.YOLO_MODEL, context)
        det_conf = self.parameterAsDouble(parameters, self.DET_CONF, context)
        keep_conf = self.parameterAsDouble(parameters, self.KEEP_CONF, context)
        min_area = self.parameterAsDouble(parameters, self.MIN_AREA, context)
        tile_m = self.parameterAsDouble(parameters, self.TILE_M, context)
        overlap = self.parameterAsDouble(parameters, self.OVERLAP, context)
        center_tol = self.parameterAsDouble(parameters, self.CENTER_TOL_M, context)
        redetect = True   # re-detection verification is always on (built-in)

        if raster is None:
            raise QgsProcessingException(tr("e_invalid_raster"))

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("confidence", QVariant.Double))
        fields.append(QgsField("area_m2", QVariant.Double))
        # Reference-only columns from re-detection (NULL for un-re-detected
        # crowns); they do not affect keep/drop, which is decided by centre-hit.
        fields.append(QgsField("redet_conf", QVariant.Double))
        fields.append(QgsField("redet_iou", QVariant.Double))
        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.Polygon, raster.crs())
        if sink is None:
            raise QgsProcessingException(tr("e_sink"))

        feedback.pushInfo("Loading YOLO-seg ONNX model...")
        sess = _load_ort_session(model_path, feedback)
        inp_meta = sess.get_inputs()[0]
        inp_name = inp_meta.name
        yshape = inp_meta.shape
        in_h = yshape[2] if (len(yshape) == 4 and isinstance(yshape[2], int) and yshape[2] > 0) else 1024
        in_w = yshape[3] if (len(yshape) == 4 and isinstance(yshape[3], int) and yshape[3] > 0) else 1024
        feedback.pushInfo(f"Model input size: {in_w} x {in_h}")

        from osgeo import gdal
        ds_meta = gdal.Open(raster.source(), gdal.GA_ReadOnly)
        if ds_meta is None:
            raise QgsProcessingException(tr("e_open_raster").format(raster.source()))
        raster_w = ds_meta.RasterXSize
        raster_h = ds_meta.RasterYSize
        gt = ds_meta.GetGeoTransform()
        ds_meta = None
        gsd_x = abs(gt[1]); gsd_y = abs(gt[5])
        if gsd_x <= 0 or gsd_y <= 0:
            raise QgsProcessingException(tr("e_geotransform"))

        # Ground-meter window -> native pixels -> resize to model input.
        tile_px_x = max(8, int(round(tile_m / gsd_x)))
        tile_px_y = max(8, int(round(tile_m / gsd_y)))
        step_x = max(1, int(round(tile_px_x * (1.0 - overlap))))
        step_y = max(1, int(round(tile_px_y * (1.0 - overlap))))
        cols = list(range(0, max(1, raster_w - tile_px_x + 1), step_x))
        rows = list(range(0, max(1, raster_h - tile_px_y + 1), step_y))
        if not cols or cols[-1] != raster_w - tile_px_x:
            cols.append(max(0, raster_w - tile_px_x))
        if not rows or rows[-1] != raster_h - tile_px_y:
            rows.append(max(0, raster_h - tile_px_y))
        cols = sorted(set(c for c in cols if c >= 0))
        rows = sorted(set(r for r in rows if r >= 0))
        total = len(cols) * len(rows)
        feedback.pushInfo(
            f"Native GSD={gsd_x:.4f} m  block={tile_m} m -> {tile_px_x} px "
            f"-> resize {in_w} px  | ~{total} tiles")
        feedback.setProgress(5)

        # ==================== Phase 1: tiled inference (prefetch) ====================
        tile_queue = queue.Queue(maxsize=8)
        SENTINEL = None
        polys = []                  # [(QgsGeometry, conf)]
        producer_err = []

        def producer():
            try:
                _ds = gdal.Open(raster.source(), gdal.GA_ReadOnly)
                if _ds is None:
                    producer_err.append("producer could not open raster")
                    return
                for ty in rows:
                    for tx in cols:
                        ntw = min(tile_px_x, raster_w - tx)
                        nth = min(tile_px_y, raster_h - ty)
                        if ntw <= 0 or nth <= 0:
                            continue
                        try:
                            bands = []
                            ok = True
                            for b in range(1, 4):
                                a = _ds.GetRasterBand(b).ReadAsArray(tx, ty, ntw, nth)
                                if a is None:
                                    ok = False; break
                                bands.append(a)
                            if not ok:
                                continue
                            tile_rgb = np.stack(bands, axis=-1)
                        except Exception:
                            continue
                        if float(tile_rgb.mean()) < 8:
                            continue
                        if (np.all(tile_rgb == 0, axis=-1).mean() > 0.9):
                            continue
                        resized = _imresize(tile_rgb.astype(np.float32), in_w, in_h)
                        inp = (resized / 255.0).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
                        sx = ntw / float(in_w); sy = nth / float(in_h)
                        tile_queue.put((tx, ty, sx, sy, inp))
                _ds = None
            except Exception as exc:
                producer_err.append(str(exc))
            finally:
                tile_queue.put(SENTINEL)

        th = threading.Thread(target=producer, daemon=True)
        th.start()

        tile_n = 0
        while True:
            item = tile_queue.get()
            if item is SENTINEL:
                break
            tx, ty, sx, sy, inp = item
            tile_n += 1
            if feedback.isCanceled():
                while True:
                    try:
                        tile_queue.get_nowait()
                    except queue.Empty:
                        break
                th.join(timeout=5)
                return {self.OUTPUT: dest_id}
            if tile_n % 25 == 0:
                feedback.setProgress(5 + int(70 * tile_n / max(total, 1)))
                feedback.pushInfo(f"  [inference] {tile_n}/{total}")

            try:
                outs = sess.run(None, {inp_name: inp})
                if outs[0].ndim == 4:            # ensure [dets, protos] order
                    outs = [outs[1], outs[0]]
                dets = _parse_yolo_seg(outs, det_conf, in_w, in_h)
                dets = _nms_per_tile(dets, 0.7)
                for det in dets:
                    for ring in _det_to_input_rings(det, in_w, in_h):
                        if len(ring) < 3:
                            continue
                        ptsxy = []
                        for (cx, cy) in ring:
                            X = tx + cx * sx; Y = ty + cy * sy
                            wx = gt[0] + X * gt[1] + Y * gt[2]
                            wy = gt[3] + X * gt[4] + Y * gt[5]
                            ptsxy.append(QgsPointXY(wx, wy))
                        g = QgsGeometry.fromPolygonXY([ptsxy])
                        if g and not g.isEmpty():
                            if not g.isGeosValid():
                                g = g.makeValid()
                            polys.append((g, det["conf"]))
            except Exception as exc:
                feedback.pushWarning(f"  tile ({tx},{ty}) failed: {exc}")
                continue

        th.join(timeout=5)
        if producer_err:
            feedback.pushWarning("producer: " + "; ".join(producer_err))

        feedback.pushInfo(f"Raw polygons: {len(polys)}  -> merging...")
        feedback.setProgress(80)
        if not polys:
            feedback.pushWarning("No detections. Lower det-conf or check model/imagery.")
            return {self.OUTPUT: dest_id}

        # ==================== Phase 2: dissolve + re-detect + filter ==========
        index = QgsSpatialIndex()
        feats_geom = []
        for i, (g, c) in enumerate(polys):
            f = QgsFeature(i)
            f.setGeometry(g)
            index.addFeature(f)
            feats_geom.append((g, c))

        dissolved = QgsGeometry.unaryUnion([g for g, _ in polys])
        parts = dissolved.asGeometryCollection() if dissolved and not dissolved.isEmpty() else []
        feedback.pushInfo(f"Connected components: {len(parts)}")

        # Invert the geotransform once (world -> pixel) for re-detection reads,
        # and open the raster for the re-sampled windows.
        inv_gt = None
        ds_redetect = None
        if redetect:
            _inv = gdal.InvGeoTransform(gt)
            if _inv is None:
                feedback.pushWarning("Cannot invert geotransform; re-detection disabled.")
                redetect = False
            elif len(_inv) == 2 and isinstance(_inv[0], int):
                inv_gt = _inv[1]                       # legacy (success, gt) binding
            else:
                inv_gt = _inv
            if redetect:
                ds_redetect = gdal.Open(raster.source(), gdal.GA_ReadOnly)
                if ds_redetect is None:
                    feedback.pushWarning("Cannot reopen raster; re-detection disabled.")
                    redetect = False

        out_id = 0
        n_conf = n_area = n_shape = 0
        n_verified = n_rejected = n_single = n_cut = 0
        for part in parts:
            if part is None or part.isEmpty() or part.type() != QgsWkbTypes.PolygonGeometry:
                continue

            # Fragments that built this component + their best confidence.
            cand = index.intersects(part.boundingBox())
            contributors = [j for j in cand if feats_geom[j][0].intersects(part)]
            n_frag = len(contributors)
            best_raw = max((feats_geom[j][1] for j in contributors), default=0.0)

            use_geom = part
            use_conf = best_raw
            rdc = rdi = None            # re-detection conf / IoU (reference only)

            if redetect and n_frag >= 2:
                # --- stitched cluster: re-sample a SAME-SCALE window on the
                # bbox centre and verify by CENTRE-HIT (spatial association only,
                # no confidence re-gate: the fragment stage already validated
                # confidence). bbox centre (not area centroid) so a crown found
                # in only 3 of 4 quadrants still centres on the true centre. ---
                ctr = part.boundingBox().center()
                redets = _infer_window(
                    sess, inp_name, ds_redetect, gt, inv_gt,
                    in_w, in_h, raster_w, raster_h,
                    tile_px_x, tile_px_y, ctr.x(), ctr.y(), det_conf)

                center_pt = QgsGeometry.fromPointXY(ctr)
                chosen = None
                chosen_dist = None
                for (gr, cr) in redets:
                    d = gr.distance(center_pt)          # 0 when the centre is inside gr
                    if d <= center_tol and (chosen is None or d < chosen_dist):
                        chosen, chosen_dist = (gr, cr), d

                if chosen is None:
                    n_rejected += 1
                    continue
                use_geom = chosen[0]
                use_conf = best_raw
                rdc = chosen[1]
                rdi = _geom_iou(part, chosen[0])
                n_verified += 1
            elif n_frag < 2:
                n_single += 1                              # lone detection
                # A lone fragment never goes through re-detection, so a crown
                # clipped at a tile boundary (long straight cut edge + boxy
                # outline, like the staircase L-shapes) would pass straight
                # through. Drop those here; organic crowns are unaffected.
                if _is_tile_cut_fragment(use_geom):
                    n_cut += 1
                    continue

            # Final filters: confidence, area, aspect (aspect is internal).
            if use_conf < keep_conf:
                n_conf += 1; continue
            g = use_geom
            if not g.isGeosValid():
                g = g.makeValid()
            g = g.simplify(0.2) or g
            g = _clean_polygon(g)
            if g is None or g.isEmpty():
                n_shape += 1; continue
            area = g.area()
            if area < min_area:
                n_area += 1; continue
            if _aspect_ratio(g) > MAX_ASPECT_INTERNAL:
                n_shape += 1; continue

            out_id += 1
            feat = QgsFeature(fields)
            feat.setGeometry(g)
            feat.setAttributes([
                out_id,
                round(float(use_conf), 4),
                round(float(area), 1),
                None if rdc is None else round(float(rdc), 4),
                None if rdi is None else round(float(rdi), 4),
            ])
            sink.addFeature(feat)

        if ds_redetect is not None:
            ds_redetect = None

        if redetect:
            feedback.pushInfo(
                f"Re-detect: verified {n_verified}, "
                f"rejected {n_rejected} (no crown at centre), "
                f"single kept {n_single}")
        feedback.pushInfo(
            f"Filtered: low-conf {n_conf}, too-small {n_area}, sliver {n_shape}, "
            f"tile-cut {n_cut} -> {out_id} crowns")
        feedback.setProgress(100)

        # Auto-style the result on load (outline + confidence labels) — built-in.
        if context.willLoadLayerOnCompletion(dest_id):
            context.layerToLoadOnCompletionDetails(dest_id).setPostProcessor(
                _CrownStyler.create())

        return {self.OUTPUT: dest_id}