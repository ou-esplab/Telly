#!/usr/bin/env python3
"""
Build Telly_stationary_waves.pptx: an editable PowerPoint version of the lecture deck.

    pip install python-pptx            # once
    python3 build_pptx.py              # writes Telly_stationary_waves.pptx next to this script

Everything is native and editable: titles are real title placeholders (so the outline view
works), text is in ordinary text boxes, tables are real tables, the heating profile is a
native chart, and the figures are pictures from images/. Colors and fonts come from the
theme (Design > Variants > Colors/Fonts), so they can be restyled in one place.

Edit slide text in PowerPoint; to change the figures, replace the PNGs in images/ and re-run.
"""
import math
import os
import re
import sys

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.dml import MSO_THEME_COLOR as TC
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = lambda n: os.path.join(HERE, "images", n)
OUT = os.path.join(HERE, "Telly_stationary_waves.pptx")

# ---- theme colors (scheme slots) --------------------------------------------------------
NAVY, CARD, SOFT, PAGE = TC.TEXT_1, TC.BACKGROUND_1, TC.TEXT_2, TC.BACKGROUND_2
TEAL, RUST, ORG, LORG, PALE, TAN = (TC.ACCENT_1, TC.ACCENT_2, TC.ACCENT_3,
                                    TC.ACCENT_4, TC.ACCENT_5, TC.ACCENT_6)
THEME_COLORS = {"dk1": "14233A", "lt1": "FBFAF6", "dk2": "4A5568", "lt2": "F4F1EA",
                "accent1": "14707C", "accent2": "B8501A", "accent3": "D9692B",
                "accent4": "E8894A", "accent5": "C5D3E0", "accent6": "D9D4C7",
                "hlink": "14707C", "folHlink": "4A5568"}
SERIF = "+mj-lt"      # theme heading font (Cambria): titles, equations, big numbers

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
LAYOUT = prs.slide_layouts[5]          # "Title Only"
X0, W = 0.75, 11.83
WARN = []

# ---- theme: fonts + colors ------------------------------------------------------------------
def set_theme():
    part = prs.slide_master.part.part_related_by(RT.THEME)
    root = etree.fromstring(part.blob)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    clr = root.find(".//a:clrScheme", ns)
    clr.set("name", "Telly")
    for child in list(clr):
        tag = etree.QName(child).localname
        if tag in THEME_COLORS:
            for g in list(child):
                child.remove(g)
            etree.SubElement(child, "{%s}srgbClr" % ns["a"]).set("val", THEME_COLORS[tag])
    fs = root.find(".//a:fontScheme", ns)
    fs.set("name", "Telly")
    fs.find("a:majorFont/a:latin", ns).set("typeface", "Cambria")
    fs.find("a:minorFont/a:latin", ns).set("typeface", "Calibri")
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


set_theme()
prs.slide_master.background.fill.solid()
prs.slide_master.background.fill.fore_color.theme_color = PAGE

# ---- text helpers ----------------------------------------------------------------------------
def _run(p, spec, base):
    text, o = (spec, {}) if isinstance(spec, str) else spec
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(o.get("size", base["size"]))
    f.bold = o.get("bold", base["bold"])
    f.italic = o.get("italic", base["italic"])
    if o.get("font", base["font"]):
        f.name = o.get("font", base["font"])
    f.color.theme_color = o.get("color", base["color"])
    if base.get("spc"):
        r._r.get_or_add_rPr().set("spc", str(base["spc"]))


def _estimate(paras, w, size, bold, font):
    """Rough text height (inches) for a box of width w: avg char width ~0.5 em (sans) / 0.55 em (serif)."""
    cw = (0.55 if font else 0.50) * (1.07 if bold else 1.0)
    total = 0.0
    for p in paras:
        runs = p["runs"] if isinstance(p, dict) else p
        runs = [runs] if isinstance(runs, (str, tuple)) else runs
        text = "".join(r if isinstance(r, str) else r[0] for r in runs)
        psize = size
        for r in runs:
            if isinstance(r, tuple) and "size" in r[1]:
                psize = r[1]["size"]
        indent = 0.3 if isinstance(p, dict) and p.get("bullet") else 0
        lines = max(1, math.ceil(len(text) * cw * psize / 72.0 / max(w - indent, 0.3)))
        total += lines * psize * 1.2 / 72.0 + (0.08 if isinstance(p, dict) and p.get("bullet") else 0)
    return total


def text(slide, x, y, w, h, paras, size=14, color=NAVY, bold=False, italic=False, font=None,
         align=None, anchor=MSO_ANCHOR.TOP, gap=0, spc=None, name=None, check=True):
    """paras: str | list of (str | [runs] | {"runs":..., "bullet":True}); a run is str or (str, {opts})."""
    if isinstance(paras, (str, tuple)):
        paras = [paras]
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    base = dict(size=size, color=color, bold=bold, italic=italic, font=font, spc=spc)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = align
        if gap:
            p.space_after = Pt(gap)
        runs, bullet = (para["runs"], para.get("bullet")) if isinstance(para, dict) else (para, False)
        runs = [runs] if isinstance(runs, (str, tuple)) else runs
        for rs in runs:
            _run(p, rs, base)
        if bullet:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", "285750")
            pPr.set("indent", "-285750")
            etree.SubElement(pPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}buChar").set("char", "•")
    if check:
        est = _estimate(paras, w, size, bold, font) + gap * len(paras) / 72.0
        if est > h * 1.05:
            WARN.append(f"slide {len(prs.slides)}: text may overflow ({est:.2f} in needed, {h:.2f} in box): "
                        f"{str(paras[0])[:50]}")
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=NAVY, gap=6):
    return text(slide, x, y, w, h, [{"runs": it, "bullet": True} for it in items], size=size,
                color=color, gap=gap)


def card(slide, x, y, w, h, fill=CARD, line=TAN, line_w=0.75, radius=0.05):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.theme_color = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.theme_color = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def arrow(slide, x, y, w=0.5, h=0.4):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.theme_color = TEAL
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def eq(slide, x, y, w, h, lines, size=22, color=NAVY, italic=True, bold=False, gap=4, align=None):
    return text(slide, x, y, w, h, lines, size=size, color=color, italic=italic, bold=bold,
                font=SERIF, gap=gap, align=align)


def picture(slide, path, x, y, w, h, alt):
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def slide_number(slide, light=False):
    tb = slide.shapes.add_textbox(Inches(12.0), Inches(7.0), Inches(0.6), Inches(0.3))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(p._p, "{http://schemas.openxmlformats.org/drawingml/2006/main}fld")
    fld.set("id", "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}")
    fld.set("type", "slidenum")
    rpr = etree.SubElement(fld, "{http://schemas.openxmlformats.org/drawingml/2006/main}rPr")
    rpr.set("lang", "en-US")
    rpr.set("sz", "1000")
    sf = etree.SubElement(rpr, "{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill")
    etree.SubElement(sf, "{http://schemas.openxmlformats.org/drawingml/2006/main}schemeClr").set(
        "val", "bg2" if light else "tx2")
    etree.SubElement(fld, "{http://schemas.openxmlformats.org/drawingml/2006/main}t").text = "‹#›"


def set_title(slide, title, x, y, w, h, size, color, bold=True):
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(x), Inches(y), Inches(w), Inches(h)
    tf = t.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.text = ""
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = SERIF
    r.font.color.theme_color = color
    est = _estimate([title], w, size, True, SERIF)
    if est > h * 1.05:
        WARN.append(f"slide {len(prs.slides)}: title may wrap past its box: {title[:50]}")


def new_slide(dark=False, accent=False):
    s = prs.slides.add_slide(LAYOUT)
    if dark or accent:
        s.background.fill.solid()
        s.background.fill.fore_color.theme_color = NAVY if dark else ORG
    return s


def content(eyebrow, title, caption=None):
    s = new_slide()
    text(s, X0, 0.5, W, 0.3, eyebrow.upper(), size=12, color=TEAL, bold=True, spc=300, check=False)
    set_title(s, title, X0, 0.85, W, 0.95, 36, NAVY)
    text(s, X0, 7.0, 7.0, 0.3, "Telly · stationary Rossby waves", size=10, color=SOFT, check=False)
    slide_number(s)
    if caption:
        text(s, X0, 6.62, 11.0, 0.3, caption, size=11, color=SOFT)
    return s


def divider(part, title, sub, note=None):
    s = new_slide(dark=True)
    text(s, 0.9, 3.55, 11, 0.4, part.upper(), size=14, color=LORG, bold=True, spc=400, check=False)
    set_title(s, title, 0.9, 4.0, 11.5, 1.2, 60, PAGE)
    text(s, 0.9, 5.35, 11, 0.6, sub, size=22, color=PALE)
    if note:
        text(s, 0.9, 6.2, 11, 0.4, note, size=14, color=LORG)
    return s


def statement(eyebrow, title, sub):
    s = new_slide(dark=True)
    text(s, 0.9, 1.3, 11, 0.4, eyebrow.upper(), size=14, color=LORG, bold=True, spc=400, check=False)
    set_title(s, title, 0.9, 1.95, 10.8, 2.9, 44, PAGE)
    text(s, 0.9, 5.2, 9.8, 1.2, sub, size=20, color=PALE)
    text(s, X0, 7.0, 7.0, 0.3, "Telly · stationary Rossby waves", size=10, color=PALE, check=False)
    return s


def table(slide, x, y, w, colw, header, rows, size=16, row_h=0.5, bold_first=True, colors=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * (len(rows) + 1)))
    tbl = shape.table
    tot = sum(colw)
    for i, cw in enumerate(colw):
        tbl.columns[i].width = Inches(w * cw / tot)
    for r in tbl.rows:
        r.height = Inches(row_h)

    def put(cell, content_, fill, color, bold, fill_bright=0.0):
        cell.fill.solid()
        cell.fill.fore_color.theme_color = fill
        if fill_bright:
            cell.fill.fore_color.brightness = fill_bright
        cell.margin_left = cell.margin_right = Inches(0.12)
        cell.margin_top = cell.margin_bottom = Inches(0.05)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        spec = content_ if isinstance(content_, tuple) else (content_, {})
        _run(p, spec, dict(size=size, color=spec[1].get("color", color), bold=spec[1].get("bold", bold),
                           italic=False, font=None))

    for j, h in enumerate(header):
        put(tbl.cell(0, j), h, NAVY, PAGE, True)
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            put(tbl.cell(i, j), v, CARD if i % 2 else TAN, NAVY, bold_first and j == 0,
                0.0 if i % 2 else 0.55)
    return tbl


S1 = "Telly · stationary Rossby waves"
# ==================================================================================== 1 cover
s = new_slide(dark=True)
text(s, 0.9, 0.9, 11.5, 0.4, "TELLY ATMOSPHERIC TELECONNECTION MODEL", size=14, color=LORG, bold=True, spc=400, check=False)
set_title(s, "Why Are There Waves in the Time-Mean Flow?", 0.9, 2.1, 11.2, 2.4, 54, PAGE)
text(s, 0.9, 4.75, 10.5, 1.0, "From ERA5 observations, to simple equations, to the Telly model", size=22, color=PALE)
text(s, 0.9, 6.55, 11, 0.4, "[Presenter name] · [Course] · [Date]", size=14, color=PALE)

# ==================================================================================== 2 roadmap
s = content("Overview", "Roadmap")
cards = [("1", "Observe", "What does the time-mean atmosphere look like?"),
         ("2", "Understand", "Why? Simple equations from class"),
         ("3", "Telly", "From the class equations to a model"),
         ("4", "Experiments", "Three forcings switched on one at a time"),
         ("5", "Results", "What each forcing does to the pattern")]
cw = (W - 4 * 0.2) / 5
for i, (n, t_, d) in enumerate(cards):
    x = X0 + i * (cw + 0.2)
    card(s, x, 2.0, cw, 4.4)
    text(s, x + 0.2, 2.15, cw - 0.4, 1.0, n, size=54, color=TEAL, bold=True, font=SERIF, check=False)
    text(s, x + 0.2, 4.55, cw - 0.4, 0.45, t_, size=18, bold=True)
    text(s, x + 0.2, 5.05, cw - 0.4, 1.2, d, size=14, color=SOFT)

# ==================================================================================== 3 sec-obs
divider("Part 1", "Observations", "What does the time-mean atmosphere look like?")

# ==================================================================================== 4 obs-map
s = content("Part 1 · Observations", "ERA5: Time-Mean 500 hPa Height",
            "ERA5 daily 500 hPa height, 1940–2024: the time mean minus its zonal mean (class notebook)")
picture(s, IMG("era5_z500.png"), X0, 1.95, 7.2, 3.625,
        "Map of the ERA5 time-mean 500 hPa height with the zonal mean removed, 1940 to 2024: highs and lows of "
        "100 m or more over the North Pacific, North America, Europe, East Asia and the Southern Ocean")
text(s, 8.25, 1.95, 4.33, 0.5, "What do you see?", size=22, color=TEAL, bold=True)
bullets(s, 8.25, 2.6, 4.33, 3.6,
        ["The zonal mean has been removed: a zonally symmetric atmosphere would give a blank map",
         "Highs and lows of 100 m or more remain",
         "They stay at the same places in the time mean"], size=16, gap=10)

# ==================================================================================== 4b obs-count
s = content("Part 1 · Observations", "How Many Waves Fit Around the Globe?",
            "ERA5 daily 500 hPa height, 1940–2024: time mean minus zonal mean")
picture(s, IMG("era5_count.png"), X0, 1.95, 7.7, 7.7 * 930 / 1560,
        "ERA5 time-mean 500 hPa height departure from the zonal mean with a dashed band along 50 to 60 degrees north, marked with two highs and two lows, and below it the height along that band plotted against longitude, with the same two highs and two lows")
text(s, 8.65, 1.95, 3.9, 0.3, "Count the highs and lows", size=13, color=TEAL, bold=True)
text(s, 8.65, 2.3, 3.9, 0.9, "Follow the dashed band once around the globe: 2 highs (H) and 2 lows (L).", size=14)
text(s, 8.65, 3.3, 3.9, 0.3, "So about 2 waves", size=13, color=TEAL, bold=True)
text(s, 8.65, 3.65, 3.9, 0.9, "Their sizes and spacing are uneven, so waves 1 and 3 are present too.", size=14)
card(s, 8.65, 4.7, 3.9, 1.7, fill=NAVY, line=None)
text(s, 8.85, 4.82, 3.5, 0.3, "IN WAVENUMBERS", size=11, color=LORG, bold=True, spc=300)
text(s, 8.85, 5.15, 3.5, 1.2, "2 waves around 55°N is a wavelength of about 11,500 km: k ≈ 5 × 10⁻⁷ per metre.", size=14, color=PAGE)

# ==================================================================================== 5 obs-why
statement("A question for the class", "Why is the time-mean atmosphere not the same at every longitude?",
          "We will look for the answer with a hierarchy of models: simple equations first, then Telly.")

# ==================================================================================== 6 sec-class
divider("Part 2", "Simple Equations", "Understanding it with the equations from class")

# ==================================================================================== 7 class-split
s = content("Part 2 · Simple equations", "Splitting the Flow")
eq(s, X0, 1.95, 5.5, 0.8, [[("u = ū + ", {}), ("u*", {"color": TEAL}), (" + u′", {})]], size=32)
eq(s, 6.9, 1.95, 5.5, 0.8, [[("v = v̄ + ", {}), ("v*", {"color": TEAL}), (" + v′", {})]], size=32)
cw = (W - 0.4) / 3
for i, (sym, ttl, body, hl) in enumerate([
        ("ū", "Zonal mean", "The average around a latitude circle: the same at every longitude.", False),
        ("u*", "Stationary wave", "The part that stays put: the waves in the ERA5 map.", True),
        ("u′", "Transient eddies", "Day-to-day weather that comes and goes.", False)]):
    x = X0 + i * (cw + 0.2)
    card(s, x, 3.0, cw, 2.7, line=TEAL if hl else TAN, line_w=2.25 if hl else 0.75)
    text(s, x + 0.25, 3.15, cw - 0.5, 0.9, sym, size=40, color=TEAL if hl else NAVY, italic=True, font=SERIF, check=False)
    text(s, x + 0.25, 4.15, cw - 0.5, 0.4, ttl, size=18, bold=True)
    text(s, x + 0.25, 4.6, cw - 0.5, 1.0, body, size=14, color=SOFT)
text(s, X0, 6.0, W, 0.5, "η = ζ + f is the absolute vorticity; on the β-plane, f = f₀ + βy.", size=16)

# ==================================================================================== 8 class-vort
s = content("Part 2 · Simple equations", "Vorticity of the Stationary Wave")
card(s, X0, 1.95, W, 1.1)
eq(s, X0, 2.15, W, 0.8, [["∂ζ*/∂t + ū ∂ζ*/∂x + β v* = 0"]], size=32, align=PP_ALIGN.CENTER)
for i, (head, lines) in enumerate([
        ("Reading the terms", [[("∂ζ*/∂t", {"bold": True}), ": how the wave’s vorticity changes"],
                               [("ū ∂ζ*/∂x", {"bold": True}), ": carried eastward by the mean wind"],
                               [("β v*", {"bold": True}), ": changes as air moves north or south, where the planet’s own vorticity is different"]]),
        ("What we assumed", ["The waves are small, so products of wave terms are dropped",
                             "Day-to-day weather is ignored", "The flow is nearly non-divergent"])]):
    x = X0 + i * 6.03
    card(s, x, 3.3, 5.8, 3.15)
    text(s, x + 0.3, 3.45, 5.2, 0.4, head, size=18, color=TEAL, bold=True)
    text(s, x + 0.3, 3.95, 5.2, 2.4, lines, size=15, gap=8)

# ==================================================================================== 9 class-wave
s = content("Part 2 · Simple equations", "Streamfunction and the Wave Equation",
            "Here ω is the wave frequency, K² = k² + l², and β = 2Ω cos φ / a")
steps = [(1.95, "1 · Define a streamfunction", ["u* = −∂ψ*/∂y,  v* = ∂ψ*/∂x,  ζ* = ∇²ψ*"], 20, NAVY),
         (2.8, "2 · Substitute into the vorticity equation", ["(∂/∂t + ū ∂/∂x) ∇²ψ* + β ∂ψ*/∂x = 0"], 20, NAVY),
         (3.65, "3 · Try a wave, and find its frequency and phase speed",
          ["ψ* = A e^{i(kx + ly − ωt)}", "ω = ūk − βk/K²,   c = ω/k = ū − β/K²"], 20, NAVY),
         (5.0, "4 · Which waves can stand still? Set c = 0", ["k² + l² = Ks² = β/ū"], 24, TEAL)]
for y, lab, lines, sz, col in steps:
    text(s, X0, y, 7.2, 0.3, lab, size=13, color=TEAL, bold=True)
    eq(s, X0, y + 0.33, 7.2, 0.5 * len(lines), lines, size=sz, color=col, bold=(col == TEAL), gap=2)
text(s, X0, 5.93, 7.2, 0.55, "Only east–west (k) and north–south (l) wavenumbers that satisfy this stay put. Ks (s for stationary) is the total wavenumber of a stationary wave.",
     size=14, color=SOFT)
card(s, 8.3, 1.95, 4.28, 2.55, line=TEAL, line_w=2.0)
text(s, 8.55, 2.1, 3.8, 0.5, "STATIONARY DOES NOT MEAN NO MOTION", size=11, color=TEAL, bold=True, spc=100)
text(s, 8.55, 2.65, 3.8, 1.8, "Air still flows through the pattern at speed ū. The pattern stays put because "
                                "westward wave propagation, β/K², cancels the eastward flow.", size=14)
card(s, 8.3, 4.7, 4.28, 1.75, fill=NAVY, line=None)
text(s, 8.55, 4.85, 3.8, 0.3, "EXAMPLE", size=11, color=LORG, bold=True, spc=300)
text(s, 8.55, 5.2, 3.8, 1.2, "ū = 20 m/s at 45°N gives Ks ≈ 9 × 10⁻⁷ per metre. Two waves around the globe use k ≈ 4 × 10⁻⁷, which leaves l ≈ 8 × 10⁻⁷.",
     size=14, color=PAGE)

# ==================================================================================== 9b class-check
s = content("Part 2 · Simple equations", "Do the ERA5 Waves Fit?",
            "ERA5 1991–2020 time mean; 30°N–70°N; ū is the 500 hPa zonal-mean wind")
picture(s, IMG("era5_k_l.png"), X0, 1.95, 5.6, 5.6 * 960 / 1260,
        "Plot of east-west wavenumber k against north-south wavenumber l: a quarter circle for k squared plus l squared equal to Ks squared, a shaded band around it, and orange dots for the ERA5 stationary waves that cluster at one to three waves around the globe, many of them near the circle")
text(s, 6.95, 1.95, 5.6, 0.3, "1 · Measure k and l in ERA5", size=13, color=TEAL, bold=True)
text(s, 6.95, 2.3, 5.6, 0.8, "Split the 500 hPa pattern between 30°N and 70°N into waves around the globe (k) and waves from south to north (l).", size=14)
text(s, 6.95, 3.2, 5.6, 0.3, "2 · Draw the stationary circle", size=13, color=TEAL, bold=True)
text(s, 6.95, 3.55, 5.6, 0.8, "With the observed ū ≈ 10 m/s, k² + l² = β/ū is a circle with Ks ≈ 1.2 × 10⁻⁶ per metre.", size=14)
card(s, 6.95, 4.5, 5.6, 1.95, fill=NAVY, line=None)
text(s, 7.2, 4.62, 5.1, 0.3, "WHAT WE FIND", size=11, color=LORG, bold=True, spc=300)
text(s, 7.2, 4.95, 5.1, 1.45, "Almost all of the pattern is 1 to 3 waves around the globe, with one or two half-waves from south to north. "
                              "About 60% of it lies within 50% of the circle. The rest is larger scale.", size=14, color=PAGE)

# ==================================================================================== 10 class-source
s = content("Part 2 · Simple equations", "Adding a Source")
card(s, X0, 1.95, W, 1.65)
eq(s, X0 + 0.3, 2.05, W - 0.6, 0.7, [["∂ζ*/∂t + ū ∂ζ*/∂x + β v* = ", ("S", {"color": RUST})]], size=28)
text(s, X0 + 0.3, 2.85, W - 0.6, 0.7,
     [[("S", {"bold": True, "color": RUST}),
       " is a push on the wave: anything that squeezes or stretches columns of air, and so changes their spin."]],
     size=15, color=SOFT)
cw = (W - 0.4) / 3
for i, (ttl, body) in enumerate([
        ("Air forced over a slope", "A column pushed up a mountain is squeezed, which changes its spin."),
        ("Heating that lifts air", "Columns are stretched below the heating and squeezed above it."),
        ("Land–sea contrast", "Land and ocean warm differently: like heating over one and cooling over the other.")]):
    x = X0 + i * (cw + 0.2)
    card(s, x, 3.8, cw, 1.6)
    text(s, x + 0.2, 3.92, cw - 0.4, 0.4, ttl, size=16, bold=True)
    text(s, x + 0.2, 4.35, cw - 0.4, 1.0, body, size=14, color=SOFT)
card(s, X0, 5.6, W, 1.0, fill=NAVY, line=None)
eq(s, X0 + 0.3, 5.75, 4.3, 0.7, [["ψ* = S / [ik(β − ūK²)]"]], size=22, color=PAGE)
text(s, X0 + 4.8, 5.7, 6.7, 0.85,
     "The steady response is strongest when the size of the push matches Ks, the stationary wavenumber from the "
     "last slide: Ks = √(β/ū), where c = 0.", size=14, color=PALE)

# ==================================================================================== 11 class-thermo
s = content("Part 2 · Simple equations", "Heat, Pressure and Air Motion")
rows3 = [("Temperature", "The wind carries temperature; heating Q warms the air. There is no compression term yet.",
          [["∂T/∂t + u ∂T/∂x + v ∂T/∂y + ω ∂T/∂p = ", ("Q/cp", {"color": RUST})]]),
         ("Hydrostatic", "Temperature sets how far apart the pressure surfaces are.", [["∂Φ/∂p = −RT/p"]]),
         ("Continuity", "Air that converges sideways must move vertically.", [["∂u/∂x + ∂v/∂y + ∂ω/∂p = 0"]])]
for i, (ttl, body, eqs) in enumerate(rows3):
    y = 1.95 + i * 1.5
    card(s, X0, y, W, 1.35)
    text(s, X0 + 0.3, y + 0.15, 4.4, 0.4, ttl, size=18, color=TEAL, bold=True)
    text(s, X0 + 0.3, y + 0.58, 4.4, 0.75, body, size=14, color=SOFT)
    eq(s, X0 + 5.1, y + 0.3, 6.5, 0.8, eqs, size=22)

# ==================================================================================== 11b class-q
s = content("Part 2 · Simple equations", "What Is in Q?")
card(s, X0, 1.95, W, 0.85)
eq(s, X0, 2.1, W, 0.6, [[("Q = Qrad + ", {}), ("Qlatent", {"color": RUST}), (" + Qsurface", {})]], size=28, align=PP_ALIGN.CENTER)
table(s, X0, 3.0, W, [1.9, 5.0, 4.93], ["Process", "What it is", "In Telly"], [
    ["Radiation", "Sunlight absorbed, infrared emitted: warms the tropics, cools the poles",
     "Not computed; the pull toward the background temperature stands in for it"],
    [("Latent heating", {"color": RUST, "bold": True}), "Heat released when water vapor condenses into rain",
     "Prescribed from observed rainfall: the Q/cp term"],
    ["Surface heating", "Warm land or ocean heats the air above it by mixing",
     "Not computed; carried by the background temperature (the land–sea contrast)"]], size=14, row_h=0.5)
text(s, X0, 5.55, W, 0.4, "So far we have thought of Q as radiative heating. In Telly, Q is the latent heating only.",
     size=17, color=TEAL, bold=True)
text(s, X0, 6.0, W, 0.7, "Telly has no solar heating or outgoing longwave radiation, so no energy balance: the equator-to-pole "
                        "temperature contrast is imposed by the background state, not produced.", size=14, color=SOFT)

# ==================================================================================== 12 sec-model
divider("Part 3", "Telly", "From the class equations to a model")

# ==================================================================================== 13 hierarchy
s = content("Part 3 · Telly", "Where Telly Fits")
cw, gx = 3.45, 0.74
for i, (lab, ttl, body, hl) in enumerate([
        ("Observations", "Real atmosphere", "Everything acts at once, so the causes are hard to separate.", False),
        ("Class", "Simple equations", "Easy to understand, but many things are dropped.", False),
        ("Today", "Telly", "Keeps the main physics, and each forcing can be switched on or off.", True)]):
    x = X0 + i * (cw + gx)
    card(s, x, 1.95, cw, 3.2, line=TEAL if hl else TAN, line_w=2.25 if hl else 0.75)
    text(s, x + 0.25, 2.1, cw - 0.5, 0.3, lab.upper(), size=12, color=TEAL if hl else SOFT, bold=True, spc=300)
    text(s, x + 0.25, 2.5, cw - 0.5, 0.6, ttl, size=20, bold=True)
    text(s, x + 0.25, 4.0, cw - 0.5, 1.1, body, size=15, color=SOFT)
    if i < 2:
        arrow(s, x + cw + 0.12, 3.35, 0.5, 0.42)
text(s, X0, 5.5, 10.5, 1.0, "Each step answers a different question: what do we see, why does it happen, and how much "
                           "does each cause matter?", size=17, color=TEAL, bold=True)

# ==================================================================================== 14 what-is
s = content("Part 3 · Telly", "What Telly Is", "Telly · Kirtman et al. (2025), Bull. Amer. Meteor. Soc., 106, E2073–E2086")
table(s, X0, 1.95, W, [3.0, 8.83], ["Property", "Description"], [
    ["The physics", "The class equations for motion, heat and mass, without the simplifications"],
    ["The grid", "Boxes about 2° wide (about 200 km), in 26 layers up to about 2 hPa"],
    ["What it tracks", "Vorticity, divergence, temperature and surface pressure"],
    ["Simple stand-ins", "Friction, smoothing, a pull toward a background state, and heating from prescribed rainfall patterns"],
    ["Left out", "Detailed treatment of radiation, clouds, the boundary layer and the land surface"]], size=16, row_h=0.7)

# ==================================================================================== 15 anatomy
s = content("Part 3 · Telly", "Model Anatomy")
for i, (ttl, col, body) in enumerate([
        ("Heating, Q", RUST, "Rainfall with a prescribed pattern warms the air"),
        ("Topography, Φs", RUST, "Ground height sets the bottom of the atmosphere"),
        ("Background state", TEAL, "Average winds and temperatures the model is gently pulled toward")]):
    y = 1.95 + i * 1.5
    card(s, X0, y, 3.5, 1.35)
    text(s, X0 + 0.2, y + 0.12, 3.1, 0.4, ttl, size=18, color=col, bold=True)
    text(s, X0 + 0.2, y + 0.55, 3.1, 0.8, body, size=14, color=SOFT)
arrow(s, 4.4, 3.75, 0.6, 0.5)
card(s, 5.15, 1.95, 3.6, 4.35, fill=NAVY, line=None)
text(s, 5.4, 2.15, 3.1, 0.85, "What the model tracks", size=20, color=PAGE, bold=True)
eq(s, 5.4, 3.4, 3.1, 1.0, [["ζ, D, T, ln ps"]], size=26, color=LORG)
text(s, 5.4, 5.3, 3.1, 0.8, "Friction and smoothing keep it stable", size=14, color=PALE)
arrow(s, 8.9, 3.75, 0.6, 0.5)
card(s, 9.65, 1.95, 2.93, 4.35)
text(s, 9.85, 2.15, 2.55, 0.5, "Daily output", size=20, bold=True)
text(s, 9.85, 3.4, 2.55, 2.7, "Winds, temperature, height, heating and surface pressure, every day, in 30-day files",
     size=14, color=SOFT)

# ==================================================================================== 15b grid
s = content("Part 3 · Telly", "The Model Grid")
picture(s, IMG("model_grid.png"), X0, 1.95, W, W * 728 / 2158,
        "Two panels: on the left, the model's grid boxes about 2 degrees wide over North America, shaded land or ocean with "
        "one box highlighted; on the right, a vertical cross-section with 26 layers that follow the ground, which is about "
        "560 hPa over a high plateau and 1000 hPa over the ocean")
text(s, X0, 6.15, 11.0, 0.5, "About 480,000 boxes in all: 192 × 96 around the globe, in 26 layers.", size=16)

# ==================================================================================== 16 step-flow
s = content("Part 3 · From class to Telly", "Step 1: Keep the Whole Flow")
card(s, X0, 1.95, 5.5, 1.5)
text(s, X0 + 0.3, 2.08, 4.9, 0.3, "CLASS", size=11, color=SOFT, bold=True, spc=300)
eq(s, X0 + 0.3, 2.5, 4.9, 0.8, [["∂ζ*/∂t + ū ∂ζ*/∂x + β v* = 0"]], size=20)
arrow(s, 6.45, 2.45, 0.55, 0.42)
card(s, 7.1, 1.95, 5.48, 1.5, line=TEAL, line_w=2.25)
text(s, 7.4, 2.08, 4.9, 0.3, "TELLY", size=11, color=TEAL, bold=True, spc=300)
eq(s, 7.4, 2.5, 4.9, 0.8, [["∂ζ/∂t + u ∂ζ/∂x + v ∂ζ/∂y + β v = …"]], size=20)
bullets(s, X0, 3.8, 11.0, 2.7,
        ["No split into ū, u*, u′: the model follows the total wind, so waves and weather act on each other",
         "No small-wave assumption: every advection term is kept",
         "The Earth is a sphere: f = 2Ω sin φ replaces f₀ + βy",
         "The stationary wave is found afterwards: the time mean minus the zonal mean"], size=16, gap=10)

# ==================================================================================== 17 step-stretch
s = content("Part 3 · From class to Telly", "Step 2: Let Columns Stretch")
card(s, X0, 1.95, W, 1.5)
eq(s, X0 + 0.3, 2.05, W - 0.6, 0.7, [["∂ζ/∂t + V·∇ζ + β v = ", ("−(ζ + f) D", {"color": TEAL})]], size=28)
text(s, X0 + 0.3, 2.9, W - 0.6, 0.4, "D = ∂u/∂x + ∂v/∂y is the divergence of the wind.", size=14, color=SOFT)
bullets(s, X0, 3.75, 11.2, 1.8,
        ["Class: the flow is nearly non-divergent, so this term is dropped",
         "Telly keeps it: converging air stretches a column and spins it up; diverging air squeezes it",
         "Mountains and heating act through D, so this is where the source S of the class equation appears in Telly"],
        size=16, gap=8)
card(s, X0, 5.7, W, 0.85, fill=NAVY, line=None)
eq(s, X0 + 0.3, 5.82, 6.4, 0.6, [["−∇·(ηV) = −V·∇ζ − β v − (ζ + f) D"]], size=20, color=PAGE)
text(s, X0 + 7.0, 5.9, 4.6, 0.5, "Telly computes all three terms at once, as one quantity: −∇·(ηV).", size=15, color=PAGE)

# ==================================================================================== 18 step-compress
s = content("Part 3 · From class to Telly", "Step 3: Add Compression", "Typical values: T = 250 K, lapse rate 6.5 K/km")
card(s, X0, 1.95, W, 0.9, line=TEAL, line_w=2.0)
text(s, X0 + 0.3, 2.1, W - 0.6, 0.65,
     [[("Why: ", {"color": TEAL}), "air that moves up or down changes pressure, and that changes its temperature even with no heating."]],
     size=18, bold=True)
for y, lab, line_ in [
        (3.05, "1 · First law for a moving air parcel", [["cp DT/Dt − αω = Q"]]),
        (4.1, "2 · Use the ideal gas law, α = RT/p",
         [["DT/Dt = ", ("κTω/p", {"color": TEAL}), " + ", ("Q/cp", {"color": RUST}), ",  κ = R/cp"]]),
        (5.15, "3 · Write out DT/Dt",
         [["∂T/∂t + u ∂T/∂x + v ∂T/∂y + ω ∂T/∂p ", ("− κTω/p", {"color": TEAL}), " = ", ("Q/cp", {"color": RUST})]])]:
    text(s, X0, y, 6.9, 0.3, lab, size=13, color=TEAL, bold=True)
    eq(s, X0, y + 0.34, 6.9, 0.6, line_, size=20 if y < 5 else 17)
card(s, 7.95, 3.05, 4.63, 3.45, fill=NAVY, line=None)
text(s, 8.2, 3.2, 4.1, 0.3, "WHAT IT MEANS", size=11, color=LORG, bold=True, spc=300)
text(s, 8.2, 3.6, 4.15, 2.85,
     ["Rising air (ω < 0) expands and cools. Sinking air (ω > 0) is compressed and warms.",
      "The class term ω ∂T/∂p only carries temperature up or down. It has no expansion or compression in it.",
      [("At 500 hPa, ∂T/∂p ≈ 0.095 K/hPa but κT/p ≈ 0.14 K/hPa: leaving compression out gets the vertical response wrong by a factor of 2.", {"color": LORG})]],
     size=14, color=PAGE, gap=8)

# ==================================================================================== 19 step-layers
s = content("Part 3 · From class to Telly", "Step 4: Stack Layers")
card(s, X0, 1.95, W, 1.3)
eq(s, X0 + 0.3, 2.05, W - 0.6, 1.1,
   [["Φ = ", ("Φs", {"color": RUST}), " + R ∫ T d(ln p)"], ["∂u/∂x + ∂v/∂y + ∂ω/∂p = 0"]], size=22, gap=2)
cw = (W - 0.4) / 3
for i, (ttl, body) in enumerate([
        ("Hydrostatic", "The temperatures of all 26 layers give the height of every pressure surface. The ground height Φs is the bottom of the sum: this is where mountains enter."),
        ("Continuity", "Convergence in each layer gives the vertical motion ω between layers."),
        ("All layers at once", "Each layer has its own wind, temperature and ω. The terms ω ∂T/∂p and compression link neighboring layers.")]):
    x = X0 + i * (cw + 0.2)
    card(s, x, 3.5, cw, 2.9)
    text(s, x + 0.25, 3.65, cw - 0.5, 0.4, ttl, size=16, color=TEAL, bold=True)
    text(s, x + 0.25, 4.15, cw - 0.5, 2.1, body, size=14, color=SOFT)

# ==================================================================================== 20 physics
s = content("Part 3 · From class to Telly", "Step 5: Simple Damping")
text(s, X0, 1.95, 11.0, 0.7, "Friction, smoothing and a gentle pull toward a background state keep the model realistic and stable.", size=17)
tb = table(s, X0, 2.85, W, [5.4, 3.4, 3.03], ["What it does", "Written as", "How fast"], [
    ["Pulls temperature and surface pressure toward the background state", ("−(T − Tref)/τ", {"color": TEAL, "bold": False}), "40 days"],
    ["Friction: pulls vorticity and divergence toward the background state", ("−(ζ − ζref)/τ", {"color": TEAL, "bold": False}), "150 days"],
    ["Smooths the smallest scales", ("diffusion", {"color": TEAL, "bold": False}), "21 to 28 hours"]], size=15, row_h=0.62)
text(s, X0, 5.25, 11.0, 0.5, "The pull is faster for the zonal mean (7 days) and for the lowest two layers (2 to 3 days).", size=14, color=SOFT)
text(s, X0, 5.95, 11.0, 0.6, "The background state is a choice, and our experiments change it.", size=18, color=TEAL, bold=True)

# ==================================================================================== 21 eq-vort-div
LEG = [[("■", {"color": NAVY}), " from class        ", ("■", {"color": TEAL}), " added in Telly        ", ("■", {"color": RUST}), " forcing"]]
s = content("Part 3 · Telly", "Telly: Vorticity and Divergence", "Some vertical terms are left out of this view")
card(s, X0, 1.95, W, 2.0)
eq(s, X0 + 0.3, 2.1, W - 0.6, 1.7, [
    [("∂ζ/∂t = −∇·(ηV)", {"color": NAVY}), (" − R k·(∇T × ∇ln ps) − (ζ − ζref)/τ + smoothing", {"color": TEAL})],
    [("∂D/∂t = k·∇×(ηV) − ∇²(", {"color": TEAL}), ("Φ", {"color": RUST}),
     (" + ½V·V) − R ∇·(T ∇ln ps) − (D − Dref)/τ + smoothing", {"color": TEAL})]], size=19, gap=10)
text(s, X0, 4.15, W, 0.4, LEG, size=14, italic=False)
text(s, X0, 4.7, 11.0, 1.8,
     ["The first term is steps 1 and 2 together. Divergence D now has its own equation: it is how Telly follows columns stretching and squeezing.",
      "The height Φ includes the ground height Φs, so terrain pushes on D."], size=15, gap=8)

# ==================================================================================== 22 eq-thermo
s = content("Part 3 · Telly", "Telly: Temperature and Pressure", "Telly writes the advection terms in an equivalent form")
card(s, X0, 1.95, W, 2.4)
eq(s, X0 + 0.3, 2.1, W - 0.6, 2.1, [
    [("∂T/∂t = −V·∇T − ω ∂T/∂p", {"color": NAVY}), (" + κTω/p", {"color": TEAL}), (" + Q/cp", {"color": RUST}),
     (" − (T − Tref)/τ + smoothing", {"color": TEAL})],
    [("∂(ln ps)/∂t = −V·∇(ln ps) − (sum of D over all layers) − (ln ps − ln psref)/τ", {"color": TEAL})],
    [("Φ = ", {"color": NAVY}), ("Φs", {"color": RUST}), (" + R ∫ T d(ln p)", {"color": NAVY})]], size=19, gap=10)
text(s, X0, 4.55, W, 0.4, LEG, size=14, italic=False)
text(s, X0, 5.1, 11.0, 1.4,
     ["The temperature equation is the class equation plus compression, a pull toward the background state and smoothing.",
      "Surface pressure is new: it changes where columns converge or diverge."], size=15, gap=8)

# ==================================================================================== 23 forcings-enter
s = content("Part 3 · Telly", "Where the Forcings Enter")
cw = (W - 0.4) / 3
for i, (ttl, sym, ssz, col, body) in enumerate([
        ("Mountains", "Φs", 54, RUST, "Ground height is the bottom of the height sum, so it pushes on the divergence"),
        ("Land–sea contrast", "−(T − Tref)/τ", 28, TEAL, "Land and ocean are pulled toward different background temperatures, like heating and cooling"),
        ("Diabatic heating", "Q/cp", 54, RUST, "Rainfall heating in the temperature equation")]):
    x = X0 + i * (cw + 0.2)
    card(s, x, 1.95, cw, 3.95)
    text(s, x + 0.3, 2.15, cw - 0.6, 0.5, ttl, size=20, bold=True)
    eq(s, x + 0.3, 3.1, cw - 0.6, 1.0, [[sym]], size=ssz, color=col)
    text(s, x + 0.3, 4.4, cw - 0.6, 1.4, body, size=14, color=SOFT)
text(s, X0, 6.1, 11.0, 0.5, "Each one is a kind of source S from the class equation.", size=16, bold=True)

# ==================================================================================== 24 heating
s = content("Part 3 · Telly", "Heating from Observed Rainfall Patterns")
card(s, X0, 1.95, 6.7, 1.45)
eq(s, X0 + 0.3, 2.05, 6.1, 1.3, [["P ~ Gamma(k, θ)"], [("Q/cp", {"color": RUST}), " = β · exp(−φ²/1000) · P · s"]], size=22, gap=4)
bullets(s, X0, 3.6, 6.7, 3.0,
        ["The pattern is prescribed: k and θ are fitted to daily satellite rainfall at every location and day of the year, so the average rain, kθ, follows the observed pattern",
         "Only the day-to-day amount varies: each day’s rain is drawn around that average, with a spread set by k and θ",
         "The rain does not respond to the model’s own weather",
         "The factor exp(−φ²/1000), with latitude φ in degrees, confines heating to the tropics; β turns rain into heating and s spreads it in the vertical"],
        size=14, gap=6)
text(s, 7.8, 1.95, 4.78, 0.55, "Vertical profile s, top of the model to the ground", size=14, color=SOFT, bold=True)
cd = CategoryChartData()
cd.categories = [str(i) for i in range(1, 27)]
cd.add_series("Heating profile", [0, 0, 0, 0, 0, .25, .5, .75, 1, 1.5, 1.75, 1.75, 1.75, 2, 2, 2, 2,
                                  1.75, 1.75, 1.5, 1.25, .75, .5, .25, 0, 0])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(7.8), Inches(2.55), Inches(4.78), Inches(3.65), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = False
ch.plots[0].gap_width = 25
ser = ch.plots[0].series[0]
ser.format.fill.solid()
ser.format.fill.fore_color.theme_color = RUST
ch.category_axis.reverse_order = True
ch.category_axis.visible = False
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
gf._element.nvGraphicFramePr.cNvPr.set("descr", "Bar chart of the vertical profile of the heating, peaking in the middle of the troposphere")
text(s, 7.8, 6.25, 4.78, 0.5, "Peaks in the middle of the troposphere; scaled so it adds up to 1.", size=12, color=SOFT)

# ==================================================================================== 25 numerics
s = content("Part 3 · Telly", "How the Model Runs")
cw = (W - 3 * 0.25) / 4
for i, (big, body) in enumerate([("2°", "Grid boxes about 200 km wide"), ("26", "Layers in the vertical"),
                                 ("4 min", "Time step: the equations are advanced every 4 minutes"),
                                 ("1 day", "of computer time for a 5-year run")]):
    x = X0 + i * (cw + 0.25)
    card(s, x, 1.95, cw, 1.95)
    text(s, x + 0.25, 2.05, cw - 0.5, 0.9, big, size=44, color=TEAL, bold=True, font=SERIF, check=False)
    text(s, x + 0.25, 3.05, cw - 0.5, 0.8, body, size=14, color=SOFT)
bullets(s, X0, 4.25, 11.2, 2.3,
        ["At each step the model works out how the wind, temperature and pressure change, using the equations from the last few slides",
         "A 5-year run is about 600,000 steps",
         "Rainfall varies from day to day, so we average over years to see the stationary waves"], size=16, gap=10)

# ==================================================================================== 26 inputs
s = content("Part 3 · Telly", "What Goes In",
            "Monthly averages are interpolated to every day of the year. These inputs are prepared once and reused by every run.")
table(s, X0, 1.95, W, [2.9, 5.4, 3.53], ["Input", "What it describes", "Where it comes from"], [
    ["Ground height", "The height of the terrain at each point", "A gridded topography data set"],
    ["Surface pressure", "Average surface pressure for each day of the year", "NCEP reanalysis, 1994–2024"],
    ["Winds and temperature", "Average through the year in every layer: the background state", "NCEP reanalysis, 1994–2024"],
    ["Starting temperature", "Where the model begins", "NCEP plus a typical decrease with height"],
    ["Rainfall statistics", "How much it rains each day, and how variable it is", "Fit to daily satellite rainfall"],
    ["Land–sea map", "The fraction of land in each grid box", "ERA5 land–sea mask"]], size=15, row_h=0.62)

# ==================================================================================== 27 reference-state
statement("Why the background state matters",
          "The average state the model is pulled toward already contains the observed stationary waves.",
          "A run that keeps the observed background state is pulled toward the answer. These experiments use a "
          "background state that is the same at every longitude, and add each forcing separately.")

# ==================================================================================== 28 pipeline
s = content("Part 3 · Telly", "Running an Experiment")
cw, gap_ = 2.6, (W - 4 * 2.6) / 3
for i, (n, ttl, body) in enumerate([("1", "Prepare inputs", "Put the data on the model grid"),
                                    ("2", "Run the model", "Step the equations forward in time"),
                                    ("3", "Convert", "Interpolate to standard pressure levels"),
                                    ("4", "Make maps", "Time means and differences")]):
    x = X0 + i * (cw + gap_)
    card(s, x, 1.95, cw, 1.85)
    text(s, x + 0.2, 2.02, 1, 0.6, n, size=32, color=TEAL, bold=True, font=SERIF, check=False)
    text(s, x + 0.2, 2.65, cw - 0.4, 0.35, ttl, size=16, bold=True)
    text(s, x + 0.2, 3.05, cw - 0.4, 0.7, body, size=14, color=SOFT)
    if i < 3:
        arrow(s, x + cw + (gap_ - 0.3) / 2, 2.75, 0.3, 0.32)
card(s, X0, 4.05, 5.6, 2.45, fill=NAVY, line=None)
text(s, X0 + 0.3, 4.2, 5.0, 0.3, "ONE SETTINGS FILE PER EXPERIMENT", size=11, color=LORG, bold=True, spc=300)
text(s, X0 + 0.3, 4.65, 5.0, 1.7, ["Mountains: real", "Land–sea contrast: on", "Heating: observed", "Length: 5 years"],
     size=16, color=PAGE, gap=4)
bullets(s, 6.7, 4.1, 5.88, 2.4,
        ["Inputs are prepared once and reused by every run", "Output is saved in 30-day files, so a run can be extended",
         "Each step can be run on its own"], size=16, gap=10)

# ==================================================================================== 29 outputs
s = content("Part 3 · Telly", "What Comes Out")
table(s, X0, 1.95, W, [3.0, 7.0, 1.83], ["Output", "Contents", "Step"], [
    ["Daily model fields", "Winds, temperature, height, heating and surface pressure in every layer, one file per 30 days", "2"],
    ["Saved state", "Lets a run be continued where it stopped", "2"],
    ["Pressure-level fields", "Winds, temperature and height at 850, 500, 300 and 200 hPa", "3"],
    ["Figures", "Time-mean maps and differences from a control run", "4"]], size=15, row_h=0.62)
cw = (W - 0.3) / 2
for i, (big, body) in enumerate([("61", "30-day files make up a 5-year run"), ("≈ 0.5 GB", "per file, about 33 GB for 5 years")]):
    x = X0 + i * (cw + 0.3)
    card(s, x, 5.0, cw, 1.5)
    text(s, x + 0.3, 5.1, 3.0, 0.8, big, size=36, color=TEAL, bold=True, font=SERIF, check=False)
    text(s, x + 0.3, 5.95, cw - 0.6, 0.45, body, size=14, color=SOFT)

# ==================================================================================== 30 sec-exp
divider("Part 4", "Experiments", "Three forcings switched on one at a time")

# ==================================================================================== 31 switches
s = content("Part 4 · Experiments", "Three Forcings, Each On or Off")
table(s, X0, 1.95, W, [2.4, 3.3, 3.3, 2.83], ["Forcing", "Off", "On", "What it changes"], [
    ["Mountains", "Flat: ground height set to zero", "Real topography", "The ground under the air"],
    ["Land–sea contrast", "Background temperature the same at every longitude", "Observed land–ocean temperature differences added", "The background temperature"],
    ["Diabatic heating", "Rainfall the same at every longitude", "Observed rainfall", "The latent heating, Q/cp"]], size=15, row_h=0.8)
text(s, X0, 5.3, W, 0.8, "Each switch changes only its own part of the model. The next three slides show what each one looks like.",
     size=18, color=TEAL, bold=True)

# ==================================================================================== 32 exp-mtn, exp-landsea, exp-heat
def forcing_slide(n, title, off, on, bar):
    s = content("Part 4 · Experiments · forcing %d of 3" % n, title)
    for i, (ttl, col, img, alt, body) in enumerate([(off[0], SOFT) + off[1:], (on[0], RUST) + on[1:]]):
        x = X0 + i * 6.03
        card(s, x, 1.95, 5.8, 3.75)
        text(s, x + 0.2, 2.05, 5.4, 0.45, ttl, size=18, color=col, bold=True)
        picture(s, IMG(img), x + 0.2, 2.55, 5.4, 5.4 * 3.95 / 10, alt)
        text(s, x + 0.2, 4.8, 5.4, 0.9, body, size=14, color=SOFT)
    card(s, X0, 5.9, W, 0.85, fill=NAVY, line=NAVY)
    text(s, X0 + 0.3, 6.0, W - 0.6, 0.7, [[("What it changes: ", {"color": LORG, "bold": True}), (bar, {"color": PAGE})]], size=15, check=False)


forcing_slide(1, "Mountains: Off and On",
    ("Off: flat ground", "exp_mtn_off.png", "Map of the ground height with the value zero everywhere: the coastlines are drawn but the ground is at sea level",
     "Ground height is zero at every point: no Himalayas, Rockies or Andes. The wind meets no slopes."),
    ("On: real ground height", "exp_mtn_on.png", "Map of the real ground height in metres, with the Tibetan Plateau, the Andes, the Rockies and Greenland standing out",
     "Real terrain, with surface pressure lowered to match (about 560 hPa over Tibet, not 1010)."),
    "The ground under the air. Air forced over a slope stretches and squeezes columns of air: a source S for waves.")
forcing_slide(2, "Land–Sea Contrast: Off and On",
    ("Off: same at every longitude", "exp_ls_off.png", "Map of the lower-troposphere temperature in mid-July: smooth bands that depend only on latitude",
     "The background temperature depends on latitude only: warm tropics, cold poles."),
    ("On: land differs from ocean", "exp_ls_on.png", "Map of the lower-troposphere temperature in mid-July with hot spots over Asia, North Africa and the southwestern United States",
     "Observed land departures added over land only. The zonal average is unchanged."),
    "The background temperature the model is pulled toward. Warm or cool land acts like a heating or cooling pattern.")
forcing_slide(3, "Diabatic Heating: Off and On",
    ("Off: rain the same at every longitude", "exp_ht_off.png", "Map of the mean rain in June to August, zonally symmetric: a single band of heavy rain just north of the equator",
     "The same average rain all the way around each latitude circle: one tropical band."),
    ("On: observed rainfall", "exp_ht_on.png", "Map of the observed mean rain in June to August, with heavy rain over the Indian Ocean, the western Pacific and the Americas",
     "Rain falls where it really does: the Indian Ocean, western Pacific, and the Americas."),
    "Where latent heat is released. Heating in some places and not others pushes on the waves.")

# ==================================================================================== 33 five-runs
s = content("Part 4 · Experiments", "The Five Experiments")
HL = {"color": RUST, "bold": True}
table(s, X0, 1.95, W, [2.6, 1.7, 1.7, 1.7, 4.13], ["Run", "Mountains", "Land–sea", "Heating", "What it shows"], [
    ["Baseline", "off", "off", "off", "No longitude-dependent forcing at all"],
    ["Mountains only", ("ON", HL), "off", "off", "The effect of mountains"],
    ["Land–sea only", "off", ("ON", HL), "off", "The effect of land–sea contrast"],
    ["Heating only", "off", "off", ("ON", HL), "The effect of diabatic heating"],
    ["All three", ("ON", HL), ("ON", HL), ("ON", HL), "The full idealized model: the reference"]], size=15, row_h=0.62)
text(s, X0, 5.95, W, 0.8, "Run names: AC_SW_base, _mtn, _landsea, _heat and _all. For comparison, AC_Test is the same model run with the observed background state.",
     size=14, color=SOFT)


# ==================================================================================== 34 reading
s = content("Part 4 · Experiments", "Reading the Results")
for i, (lab, e) in enumerate([("Stationary eddy", "z* = z − [z]"), ("Response to a forcing", "z*(run) − z*(base)"),
                              ("Nonlinearity", "all − mtn − landsea − heat + 2·base")]):
    y = 1.95 + i * 1.2
    card(s, X0, y, W, 1.05)
    text(s, X0 + 0.3, y + 0.3, 3.4, 0.5, lab, size=16, bold=True)
    eq(s, X0 + 3.9, y + 0.2, 7.6, 0.7, [[e]], size=26, color=TEAL)
bullets(s, X0, 5.6, 11.2, 1.1,
        ["[z] is the zonal mean; the maps show 500 hPa height, averaged over a season, like the ERA5 map",
         "The first months of each run are discarded while the model settles",
         "Each difference map shows what one forcing does on its own"], size=14, gap=4)

# ==================================================================================== 35 sec-res
divider("Part 5", "Results", "What each forcing does to the wave pattern",
        "Preliminary: about four model years of the five-year runs, winter and summer")


# ==================================================================================== 36-41 results
def result(title, img, alt, look, see):
    s = content("Part 5 · Results · preliminary", title,
                "Preliminary: about 3.75 years after spin-up; winter and summer; 500 hPa height; runs still in progress")
    picture(s, IMG(img), X0, 1.95, 7.2, 7.2 * 990 / 1560, alt)
    text(s, 8.25, 1.95, 4.33, 0.35, "What to look for", size=16, color=TEAL, bold=True)
    bullets(s, 8.25, 2.35, 4.33, 1.4, look, size=14, gap=4)
    text(s, 8.25, 3.85, 4.33, 0.35, "What we see so far", size=16, color=RUST, bold=True)
    bullets(s, 8.25, 4.25, 4.33, 2.3, see, size=14, gap=4)


result("Baseline: No Zonal Asymmetry", "base.png", "Map of the stationary eddy in the baseline run, winter and summer, 500 hPa",
       ["Is z* close to zero?", "How big is the leftover pattern?"],
       ["Very weak leftover patterns in both seasons",
        "Day-to-day rainfall leaves small random wiggles",
        "This is the starting point for every other panel"])
result("Mountains Alone", "mtn.png", "Map of the 500 hPa height response to mountains alone, winter and summer",
       ["A wave train downstream of the Rockies and the Tibetan Plateau?", "Is the response stronger in winter than in summer?"],
       ["Winter: a deep low over northeast Asia, with highs over central Asia and Greenland",
        "Summer: weaker in the north; the largest features are around Antarctica",
        "The strongest of the three forcings in both seasons"])
result("Land–Sea Contrast Alone", "landsea.png", "Map of the 500 hPa height response to land-sea contrast alone, winter and summer",
       ["Is it larger or smaller than the mountain response?", "Are its centres over the continents or the oceans?"],
       ["Winter: weak highs and lows over the North Pacific and Asia", "Summer: very weak",
        "Much weaker than the mountain response"])
result("Diabatic Heating Alone", "heat.png", "Map of the 500 hPa height response to diabatic heating alone, winter and summer",
       ["A response near the tropical rain regions?", "Do wave trains extend into the extratropics?"],
       ["Winter: a weak wave train from the North Pacific to North America",
        "Summer: very weak",
        "Much weaker than the mountain response"])
result("All Three Together", "all.png",
       "Map of the stationary eddy at 500 hPa in the run with mountains, land-sea contrast and heating, winter and summer",
       ["Does it look like the ERA5 map?", "Which features are missing, and why?"],
       ["Winter: a deep low over northeast Asia and a high over central Asia, as in ERA5",
        "Summer: weaker in the north, strong around Antarctica",
        "Almost the same as the observed-background run (correlation 0.96 in each season)"])
result("Is the Whole the Sum of the Parts?", "nonlin.png",
       "Map of the nonlinear residual, all minus mountains minus land-sea minus heating plus two times baseline, 500 hPa, winter and summer",
       ["Where does the combined response differ from the sum of the three?",
        "Is that difference small compared with each single response?"],
       ["The combined response differs from the sum of the three", "Large in northern winter over the North Atlantic, Arctic and North Pacific", "Mountains plus heating and land–sea do not simply add"])

# ==================================================================================== 42 discussion
s = new_slide(dark=True)
text(s, X0, 0.5, W, 0.3, "WRAP-UP", size=12, color=LORG, bold=True, spc=300, check=False)
set_title(s, "Discussion", X0, 0.85, W, 0.95, 36, PAGE)
for i, q in enumerate(["Which forcing dominates the pattern at 500 hPa in winter, and which in summer?",
                       "Why must the surface pressure be placed on the terrain before mountains can be added?",
                       "How would you change the experiments to isolate ocean-only asymmetries such as the warm pool?"]):
    y = 2.2 + i * 1.5
    text(s, X0, y, 0.9, 0.9, str(i + 1), size=44, color=LORG, bold=True, font=SERIF, check=False)
    text(s, X0 + 1.1, y + 0.1, 10.5, 1.0, q, size=22, color=PAGE)
text(s, X0, 7.0, 7.0, 0.3, "Telly · stationary Rossby waves", size=10, color=PALE, check=False)
slide_number(s, light=True)

# ==================================================================================== 43 refs
s = content("Appendix", "References and Code")
card(s, X0, 1.95, 5.8, 4.4)
text(s, X0 + 0.3, 2.1, 5.2, 0.5, "Model paper", size=20, bold=True)
text(s, X0 + 0.3, 2.75, 5.2, 2.3, "Kirtman, B. P., and coauthors, 2025: A simplified-physics atmosphere general circulation model "
                                  "for idealized climate dynamics studies. Bull. Amer. Meteor. Soc., 106, E2073–E2086.", size=15, color=SOFT)
text(s, X0 + 0.3, 5.2, 5.2, 0.4, "doi.org/10.1175/BAMS-D-24-0196.1", size=13, color=TEAL, font="Courier New")
card(s, 6.78, 1.95, 5.8, 4.4, fill=NAVY, line=None)
text(s, 7.08, 2.1, 5.2, 0.5, "Run it yourself", size=20, color=PAGE, bold=True)
text(s, 7.08, 2.75, 5.2, 2.0, ["github.com/ou-esplab/Telly", "config/experiments/AC_SW_*.yaml", "scripts/stationary_waves.py", "EXPERIMENTS.md"],
     size=13, color=PAGE, font="Courier New", gap=6)
text(s, 7.08, 5.0, 5.2, 1.1, "Original model: github.com/jsb288/Atmospheric-Teleconnection-Model", size=14, color=PALE)

# ---- save + checks -------------------------------------------------------------------------------
prs.save(OUT)
print(f"Wrote {OUT} ({len(prs.slides)} slides)")
if WARN:
    print("\nLayout warnings (estimates, check these slides in PowerPoint):")
    for w_ in WARN:
        print("  -", w_)
