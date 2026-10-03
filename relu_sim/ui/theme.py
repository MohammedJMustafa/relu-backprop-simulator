"""Light and dark themes: colours, fonts and the Qt style sheet.

The node colours follow Figure 1 of the homework (inputs lavender, hidden
neurons peach, output mint) and every math symbol is coloured like its node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath, QPalette, QPen

from ..paths import data_dir

# Windows fonts first; the other names are the closest fonts on macOS and Linux.
UI_FAMILIES = ["Segoe UI Variable", "Segoe UI", ".AppleSystemUIFont", "Helvetica Neue", "Ubuntu", "Cantarell",
               "Noto Sans", "DejaVu Sans"]
MATH_FAMILIES = ["Cambria", "STIX Two Text", "STIXGeneral", "Times New Roman", "Noto Serif", "DejaVu Serif"]
UI_FAMILY = UI_FAMILIES[0]
MATH_FAMILY = MATH_FAMILIES[0]
ICON_FAMILY = "Segoe Fluent Icons"
ICON_FALLBACK = "Segoe MDL2 Assets"


@dataclass(frozen=True)
class Theme:
    name: str
    dark: bool
    window: str
    surface: str
    surface2: str
    surface3: str
    border: str
    border2: str
    text: str
    text2: str
    muted: str
    accent: str
    accent_hover: str
    heading: str
    roles: dict[str, str] = field(default_factory=dict)
    node_fill: dict[str, str] = field(default_factory=dict)
    node_border: dict[str, str] = field(default_factory=dict)
    edge_pos: str = "#5B7DB8"
    edge_neg: str = "#D0644A"
    edge_neutral: str = "#8A94A6"
    signal: str = "#2F6AF5"
    gradient: str = "#E5396F"
    grid_dot: str = "#DCE1EA"
    shadow: str = "#000000"
    chart_relu: str = "#2F6AF5"
    chart_sigmoid: str = "#E8833A"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Theme) and other.name == self.name

    def c(self, value: str, alpha: float | None = None) -> QColor:
        """A QColor from a theme colour name ("accent") or a hex string."""
        color = QColor(getattr(self, value) if not value.startswith("#") else value)
        if alpha is not None:
            color.setAlphaF(alpha)
        return color

    def role(self, role: str | None, alpha: float | None = None) -> QColor:
        color = QColor(self.roles.get(role, self.text) if role else self.text)
        if alpha is not None:
            color.setAlphaF(alpha)
        return color

    def fill(self, kind: str) -> QColor:
        return QColor(self.node_fill.get(kind, self.surface2))

    def stroke(self, kind: str) -> QColor:
        return QColor(self.node_border.get(kind, self.border2))


LIGHT = Theme(
    name="light", dark=False,
    window="#F3F5F9", surface="#FFFFFF", surface2="#F7F8FB", surface3="#ECEFF5",
    border="#E2E6EE", border2="#CFD6E2", text="#172033", text2="#3E4A5F", muted="#6B7689",
    accent="#2F6AF5", accent_hover="#2457D6", heading="#2C5F8E",
    roles={"input": "#5354C9", "hidden": "#C9671F", "output": "#1E8A51", "error": "#D2404C",
           "target": "#7C4DDB", "grad": "#D92D63", "good": "#15803D", "bad": "#DC2626", "warn": "#B45309",
           "muted": "#6B7689", "accent": "#2F6AF5"},
    node_fill={"input": "#EDEDFC", "hidden": "#FDF0E2", "output": "#E3F5EA", "error": "#FCE7E9",
               "bias": "#F3F5F9"},
    node_border={"input": "#8B8BE6", "hidden": "#E5A266", "output": "#62BC88", "error": "#EB8790",
                 "bias": "#C5CDDA"},
    edge_pos="#5876B0", edge_neg="#CF6448", edge_neutral="#8A94A6",
    signal="#2F6AF5", gradient="#E0336C", grid_dot="#DCE1EA",
    chart_relu="#2F6AF5", chart_sigmoid="#E8833A",
)

DARK = Theme(
    name="dark", dark=True,
    window="#0D121C", surface="#141A26", surface2="#1A2130", surface3="#232B3D",
    border="#252E40", border2="#344057", text="#E7EAF1", text2="#B9C1D0", muted="#8792A7",
    accent="#5B8CFF", accent_hover="#7DA3FF", heading="#8DB4F0",
    roles={"input": "#A5A6FF", "hidden": "#FFAE66", "output": "#5ED393", "error": "#FF7782",
           "target": "#C6A4FF", "grad": "#FF6F9E", "good": "#4ADE80", "bad": "#F87171", "warn": "#FBBF24",
           "muted": "#8792A7", "accent": "#5B8CFF"},
    node_fill={"input": "#23234A", "hidden": "#3A2716", "output": "#12301F", "error": "#3A161B",
               "bias": "#1B2232"},
    node_border={"input": "#6E6FD9", "hidden": "#C47D3E", "output": "#3EA86B", "error": "#D9545F",
                 "bias": "#3A4560"},
    edge_pos="#7193D4", edge_neg="#E57C61", edge_neutral="#5D6A82",
    signal="#6E9BFF", gradient="#FF6F9E", grid_dot="#1F2737",
    chart_relu="#6E9BFF", chart_sigmoid="#FFA25C",
)

THEMES = {LIGHT.name: LIGHT, DARK.name: DARK}


def system_theme() -> Theme:
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        return DARK if scheme == Qt.ColorScheme.Dark else LIGHT
    except Exception:
        return LIGHT


def ui_font(point_size: float = 10.0, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(UI_FAMILY)
    font.setFamilies(UI_FAMILIES)
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    return font


def palette(theme: Theme) -> QPalette:
    pal = QPalette()
    c = theme.c
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        pal.setColor(group, QPalette.ColorRole.Window, c("window"))
        pal.setColor(group, QPalette.ColorRole.WindowText, c("text"))
        pal.setColor(group, QPalette.ColorRole.Base, c("surface2"))
        pal.setColor(group, QPalette.ColorRole.AlternateBase, c("surface3"))
        pal.setColor(group, QPalette.ColorRole.Text, c("text"))
        pal.setColor(group, QPalette.ColorRole.Button, c("surface"))
        pal.setColor(group, QPalette.ColorRole.ButtonText, c("text"))
        pal.setColor(group, QPalette.ColorRole.Highlight, c("accent"))
        pal.setColor(group, QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
        pal.setColor(group, QPalette.ColorRole.ToolTipBase, c("surface"))
        pal.setColor(group, QPalette.ColorRole.ToolTipText, c("text"))
        pal.setColor(group, QPalette.ColorRole.PlaceholderText, c("muted"))
        pal.setColor(group, QPalette.ColorRole.Link, c("accent"))
        pal.setColor(group, QPalette.ColorRole.Mid, c("border2"))
        pal.setColor(group, QPalette.ColorRole.Midlight, c("border"))
        pal.setColor(group, QPalette.ColorRole.Light, c("surface"))
        pal.setColor(group, QPalette.ColorRole.Dark, c("border2"))
        pal.setColor(group, QPalette.ColorRole.Shadow, QColor(0, 0, 0, 80))
    disabled = QPalette.ColorGroup.Disabled
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        pal.setColor(disabled, role, c("muted"))
    pal.setColor(disabled, QPalette.ColorRole.Base, c("surface3"))
    pal.setColor(disabled, QPalette.ColorRole.Button, c("surface3"))
    return pal


# --------------------------------------------------------------------------- style sheet images

def _image_dir():
    folder = data_dir() / "style"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@lru_cache(maxsize=None)
def style_images(theme: Theme) -> dict[str, str]:
    """Small chevron and check-mark images the style sheet refers to (drawn once per theme)."""
    folder = _image_dir()
    out = {}
    specs = {
        "up": (theme.text2, [(3.0, 9.5), (7.0, 5.5), (11.0, 9.5)]),
        "down": (theme.text2, [(3.0, 5.5), (7.0, 9.5), (11.0, 5.5)]),
        "check": ("#FFFFFF", [(3.0, 7.4), (5.8, 10.0), (11.2, 4.2)]),
    }
    for name, (color, points) in specs.items():
        image = QImage(28, 28, QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(2.0)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color), 1.7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(*point))
        painter.drawPath(path)
        painter.end()
        file = folder / f"{theme.name}-{name}.png"
        image.save(str(file))
        out[name] = file.as_posix()
    return out


def style_sheet(theme: Theme) -> str:
    img = style_images(theme)
    t = theme
    accent_soft = t.c("accent", 0.14 if t.dark else 0.10).name(QColor.NameFormat.HexArgb)
    return f"""
QWidget {{ color: {t.text}; }}
QMainWindow, QWidget#Root, QDialog {{ background: {t.window}; }}
QToolTip {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border2}; padding: 6px 8px;
           border-radius: 6px; }}

QFrame#Card {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 12px; }}
QFrame#Paper {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 6px; }}
QFrame#Divider {{ background: {t.border}; max-height: 1px; min-height: 1px; border: none; }}
QWidget#Header {{ background: {t.surface}; border-bottom: 1px solid {t.border}; }}
QWidget#Sidebar {{ background: {t.surface}; border-right: 1px solid {t.border}; }}

QLabel#AppTitle {{ font-size: 13.5pt; font-weight: 700; }}
QLabel#AppSubtitle {{ color: {t.muted}; font-size: 9pt; }}
QLabel#PageTitle {{ font-size: 15pt; font-weight: 700; }}
QLabel#PageSubtitle {{ color: {t.muted}; }}
QLabel#CardTitle {{ font-size: 11pt; font-weight: 700; }}
QLabel#Caption {{ color: {t.muted}; font-size: 8pt; font-weight: 700; letter-spacing: 0.7px; }}
QLabel#Muted {{ color: {t.muted}; }}
QLabel#StatValue {{ font-size: 17pt; font-weight: 700; }}
QLabel#StatNote {{ color: {t.muted}; font-size: 8.5pt; }}
QLabel#StudentName {{ font-weight: 700; }}
QLabel#StudentMeta {{ color: {t.muted}; font-size: 8.5pt; }}

QWidget#NavBar, QWidget#Segmented {{ background: {t.surface3}; border-radius: 9px; }}
QToolButton#NavButton {{ background: transparent; border: none; border-radius: 7px; padding: 6px 13px;
                         color: {t.text2}; font-weight: 600; }}
QToolButton#NavButton:hover {{ color: {t.text}; background: {accent_soft}; }}
QToolButton#NavButton:checked {{ background: {t.surface}; color: {t.accent}; }}
QToolButton#Segment {{ background: transparent; border: none; border-radius: 6px; padding: 5px 10px;
                       color: {t.text2}; font-weight: 600; }}
QToolButton#Segment:hover {{ color: {t.text}; }}
QToolButton#Segment:checked {{ background: {t.surface}; color: {t.accent}; }}

QPushButton {{ background: {t.surface}; border: 1px solid {t.border2}; border-radius: 8px; padding: 6px 14px;
               font-weight: 600; }}
QPushButton:hover {{ border-color: {t.accent}; color: {t.accent}; }}
QPushButton:pressed {{ background: {t.surface3}; }}
QPushButton:disabled {{ color: {t.muted}; border-color: {t.border}; background: {t.surface2}; }}
QPushButton#Primary {{ background: {t.accent}; border-color: {t.accent}; color: #FFFFFF; }}
QPushButton#Primary:hover {{ background: {t.accent_hover}; border-color: {t.accent_hover}; color: #FFFFFF; }}
QPushButton#Primary:disabled {{ background: {t.surface3}; border-color: {t.border}; color: {t.muted}; }}
QToolButton#IconButton {{ background: transparent; border: none; border-radius: 8px; padding: 5px; }}
QToolButton#IconButton:hover {{ background: {t.surface3}; }}
QToolButton#IconButton:pressed {{ background: {t.border}; }}
QToolButton#IconButton:disabled {{ background: transparent; }}
QToolButton#IconButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#PlayButton {{ background: {t.accent}; border: none; border-radius: 21px; }}
QToolButton#PlayButton:hover {{ background: {t.accent_hover}; }}

QAbstractSpinBox, QComboBox {{ background: {t.surface2}; border: 1px solid {t.border2}; border-radius: 7px;
                               padding: 3px 6px; min-height: 22px; selection-background-color: {t.accent};
                               selection-color: #FFFFFF; }}
QAbstractSpinBox:hover, QComboBox:hover {{ border-color: {t.muted}; }}
QAbstractSpinBox:focus, QComboBox:focus {{ border-color: {t.accent}; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 16px;
                               border: none; background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin: 2px 2px 0 0; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 2px 2px 0; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {t.surface3};
                               border-radius: 4px; }}
QAbstractSpinBox::up-arrow {{ image: url("{img['up']}"); width: 9px; height: 9px; }}
QAbstractSpinBox::down-arrow {{ image: url("{img['down']}"); width: 9px; height: 9px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{img['down']}"); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{ background: {t.surface}; border: 1px solid {t.border2}; padding: 4px;
                               selection-background-color: {accent_soft}; selection-color: {t.text};
                               outline: none; }}

QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {t.border2};
                        background: {t.surface2}; }}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; image: url("{img['check']}"); }}

QSlider::groove:horizontal {{ height: 4px; background: {t.surface3}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t.surface}; border: 2px solid {t.accent}; width: 12px; height: 12px;
                             margin: -6px 0; border-radius: 8px; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px 2px 2px 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0 2px 2px 2px; }}
QScrollBar::handle:vertical {{ background: {t.border2}; border-radius: 4px; min-height: 32px; }}
QScrollBar::handle:horizontal {{ background: {t.border2}; border-radius: 4px; min-width: 32px; }}
QScrollBar::handle:hover {{ background: {t.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QMenu {{ background: {t.surface}; border: 1px solid {t.border2}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 22px 7px 10px; border-radius: 6px; }}
QMenu::item:selected {{ background: {accent_soft}; color: {t.text}; }}
QMenu::separator {{ height: 1px; background: {t.border}; margin: 5px 8px; }}
QMenu::icon {{ padding-left: 8px; }}

QStatusBar {{ background: {t.surface}; border-top: 1px solid {t.border}; color: {t.muted}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {t.muted}; padding: 0 6px; }}
"""
