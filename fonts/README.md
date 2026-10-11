The font resource `TimesNewRoman-ManuscriptSubset.ttf` contains only the regular
Times New Roman glyphs already embedded in this project's `main.pdf`. It is used
by `draw_time_performance.py` and `draw_weights.py` to match the manuscript's actual body font on this
server, where Times New Roman is not installed.

The PDF's ToUnicode mapping was restored as a TrueType Unicode cmap, and its
existing space glyph was mapped to U+0020. Its omitted advance width was
restored to Times New Roman's quarter-em space. The original glyph outlines,
font names, other metrics, and embedding permission flags were preserved. This resource
covers the current English figure labels and numeric ticks; it is not a full
general-purpose font. Supply `--font-file /path/to/font.ttf` for another font or
additional characters.

The default lettering is 9.9626 PDF points, matching the manuscript's 10-pt
LaTeX text. The default figure is 114 x 85.5 mm, matching Figure 6's current
114-mm import width. Import the generated PDF at 114 mm width to preserve the
lettering size. For a different manuscript layout, generate at its final width
and body font size rather than resizing the exported figure afterward.

Figure 7 (`draw_weights.py`) defaults to 114 x 81.6 mm, matching its current
import dimensions. Its values, legend, axes, and tick labels also use 9.9626 pt.
The plus signs in the condition labels use Computer Modern mathematical glyphs,
as in the manuscript's mathematics; the surrounding text remains Times New Roman.
