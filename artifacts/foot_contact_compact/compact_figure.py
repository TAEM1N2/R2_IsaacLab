from pathlib import Path
import re
import xml.etree.ElementTree as ET
import gi
import cairo

gi.require_version('Rsvg', '2.0')
from gi.repository import Rsvg
out = Path(__file__).resolve().parent
ns = '{http://www.w3.org/2000/svg}'
ET.register_namespace('', ns[1:-1])
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
root = ET.parse(out / 'source.svg').getroot()
w, h = map(float, root.attrib['viewBox'].split()[2:])
scale = 0.77
surface = root.find(f'{ns}g')
surface.set('transform', f'scale(1,{scale})')
# Preserve letter proportions while moving them with the compressed layout.
for elem in surface.iter():
    if elem.tag == ns+'use':
        y = float(elem.get('y','0'))
        elem.set('transform', f'translate(0,{y}) scale(1,{1/scale}) translate(0,{-y})')
    elif elem.tag == ns+'path' and 'stroke:none' in elem.get('style','').replace(' ',''):
        d = elem.get('d','')
        coords = list(map(float,re.findall(r'-?\d+(?:\.\d+)?',d)))
        # Filled letter outlines have curves or more vertices than rectangles.
        is_text = 'C' in d or len(coords)>12
        if is_text and not elem.get('transform'):
            ys=coords[1::2]
            y=(min(ys)+max(ys))/2
            elem.set('transform', f'translate(0,{y}) scale(1,{1/scale}) translate(0,{-y})')
root.set('height', f'{h*scale}pt')
root.set('viewBox',f'0 0 {w} {h*scale}')
svg=out/'foot_contact_pattern_shared_axis.svg'
ET.ElementTree(root).write(svg,encoding='utf-8',xml_declaration=True)
handle=Rsvg.Handle.new_from_file(str(svg))
pdf=cairo.PDFSurface(str(out/'foot_contact_pattern_shared_axis.pdf'),w,h*scale)
ctx=cairo.Context(pdf)
viewport=Rsvg.Rectangle()
viewport.x=0
viewport.y=0
viewport.width=w
viewport.height=h*scale
handle.render_document(ctx,viewport)
pdf.finish()
print(f'Page height: {h:.2f} -> {h*scale:.2f} pt; width: {w:.2f} pt')
