"""Rebuild the report's fixed-image crops from the included original PNGs."""
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent / 'assets'
originals, output = root / 'originals', root / 'casebook'
output.mkdir(exist_ok=True)
cases = [('recovered_1037', '1037', 98, 180),
         ('missed_1037', '1037', 43, 37),
         ('lost_0776', '0776', 148, 26)]
for name, image, row, col in cases:
    c = Image.open(originals / f'p22_c_{image}.png').convert('RGB')
    m = Image.open(originals / f'p22_m_{image}.png').convert('RGB')
    x0, y0 = max(0, min(176, col - 40)), max(0, min(176, row - 40))
    result = Image.new('RGB', (1220, 400), 'white')
    for i, (image_data, offset) in enumerate(((c, 0), (c, 768), (m, 768))):
        tile = image_data.crop((offset+x0, 42+y0, offset+x0+80, 42+y0+80))
        tile = tile.resize((400, 400), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(tile)
        x, y = (col-x0)*5, (row-y0)*5
        draw.ellipse((x-27, y-27, x+27, y+27), outline='white', width=5)
        draw.ellipse((x-23, y-23, x+23, y+23), outline='black', width=3)
        result.paste(tile, (i*410, 0))
    result.save(output / f'{name}.png')

for name in ('p21_c_1025', 'p21_a_1025', 'p24_c_1037', 'p24_l_1037'):
    image = Image.open(originals / f'{name}.png').convert('RGB')
    image.crop((768, 42, 1024, 298)).resize((768, 768), Image.Resampling.NEAREST).save(output / f'{name}.png')
