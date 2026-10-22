from pathlib import Path

import fitz


class PageRenderer:

    def __init__(self, dpi=300):

        self.dpi = dpi

    def render(self, page, output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        zoom = self.dpi / 72

        matrix = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=matrix)

        pix.save(output_path)

        return output_path
