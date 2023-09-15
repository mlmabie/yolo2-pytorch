
class TileDimensions:
    def __init__(self, sw=0, sh=0, ew=0, eh=0):
        self.startw = sw
        self.starth = sh
        self.endw = ew
        self.endh = eh
        self.width = self.endw - self.startw + 1
        self.height = self.endh - self.starth + 1

    def __str__(self):
        return (
            f"TileDimensions(\n"
            f"\tstartw={self.startw},\n"
            f"\tstarth={self.starth},\n"
            f"\tendw={self.endw},\n"
            f"\tendh={self.endh},\n"
            f"\twidth={self.width},\n"
            f"\theight={self.height}\n"
            ")"
        )

class LayerTile:
    def __init__(self):
        self.input = None
        self.output = None

    def __str__(self):
        return f"LayerTile(input={self.input}, output={self.output})"


class FusedTilePartitioner:
    def __init__(self, model, partition_width, partition_height, n_fused_layers):
        self.model = model
        self.partition_width = partition_width
        self.partition_height = partition_height
        self.n_fused_layers = n_fused_layers
        self.tiles = {}

    def get_tile_input_dimensions(self, h, w, l):
        pass  # Implement this method

    def get_tile_output_dimensions(self, h, w, l):
        pass  # Implement this method

    def crop_data_from_input(self, h, w, l, layerh, layerw, layerc, data, len):
        pass  # Implement this method

    def crop_data_from_relative(self, h, w, l, layerh, layerw, layerc, data, len):
        pass  # Implement this method

    def print(self):
        pass  # Implement this method

    def setup_tiles(self):
        pass  # Implement this method

    def create_input_tile(self, output_tile, l):
        pass  # Implement this method

    def crop_data(self, dims, layerh, layerw, layerc, data, len):
        pass  # Implement this method

    def partition(self, data, dims):
        data_avg = 0
        cropped_avg = 0
        cropped_data = [0] * len(data)

        for c in range(dims.depth):
            for h in range(dims.starth, dims.starth + dims.height):
                for w in range(dims.startw, dims.startw + dims.width):
                    data_idx = w + dims.layerw * (h + dims.layerh * c)
                    cropped_idx = (w - dims.startw) + dims.width * (h - dims.starth) + dims.height * dims.width * c
                    cropped_data[cropped_idx] = data[data_idx]

                    data_avg += data_idx
                    cropped_avg += cropped_idx

        print(f"[ftp] dataIdx={data_avg} croppedIdx={cropped_avg}")

        return cropped_data

