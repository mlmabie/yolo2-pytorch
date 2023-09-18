"""
Python implementation of Fused Tile Partitioner
    Author: Malachi Mabie
    Copyright RAM Laboratories 2023-present
    All commercial rights reserved
"""
import math
import logging

logger = logging.getLogger('dist.FTP')
logger.setLevel(logging.DEBUG)  # or whatever level you want

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
    def __str__(self):
        return f"[{self.startw:3},{self.starth:3}] -> [{self.endw:3},{self.endh:3}]"


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
        
        setup_tiles()

    """
    DeepThings/src/ftp.c @ preform_ftp [sic]

    This function sets up the tile partitions boundaries based on the parameters (partition 
    width and height and the number of fused layers) and the CNN network loaded. Note that
    it sets up the output tiles of a layer based on the input layers of the previous layers.
    It sets up the initial output tiles of the last fused layer (DT function: grid) and then
    based on that sets up all the other layers partitions (DT function: traversal).
    """
    def setup_tiles(self):
        # setup the initial output tiles
        layerw = self.model.get_layer(self.n_fused_layers - 1).out_w
        layerh = self.model.get_layer(self.n_fused_layers - 1).out_h
        stridew = math.ceil(layerw / self.partition_width)
        strideh = math.ceil(layerh / self.partition_height)

        starth = 0
        endh = strideh - 1
        for h in range(self.partition_height):
            startw = 0
            endw = stridew - 1
            for w in range(self.partition_width):
                self.tiles[h][w][self.n_fused_layers - 1].output = TileDimensions(startw, starth, endw, endh)
                startw = endw + 1
                endw = min(endw + stridew, layerw - 1)
            starth = endh + 1
            endh = min(endh + strideh, layerh - 1)

        # setup the rest of the tiles
        for h in range(self.partition_height):
            for w in range(self.partition_width):
                for l in range(self.n_fused_layers - 1, -1, -1):
                    layer = self.model.get_layer(l)
                    self.tiles[h][w][l].input = self.create_input_tile(self.tiles[h][w][l].output, layer)
                    if l > 0:
                        self.tiles[h][w][l - 1].output = self.tiles[h][w][l].input

    """
    DeepThings/src/ftp.c @ traversal

    Sets a tile boundaries based on whether or not the layer is a 
    CONVULATIONAL or MAXPOOL layer.
    """
    def create_input_tile(self, output_tile, l):
        tile = TileDimensions()

        if l.type == 'CONVOLUTIONAL':
            tile.startw = max((output_tile.startw * l.stride) - (l.size // 2), 0)
            tile.endw = min((output_tile.endw * l.stride) + (l.size // 2), l.w - 1)
            tile.starth = max((output_tile.starth * l.stride) - (l.size // 2), 0)
            tile.endh = min((output_tile.endh * l.stride) + (l.size // 2), l.h - 1)
        elif l.type == 'MAXPOOL':
            tile.startw = output_tile.startw * l.stride
            tile.endw = output_tile.endw * l.stride + l.stride - 1
            tile.starth = output_tile.starth * l.stride
            tile.endh = output_tile.endh * l.stride + l.stride - 1

        tile.width = tile.endw - tile.startw + 1
        tile.height = tile.endh - tile.starth + 1

        return tile

    def crop_data_from_input(self, h, w, l, layerh, layerw, layerc, data, len):
        return self.crop_data(self.tiles[h][w][l].input, layerh, layerw, layerc, data, len)

    def crop_data_from_relative(self, h, w, l, layerh, layerw, layerc, data, len):
        input = self.tiles[h][w][l].input
        output = self.tiles[h][w][l].output
        relative = TileDimensions()
        relative.startw = output.startw - input.startw
        relative.endw = relative.startw + (output.endw - output.startw)
        relative.starth = output.starth - input.starth
        relative.endh = relative.starth + (output.endh - output.starth)
        relative.width = relative.endw - relative.startw + 1
        relative.height = relative.endh - relative.starth + 1

        return self.crop_data(relative, layerh, layerw, layerc, data, len)

    def crop_data(self, dims, layerh, layerw, layerc, data, len):
        print(f"[ftp] cropping ({dims.starth},{dims.startw}) -> ({dims.endh},{dims.endw})")

        data_avg = 0
        cropped_avg = 0

        nentries = dims.height * dims.width * layerc
        cropped_data = [0] * nentries
        for c in range(layerc):
            for h in range(dims.starth, dims.endh + 1):
                for w in range(dims.startw, dims.endw + 1):
                    data_idx = w + layerw * (h + layerh * c)
                    cropped_idx = (w - dims.startw) + dims.width * (h - dims.starth) + dims.height * dims.width * c
                    cropped_data[cropped_idx] = data[data_idx]

                    data_avg += data_idx
                    cropped_avg += cropped_idx

        print(f"[ftp] dataIdx={data_avg} croppedIdx={cropped_avg}")

        if len is not None:
            len[0] = nentries

        return cropped_data

    def get_tile_input_dimensions(self, h, w, l):
        return self.tiles[h][w][l].input

    def get_tile_output_dimensions(self, h, w, l):
        return self.tiles[h][w][l].output

    def print(self):
        for l in range(self.n_fused_layers):
            print(f"[layer {l}]")
            for h in range(self.partition_height):
                for w in range(self.partition_width):
                    print(f"({h},{w}) - {self.tiles[h][w][l]}")

    # not sure if used
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
