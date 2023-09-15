/*
 * DAICON
 * 
 * Author(s):
 *     John Geddes <jgeddes@ramlabs.com>
 *
 * Copyright RAM Laboratories, 2021-present
 * All commercial rights reserved
 */

#include <iostream>
#include <cmath>

#include "util/logger.hpp"

#include "fused-tile-partitioner.hpp"

namespace daicon {

namespace distributed {

DAICON_LOG_INIT(dist.FTP);

FusedTilePartitioner::FusedTilePartitioner(const daicon::darknet::Model& model, uint32_t partitionWidth, uint32_t partitionHeight, uint32_t nFusedLayers)
    : model_(model)
    , partitionWidth_(partitionWidth)
    , partitionHeight_(partitionHeight)
    , nFusedLayers_(nFusedLayers)
{
    setupTiles();
}

/**
 * DeepThings/src/ftp.c @ preform_ftp [sic]
 * 
 * This function setups the tile partitions boundaries based on the parameters (partition 
 * width and height and the number of fused layers) and the CNN network loaded.  Note that
 * it sets up the output tiles of a layer based on the input layers of the previous layers.
 * It sets up the initial output tiles of the last fused layer (DT function: grid) and then
 * based on that setups up all the other layers partitions (DT function: traversal).
 */
void
FusedTilePartitioner::setupTiles()
{
    DAICON_LOG_TRACE(__FUNCTION__);

    // create the tile objects
    for (int h = 0; h < partitionHeight_; h++) {
        tiles_[h] = std::map<uint32_t, std::map<uint32_t, LayerTile>>();
        for (int w = 0; w < partitionWidth_; w++) {
            tiles_[h][w] = std::map<uint32_t, LayerTile>();
            for (int l = 0; l < nFusedLayers_; l++) {
                tiles_[h][w][l] = LayerTile();
            }
        }
    }

    // DeepThings/src/ftp.c @ grid
    // setup the initial output tiles

    // for now let 'setupTiles' manipulate the darknet network object directly
    // instead of calling methods on the 'model_' object
    int32_t layerw = model_.getLayer(nFusedLayers_ - 1)->out_w;
    int32_t layerh = model_.getLayer(nFusedLayers_ - 1)->out_h;
    int32_t stridew = std::ceil((float)layerw/(float)partitionWidth_);    
    int32_t strideh = std::ceil((float)layerh/(float)partitionHeight_);    

    int32_t starth = 0;
    int32_t endh = strideh - 1;
    for (int h = 0; h < partitionHeight_; h++) {
        int32_t startw = 0;
        int32_t endw = stridew - 1;
        for (int w = 0; w < partitionWidth_; w++) {
            tiles_[h][w][nFusedLayers_ - 1].output = TileDimensions(startw, starth, endw, endh);
            startw = endw + 1;
            endw = std::min(endw + stridew, layerw - 1);
        }
        starth = endh + 1;
        endh = std::min(endh + strideh, layerh - 1);
    }

    // DeepThings/src/ftp.c @ preform_ftp
    // setup the rest of the tiles 
    for (int h = 0; h < partitionHeight_; h++) {
        for (int w = 0; w < partitionWidth_; w++) {
            for (int l = nFusedLayers_ - 1; l >= 0; l--) {
                const layer *layer = model_.getLayer(l);
                tiles_[h][w][l].input = createInputTile(tiles_[h][w][l].output, layer);
                if (l > 0) {
                    tiles_[h][w][l - 1].output = tiles_[h][w][l].input;
                }
            }
        }
    }
}

/**
 * DeepThings/src/ftp.c @ traversal
 * 
 * Sets a tile boundaries based on whether or not the layer is a 
 * CONVULATIONAL or MAXPOOL layer.
 */
TileDimensions
FusedTilePartitioner::createInputTile(const TileDimensions& outputTile, const layer* l)
{
    TileDimensions tile;

    if (l->type == CONVOLUTIONAL) {
        tile.startw = std::max((outputTile.startw * l->stride) - (l->size / 2), 0);
        tile.endw   = std::min((outputTile.endw * l->stride) + (l->size / 2), l->w - 1);
        tile.starth = std::max((outputTile.starth * l->stride)- (l->size / 2), 0);
        tile.endh   = std::min((outputTile.endh * l->stride) + (l->size / 2), l->h - 1);
    } else if (l->type == MAXPOOL) {
        tile.startw = outputTile.startw * l->stride;
        tile.endw = outputTile.endw * l->stride + l->stride - 1;
        tile.starth = outputTile.starth * l->stride;
        tile.endh = outputTile.endh * l->stride + l->stride - 1;
    }
    tile.width = tile.endw - tile.startw + 1;
    tile.height = tile.endh - tile.starth + 1;

    return tile;
}

float *
FusedTilePartitioner::cropDataFromInput(uint32_t h, uint32_t w, uint32_t l, int layerh, int layerw, int layerc, float_t *data, size_t *len)
{
    return cropData(tiles_[h][w][l].input, layerh, layerw, layerc, data, len);
}

float *
FusedTilePartitioner::cropDataFromRelative(uint32_t h, uint32_t w, uint32_t l, int layerh, int layerw, int layerc, float_t *data, size_t *len)
{
    TileDimensions input = tiles_[h][w][l].input;
    TileDimensions output = tiles_[h][w][l].output;
    TileDimensions relative;
    relative.startw = output.startw - input.startw;
    relative.endw = relative.startw + (output.endw - output.startw);
    relative.starth = output.starth - input.starth;
    relative.endh = relative.starth + (output.endh - output.starth);
    relative.width = relative.endw - relative.startw + 1;
    relative.height = relative.endh - relative.starth + 1;

    return cropData(relative, layerh, layerw, layerc, data, len);
}

float *
FusedTilePartitioner::cropData(TileDimensions dims, int layerh, int layerw, int layerc, float *data, size_t *len)
{
    DAICON_LOG_DEBUG("[ftp] cropping (" << dims.starth << "," << dims.startw << ") -> (" << dims.endh << "," << dims.endw << ")");

    uint32_t dataAvg = 0;
    uint32_t croppedAvg = 0;

    size_t nentries = dims.height * dims.width * layerc;
    float *croppedData = (float *)malloc(sizeof(float) * nentries);
    for (int c = 0; c < layerc; c++) {
        for (int h = dims.starth; h <= dims.endh; h++) {
            for (int w = dims.startw; w <= dims.endw; w++) {
                int dataIdx = w + layerw*(h + layerh*c);
                int croppedIdx = (w - dims.startw) + dims.width * (h - dims.starth) + dims.height * dims.width * c;
                croppedData[croppedIdx] = data[dataIdx];

                dataAvg += dataIdx;
                croppedAvg += croppedIdx;
            }
        }
    }

    // REFACTOR LOGGING
    DAICON_LOG_DEBUG("[ftp] dataIdx=" << dataAvg << " croppedIdx=" << croppedAvg);

    if (len) {
        *len = nentries;
    }

    return croppedData;
}


TileDimensions
FusedTilePartitioner::getTileInputDimensions(uint32_t h, uint32_t w, uint32_t l) 
{
    return tiles_[h][w][l].input;
}

TileDimensions
FusedTilePartitioner::getTileOutputDimensions(uint32_t h, uint32_t w, uint32_t l) 
{
    return tiles_[h][w][l].output;
}

void
FusedTilePartitioner::print()
{
    for (int l = 0; l < nFusedLayers_; l++) {
        DAICON_LOG_INFO("[layer " << l << "]");
        for (int h = 0; h < partitionHeight_; h++) {
            for (int w = 0; w < partitionWidth_; w++) {
                DAICON_LOG_INFO("(" << h << "," << w << ") - " << tiles_[h][w][l]);
            }
        }
    }
}

std::ostream& operator<<(std::ostream& os, const LayerTile& t)
{
    os << "input: " << t.input << " output: " << t.output;
    return os;
}

std::ostream& operator<<(std::ostream& os, const TileDimensions& t)
{
    //os << std::resetiosflags(std::ios::right) << std::setiosflags(std::ios::left)
    os << "[" << std::setw(3) << t.startw << std::setw(0) << "," << std::setw(3) << t.starth
       << std::setw(0) << "] -> [" << std::setw(3) << t.endw << std::setw(0) << "," 
       << std::setw(3) << t.endh << std::setw(0) << "]";
    return os;
}

} // namespace distributed

} // namespace daicon
