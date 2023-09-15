/*
 * DAICON
 * 
 * Author(s):
 *     John Geddes <jgeddes@ramlabs.com>
 *
 * Copyright RAM Laboratories, 2023-present
 * All commercial rights reserved
 */

#ifndef DAICON_FUSED_TILE_PARTIONER_HPP
#define DAICON_FUSED_TILE_PARTIONER_HPP

#include <iostream>
#include <string>
#include <map>
#include <iomanip>

#include "darknet/model.hpp"

namespace daicon {

namespace distributed {

class TileDimensions
{
public:
    TileDimensions()
      : startw(0)
      , endw(0)
      , starth(0)
      , endh(0)
      , width(0)
      , height(0)
    {}

    TileDimensions(int sw, int sh, int ew, int eh)
      : startw(sw)
      , starth(sh)
      , endw(ew)
      , endh(eh)
      , width(endw - startw + 1)
      , height(endh - starth + 1)
    {}

    int startw, endw;
    int starth, endh;
    int width;
    int height;

    friend std::ostream& operator<<(std::ostream& os, const TileDimensions& t);
};


class LayerTile
{
public:
    LayerTile() {}
    TileDimensions input;
    TileDimensions output;

    friend std::ostream& operator<<(std::ostream& os, const LayerTile& t);
};

class FusedTilePartitioner
{
public:
    FusedTilePartitioner(const daicon::darknet::Model& model, uint32_t partitionWidth, uint32_t partitionHeight, uint32_t nFusedLayers);

    TileDimensions
    getTileInputDimensions(uint32_t h, uint32_t w, uint32_t l);

    TileDimensions
    getTileOutputDimensions(uint32_t h, uint32_t w, uint32_t l);

    float *
    cropDataFromInput(uint32_t h, uint32_t w, uint32_t l, int layerh, int layerw, int layerc, float_t *data, size_t *len);

    float *
    cropDataFromRelative(uint32_t h, uint32_t w, uint32_t l, int layerh, int layerw, int layerc, float_t *data, size_t *len);

    void
    print();

private:
    void
    setupTiles();

    TileDimensions
    createInputTile(const TileDimensions& outputTile, const layer* l);

    float *
    cropData(TileDimensions dims, int layerh, int layerw, int layerc, float *data, size_t *len);

private:
    const daicon::darknet::Model& model_;

    uint32_t partitionHeight_;
    uint32_t partitionWidth_;
    uint32_t nFusedLayers_;

    std::map<uint32_t, std::map<uint32_t, std::map<uint32_t, LayerTile>>> tiles_;
};

} // namespace distributed

} // namespace daicon

#endif /* DAICON_FUSED_TILE_PARTIONER_HPP */
