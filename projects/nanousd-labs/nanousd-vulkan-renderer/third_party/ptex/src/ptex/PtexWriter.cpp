/*
PTEX SOFTWARE
Copyright 2014 Disney Enterprises, Inc.  All rights reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * The names "Disney", "Walt Disney Pictures", "Walt Disney Animation
    Studios" or the names of its contributors may NOT be used to
    endorse or promote products derived from this software without
    specific prior written permission from Walt Disney Pictures.

Disclaimer: THIS SOFTWARE IS PROVIDED BY WALT DISNEY PICTURES AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, NONINFRINGEMENT AND TITLE ARE DISCLAIMED.
IN NO EVENT SHALL WALT DISNEY PICTURES, THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND BASED ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
*/

#include "PtexPlatform.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#if defined(__FreeBSD__)
    #include <unistd.h>
    #include <stddef.h>
#endif
#include <libdeflate.h>

#include "Ptexture.h"
#include "PtexUtils.h"
#include "PtexWriter.h"

PTEX_NAMESPACE_BEGIN

namespace {

    std::string fileError(const char* message, const char* path)
    {
        std::stringstream str;
        str << message << path << "\n" << strerror(errno);
        return str.str();
    }

    bool checkFormat(Ptex::MeshType mt, Ptex::DataType dt, int nchannels, int alphachan,
                     Ptex::String& error)
    {
        // check to see if given file attributes are valid
        if (!LittleEndian()) {
            error = "PtexWriter doesn't currently support big-endian cpu's";
            return 0;
        }

        if (mt < Ptex::mt_triangle || mt > Ptex::mt_quad) {
            error = "PtexWriter error: Invalid mesh type";
            return 0;
        }

        if (dt < Ptex::dt_uint8 || dt > Ptex::dt_float) {
            error = "PtexWriter error: Invalid data type";
            return 0;
        }

        if (nchannels <= 0) {
            error = "PtexWriter error: Invalid number of channels";
            return 0;
        }

        if (alphachan != -1 && (alphachan < 0 || alphachan >= nchannels)) {
            error = "PtexWriter error: Invalid alpha channel";
            return 0;
        }

        return 1;
    }
}


PtexWriter* PtexWriter::open(const char* path,
                             Ptex::MeshType mt, Ptex::DataType dt,
                             int nchannels, int alphachan, int nfaces,
                             Ptex::String& error, bool genmipmaps)
{
    if (!checkFormat(mt, dt, nchannels, alphachan, error))
        return 0;

    PtexMainWriter* w = new PtexMainWriter(path, 0,
                                           mt, dt, nchannels, alphachan, nfaces,
                                           genmipmaps);
    if (!w->ok(error)) {
        w->release();
        return 0;
    }
    return w;
}


PtexWriter* PtexWriter::edit(const char* path,
                             Ptex::MeshType mt, Ptex::DataType dt,
                             int nchannels, int alphachan, int nfaces,
                             Ptex::String& error, bool genmipmaps)
{
    if (!checkFormat(mt, dt, nchannels, alphachan, error))
        return 0;

    // try to open existing file (it might not exist)
    FILE* fp = fopen(path, "rb+");
    if (!fp && errno != ENOENT) {
        error = fileError("Can't open ptex file for update: ", path).c_str();
    }

    PtexTexture* tex = 0;
    if (fp) {
        // got an existing file, close and reopen with PtexReader
        fclose(fp);

        // open reader for existing file
        tex = PtexTexture::open(path, error);
        if (!tex) return 0;

        // make sure header matches
        bool headerMatch = (mt == tex->meshType() &&
                            dt == tex->dataType() &&
                            nchannels == tex->numChannels() &&
                            alphachan == tex->alphaChannel() &&
                            nfaces == tex->numFaces());
        if (!headerMatch) {
            std::stringstream str;
            str << "PtexWriter::edit error: header doesn't match existing file, "
                << "conversions not supported";
            error = str.str().c_str();
            return 0;
        }
    }
    PtexMainWriter* w = new PtexMainWriter(path, tex, mt, dt, nchannels, alphachan,
                                           nfaces, genmipmaps);

    if (!w->ok(error)) {
        w->release();
        return 0;
    }
    return w;
}

PtexWriter* PtexWriter::edit(const char* path, bool /*incremental*/,
                             Ptex::MeshType mt, Ptex::DataType dt,
                             int nchannels, int alphachan, int nfaces,
                             Ptex::String& error, bool genmipmaps)
{
    // This function is deprecated
    return edit(path, mt, dt, nchannels, alphachan, nfaces, error, genmipmaps);
}

bool PtexWriter::applyEdits(const char*, Ptex::String&)
{
    // This function is obsolete
    return 1;
}


void PtexMainWriter::release()
{
    Ptex::String error;
    // close writer if app didn't, and report error if any
    if (!close(error))
        std::cerr << error.c_str() << std::endl;
    delete this;
}


bool PtexMainWriter::storeFaceInfo(int faceid, FaceInfo& f, const FaceInfo& src, int flags)
{
    if (faceid < 0 || size_t(faceid) >= _header.nfaces) {
        setError("PtexWriter error: faceid out of range");
        return 0;
    }

    if (_header.meshtype == mt_triangle && (f.res.ulog2 != f.res.vlog2)) {
        setError("PtexWriter error: asymmetric face res not supported for triangle textures");
        return 0;
    }

    // copy all values
    f = src;

    // and clear extraneous ones
    if (_header.meshtype == mt_triangle) {
        f.flags = 0; // no user-settable flags on triangles
        f.adjfaces[3] = -1;
        f.adjedges &= 0x3f; // clear all but bottom six bits
    }
    else {
        // clear non-user-settable flags
        f.flags &= FaceInfo::flag_subface;
    }

    // set new flags
    f.flags |= (uint8_t)flags;
    return 1;
}


void PtexMainWriter::writeMeta(const char* key, const char* value)
{
    addMetaData(key, mdt_string, value, int(strlen(value)+1));
}


void PtexMainWriter::writeMeta(const char* key, const int8_t* value, int count)
{
    addMetaData(key, mdt_int8, value, count);
}


void PtexMainWriter::writeMeta(const char* key, const int16_t* value, int count)
{
    addMetaData(key, mdt_int16, value, count*(int)sizeof(int16_t));
}


void PtexMainWriter::writeMeta(const char* key, const int32_t* value, int count)
{
    addMetaData(key, mdt_int32, value, count*(int)sizeof(int32_t));
}


void PtexMainWriter::writeMeta(const char* key, const float* value, int count)
{
    addMetaData(key, mdt_float, value, count*(int)sizeof(float));
}


void PtexMainWriter::writeMeta(const char* key, const double* value, int count)
{
    addMetaData(key, mdt_double, value, count*(int)sizeof(double));
}


void PtexMainWriter::writeMeta(PtexMetaData* data)
{
    int nkeys = data->numKeys();
    for (int i = 0; i < nkeys; i++) {
        const char* key = 0;
        MetaDataType type;
        data->getKey(i, key, type);
        int count;
        switch (type) {
        case mdt_string:
            {
                const char* val=0;
                data->getValue(key, val);
                writeMeta(key, val);
            }
            break;
        case mdt_int8:
            {
                const int8_t* val=0;
                data->getValue(key, val, count);
                writeMeta(key, val, count);
            }
            break;
        case mdt_int16:
            {
                const int16_t* val=0;
                data->getValue(key, val, count);
                writeMeta(key, val, count);
            }
            break;
        case mdt_int32:
            {
                const int32_t* val=0;
                data->getValue(key, val, count);
                writeMeta(key, val, count);
            }
            break;
        case mdt_float:
            {
                const float* val=0;
                data->getValue(key, val, count);
                writeMeta(key, val, count);
            }
            break;
        case mdt_double:
            {
                const double* val=0;
                data->getValue(key, val, count);
                writeMeta(key, val, count);
            }
            break;
        }
    }
}


void PtexMainWriter::addMetaData(const char* key, MetaDataType t,
                                 const void* value, int size)
{
    if (strlen(key) > 255) {
        std::stringstream str;
        str << "PtexWriter error: meta data key too long (max=255) \"" << key << "\"";
        setError(str.str());
        return;
    }
    if (size <= 0) {
        std::stringstream str;
        str << "PtexWriter error: meta data size <= 0 for \"" << key << "\"";
        setError(str.str());
    }
    std::map<std::string,int>::iterator iter = _metamap.find(key);
    int index;
    if (iter != _metamap.end()) {
        // see if we already have this entry - if so, overwrite it
        index = iter->second;
    }
    else {
        // allocate a new entry
        index = (int)_metadata.size();
        _metadata.resize(index+1);
        _metamap[key] = index;
    }
    MetaEntry& m = _metadata[index];
    m.key = key;
    m.datatype = t;
    m.data.resize(size);
    memcpy(&m.data[0], value, size);
}


size_t PtexMainWriter::writeBlock(FILE* fp, const void* data, size_t size)
{
    if (!_ok) return 0;
    if (!fwrite(data, size, 1, fp)) {
        setError("PtexWriter error: file write failed");
        return 0;
    }
    return size;
}


void PtexMainWriter::addToDataBlock(std::vector<std::byte>& dataBlock, const void* data, size_t size)
{
    size_t previousSize = dataBlock.size();
    dataBlock.resize(previousSize+size);
    memcpy(dataBlock.data() + previousSize, data, size);
}


libdeflate_compressor* PtexMainWriter::getCompressor()
{
    AutoMutex lock(_compressorMutex);
    libdeflate_compressor* compressor;
    if (!_compressors.empty()) {
        compressor = _compressors.back();
        _compressors.pop_back();
    } else {
        const int compressionLevel = 4;
        compressor = libdeflate_alloc_compressor(compressionLevel);
    }
    return compressor;
}


void PtexMainWriter::releaseCompressor(libdeflate_compressor* compressor)
{
    AutoMutex lock(_compressorMutex);
    _compressors.push_back(compressor);
}


void PtexMainWriter::compressDataBlock(libdeflate_compressor* compressor, std::vector<std::byte>& compressedData, const void* data, size_t size)
{
    compressedData.resize(libdeflate_zlib_compress_bound(compressor, size));
    int compressedSize = int(libdeflate_zlib_compress(compressor, data, size, compressedData.data(), compressedData.size()));
    if (!compressedSize) {
        setError("PtexWriter error: compression failed");
    }
    compressedData.resize(compressedSize);
}


Ptex::Res PtexMainWriter::calcTileRes(Res faceres)
{
    // desired number of tiles = floor(log2(facesize / tilesize))
    size_t facesize = faceres.size64() * _pixelSize;
    int ntileslog2 = PtexUtils::floor_log2(facesize/TileSize);
    if (ntileslog2 == 0) return faceres;

    // number of tiles is defined as:
    //   ntileslog2 = ureslog2 + vreslog2 - (tile_ureslog2 + tile_vreslog2)
    // rearranging to solve for the tile res:
    //   tile_ureslog2 + tile_vreslog2 = ureslog2 + vreslog2 - ntileslog2
    int n = faceres.ulog2 + faceres.vlog2 - ntileslog2;

    // choose u and v sizes for roughly square result (u ~= v ~= n/2)
    // and make sure tile isn't larger than face
    Res tileres;
    tileres.ulog2 = (int8_t)std::min(int((n+1)/2), int(faceres.ulog2));
    tileres.vlog2 = (int8_t)std::min(int(n - tileres.ulog2), int(faceres.vlog2));
    return tileres;
}


void PtexMainWriter::compressFaceDataBlock(libdeflate_compressor* compressor, std::vector<std::byte>& compressedData, FaceDataHeader& fdh,
                                           Res res, const void* uncompressedData, int stride)
{
    // compress a single face data block; could be a whole face or just a tile

    // first, copy to temp buffer, and deinterleave
    int ures = res.u(), vres = res.v();
    size_t tempSize = ures*vres*_pixelSize;
    std::vector<std::byte> temp(tempSize);
    PtexUtils::deinterleave(uncompressedData, stride, ures, vres, temp.data(),
                            ures*DataSize(datatype()),
                            datatype(), _header.nchannels);

    // difference if needed
    bool diff = (datatype() == dt_uint8 ||
                 datatype() == dt_uint16);
    if (diff) PtexUtils::encodeDifference(temp.data(), tempSize, datatype());

    // compress
    compressDataBlock(compressor, compressedData, temp.data(), tempSize);

    // record compressed size and encoding in data header
    fdh.set(compressedData.size(), diff ? enc_diffzipped : enc_zipped);
}


void PtexMainWriter::compressFaceData(libdeflate_compressor* compressor, std::vector<std::byte>& compressedData, FaceDataHeader& fdh,
                                      Res res, const void* uncompressedData)
{
    // determine whether to break into tiles
    int stride = res.u() * _pixelSize;
    Res tileRes = calcTileRes(res);
    size_t ntilesu = res.ntilesu(tileRes);
    size_t ntilesv = res.ntilesv(tileRes);
    size_t ntiles = ntilesu * ntilesv;
    if (ntiles == 1) {
        // output single block
        compressFaceDataBlock(compressor, compressedData, fdh, res, uncompressedData, stride);
    } else {
        // alloc tiles
        std::vector<std::vector<std::byte>> tiles(ntiles);
        std::vector<FaceDataHeader> tileHeader(ntiles);
        size_t tileures = tileRes.u();
        size_t tilevres = tileRes.v();
        size_t tileustride = tileures*_pixelSize;
        size_t tilevstride = tilevres*stride;

        // compress tiles
        std::vector<std::byte>* tile = tiles.data();
        FaceDataHeader* tdh = tileHeader.data();
        const std::byte* rowp = reinterpret_cast<const std::byte*>(uncompressedData);
        const std::byte* rowpend = rowp + ntilesv * tilevstride;
        for (; rowp != rowpend; rowp += tilevstride) {
            const std::byte* p = rowp;
            const std::byte* pend = p + ntilesu * tileustride;
            for (; p != pend; tile++, tdh++, p += tileustride) {
                // determine if tile is constant
                if (PtexUtils::isConstant(p, stride, tileures, tilevres, _pixelSize)) {
                    // output a const tile
                    tile->assign(p, p + _pixelSize);
                    tdh->set(_pixelSize, enc_constant);
                } else {
                    // output a compressed tile
                    compressFaceDataBlock(compressor, *tile, *tdh, tileRes, p, stride);
                }
            }
        }

        // compress tile header
        std::vector<std::byte> compressedTileHeader;
        compressDataBlock(compressor, compressedTileHeader, reinterpret_cast<std::byte*>(tileHeader.data()),
                          ntiles * sizeof(FaceDataHeader));
        uint32_t compressedTileHeaderSize = compressedTileHeader.size();

        size_t totalSize = sizeof(tileRes) + sizeof(compressedTileHeaderSize) + compressedTileHeaderSize;
        for (auto& tile : tiles) {
            totalSize += tile.size();
        }
        compressedData.reserve(totalSize);
        addToDataBlock(compressedData, &tileRes, sizeof(tileRes));
        addToDataBlock(compressedData, &compressedTileHeaderSize, sizeof(compressedTileHeaderSize));
        addToDataBlock(compressedData, compressedTileHeader.data(), compressedTileHeaderSize);
        for (auto& tile : tiles) {
            addToDataBlock(compressedData, tile.data(), tile.size());
        }

        fdh.set(totalSize, enc_tiled);
    }
}


void PtexMainWriter::addToMetaDataBlock(std::vector<std::byte>& metaDataBlock, const MetaEntry& val)
{
    uint8_t keysize = uint8_t(val.key.size()+1);
    uint8_t datatype = val.datatype;
    uint32_t datasize = uint32_t(val.data.size());
    addToDataBlock(metaDataBlock, &keysize, sizeof(keysize));
    addToDataBlock(metaDataBlock, val.key.c_str(), keysize);
    addToDataBlock(metaDataBlock, &datatype, sizeof(datatype));
    addToDataBlock(metaDataBlock, &datasize, sizeof(datasize));
    addToDataBlock(metaDataBlock, &val.data[0], datasize);
}


PtexMainWriter::PtexMainWriter(const char* path, PtexTexture* tex,
                               Ptex::MeshType mt, Ptex::DataType dt,
                               int nchannels, int alphachan, int nfaces, bool genmipmaps)
    : _ok(true),
      _path(path),
      _genmipmaps(genmipmaps),
      _reader(0)
{
    memset(&_header, 0, sizeof(_header));
    _header.magic = Magic;
    _header.version = PtexFileMajorVersion;
    _header.minorversion = PtexFileMinorVersion;
    _header.meshtype = mt;
    _header.datatype = dt;
    _header.alphachan = alphachan;
    _header.nchannels = (uint16_t)nchannels;
    _header.nfaces = nfaces;
    _header.extheadersize = sizeof(_extheader);
    _pixelSize = _header.pixelSize();

    memset(&_extheader, 0, sizeof(_extheader));

    if (mt == mt_triangle)
        _reduceFn = &PtexUtils::reduceTri;
    else
        _reduceFn = &PtexUtils::reduce;

    // data will be written to a ".new" path and then renamed to final location
    _newpath = path; _newpath += ".new";

    // init faceinfo and set flags to -1 to mark as uninitialized
    _faceinfo.resize(nfaces);
    for (int i = 0; i < nfaces; i++) _faceinfo[i].flags = uint8_t(-1);
    _faces.resize(nfaces);
    _constdata.resize(nfaces*_pixelSize);

    if (tex) {
        // access reader implementation
        // Note: we can assume we have a PtexReader because we opened the tex from the cache
        _reader = static_cast<PtexReader*>(tex);

        // copy border modes
        setBorderModes(tex->uBorderMode(), tex->vBorderMode());

        // copy edge filter mode
        setEdgeFilterMode(tex->edgeFilterMode());

        // copy meta data from existing file
        PtexPtr<PtexMetaData> meta ( _reader->getMetaData() );
        writeMeta(meta);
    }
}


PtexMainWriter::~PtexMainWriter()
{
    for (libdeflate_compressor* compressor : _compressors) {
        libdeflate_free_compressor(compressor);
    }
    if (_reader) _reader->release();
}


bool PtexMainWriter::close(Ptex::String& error)
{
    if (_ok) finish();
    if (_reader) {
        if (!_reader->ok()) {
            _ok = false;
        }
        _reader->release();
        _reader = 0;
    }
    if (_ok) {
        // rename temppath into final location
        unlink(_path.c_str());
        if (rename(_newpath.c_str(), _path.c_str()) == -1) {
            setError(fileError("Can't write to ptex file: ", _path.c_str()).c_str());
            unlink(_newpath.c_str());
        }
    }
    if (!_ok) getError(error);
    return _ok;
}

bool PtexMainWriter::writeFace(int faceid, const FaceInfo& f, const void* data, int stride)
{
    if (!_ok) return false;

    FaceRec& face = _faces[faceid];
    AutoMutex lock(face.mutex);

    // reset face data in case it was written previously
    face.faceData.clear();
    face.fdh.clear();

    // auto-compute stride
    if (stride == 0) stride = f.res.u()*_pixelSize;

    // handle constant case
    if (PtexUtils::isConstant(data, stride, f.res.u(), f.res.v(), _pixelSize)) {
        return writeConstantFace(faceid, f, data);
    }

    // non-constant case, ...

    // check and store face info
    if (!storeFaceInfo(faceid, _faceinfo[faceid], f)) return false;

    // determine how many mipmap levels for face
    int nlevels = 1;
    if (_genmipmaps) {
        nlevels += std::max(0, std::min(f.res.ulog2, f.res.vlog2) - MinReductionLog2);
    }
    face.faceData.resize(nlevels);
    face.fdh.resize(nlevels);

    // copy data into face level 0
    Ptex::Res res = f.res;
    size_t rowlen = res.u() * _pixelSize, nrows = res.v();
    face.faceData[0].resize(rowlen * nrows);
    PtexUtils::copy(data, stride, face.faceData[0].data(), rowlen, nrows, rowlen);
    data = face.faceData[0].data();
    stride = rowlen;

    // premultiply into temp copy (if needed)
    std::vector<std::byte> premultData;
    if (_header.hasAlpha()) {
        // copy to temp buffer, and premultiply alpha
        premultData = face.faceData[0];
        PtexUtils::multalpha(premultData.data(), res.size64(), datatype(), _header.nchannels,
                             _header.alphachan);
        data = premultData.data();
    }

    // generate reductions (as needed)
    libdeflate_compressor* compressor = getCompressor();
    for (int level = 1; level < nlevels; level++) {
        Ptex::Res nextres((int8_t)(res.ulog2-1), (int8_t)(res.vlog2-1));
        face.faceData[level].resize(nextres.size64() * _pixelSize);
        int dstride = nextres.u() * _pixelSize;
        _reduceFn(data, stride, res.u(), res.v(), face.faceData[level].data(), dstride, datatype(), _header.nchannels);
        data = face.faceData[level].data();
        stride = dstride;
        res = nextres;
    }

    // compute and store constant value from last level (note: level 0 would be more accurate, but slower)
    storeConstValue(faceid, data, stride, res);

    // free premultData (if allocated) as it is no longer needed
    premultData.clear();
    premultData.shrink_to_fit();

    // compress face data for each level
    for (int level = 0; level < nlevels; level++) {
        Ptex::Res res((int8_t)(f.res.ulog2-level), (int8_t)(f.res.vlog2-level));
        std::vector<std::byte> compressedData;
        compressFaceData(compressor, compressedData, face.fdh[level], res, face.faceData[level].data());
        face.faceData[level] = std::move(compressedData);
    }
    releaseCompressor(compressor);

    return true;
}


bool PtexMainWriter::writeConstantFace(int faceid, const FaceInfo& f, const void* data)
{
    if (!_ok) return 0;

    // check and store face info
    if (!storeFaceInfo(faceid, _faceinfo[faceid], f, FaceInfo::flag_constant)) return 0;

    // store face value in constant block
    memcpy(&_constdata[faceid*_pixelSize], data, _pixelSize);
    return 1;
}



void PtexMainWriter::storeConstValue(int faceid, const void* data, int stride, Res res)
{
    // compute average value and store in _constdata block
    std::byte* constdata = &_constdata[faceid*_pixelSize];
    PtexUtils::average(data, stride, res.u(), res.v(), constdata,
                       datatype(), _header.nchannels);
    if (_header.hasAlpha())
        PtexUtils::divalpha(constdata, 1, datatype(), _header.nchannels, _header.alphachan);
}



void PtexMainWriter::finish()
{
    uint32_t nfaces = _header.nfaces;
    // copy missing faces from _reader
    if (_reader) {
        for (uint32_t faceid = 0; faceid < nfaces; faceid++) {
            if (_faceinfo[faceid].flags == uint8_t(-1)) {
                // copy constant data
                memcpy(&_constdata[faceid*_pixelSize], _reader->getConstantData(faceid), _pixelSize);

                // update faceinfo and copy face data
                const Ptex::FaceInfo& info = _reader->getFaceInfo(faceid);
                if (info.isConstant()) {
                    storeFaceInfo(faceid, _faceinfo[faceid], info, FaceInfo::flag_constant);
                } else if (_genmipmaps && !_reader->hasMipMaps()) {
                    // read uncompressed data and generate mipmaps
                    size_t size = _pixelSize * info.res.size64();
                    char* data = new char [size];
                    _reader->getData(faceid, data, 0);
                    writeFace(faceid, info, data, 0);
                    delete [] data;
                } else {
                    // read compressed data, including mipmaps if any
                    FaceRec& face = _faces[faceid];
                    storeFaceInfo(faceid, _faceinfo[faceid], info);
                    int nlevels = 1;
                    if (_genmipmaps) {
                        nlevels += std::max(0, std::min(info.res.ulog2, info.res.vlog2) - MinReductionLog2);
                    }
                    face.fdh.resize(nlevels);
                    face.faceData.resize(nlevels);
                    for (int level = 0; level < nlevels; level++) {
                        _reader->getCompressedData(faceid, level, face.fdh[level], face.faceData[level]);
                    }
                }
            }
        }
    }
    else {
        // just flag missing faces as constant (black)
        for (uint32_t faceid = 0; faceid < nfaces; faceid++) {
            if (_faceinfo[faceid].flags == uint8_t(-1))
                _faceinfo[faceid].flags = FaceInfo::flag_constant;
        }
    }

    // generate "rfaceids", reduction faceids, which are faceids reordered by decreasing smaller dimension
    if (_genmipmaps) {
        _rfaceids.resize(nfaces);
        _faceids_r.resize(nfaces);
        PtexUtils::genRfaceids(&_faceinfo[0], nfaces, &_rfaceids[0], &_faceids_r[0]);
    }

    // flag faces w/ constant neighborhoods
    flagConstantNeighorhoods();

    // compress face info block
    std::vector<std::byte> compressedFaceInfo;
    libdeflate_compressor* compressor = getCompressor();
    compressDataBlock(compressor, compressedFaceInfo, _faceinfo.data(), _faceinfo.size()*sizeof(FaceInfo));

    // compress const data block
    std::vector<std::byte> compressedConstData;
    compressDataBlock(compressor, compressedConstData, _constdata.data(), _constdata.size());

    // create level info and compress level headers
    std::vector<LevelInfo> levelInfo(1);
    for (auto& face : _faces) {
        size_t nlevelsThisFace = face.faceData.size();
        if (nlevelsThisFace > levelInfo.size()) {
            levelInfo.resize(nlevelsThisFace);
        }
        for (size_t level = 0; level < nlevelsThisFace; level++) {
            levelInfo[level].leveldatasize += face.faceData[level].size();
            levelInfo[level].nfaces++;
        }
    }
    levelInfo[0].nfaces = _header.nfaces; // constant faces weren't counted in the loop above, but level 0 needs them

    int nlevels = int(levelInfo.size());

    // gather fdh for faces in each level into LevelInfo, and compress level data headers
    std::vector<std::vector<std::byte>> compressedLevelDataHeaders(nlevels);
    std::vector<std::vector<size_t>> largeFaceHeaders(nlevels);
    size_t totalLevelDataSize = 0;
    for (int level = 0; level < nlevels; level++) {
        uint32_t nfacesThisLevel = levelInfo[level].nfaces;
        std::vector<FaceDataHeader> levelDataHeader(nfacesThisLevel);
        for (uint32_t f = 0; f < nfacesThisLevel; f++) {
            uint32_t faceId = level==0? f : _faceids_r[f];
            if (_faces[faceId].fdh.size() > size_t(level)) {
                levelDataHeader[f] = _faces[faceId].fdh[level];
                if (levelDataHeader[f].isLargeFace()) {
                    // large faces have a compressed size too big for the fdh; store in largeFaceHeader
                    largeFaceHeaders[level].push_back(_faces[faceId].faceData[level].size());
                }
            }
        }
        compressDataBlock(compressor, compressedLevelDataHeaders[level], levelDataHeader.data(), nfacesThisLevel * sizeof(FaceDataHeader));
        levelInfo[level].levelheadersize = uint32_t(compressedLevelDataHeaders[level].size());
        levelInfo[level].leveldatasize += levelInfo[level].levelheadersize + sizeof(size_t) * largeFaceHeaders[level].size();
        totalLevelDataSize += levelInfo[level].leveldatasize;
    }

    // compress meta data
    std::vector<MetaEntry*> lmdEntries; // large meta data items
    std::vector<std::byte> metaData, compressedMetaData;
    for (size_t i = 0, n = _metadata.size(); i < n; i++) {
        MetaEntry& e = _metadata[i];
        if (e.data.size() > MetaDataThreshold) {
            // skip large items, but record for later
            lmdEntries.push_back(&e);
        } else {
            // add small item to meta data block
            addToMetaDataBlock(metaData, e);
        }
    }
    if (!metaData.empty()) {
        compressDataBlock(compressor, compressedMetaData, metaData.data(), metaData.size());
    }

    // compress large meta data
    size_t nLmd = lmdEntries.size();
    std::vector<std::byte> lmdHeader, compressedLmdHeader;
    std::vector<std::vector<std::byte>> compressedLargeMetaData(nLmd);
    if (nLmd > 0) {
        // compress lmd data items
        for (size_t i = 0; i < nLmd; i++) {
            MetaEntry* e= lmdEntries[i];
            compressDataBlock(compressor, compressedLargeMetaData[i], &e->data[0], e->data.size());

            uint8_t keysize = uint8_t(e->key.size()+1);
            uint8_t datatype = e->datatype;
            uint32_t datasize = (uint32_t)e->data.size();
            uint32_t zipsize = (uint32_t)compressedLargeMetaData[i].size();

            addToDataBlock(lmdHeader, &keysize, sizeof(keysize));
            addToDataBlock(lmdHeader, e->key.c_str(), keysize);
            addToDataBlock(lmdHeader, &datatype, sizeof(datatype));
            addToDataBlock(lmdHeader, &datasize, sizeof(datasize));
            addToDataBlock(lmdHeader, &zipsize, sizeof(zipsize));
        }
        compressDataBlock(compressor, compressedLmdHeader, lmdHeader.data(), lmdHeader.size());
    }
    releaseCompressor(compressor);
    compressor = nullptr;

    // init header
    _header.nlevels = nlevels;
    _header.faceinfosize = compressedFaceInfo.size();
    _header.constdatasize = compressedConstData.size();
    _header.levelinfosize = LevelInfoSize * nlevels;
    _header.leveldatasize = totalLevelDataSize;
    _header.metadatamemsize = uint32_t(metaData.size());
    _header.metadatazipsize = uint32_t(compressedMetaData.size());
    _extheader.lmdheadermemsize = lmdHeader.size();
    _extheader.lmdheaderzipsize = compressedLmdHeader.size();

    // create new file
    FILE* newfp = fopen(_newpath.c_str(), "wb+");
    if (!newfp) {
        setError(fileError("Can't write to ptex file: ", _newpath.c_str()));
        return;
    }

    // write header
    writeBlock(newfp, &_header, HeaderSize);
    writeBlock(newfp, &_extheader, ExtHeaderSize);

    // write face info block
    writeBlock(newfp, compressedFaceInfo.data(), compressedFaceInfo.size());

    // write const data block
    writeBlock(newfp, compressedConstData.data(), compressedConstData.size());

    // write level info block
    writeBlock(newfp, &levelInfo[0], LevelInfoSize * nlevels);

    // write level data blocks
    for (int level = 0; level < nlevels; level++) {
        // write level data header
        writeBlock(newfp, compressedLevelDataHeaders[level].data(), compressedLevelDataHeaders[level].size());

        // write large face header, if present
        if (!largeFaceHeaders[level].empty()) {
            writeBlock(newfp, largeFaceHeaders[level].data(), sizeof(size_t) * largeFaceHeaders[level].size());
        }

        // write compressed face data for faces in level
        uint32_t nfacesThisLevel = levelInfo[level].nfaces;
        for (uint32_t rfaceId = 0; rfaceId < nfacesThisLevel; rfaceId++) {
            uint32_t faceId = level==0? rfaceId : _faceids_r[rfaceId];
            if (_faces[faceId].faceData.size() > size_t(level)) {
                writeBlock(newfp, _faces[faceId].faceData[level].data(), _faces[faceId].faceData[level].size());
            }
        }
    }

    // write meta data
    if (!compressedMetaData.empty()) {
        writeBlock(newfp, compressedMetaData.data(), compressedMetaData.size());
    }

    // write compatibility barrier
    uint64_t compatibilityBarrier = 0;
    writeBlock(newfp, &compatibilityBarrier, sizeof(compatibilityBarrier));

    // write large meta data
    if (!compressedLmdHeader.empty()) {
        writeBlock(newfp, compressedLmdHeader.data(), compressedLmdHeader.size());
        for (auto& lmd : compressedLargeMetaData) {
            writeBlock(newfp, lmd.data(), lmd.size());
        }
    }
    fclose(newfp);
}


void PtexMainWriter::flagConstantNeighorhoods()
{
    // for each constant face
    for (int faceid = 0, n = int(_faceinfo.size()); faceid < n; faceid++) {
        FaceInfo& f = _faceinfo[faceid];
        if (!f.isConstant()) continue;
        std::byte* constdata = &_constdata[faceid*_pixelSize];

        // check to see if neighborhood is constant
        bool isConst = true;
        bool isTriangle = _header.meshtype == mt_triangle;
        int nedges = isTriangle ? 3 : 4;
        for (int eid = 0; isConst && (eid < nedges); eid++) {
            bool prevWasSubface = f.isSubface();
            int prevFid = faceid;

            // traverse around vertex in CW direction
            int afid = f.adjface(eid);
            int aeid = f.adjedge(eid);
            int count = 0;
            const int maxcount = 10; // max valence (as safety valve)
            while (afid != faceid && afid >= 0 && ++count < maxcount) {
                // check if neighbor is constant, and has the same value as face
                FaceInfo& af = _faceinfo[afid];
                if (!af.isConstant() ||
                    0 != memcmp(constdata, &_constdata[afid*_pixelSize], _pixelSize))
                { isConst = false; break; }

                // if vertex is a T vertex between subface and main face, we can stop
                bool isSubface = af.isSubface();
                bool isT = !isTriangle && prevWasSubface && !isSubface && af.adjface(aeid) == prevFid;
                if (isT) break;
                prevWasSubface = isSubface;

                // traverse around vertex in CW direction
                prevFid = afid;
                aeid = (aeid + 1) % nedges;
                afid = af.adjface(aeid);
                aeid = af.adjedge(aeid);
            }

            if (afid < 0)  {
                // hit boundary edge, check boundary mode
                if (_extheader.ubordermode != Ptex::m_clamp || _extheader.vbordermode != Ptex::m_clamp) {
                    isConst = false;
                }

                // and traverse CCW neighbors too
                if (isConst) {
                    aeid = (aeid - 1 + nedges) % nedges;
                    afid = f.adjface(aeid);
                    aeid = f.adjedge(aeid);
                    count = 0;
                    while (afid != faceid && afid >= 0 && ++count < maxcount) {
                        // check if neighbor is constant, and has the same value as face
                        FaceInfo& af = _faceinfo[afid];
                        if (!af.isConstant() ||
                            0 != memcmp(constdata, &_constdata[afid*_pixelSize], _pixelSize))
                        { isConst = false; break; }

                        // traverse around vertex in CCW direction
                        prevFid = afid;
                        aeid = (aeid - 1 + nedges) % nedges;
                        afid = af.adjface(aeid);
                        aeid = af.adjedge(aeid);

                        // if traversing to a subface, switch to secondary subface (afid points to primary/CW subface)
                        bool isSubface = af.isSubface();
                        if (isSubface && !prevWasSubface) {
                            aeid = (aeid + 3) % 4;
                            afid = af.adjface(aeid);
                            aeid = (af.adjedge(aeid) + 3) % 4;
                        }
                        prevWasSubface = isSubface;
                    }
                }
            }
        }
        if (isConst) f.flags |= FaceInfo::flag_nbconstant;
    }
}


PTEX_NAMESPACE_END
