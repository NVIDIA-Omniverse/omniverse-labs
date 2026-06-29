#ifndef PtexWriter_h
#define PtexWriter_h

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
#include <map>
#include <vector>
#include <stdio.h>
#include "Ptexture.h"
#include "PtexIO.h"
#include "PtexReader.h"

struct libdeflate_compressor;

PTEX_NAMESPACE_BEGIN

class PtexMainWriter : public PtexWriter {
public:
    PtexMainWriter(const char* path, PtexTexture* tex,
                   Ptex::MeshType mt, Ptex::DataType dt,
                   int nchannels, int alphachan, int nfaces, bool genmipmaps);

    virtual void release();
    virtual bool close(Ptex::String& error);
    virtual bool writeFace(int faceid, const FaceInfo& f, const void* data, int stride);
    virtual bool writeConstantFace(int faceid, const FaceInfo& f, const void* data);

    virtual void setBorderModes(Ptex::BorderMode uBorderMode, Ptex::BorderMode vBorderMode)
    {
        _extheader.ubordermode = uBorderMode;
        _extheader.vbordermode = vBorderMode;
    }
    virtual void setEdgeFilterMode(Ptex::EdgeFilterMode edgeFilterMode)
    {
            _extheader.edgefiltermode = edgeFilterMode;
    }
    virtual void writeMeta(const char* key, const char* value);
    virtual void writeMeta(const char* key, const int8_t* value, int count);
    virtual void writeMeta(const char* key, const int16_t* value, int count);
    virtual void writeMeta(const char* key, const int32_t* value, int count);
    virtual void writeMeta(const char* key, const float* value, int count);
    virtual void writeMeta(const char* key, const double* value, int count);
    virtual void writeMeta(PtexMetaData* data);

    bool ok(Ptex::String& error) {
        if (!_ok) getError(error);
        return _ok;
    }
    void getError(Ptex::String& error) {
        error = (_error + "\nPtex file: " + _path).c_str();
    }

private:
    virtual ~PtexMainWriter();

    DataType datatype() const { return DataType(_header.datatype); }

    struct MetaEntry {
        std::string key;
        MetaDataType datatype;
        std::vector<uint8_t> data;
        MetaEntry() : datatype(MetaDataType(0)), data() {}
    };

    size_t writeBlock(FILE* fp, const void* data, size_t size);
    size_t writeBlock(FILE* fp, const std::vector<std::byte>& dataBlock) { return writeBlock(fp, dataBlock.data(), dataBlock.size()); }
    void addToDataBlock(std::vector<std::byte>& dataBlock, const void* data, size_t size);
    libdeflate_compressor* getCompressor();
    void releaseCompressor(libdeflate_compressor* compressor);
    void compressDataBlock(libdeflate_compressor* compressor, std::vector<std::byte>& compressedData, const void* data, size_t size);
    Res calcTileRes(Res faceres);
    void addMetaData(const char* key, MetaDataType t, const void* value, int size);
    void compressFaceDataBlock(libdeflate_compressor* compressor, std::vector<std::byte>& compressedData, FaceDataHeader& fdh,
                               Res res, const void* uncompressedData, int stride);
    void compressFaceData(libdeflate_compressor* compressor, std::vector<std::byte>& compressedData, FaceDataHeader& fdh,
                          Res res, const void* uncompressedData);
    void addToMetaDataBlock(std::vector<std::byte>& dataBlock, const MetaEntry& val);
    void setError(const std::string& error) { _error = error; _ok = false; }
    bool storeFaceInfo(int faceid, FaceInfo& dest, const FaceInfo& src, int flags=0);
    void finish();
    void flagConstantNeighorhoods();
    void storeConstValue(int faceid, const void* data, int stride, Res res);

    bool _ok;                                // true if no error has occurred
    std::string _error;                      // the error text (if any)
    std::string _path;                       // file path
    Header _header;                          // the file header
    ExtHeader _extheader;                    // extended header
    int _pixelSize;                          // size of a pixel in bytes
    std::vector<MetaEntry> _metadata;        // meta data waiting to be written
    std::map<std::string,int> _metamap;      // for preventing duplicate keys

    std::vector<libdeflate_compressor*> _compressors;
    Mutex _compressorMutex;

    PtexUtils::ReduceFn* _reduceFn;

    std::string _newpath;                 // path to ".new" file
    bool _genmipmaps;                     // true if mipmaps should be generated
    std::vector<FaceInfo> _faceinfo;      // info about each face
    std::vector<std::byte> _constdata;    // constant data for each face
    std::vector<uint32_t> _rfaceids;      // faceid reordering for reduction levels
    std::vector<uint32_t> _faceids_r;     // faceid indexed by rfaceid

    static const int MinReductionLog2 =2; // log2(minimum reduction size) - can tune
    struct FaceRec {
        std::vector<std::vector<std::byte>> faceData; // compressed face data, including reductions
        std::vector<FaceDataHeader> fdh;  // face data headers
        Mutex mutex;
    };
    std::vector<FaceRec> _faces;          // compressed data for each face

    PtexReader* _reader;                  // reader for accessing existing data in file
};


PTEX_NAMESPACE_END

#endif
