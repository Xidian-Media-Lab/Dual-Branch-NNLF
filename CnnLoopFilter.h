/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2018, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     CnnLoopFilter.h
    \brief    cnn loop filter class (header)
*/

#ifndef __CNNLOOPFILTER__
#define __CNNLOOPFILTER__
#include "CommonDef.h"

#if NN_FILTER

#include "Unit.h"
#include "Picture.h"
#define HAVE_INTTYPES_H 1
#define __STDC_FORMAT_MACROS
#define ENABLE_CUDA 0
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


class CnnLoopFilter
{
public:
  CnnLoopFilter();
  virtual ~CnnLoopFilter() {}
  void create(const int picWidth, const int picHeight, const ChromaFormat format, const int maxCUWidth, const int maxCUHeight, const int maxCUDepth, const int inputBitDepth[MAX_NUM_CHANNEL_TYPE]);
  void destroy();
  void PreCNNLFProcess(Picture* pic, CodingStructure& cs, CnnlfSliceParam& cnnlfSliceParam);
  void CNNLFProcess(CodingStructure& cs, CnnlfSliceParam& cnnlfSliceParam);

protected:
  uint8_t*                     m_ctuEnableFlag[MAX_NUM_COMPONENT];
  PelStorage                   m_tempCnnBuf;
  int                          m_inputBitDepth[MAX_NUM_CHANNEL_TYPE];
  int                          m_picWidth;
  int                          m_picHeight;
  int                          m_maxCUWidth;
  int                          m_maxCUHeight;
  int                          m_maxCUDepth;
  int                          m_numCTUsInWidth;
  int                          m_numCTUsInHeight;
  int                          m_numCTUsInPic;
  ChromaFormat                 m_chromaFormat;
  ClpRngs                      m_clpRngs;
  bool                         m_initFlag;

  std::vector<torch::jit::script::Module> models;

  torch::jit::script::Module m_Module;

  void filterPic(CodingStructure& cs);
  void filterBlk(PelUnitBuf &recDst, const CPelUnitBuf& recSrc, const Area& blk, const ComponentID compI, const ClpRng& clpRng);

  void initCnnModel(const int baseQP, const int iQP);
  void runCNNLF(Picture* pic, const PelUnitBuf& recUnitBuf, const PelUnitBuf &predUnitBuf, const PelUnitBuf &parUnitBuf, PelUnitBuf& cnnUnitBuf, const SliceType slice_type, const int baseQP, const int iQP, bool is_dec);
#if NN_SCALE
  void scaleResidue(CodingStructure &cs, PelUnitBuf recUnitBuf, PelUnitBuf cnnYuv, Slice *slice, bool is_dec);
#endif
};
#endif

#endif
