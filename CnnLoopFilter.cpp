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

/** \file     CnnLoopFilter.cpp
    \brief    cnn loop filter class
*/

#include "CnnLoopFilter.h"

#if NN_FILTER
#include "CodingStructure.h"
#include "Picture.h"

CnnLoopFilter::CnnLoopFilter()
{

  for( int compIdx = 0; compIdx < MAX_NUM_COMPONENT; compIdx++ )
  {
    m_ctuEnableFlag[compIdx] = nullptr;
  }
  m_initFlag = false;
}

void CnnLoopFilter::PreCNNLFProcess(Picture* pic, CodingStructure& cs, CnnlfSliceParam& cnnlfSliceParam)
{
  int baseQP = cs.slice->getPPS()->getPicInitQPMinus26() + 26;
  int SliceQP     = cs.slice->getSliceQp();
  initCnnModel(baseQP, SliceQP);

  if (!cnnlfSliceParam.enabledFlag[COMPONENT_Y] && !cnnlfSliceParam.enabledFlag[COMPONENT_Cb] && !cnnlfSliceParam.enabledFlag[COMPONENT_Cr])
  {
    return;
  }

  // set CTU enable flags
  for (int compIdx = 0; compIdx < MAX_NUM_COMPONENT; compIdx++)
  {
    m_ctuEnableFlag[compIdx] = cs.picture->getCnnlfCtuEnableFlag(compIdx);
  }

  PelUnitBuf recYuv = cs.getRecoBuf();
  m_tempCnnBuf.copyFrom(recYuv);
  PelUnitBuf cnnYuv = m_tempCnnBuf.getBuf(cs.area);

  PelUnitBuf predYuv      = cs.getTruePredBuf();
  PelUnitBuf partitionYuv = cs.picture->getPartBuf();

  // run DRNLF
  runCNNLF(pic, recYuv, predYuv, partitionYuv, cnnYuv, cs.slice->getSliceType(), baseQP, cs.slice->getSliceQp(), true);

#if NN_SCALE
  scaleResidue(cs, recYuv, cnnYuv, cs.slice, true);
#endif
}

void CnnLoopFilter::CNNLFProcess( CodingStructure& cs, CnnlfSliceParam& cnnlfSliceParam )
{
  if (!cnnlfSliceParam.enabledFlag[COMPONENT_Y] && !cnnlfSliceParam.enabledFlag[COMPONENT_Cb] && !cnnlfSliceParam.enabledFlag[COMPONENT_Cr])
  {
    return;
  }

  // set clipping range
  m_clpRngs = cs.slice->getClpRngs();

  // set CTU enable flags
  for( int compIdx = 0; compIdx < MAX_NUM_COMPONENT; compIdx++ )
  {
    m_ctuEnableFlag[compIdx] = cs.picture->getCnnlfCtuEnableFlag( compIdx );
  }

  filterPic(cs);
}

void CnnLoopFilter::create( const int picWidth, const int picHeight, const ChromaFormat format, const int maxCUWidth, const int maxCUHeight, const int maxCUDepth, const int inputBitDepth[MAX_NUM_CHANNEL_TYPE])
{
  std::memcpy( m_inputBitDepth, inputBitDepth, sizeof( m_inputBitDepth ) );
  m_picWidth = picWidth;
  m_picHeight = picHeight;
  m_maxCUWidth = maxCUWidth;
  m_maxCUHeight = maxCUHeight;
  m_maxCUDepth = maxCUDepth;
  m_chromaFormat = format;

  m_numCTUsInWidth = ( m_picWidth / m_maxCUWidth ) + ( ( m_picWidth % m_maxCUWidth ) ? 1 : 0 );
  m_numCTUsInHeight = ( m_picHeight / m_maxCUHeight ) + ( ( m_picHeight % m_maxCUHeight ) ? 1 : 0 );
  m_numCTUsInPic = m_numCTUsInHeight * m_numCTUsInWidth;


  m_tempCnnBuf.destroy();
  m_tempCnnBuf.create( format, Area( 0, 0, picWidth, picHeight ), maxCUWidth,  0, false );
}

void CnnLoopFilter::destroy()
{
  m_tempCnnBuf.destroy();
}


void CnnLoopFilter::filterPic(CodingStructure& cs)
{
  PelUnitBuf recYuv = cs.getRecoBuf();
  PelUnitBuf cnnYuv = m_tempCnnBuf.getBuf(cs.area);

  const PreCalcValues& pcv = *cs.pcv;
  int ctuIdx = 0;
  for (int yPos = 0; yPos < pcv.lumaHeight; yPos += pcv.maxCUHeight)
  {
    for (int xPos = 0; xPos < pcv.lumaWidth; xPos += pcv.maxCUWidth)
    {
      const int width = (xPos + pcv.maxCUWidth > pcv.lumaWidth) ? (pcv.lumaWidth - xPos) : pcv.maxCUWidth;
      const int height = (yPos + pcv.maxCUHeight > pcv.lumaHeight) ? (pcv.lumaHeight - yPos) : pcv.maxCUHeight;

      // LUMA
      if (m_ctuEnableFlag[COMPONENT_Y][ctuIdx] > 0)
      {
        Area blk(xPos, yPos, width, height);
        filterBlk(recYuv, cnnYuv, blk, COMPONENT_Y, m_clpRngs.comp[COMPONENT_Y]);
      }
      // CHROMA
      for (int compIdx = 1; compIdx < MAX_NUM_COMPONENT; compIdx++)
      {
        ComponentID compID       = ComponentID(compIdx);
        const int   chromaScaleX = getComponentScaleX(compID, recYuv.chromaFormat);
        const int   chromaScaleY = getComponentScaleY(compID, recYuv.chromaFormat);

        if (m_ctuEnableFlag[compIdx][ctuIdx] > 0)
        {
          Area blk(xPos >> chromaScaleX, yPos >> chromaScaleY, width >> chromaScaleX, height >> chromaScaleY);
          filterBlk(recYuv, cnnYuv, blk, compID, m_clpRngs.comp[compIdx]);
        }
      }
      ctuIdx++;
    }
  }
}

void CnnLoopFilter::filterBlk( PelUnitBuf &recUnitBuf, const CPelUnitBuf& cnnUnitBuf, const Area& blk, const ComponentID compId,  const ClpRng& clpRng )
{
  const CPelBuf cnnBlk = cnnUnitBuf.get(compId).subBuf(blk.pos(), blk.size());
  PelBuf recBlk = recUnitBuf.get(compId).subBuf(blk.pos(), blk.size());
  recBlk.copyFrom(cnnBlk);
}

void CnnLoopFilter::runCNNLF(Picture* pic, const PelUnitBuf& recUnitBuf, const PelUnitBuf &predUnitBuf, const PelUnitBuf &parUnitBuf, PelUnitBuf& cnnUnitBuf, const SliceType slice_type, const int baseQP, const int iQP, bool is_dec)
{ 
  int patchSize = 128;
  int padSize   = 8;
  double maxValue  = 1023.0;
  bool       isInter     = slice_type != I_SLICE;
  int picWidth = pic->getPicWidthInLumaSamples();
  int picHeight = pic->getPicHeightInLumaSamples();

  int picWidthInPatchs = ceil((double)picWidth / patchSize);
  int picHeightInPatchs = ceil((double)picHeight / patchSize);


  int ctuRsAddr = 0;
  for (int y = 0; y < picHeightInPatchs; y++)
  {
    for (int x = 0; x < picWidthInPatchs; x++)
    {
      if (is_dec)
      {
        if (!m_ctuEnableFlag[COMPONENT_Y][ctuRsAddr] && !m_ctuEnableFlag[COMPONENT_Cb][ctuRsAddr] && !m_ctuEnableFlag[COMPONENT_Cr][ctuRsAddr])
        {
          ctuRsAddr++;
          continue;
        }
      }

      int pix_y = y * patchSize;
      int pix_x = x * patchSize;

      int pix_y_end = (y+1) * patchSize > picHeight ? picHeight - 1 : (y + 1) * patchSize - 1;
      int pix_x_end = (x+1) * patchSize > picWidth  ? picWidth - 1 : (x + 1) * patchSize - 1;
      
      int st_h = pix_y - padSize < 0 ? 0 : pix_y - padSize;
      int ed_h = pix_y_end + padSize > picHeight ? picHeight -1  : pix_y_end + padSize;
      int st_w = pix_x - padSize < 0 ? 0 : pix_x - padSize;
      int ed_w = pix_x_end + padSize > picWidth ? picWidth - 1 : pix_x_end + padSize;

      int actualPatchSizeH = ed_h - st_h + 1;
      int actualPatchSizeW = ed_w - st_w + 1;
     
      int actualPatchSizeH_chroma = actualPatchSizeH >> 1;
      int actualPatchSizeW_chroma = actualPatchSizeW >> 1;

      int st_w1 = (st_w >> 1);
      int st_h1 = (st_h >> 1);

      int channel_y = 4, channel_uv = 8;
      if (isInter) {
          channel_y += 1;
          channel_uv+= 1;
      }

      torch::Tensor Patch_y = torch::ones({ 1, channel_y, actualPatchSizeH, actualPatchSizeW });
      torch::Tensor Patch_uv = torch::ones({ 1, channel_uv, actualPatchSizeH_chroma, actualPatchSizeW_chroma });

      float *pPatch_y = Patch_y.data_ptr<float>();
      float* pPatch_uv = Patch_uv.data_ptr<float>();

      cv::Mat cnn_recY(actualPatchSizeH, actualPatchSizeW, CV_32FC1);
      cv::Mat cnn_recU(actualPatchSizeH_chroma, actualPatchSizeW_chroma, CV_32FC1);
      cv::Mat cnn_recV(actualPatchSizeH_chroma, actualPatchSizeW_chroma, CV_32FC1);
      cv::Mat cnn_predU(actualPatchSizeH_chroma, actualPatchSizeW_chroma, CV_32FC1);
      cv::Mat cnn_predV(actualPatchSizeH_chroma, actualPatchSizeW_chroma, CV_32FC1);
      cv::Mat cnn_parU(actualPatchSizeH_chroma, actualPatchSizeW_chroma, CV_32FC1);
      cv::Mat cnn_parV(actualPatchSizeH_chroma, actualPatchSizeW_chroma, CV_32FC1);

      cv::Mat cnn_recY_down(actualPatchSizeH_chroma, actualPatchSizeW_chroma, CV_32FC1);
      cv::Mat cnn_recU_up(actualPatchSizeH, actualPatchSizeW, CV_32FC1);
      cv::Mat cnn_recV_up(actualPatchSizeH, actualPatchSizeW, CV_32FC1);
      cv::Mat cnn_predU_up(actualPatchSizeH, actualPatchSizeW, CV_32FC1);
      cv::Mat cnn_predV_up(actualPatchSizeH, actualPatchSizeW, CV_32FC1);
      cv::Mat cnn_parU_up(actualPatchSizeH, actualPatchSizeW, CV_32FC1);
      cv::Mat cnn_parV_up(actualPatchSizeH, actualPatchSizeW, CV_32FC1);

      for (int yy = 0; yy < actualPatchSizeH_chroma; yy++)
      {
        for (int xx = 0; xx < actualPatchSizeW_chroma; xx++)
        {
          int id_x                    = (st_w >> 1) + xx;
          int id_y                    = (st_h >> 1) + yy;
          cnn_recU.at<float>(yy, xx)  = recUnitBuf.get(COMPONENT_Cb).at(id_x, id_y);
          cnn_recV.at<float>(yy, xx)  = recUnitBuf.get(COMPONENT_Cr).at(id_x, id_y);
          cnn_predU.at<float>(yy, xx) = predUnitBuf.get(COMPONENT_Cb).at(id_x, id_y);
          cnn_predV.at<float>(yy, xx) = predUnitBuf.get(COMPONENT_Cr).at(id_x, id_y);
          cnn_parU.at<float>(yy, xx) = parUnitBuf.get(COMPONENT_Cb).at(id_x, id_y);
          cnn_parV.at<float>(yy, xx) = parUnitBuf.get(COMPONENT_Cr).at(id_x, id_y);
        }
      }

      cv::resize(cnn_recU, cnn_recU_up, cnn_recU_up.size(), 0, 0, cv::INTER_LANCZOS4);
      cv::resize(cnn_recV, cnn_recV_up, cnn_recV_up.size(), 0, 0, cv::INTER_LANCZOS4);
      cv::resize(cnn_predU, cnn_predU_up, cnn_predU_up.size(), 0, 0, cv::INTER_LANCZOS4);
      cv::resize(cnn_predV, cnn_predV_up, cnn_predV_up.size(), 0, 0, cv::INTER_LANCZOS4);
      cv::resize(cnn_parU, cnn_parU_up, cnn_parU_up.size(), 0, 0, cv::INTER_LANCZOS4);
      cv::resize(cnn_parV, cnn_parV_up, cnn_parV_up.size(), 0, 0, cv::INTER_LANCZOS4);

      for (int yy = 0; yy < actualPatchSizeH; yy++)
      {
          for (int xx = 0; xx < actualPatchSizeW; xx++)
          {
              int id_x = st_w + xx;
              int id_y = st_h + yy;
              cnn_recY.at<float>(yy, xx) = recUnitBuf.get(COMPONENT_Y).at(id_x, id_y);
          }
      }
      cv::resize(cnn_recY, cnn_recY_down, cnn_recY_down.size(), 0, 0);

      int actualPatchArea = actualPatchSizeH * actualPatchSizeW;
      int actualPatchArea_chroma = actualPatchSizeH_chroma * actualPatchSizeW_chroma;

      for (int yy = 0; yy < actualPatchSizeH; yy++)
      {
        for (int xx = 0; xx < actualPatchSizeW; xx++)
        {
          int id_x = st_w + xx;
          int id_y = st_h + yy;
          pPatch_y[yy * actualPatchSizeW + xx]                   = recUnitBuf.get(COMPONENT_Y).at(id_x, id_y) / maxValue;
          pPatch_y[actualPatchArea + yy * actualPatchSizeW + xx] = predUnitBuf.get(COMPONENT_Y).at(id_x, id_y) / maxValue;
          pPatch_y[2 * actualPatchArea + yy * actualPatchSizeW + xx] = parUnitBuf.get(COMPONENT_Y).at(id_x, id_y) / maxValue;
          pPatch_y[3 * actualPatchArea + yy * actualPatchSizeW + xx] = baseQP / 63.0;
          if (isInter) {
              pPatch_y[4 * actualPatchArea + yy * actualPatchSizeW + xx] = iQP / 63.0;
          }
        }
      }
      for (int yy = 0; yy < actualPatchSizeH_chroma; yy++)
      {
          for (int xx = 0; xx < actualPatchSizeW_chroma; xx++)
          {
              pPatch_uv[yy * actualPatchSizeW_chroma + xx] = cnn_recY_down.at<float>(yy, xx) / maxValue;
              pPatch_uv[actualPatchArea_chroma + yy * actualPatchSizeW_chroma + xx] = cnn_recU.at<float>(yy, xx) / maxValue;
              pPatch_uv[2 * actualPatchArea_chroma + yy * actualPatchSizeW_chroma + xx] = cnn_recV.at<float>(yy, xx) / maxValue;
              pPatch_uv[3 * actualPatchArea_chroma + yy * actualPatchSizeW_chroma + xx] = cnn_predU.at<float>(yy, xx) / maxValue;
              pPatch_uv[4 * actualPatchArea_chroma + yy * actualPatchSizeW_chroma + xx] = cnn_predV.at<float>(yy, xx) / maxValue;
              pPatch_uv[5 * actualPatchArea_chroma + yy * actualPatchSizeW_chroma + xx] = cnn_parU.at<float>(yy, xx) / maxValue;
              pPatch_uv[6 * actualPatchArea_chroma + yy * actualPatchSizeW_chroma + xx] = cnn_parV.at<float>(yy, xx) / maxValue;
              pPatch_uv[7 * actualPatchArea_chroma + yy * actualPatchSizeW_chroma + xx] = baseQP / 63.0;
              if (isInter) {
                  pPatch_uv[8 * actualPatchArea_chroma + yy * actualPatchSizeW_chroma + xx] = iQP / 63.0;
              }
          }
      }
#if ENABLE_CUDA
      torch::Tensor input_y = Patch_y.to(at::kCUDA);
      torch::Tensor input_uv = Patch_uv.to(at::kCUDA);
#else
      torch::Tensor input_y = Patch_y;
      torch::Tensor input_uv = Patch_uv;
#endif
      torch::NoGradGuard no_grad_guard;
      torch::globalContext().setFlushDenormal(true);

      torch::Tensor output_y, output_uv;
      if (!isInter)
      {
        output_y = (models[0].forward({ input_y })).toTensor();
        output_uv = (models[1].forward({ input_uv })).toTensor();
      }
      else
      {
        output_y = (models[2].forward({ input_y })).toTensor();
        output_uv = (models[3].forward({ input_uv })).toTensor();
      }
      
      
#if ENABLE_CUDA
      output_y = output_y.to(at::kCPU);
      output_uv = output_uv.to(at::kCPU);
#endif

      float *pOutput_y = output_y.data_ptr<float>();
      float* pOutput_uv = output_uv.data_ptr<float>();

      int centerH = pix_y_end - pix_y + 1;
      int centerW = pix_x_end - pix_x + 1;
      for (int yy = 0; yy < centerH; yy++)
      {
          for (int xx = 0; xx < centerW; xx++)
          {
              int id_x = pix_x + xx;
              int id_y = pix_y + yy;

              cnnUnitBuf.get(COMPONENT_Y).at(id_x, id_y) = Pel(Clip3<int>(0, 1023, int(pOutput_y[(id_y - st_h) * actualPatchSizeW + (id_x - st_w)] * maxValue + 0.5)));
          }
      }
      int centerH_chroma = centerH >> 1;
      int centerW_chroma = centerW >> 1;


      for (int yy = 0; yy < centerH_chroma; yy++)
      {
        for (int xx = 0; xx < centerW_chroma; xx++)
        {
          int id_x = (pix_x >> 1) + xx;
          int id_y = (pix_y >> 1) + yy;

          cnnUnitBuf.get(COMPONENT_Cb).at(id_x, id_y) = Pel(Clip3<int>(0, 1023, int(pOutput_uv[(id_y - st_h1) * actualPatchSizeW_chroma + (id_x - st_w1)] * maxValue + 0.5)));
          cnnUnitBuf.get(COMPONENT_Cr).at(id_x, id_y) = Pel(Clip3<int>(0, 1023, int(pOutput_uv[actualPatchArea_chroma + (id_y - st_h1) * actualPatchSizeW_chroma + (id_x - st_w1)] * maxValue + 0.5)));
        }
      }
      ctuRsAddr++;
    }
  }
}

#if NN_SCALE
void CnnLoopFilter::scaleResidue(CodingStructure &cs, PelUnitBuf recUnitBuf, PelUnitBuf cnnYuv, Slice *slice,
                                 bool is_dec)
{
  const PreCalcValues &pcv = *cs.pcv;
  for (int compIdx = 0; compIdx < MAX_NUM_COMPONENT; compIdx++)
  {
    ComponentID compID       = ComponentID(compIdx);
    const int   chromaScaleX = getComponentScaleX(compID, recUnitBuf.chromaFormat);
    const int   chromaScaleY = getComponentScaleY(compID, recUnitBuf.chromaFormat);

    PelBuf recBuf = recUnitBuf.get(compID);
    PelBuf cnnBuf = cnnYuv.get(compID);

    const int scale  = slice->getNnScale(compID);
    int       shift  = NN_SCALE_SHIFT + NN_SCALE_EXT_SHIFT;
    int       offset = (1 << shift) / 2;

    int ctuIdx = 0;
    for (int yPos = 0; yPos < pcv.lumaHeight; yPos += pcv.maxCUHeight)
    {
      for (int xPos = 0; xPos < pcv.lumaWidth; xPos += pcv.maxCUWidth)
      {
        if (is_dec && !m_ctuEnableFlag[compID][ctuIdx])
        {
          ctuIdx++;
          continue;
        }

        int width  = (xPos + pcv.maxCUWidth > pcv.lumaWidth) ? (pcv.lumaWidth - xPos) : pcv.maxCUWidth;
        int height = (yPos + pcv.maxCUHeight > pcv.lumaHeight) ? (pcv.lumaHeight - yPos) : pcv.maxCUHeight;

        int x_start = xPos >> chromaScaleX;
        int y_start = yPos >> chromaScaleY;
        int x_end   = x_start + (width >> chromaScaleX);
        int y_end   = y_start + (height >> chromaScaleY);

        for (int y = y_start; y < y_end; y++)
        {
          for (int x = x_start; x < x_end; x++)
          {
            cnnBuf.at(x, y) =
              Clip3(0, 1023,
                    recBuf.at(x, y)
                      + (((cnnBuf.at(x, y) - (recBuf.at(x, y) << NN_SCALE_EXT_SHIFT)) * scale + offset) >> shift));
          }
        }

        ctuIdx++;
      }
    }
  }
}
#endif

void CnnLoopFilter::initCnnModel(const int baseQP, const int iQP)
{
  if (m_initFlag)
  {
    return;
  }

  at::set_num_threads(1);
  at::set_num_interop_threads(1);

   /*
  std::string rootPath = "/media/lab/fengzhen/VVCSoftware_VTM/model/";

  models.push_back(torch::jit::load(rootPath + "AI/filter_Y.pt"));
  models.push_back(torch::jit::load(rootPath + "AI/filter_UV.pt"));
  models.push_back(torch::jit::load(rootPath + "RA/filter_Y.pt"));
  models.push_back(torch::jit::load(rootPath + "RA/filter_UV.pt"));
  */
  models.push_back(torch::jit::load("/media/media-318/c6f59dc7-a6d6-4e6c-8c51-01d58118908e/fz/VVCSoftware_VTM/model/AI/filter_Y.pt"));
  models.push_back(torch::jit::load("/media/media-318/c6f59dc7-a6d6-4e6c-8c51-01d58118908e/fz/VVCSoftware_VTM/model/AI/filter_UV.pt"));
  models.push_back(torch::jit::load("/media/media-318/c6f59dc7-a6d6-4e6c-8c51-01d58118908e/fz/VVCSoftware_VTM/model/RA/filter_Y.pt"));
  models.push_back(torch::jit::load("/media/media-318/c6f59dc7-a6d6-4e6c-8c51-01d58118908e/fz/VVCSoftware_VTM/model/RA/filter_UV.pt"));

  
  
  for (int i = 0; i < models.size(); i++)
  {
#if ENABLE_CUDA
    models[i].to(at::kCUDA);
#endif
    models[i].eval();
  }
  m_initFlag = true;
}
#endif
