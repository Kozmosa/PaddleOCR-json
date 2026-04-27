// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "include/utility.h"

namespace PaddleOCR
{
    // PIMPL 前向声明，隐藏 v3.5.0 实现细节
    class PPOCRImpl;

    class PPOCR
    {
    public:
        explicit PPOCR();
        ~PPOCR();

        // OCR方法，处理图像列表，返回每个图像的OCR结果向量
        std::vector<std::vector<OCRPredictResult>> ocr(std::vector<cv::Mat> img_list,
                                                       bool det = true,
                                                       bool rec = true,
                                                       bool cls = true);
        // OCR方法，处理单个图像，返回OCR结果
        std::vector<OCRPredictResult> ocr(cv::Mat img, bool det = true,
                                          bool rec = true, bool cls = true);

        void reset_timer();
        void benchmark_log(int img_num);

    private:
        std::unique_ptr<PPOCRImpl> impl_;

        // 时间信息
        std::vector<double> time_info_det = {0, 0, 0};
        std::vector<double> time_info_rec = {0, 0, 0};
        std::vector<double> time_info_cls = {0, 0, 0};
    };
} // namespace PaddleOCR
