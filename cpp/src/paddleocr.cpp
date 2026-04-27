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

// 先包含 v3.5.0 的 ocr.h（定义全局 class PaddleOCR），避免与 namespace PaddleOCR 冲突
#include "src/api/pipelines/ocr.h"
#include "src/pipelines/ocr/result.h"
#include "src/pipelines/ocr/pipeline.h"

#include <include/paddleocr.h>
#include <include/args.h>
#include <include/utility.h>
#include <iostream>
#include <cstdio>

#if defined(_WIN32)
#include <process.h>
#define GETPID() _getpid()
#else
#include <unistd.h>
#define GETPID() getpid()
#endif

namespace PaddleOCR
{
    // 内部实现类
    class PPOCRImpl
    {
    public:
        PPOCRImpl();
        ~PPOCRImpl() = default;

        std::vector<OCRPredictResult> ocr(cv::Mat img, bool det, bool rec, bool cls);

    private:
        ::PaddleOCR_v35 engine_;
        static int temp_counter_;

        std::string save_temp_image(const cv::Mat& img);
        std::vector<OCRPredictResult> convert_results(
            const std::vector<std::unique_ptr<BaseCVResult>>& results);
    };

    int PPOCRImpl::temp_counter_ = 0;

    // 从路径中提取目录名作为模型名
    static std::string extract_model_name(const std::string& path) {
        size_t pos = path.find_last_of("/\\");
        if (pos != std::string::npos) {
            return path.substr(pos + 1);
        }
        return path;
    }

    PPOCRImpl::PPOCRImpl()
        : engine_([]() {
            PaddleOCRParams params;

            // 设置模型路径和名称
            if (!FLAGS_det_model_dir.empty()) {
                params.text_detection_model_dir = FLAGS_det_model_dir;
                params.text_detection_model_name = extract_model_name(FLAGS_det_model_dir);
            }
            if (!FLAGS_rec_model_dir.empty()) {
                params.text_recognition_model_dir = FLAGS_rec_model_dir;
                params.text_recognition_model_name = extract_model_name(FLAGS_rec_model_dir);
            }
            if (!FLAGS_cls_model_dir.empty()) {
                params.textline_orientation_model_dir = FLAGS_cls_model_dir;
                params.textline_orientation_model_name = extract_model_name(FLAGS_cls_model_dir);
            }

            // 设置设备
            std::string device = FLAGS_use_gpu ? "gpu:" + std::to_string(FLAGS_gpu_id) : "cpu";
            params.device = device;

            // 设置推理选项
            params.precision = FLAGS_precision;
            params.enable_mkldnn = FLAGS_enable_mkldnn;
            params.cpu_threads = FLAGS_cpu_threads;

            // 设置检测参数
            params.text_det_limit_side_len = FLAGS_limit_side_len;
            params.text_det_limit_type = FLAGS_limit_type;
            params.text_det_thresh = static_cast<float>(FLAGS_det_db_thresh);
            params.text_det_box_thresh = static_cast<float>(FLAGS_det_db_box_thresh);
            params.text_det_unclip_ratio = static_cast<float>(FLAGS_det_db_unclip_ratio);

            // 设置识别参数
            if (FLAGS_rec_batch_num > 0) {
                params.text_recognition_batch_size = FLAGS_rec_batch_num;
            }

            // 设置方向分类
            params.use_textline_orientation = FLAGS_use_angle_cls;

            // 禁用不需要的功能以保持兼容
            params.use_doc_orientation_classify = false;
            params.use_doc_unwarping = false;

            return params;
        }())
    {
    }

    std::string PPOCRImpl::save_temp_image(const cv::Mat& img)
    {
        static const std::string TEMP_PREFIX = "/tmp/paddleocr_json_";
        std::string temp_path = TEMP_PREFIX + std::to_string(GETPID()) + "_" +
                                std::to_string(temp_counter_++) + ".png";
        if (!cv::imwrite(temp_path, img)) {
            std::cerr << "[ERROR] Failed to save temporary image: " << temp_path << std::endl;
            return "";
        }
        return temp_path;
    }

    std::vector<OCRPredictResult> PPOCRImpl::convert_results(
        const std::vector<std::unique_ptr<BaseCVResult>>& results)
    {
        std::vector<OCRPredictResult> ocr_results;

        if (results.empty()) {
            return ocr_results;
        }

        // 向下转型为 OCRResult
        for (const auto& result : results) {
            OCRResult* ocr_result = dynamic_cast<OCRResult*>(result.get());
            if (!ocr_result) {
                continue;
            }

            // 获取 pipeline 结果
            const OCRPipelineResult& pipeline_result = ocr_result->GetPipelineResult();

            // 转换每个识别结果
            size_t num_results = pipeline_result.rec_texts.size();
            for (size_t i = 0; i < num_results; ++i) {
                OCRPredictResult res;
                res.text = pipeline_result.rec_texts[i];
                res.score = pipeline_result.rec_scores[i];

                // 转换检测框 (rec_polys 优先，方向校正后的)
                if (i < pipeline_result.rec_polys.size()) {
                    const auto& poly = pipeline_result.rec_polys[i];
                    for (const auto& pt : poly) {
                        res.box.push_back({static_cast<int>(pt.x), static_cast<int>(pt.y)});
                    }
                } else if (i < pipeline_result.dt_polys.size()) {
                    const auto& poly = pipeline_result.dt_polys[i];
                    for (const auto& pt : poly) {
                        res.box.push_back({static_cast<int>(pt.x), static_cast<int>(pt.y)});
                    }
                }

                // 转换方向分类结果
                if (i < pipeline_result.textline_orientation_angles.size()) {
                    int angle = pipeline_result.textline_orientation_angles[i];
                    // v3.5.0 textline_orientation 返回 0(0度) 或 1(180度)
                    res.cls_label = angle;
                    res.cls_score = 1.0f; // v3.5.0 不直接提供分类分数
                } else {
                    res.cls_label = -1;
                    res.cls_score = 0.0f;
                }

                ocr_results.push_back(res);
            }
        }

        // 按位置排序
        Utility::sorted_boxes(ocr_results);

        return ocr_results;
    }

    std::vector<OCRPredictResult> PPOCRImpl::ocr(cv::Mat img, bool det, bool rec, bool cls)
    {
        // 保存为临时文件
        std::string temp_path = save_temp_image(img);
        if (temp_path.empty()) {
            return std::vector<OCRPredictResult>();
        }

        // 调用 v3.5.0 API
        std::vector<std::unique_ptr<BaseCVResult>> results;

        try {
            results = engine_.Predict(temp_path);
        } catch (...) {
            // 清理临时文件
            std::remove(temp_path.c_str());
            return std::vector<OCRPredictResult>();
        }

        // 清理临时文件
        std::remove(temp_path.c_str());

        // 转换结果
        std::vector<OCRPredictResult> ocr_result = convert_results(results);

        // 根据 det/rec/cls 参数调整结果
        if (!det) {
            // 不做检测，将所有框设为整张图片
            for (auto& res : ocr_result) {
                res.box = {{0, 0}, {img.cols - 1, 0}, {img.cols - 1, img.rows - 1}, {0, img.rows - 1}};
            }
            // 如果没有结果，添加一个默认结果
            if (ocr_result.empty()) {
                OCRPredictResult res;
                res.box = {{0, 0}, {img.cols - 1, 0}, {img.cols - 1, img.rows - 1}, {0, img.rows - 1}};
                ocr_result.push_back(res);
            }
        }

        if (!rec) {
            // 不做识别，清空文本和分数
            for (auto& res : ocr_result) {
                res.text = "";
                res.score = -1.0f;
            }
        }

        if (!cls) {
            // 不做分类，将分类信息设为默认值
            for (auto& res : ocr_result) {
                res.cls_label = -1;
                res.cls_score = 0.0f;
            }
        }

        return ocr_result;
    }

    // ==================== PPOCR 公共接口 ====================

    PPOCR::PPOCR() : impl_(new PPOCRImpl()) {}
    PPOCR::~PPOCR() = default;

    std::vector<std::vector<OCRPredictResult>>
    PPOCR::ocr(std::vector<cv::Mat> img_list, bool det, bool rec, bool cls)
    {
        std::vector<std::vector<OCRPredictResult>> ocr_results;

        for (auto& img : img_list) {
            ocr_results.push_back(ocr(img, det, rec, cls));
        }

        return ocr_results;
    }

    std::vector<OCRPredictResult>
    PPOCR::ocr(cv::Mat img, bool det, bool rec, bool cls)
    {
        return impl_->ocr(img, det, rec, cls);
    }

    void PPOCR::reset_timer()
    {
        time_info_det = {0, 0, 0};
        time_info_rec = {0, 0, 0};
        time_info_cls = {0, 0, 0};
    }

    void PPOCR::benchmark_log(int img_num)
    {
        // v3.5.0 的 benchmark 逻辑不同，这里保持兼容
        // TODO: 如果需要，可以实现新的 benchmark 逻辑
    }

} // namespace PaddleOCR
