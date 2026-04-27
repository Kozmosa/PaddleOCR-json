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

// 版本信息
#define PROJECT_VER "v1.4.1 dev.1"
#define PROJECT_NAME "PaddleOCR-json " PROJECT_VER

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

#include <include/args.h>
#include <include/paddleocr.h>
#include <include/task.h>

using namespace PaddleOCR;

int main(int argc, char **argv)
{
    std::cout << PROJECT_NAME << std::endl; // 版本提示
    // 设置gflags并读取命令行
    google::SetUsageMessage("PaddleOCR-json [FLAG1=ARG1] [FLAG2=ARG2]");
    google::SetVersionString(PROJECT_VER);
    google::ParseCommandLineFlags(&argc, &argv, true);
    // 读取配置文件
    std::string configMsg = read_config();
    if (!configMsg.empty())
    {
        std::cerr << configMsg << std::endl;
    }
    // 检查参数合法性
    std::string checkMsg = check_flags();
    if (!checkMsg.empty())
    {
        std::cerr << "[ERROR] " << checkMsg << std::endl;
        return 1;
    }

    // 启动任务
    Task task = Task();
    if (FLAGS_type == "ocr")
    { // OCR图片模式
        return task.ocr();
    }
    // TODO: 图表识别模式
    else if (FLAGS_type == "structure")
    {
        std::cerr << "[ERROR] structure not support. " << std::endl;
        // structure(cv_all_img_names);
    }
}
