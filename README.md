# Dual-Branch Neural-Network-Based Loop Filter (NNLF)

<img width="960" height="413" alt="VISUAL" src="https://github.com/user-attachments/assets/0fe3d4b5-e41d-4c19-8b2a-4ba60983ee69" />

(a) Uncompressed frames. (b) VTM-11.0 NNVC-3.0 anchor. (c) Proposed NNLF. Top: BlowingBubbles. Bottom: BQSquare. We obtain the results at QP 42 under the AI configuration.

## 1. Project Overiview
In this project,  we propose a neural network-based in-loop filter (NNLF) for VVC intra coding based on dual-branch collaborative architecture.  
- The PyTorch model .pth is obtained during the training stage.  
- It is compiled into a TorchScript .pt file using torch.jit.trace for C++ inference.  
- The inference depends on LibTorch (version ≥ 1.9, compatible with both CPU and CUDA).  
- The code is integrated into VVCSoftware_VTM/source/Lib/CommonLib to be compiled together with VTM.

---

## 2. Document List
```
Dual-Branch-NNLF/
├── README.md
├── model_transfer.py          # 把 .pth 转成 .pt
├── checkpoint/
│   └── AI/
│       └── REAM/
│           └── G_epoch_65.pth # 第三阶段训练得到的生成器权重
├── pt/
│   ├── AI/
│   │   ├── filter_Y.pt        # Y 分量模型（已注释掉，备用）
│   │   └── filter_UV.pt       # UV 分量模型（当前启用）
├── CnnLoopFilter.h            # 公用接口头文件
└── CnnLoopFilter.cpp          # LibTorch 推理实现
```

---

## 3. Quick Start Guide

### 3.1 Environment Setup
- Python ≥ 3.8  
- PyTorch ≥ 1.9  
- LibTorch（与 PyTorch 版本保持一致）  
- VTM-10.0 源码树已下载并可正常编译

### 3.2 Export a TorchScript (.pt) model
```bash
python model_transfer.py
```
脚本默认会把 `G_epoch_65.pth` 转成 `filter_UV.pt`，并保存在 `./pt/AI/`。  
如需生成 Y 分量模型，请取消脚本中相应注释。

### 3.3 Integration into VTM
1. 把 `CnnLoopFilter.h` / `CnnLoopFilter.cpp` 复制到  
   `~/VVCSoftware_VTM/source/Lib/CommonLib/`
2. 在 `CommonLib/CMakeLists.txt` 末尾追加：
```cmake
# --- LibTorch ---
set(CMAKE_PREFIX_PATH "/path/to/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
target_link_libraries(CommonLib Torch::Torch)
```
3. 重新编译 VTM：
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
4. 运行时请把 `filter_UV.pt` 放在可执行文件同级目录，或通过代码指定绝对路径。

---

## 4. Interface Description（C++）
```cpp
// CnnLoopFilter.h
class CnnLoopFilter
{
public:
    CnnLoopFilter(); 
    ~CnnLoopFilter();
    void init(const std::string& modelPath, bool isY = false);
    void process(Pel* dst, int dstStride, const Pel* src, int srcStride,
                 int width, int height, int ch);
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
```
- `init()`：加载 TorchScript 模型  
- `process()`：对单通道图像做滤波，`ch=0/1/2` 对应 Y/U/V  
- 已做 `torch::NoGradGuard` 加速，支持 CPU/CUDA 自动切换

---

## 5. Model Description
| 分量 | 输入维度 | 输出维度 | 模型大小 |
|------|-----------|-----------|-----------|
| Y    | (1, 4, H, W)  | (1, 1, H, W)  | ~7 MB（未启用） |
| UV   | (1, 10, H/2, W/2) | (1, 2, H/2, W/2) | ~7 MB（已启用） |

网络结构：双分支残差网络 + 通道注意力（REAM），训练细节请参见 `train/`（后续开源）。

---

## 6. Performance Metrics (Example)
- 测试序列：BasketballDrive_1920x1080_50  
- 平台：Intel i9-10900K @ 3.7 GHz，单线程  
- 滤波耗时：≈ 4.3 ms / frame（UV 联合滤波）  
- BD-rate：Y -2.1%，U -4.7%，V -4.5%（AI 配置，RA main10）

---

## 7. Frequently Asked Questions (FAQ)
**Q1. 加载模型失败？**  
→ 确认 LibTorch 版本与生成 `.pt` 时 PyTorch 版本一致；路径正确。

**Q2. 编译报错 “undefined reference to `torch::jit::load`”？**  
→ `target_link_libraries` 未链接 `Torch::Torch`，或 `CMAKE_PREFIX_PATH` 未指向 LibTorch。

**Q3. 如何开启 CUDA 推理？**  
→ 在 `model_transfer.py` 里把 `device='cuda:0'`，C++ 端会自动识别 CUDA 模型；需配合 LibTorch-cxx11-abi-cuda 版本。

---

## 8. Changelog
| 日期 | 内容 |
|------|------|
| 2025-10-21 | 第一版，支持 UV 双通道滤波 |
| …… | …… |

---

## 9. Contributor / Organization
Xidian Media Lab  

---

## 10. Reference
同 VTM 采用 **BSD-3-Clause**。使用请引用：  
> Liu et al., "Neural Network-Based In-Loop Filter Based on Dual-Branch Collaborative Architecture," 2025 (Under Review)
