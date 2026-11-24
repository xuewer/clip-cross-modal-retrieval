# 基于 CLIP 的图文双向检索系统

这是一个基于 OpenAI CLIP ViT-B/32 的 Flickr8k 图文检索项目，支持：

- 文本检索图片
- 图片检索文本
- 图像和文本向量持久化
- NumPy 暴力 Top-K 检索
- Text-to-Image 和 Image-to-Text Recall@1/5/10
- Streamlit Web Demo
- CPU 运行和 CUDA 自动检测

项目使用预训练 CLIP 推理，未训练或微调模型。检索采用 NumPy 精确余弦搜索，输出 512 维归一化向量。

## 已保存的实验结果

清洗后共 8,091 张图片、40,445 条描述；随机种子 42，按图片 ID 以 80%/20% 划分，训练与测试图片无重叠。演示索引包含训练划分的 6,473 张图片、32,355 条描述。

| 检索方向 | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| 文本 → 图片 | 47.53% | 74.34% | 83.28% |
| 图片 → 文本 | 65.08% | 86.65% | 91.84% |

指标来自已有的 [outputs/metrics.json](outputs/metrics.json)，模型为 ViT-B/32、设备为 CPU。本次仓库整理核对了数据划分、索引形状和源 CSV 哈希，未重新执行完整模型评估。

文本查询在独立测试集的 **1,618 张图片**中检索；图片查询在测试集的 **8,090 条描述**中检索。每张测试图片关联 5 条描述，Top-K 命中任意一条关联描述即记为成功。该口径是图文检索中常用的 Recall@K（查询命中率）。这是自定义测试划分，不能直接作为标准 Flickr8k 1K 测试协议的对比结果。

## 1. 环境

推荐：

- Windows 10/11
- Python 3.10 或 3.11
- PyTorch
- CPU 或 CUDA

现有本地验证环境：

- Python 3.13.5
- PyTorch 2.9.0+cpu
- CUDA 不可用
- CPU 运行

当前 Python 3.13 环境已通过 7 个核心测试和 Streamlit 页面加载测试。Python 3.11 的全新安装与完整复现未在本次整理中验证；依赖文件没有锁定全部版本。

## 2. 项目结构

```text
data/
  captions.csv
  raw/
  splits/
artifacts/
outputs/
src/
  config.py
  dataset.py
  model.py
  index.py
  retrieval.py
  metrics.py
  utils.py
tests/
app.py
download_flickr8k.py
prepare_data.py
build_index.py
evaluate.py
requirements.txt
PROJECT_SUMMARY.md
```

## 3. 从头开始复现

以下命令都在项目根目录执行：

```powershell
cd clip-cross-modal-retrieval
```

### 3.1 创建 Python 环境

推荐 Python 3.11：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 不允许激活脚本，可以直接使用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.2 安装 CLIP

项目按以下顺序选择后端：本地 OpenAI checkpoint → 可选的原生 OpenAI CLIP → Transformers。

本地权重存在于 `models/ViT-B-32.pt` 时，通过已包含在依赖中的 `open_clip_torch` 加载；准备方法见 [models/README.md](models/README.md)。已有本地索引和评估使用此后端。

后端一：原生 OpenAI CLIP：

```powershell
pip install git+https://github.com/openai/CLIP.git
```

后端二：Transformers 兼容后端。`requirements.txt` 已包含：

```text
transformers
huggingface-hub
```

代码在没有原生 `clip` 包时，会自动尝试加载：

```text
openai/clip-vit-base-patch32
```

无论使用哪种后端，第一次运行都需要下载 CLIP 模型权重。模型下载失败时，程序会明确报错，不会悄悄使用随机向量。

### 3.3 下载 Flickr8k

执行：

```powershell
python download_flickr8k.py
```

脚本会：

1. 下载图片压缩包和文本标注压缩包。
2. 解压到 `data/raw/`。
3. 读取 `Flickr8k.token.txt`。
4. 去掉文件名中的 `#0` 到 `#4` caption 编号。
5. 生成 `data/captions.csv`。

需要的主要文件：

```text
data/raw/Flicker8k_Dataset/
data/raw/Flickr8k.token.txt
```

`__MACOSX/` 和其中的 `._*.jpg` 是 macOS 元数据，不是真实图片，可以忽略。

如果图片已经解压、但下载脚本在 Windows 上收尾时间过长，可以单独生成 CSV：

```powershell
python -c "from pathlib import Path; from download_flickr8k import make_captions_csv; print(make_captions_csv(Path('data/raw/Flickr8k.token.txt'), Path('data/raw/Flicker8k_Dataset'), Path('data/captions.csv')))"
```

### 3.4 清洗数据并划分训练/测试集

正式实验使用严格图片校验：

```powershell
python prepare_data.py --input data/captions.csv --test-size 0.2 --seed 42
```

输出：

```text
data/splits/captions_clean.csv
data/splits/train.csv
data/splits/test.csv
data/splits/cleaning_stats.json
```

划分单位是 `image_id`，同一张图片的所有 caption 只会进入训练集或测试集之一，避免数据泄漏。

如果 Windows 文件系统校验大量图片速度很慢，可以临时跳过图片打开验证：

```powershell
python prepare_data.py --input data/captions.csv --test-size 0.2 --seed 42 --skip-image-validation
```

正式结果建议使用默认严格模式。

### 3.5 生成向量索引

```powershell
python build_index.py --csv data/splits/train.csv --device cpu
```

有 CUDA 时可以使用：

```powershell
python build_index.py --csv data/splits/train.csv --device cuda
```

成功后应生成：

```text
artifacts/image_vectors.npy
artifacts/text_vectors.npy
artifacts/image_metadata.csv
artifacts/text_metadata.csv
artifacts/index_config.json
```

索引包含：

- 图像向量
- 文本向量
- `image_id`
- 图片路径
- caption
- 模型名称
- 设备
- 向量维度
- 源 CSV SHA-256

向量经过 L2 归一化，检索使用内积计算余弦相似度。

### 3.6 运行 Recall@K 评测

```powershell
python evaluate.py --csv data/splits/test.csv --output outputs/metrics.json --device cpu
```

评测包括：

- Text-to-Image Recall@1/5/10
- Image-to-Text Recall@1/5/10

Text-to-Image 将相同 `image_id` 视为正确图片。Image-to-Text 将同一图片真实关联的多个 caption 都视为正确候选。

成功后生成：

```text
outputs/metrics.json
```

### 3.7 启动 Streamlit Demo

先确认 `artifacts/` 已生成，再运行：

```powershell
python -m streamlit run app.py
```

请使用安装了项目依赖的同一个 Python 解释器启动。Windows 未激活虚拟环境时，使用 `.\.venv\Scripts\python.exe -m streamlit run app.py`。直接执行 `python app.py` 不会启动网页服务。

浏览器打开 Streamlit 显示的地址，通常是：

```text
http://localhost:8501
```

页面支持：

- 文本输入检索图片
- 上传图片检索文本
- Top-K 选择
- 相似度显示
- 设备和索引数量显示

### 3.8 运行测试

```powershell
pytest -q
```

当前测试覆盖：

- L2 向量归一化
- Top-K 排序
- K 超出库大小
- Recall@K 正样本匹配
- 按图片划分防泄漏
- 异常输入

## 4. 当前真实执行结果

已经完成：

- Flickr8k 数据下载和解压
- 标准 CSV 生成
- 数据清洗
- 严格图片完整性校验
- 按 `image_id` 划分
- 核心代码实现
- Streamlit 启动检查
- `python -m pytest -q`：7 passed（2026-11-21 本地复核）
- Python 编译检查

本地已有模型权重、训练索引和评估结果。本次核对确认：训练/测试图片 ID 无交集；索引记录的源 CSV SHA-256 与当前训练 CSV 一致；图片向量形状为 `(6473, 512)`，文本向量形状为 `(32355, 512)`。

仓库仅提交代码、文档、测试、清洗统计和指标 JSON。Flickr8k 数据、模型权重、生成的向量索引及 Python 虚拟环境保留在本地，克隆仓库后需要重新准备数据和构建索引。生成文件中的图片路径基于构建机器，迁移机器后应重建索引。

当前未进行模型微调、检索延迟与吞吐量测量、ANN/FAISS 集成或生产部署；英文查询最适合当前模型与标注，中文检索效果未验证。

数据和模型来源：[Flickr8k 下载来源](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)、[OpenAI CLIP](https://github.com/openai/CLIP)、[OpenCLIP](https://github.com/mlfoundations/open_clip)、[Transformers CLIP 权重](https://huggingface.co/openai/clip-vit-base-patch32)。使用时遵循各自来源的许可与条款。

## 5. 常见问题

### `OpenAI CLIP is not installed`

安装原生 CLIP：

```powershell
pip install git+https://github.com/openai/CLIP.git
```

或者确认 `transformers` 已安装，并保证能够访问 Hugging Face 模型仓库。

### `Failed to load OpenAI CLIP weights`

说明 Python 包存在，但模型权重没有下载成功。检查：

- 网络和代理
- Hugging Face 访问权限
- 磁盘空间
- 是否可以在浏览器访问模型仓库

### Streamlit 提示索引不完整

先执行：

```powershell
python build_index.py --csv data/splits/train.csv --device cpu
```

### 图片无法打开

不要打开：

```text
data/raw/__MACOSX/Flicker8k_Dataset/._*.jpg
```

应打开：

```text
data/raw/Flicker8k_Dataset/*.jpg
```
