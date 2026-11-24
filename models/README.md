# 本地 CLIP 权重

此目录仅提交说明，不提交模型文件。

默认在线流程会通过 Transformers 下载 `openai/clip-vit-base-patch32`。如果需要复用本地 OpenAI CLIP checkpoint，可将官方 ViT-B/32 权重放在：

```text
models/ViT-B-32.pt
```

权重来源及下载方式见 [OpenAI CLIP](https://github.com/openai/CLIP)。该文件由 `open_clip_torch` 后端加载；不要将其他模型或其他格式的权重重命名后放入此处。

本项目已有的 CPU 索引和评估使用此本地后端。需要复现实验时，应记录使用的模型来源、后端和依赖版本；本仓库没有锁定完整依赖环境。
