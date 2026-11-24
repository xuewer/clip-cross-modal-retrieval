# 数据准备

项目使用标准 CSV：

```csv
image_id,image_path,caption
example_image,raw/Flicker8k_Dataset/example_image.jpg,An example image caption.
```

`image_path` 以 CSV 文件所在目录为相对路径基准，也可以使用绝对路径。同一张图片可以有多行 caption，但 `image_id` 必须保持一致。

推荐流程：

1. 将 Flickr8k 图片放在 `data/raw/Flicker8k_Dataset/`。
2. 通过 `python download_flickr8k.py` 下载并将 `Flickr8k.token.txt` 转换为 `data/captions.csv`，列名为 `image_id,image_path,caption`。
3. 运行 `python prepare_data.py --input data/captions.csv`。

清洗脚本会检查空值、重复记录、图片是否存在以及图片是否损坏，并按 `image_id` 划分 `data/splits/train.csv` 和 `data/splits/test.csv`，避免同一张图片泄漏到两个集合。
