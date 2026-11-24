from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from src.index import load_index
from src.model import ClipEncoder, ClipModelError
from src.retrieval import Retriever


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@st.cache_resource(show_spinner=False)
def load_resources(artifacts_dir: str) -> tuple[object, object, object]:
    index = load_index(Path(artifacts_dir))
    encoder = ClipEncoder(model_name=index.config["model_name"], device=None)
    return index, encoder, Retriever(index, encoder)


def render_image_result(result: dict, position: int) -> None:
    st.markdown(f"**{position}. {result['image_id']}**  ·  similarity `{result['score']:.4f}`")
    path = Path(result["image_path"])
    if path.is_file():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Image file is missing: {path}")


def main() -> None:
    st.set_page_config(page_title="CLIP 图文双向检索", page_icon="CLIP", layout="wide")
    st.title("CLIP 图文双向检索")
    st.caption("OpenAI CLIP ViT-B/32 · NumPy brute-force cosine search")

    try:
        index, encoder, retriever = load_resources(str(ARTIFACTS_DIR))
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("请先运行：`python build_index.py --csv data/splits/train.csv`")
        return
    except ClipModelError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"加载资源失败：{exc}")
        return

    max_k = max(1, min(10, len(index.image_metadata), len(index.text_metadata)))
    st.sidebar.write(f"设备：`{encoder.device_type}`")
    st.sidebar.write(f"图片向量：`{len(index.image_metadata)}`")
    st.sidebar.write(f"文本向量：`{len(index.text_metadata)}`")
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=max_k, value=min(5, max_k))

    text_tab, image_tab = st.tabs(["文本检索图片", "图片检索文本"])
    with text_tab:
        query = st.text_input("输入图片描述", placeholder="例如：a dog running on the beach")
        if st.button("检索图片", type="primary", key="text_search"):
            if not query.strip():
                st.warning("请输入非空文本。")
            else:
                with st.spinner("正在编码并检索..."):
                    try:
                        results = retriever.search_text(query, top_k)
                    except Exception as exc:
                        st.error(f"检索失败：{exc}")
                    else:
                        for position, result in enumerate(results, 1):
                            render_image_result(result, position)

    with image_tab:
        uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            st.image(image, caption="查询图片", width=320)
            if st.button("检索文本", type="primary", key="image_search"):
                with st.spinner("正在编码并检索..."):
                    try:
                        results = retriever.search_image(image, top_k)
                    except Exception as exc:
                        st.error(f"检索失败：{exc}")
                    else:
                        for position, result in enumerate(results, 1):
                            st.markdown(
                                f"**{position}. {result['caption']}**  ·  "
                                f"`{result['score']:.4f}` · `{result['image_id']}`"
                            )


if __name__ == "__main__":
    main()
