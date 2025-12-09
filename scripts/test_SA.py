"""
使用示例数据测试语义对齐度（SA），并以 SVG/HTML 形式可视化结果。
可视化输出保存在 ./test_SA/ 目录。
"""
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scoring import SAResult, SemanticAlignment


def build_samples() -> List[dict]:
    """构造若干示例样本，覆盖高度对齐、部分对齐、完全不对齐等情况。"""

    return [
        {
            "id": 1,
            "reference": "模型能够在图像中准确识别猫和狗。",
            "candidate": "该模型可以在图片里准确地识别出猫狗。",
            "note": "语义高度一致，表述略有差异",
        },
        {
            "id": 2,
            "reference": "系统需要在毫秒级延迟下完成搜索请求。",
            "candidate": "搜索请求的延迟应该控制在秒级，不要求毫秒。",
            "note": "部分矛盾，覆盖关键词但方向不同",
        },
        {
            "id": 3,
            "reference": "用户可以通过手机号或邮箱登录账户。",
            "candidate": "登录方式仅支持手机号，不支持邮箱。",
            "note": "半对齐，覆盖手机号但缺失邮箱，含否定",
        },
        {
            "id": 4,
            "reference": "应用需要支持离线缓存和夜间模式。",
            "candidate": "应用新增了多人协作和权限管理。",
            "note": "几乎不相关，语义偏离",
        },
        {
            "id": 5,
            "reference": "请将文本翻译成英语，并保持专有名词。",
            "candidate": "把这段话译成英文，专有名词要保留。",
            "note": "高相似的指令改写",
        },
        {
            "id": 6,
            "reference": "The API should return JSON formatted results for all requests.",
            "candidate": "All API responses need to be in JSON format.",
            "note": "英文高对齐",
        },
        {
            "id": 7,
            "reference": "Ensure battery life lasts at least ten hours of active use.",
            "candidate": "The device drains the battery within five hours when gaming heavily.",
            "note": "语义反向，长度差异明显",
        },
    ]


def save_results(results, output_dir: Path):
    table_data = [
        {
            "id": r["id"],
            "score": res.score,
            "cosine": res.cosine_similarity,
            "jaccard": res.jaccard_overlap,
            "ref_len": res.ref_length,
            "cand_len": res.cand_length,
            "note": r["note"],
        }
        for r, res in results
    ]

    # 保存 JSON
    (output_dir / "sa_scores.json").write_text(json.dumps(table_data, ensure_ascii=False, indent=2))

    # 保存 CSV
    lines = ["id,score,cosine,jaccard,ref_len,cand_len,note"]
    for row in table_data:
        note = row["note"].replace(",", "；")
        lines.append(
            f"{row['id']},{row['score']},{row['cosine']},{row['jaccard']},{row['ref_len']},{row['cand_len']},{note}"
        )
    (output_dir / "sa_scores.csv").write_text("\n".join(lines))


def _svg_header(width: int, height: int) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>\n"


def plot_bar(scores: Sequence[Tuple[dict, SAResult]], output_dir: Path):
    width, height = 640, 320
    margin = 40
    bar_width = (width - 2 * margin) / max(1, len(scores)) * 0.6
    max_height = height - 2 * margin
    svg_parts = [_svg_header(width, height)]
    svg_parts.append(f"<text x='{width/2}' y='24' text-anchor='middle' font-size='16'>语义对齐度柱状图</text>")

    for idx, (sample, res) in enumerate(scores):
        x = margin + idx * (bar_width * 1.6)
        h = max_height * res.score
        y = height - margin - h
        svg_parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{h:.1f}' fill='#4682b4' />")
        svg_parts.append(
            f"<text x='{x + bar_width/2:.1f}' y='{height - margin + 16}' text-anchor='middle' font-size='12'>{sample['id']}</text>"
        )
        svg_parts.append(
            f"<text x='{x + bar_width/2:.1f}' y='{y - 4:.1f}' text-anchor='middle' font-size='11'>{res.score:.2f}</text>"
        )

    svg_parts.append("</svg>")
    (output_dir / "bar_scores.svg").write_text("\n".join(svg_parts), encoding="utf-8")


def plot_histogram(scores: Sequence[Tuple[dict, SAResult]], output_dir: Path):
    width, height = 520, 320
    margin = 40
    bin_count = 5
    values = [res.score for _, res in scores]
    bins = [0 for _ in range(bin_count)]
    for v in values:
        idx = min(int(v * bin_count), bin_count - 1)
        bins[idx] += 1

    max_bin = max(bins) if bins else 1
    bar_width = (width - 2 * margin) / bin_count * 0.6
    svg_parts = [_svg_header(width, height)]
    svg_parts.append(f"<text x='{width/2}' y='24' text-anchor='middle' font-size='16'>SA 分布直方图</text>")

    for i, count in enumerate(bins):
        x = margin + i * (bar_width * 1.6)
        h = (height - 2 * margin) * (count / max_bin if max_bin else 0)
        y = height - margin - h
        svg_parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{h:.1f}' fill='#ff8c00' />")
        svg_parts.append(
            f"<text x='{x + bar_width/2:.1f}' y='{height - margin + 16}' text-anchor='middle' font-size='12'>{i/bin_count:.1f}-{(i+1)/bin_count:.1f}</text>"
        )
        svg_parts.append(
            f"<text x='{x + bar_width/2:.1f}' y='{y - 4:.1f}' text-anchor='middle' font-size='11'>{count}</text>"
        )

    svg_parts.append("</svg>")
    (output_dir / "hist_scores.svg").write_text("\n".join(svg_parts), encoding="utf-8")


def plot_length_vs_score(scores: Sequence[Tuple[dict, SAResult]], output_dir: Path):
    width, height = 520, 320
    margin = 40
    max_len = max([max(res.ref_length, res.cand_length) for _, res in scores], default=1)
    svg_parts = [_svg_header(width, height)]
    svg_parts.append(f"<text x='{width/2}' y='24' text-anchor='middle' font-size='16'>长度与分数关系</text>")

    def to_xy(length: int, score: float) -> Tuple[float, float]:
        x = margin + (width - 2 * margin) * (length / max_len)
        y = height - margin - (height - 2 * margin) * score
        return x, y

    for sample, res in scores:
        xr, yr = to_xy(res.ref_length, res.score)
        xc, yc = to_xy(res.cand_length, res.score)
        svg_parts.append(f"<circle cx='{xr:.1f}' cy='{yr:.1f}' r='5' fill='#2e8b57' />")
        svg_parts.append(f"<circle cx='{xc:.1f}' cy='{yc:.1f}' r='5' fill='#8a2be2' />")
        svg_parts.append(
            f"<text x='{xr + 6:.1f}' y='{yr - 6:.1f}' font-size='10'>ID{sample['id']}</text>"
        )

    # 坐标轴
    svg_parts.append(
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' stroke-width='1'/>"
    )
    svg_parts.append(
        f"<line x1='{margin}' y1='{height - margin}' x2='{margin}' y2='{margin}' stroke='black' stroke-width='1'/>"
    )
    svg_parts.append(f"<text x='{width/2}' y='{height - 6}' text-anchor='middle'>分词长度</text>")
    svg_parts.append(
        f"<text x='{margin - 28}' y='{height/2}' text-anchor='middle' transform='rotate(-90 {margin - 28},{height/2})'>SA 分数</text>"
    )

    svg_parts.append("</svg>")
    (output_dir / "length_vs_score.svg").write_text("\n".join(svg_parts), encoding="utf-8")


def save_html_dashboard(results: Sequence[Tuple[dict, SAResult]], output_dir: Path):
    """输出简单 HTML，方便一次性查看所有可视化。"""

    html_content = f"""
    <html>
    <head><meta charset='utf-8'><title>SA 可视化汇总</title></head>
    <body>
        <h2>语义对齐度示例结果</h2>
        <p>共 {len(results)} 条样本。点击或另存为查看 SVG 图。</p>
        <img src='bar_scores.svg' alt='bar scores' style='max-width: 600px;'><br>
        <img src='hist_scores.svg' alt='hist scores' style='max-width: 480px;'><br>
        <img src='length_vs_score.svg' alt='length vs score' style='max-width: 480px;'>
    </body>
    </html>
    """
    (output_dir / "index.html").write_text(html_content, encoding="utf-8")


def main():
    output_dir = Path("test_SA")
    output_dir.mkdir(exist_ok=True)

    samples = build_samples()
    sa = SemanticAlignment()
    results = []
    for sample in samples:
        res = sa.detailed_score(sample["reference"], sample["candidate"])
        results.append((sample, res))

    save_results(results, output_dir)
    plot_bar(results, output_dir)
    plot_histogram(results, output_dir)
    plot_length_vs_score(results, output_dir)
    save_html_dashboard(results, output_dir)

    aggregated = sa.aggregate([res for _, res in results])
    summary_path = output_dir / "summary.txt"
    summary_lines = [
        "语义对齐度测试汇总",
        f"样本数量: {len(results)}",
        f"平均分: {aggregated['mean_score']}",
        f"平均余弦: {aggregated['mean_cosine']}",
        f"平均Jaccard: {aggregated['mean_jaccard']}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
