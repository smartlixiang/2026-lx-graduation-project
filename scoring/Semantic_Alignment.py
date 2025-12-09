"""
实现语义对齐度（Semantic Alignment, SA）评分。

该实现基于简单的词频向量与集合重合度计算语义相似度，
输出范围为[0,1]，数值越大表示候选文本与参考文本的语义越接近。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class SAResult:
    """存放单条样本的语义对齐度评分细节。"""

    reference: str
    candidate: str
    score: float
    cosine_similarity: float
    jaccard_overlap: float
    ref_length: int
    cand_length: int


class SemanticAlignment:
    """语义对齐度（SA）计算器。

    设计目标：
    1. 轻量化依赖：无需外部模型，仅依赖基础 Python 库。
    2. 同时考虑词频权重（余弦相似）与集合覆盖度（Jaccard），降低纯字面匹配的偏差。
    """

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def _tokenize(self, text: str) -> List[str]:
        """简单分词：
        - 英文按照单词划分
        - 中文按单字划分
        - 数字作为整体 token
        """

        if not text:
            return []

        processed = text.strip()
        if self.lowercase:
            processed = processed.lower()

        # 匹配中文单字或英文/数字单词
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+(?:'[a-z0-9]+)?", processed, flags=re.IGNORECASE)
        return tokens

    def _vectorize(self, tokens: Iterable[str]) -> Counter:
        return Counter(tokens)

    def _cosine_similarity(self, v1: Counter, v2: Counter) -> float:
        intersection = set(v1.keys()) & set(v2.keys())
        numerator = sum(v1[t] * v2[t] for t in intersection)
        if numerator == 0:
            return 0.0

        sum1 = sum(freq ** 2 for freq in v1.values())
        sum2 = sum(freq ** 2 for freq in v2.values())
        denominator = math.sqrt(sum1 * sum2)
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _jaccard_overlap(self, t1: Sequence[str], t2: Sequence[str]) -> float:
        set1, set2 = set(t1), set(t2)
        if not set1 and not set2:
            return 1.0
        union = set1 | set2
        if not union:
            return 0.0
        return len(set1 & set2) / len(union)

    def score(self, reference: str, candidate: str, alpha: float = 0.6) -> float:
        """计算单条样本的语义对齐度。

        参数
        ----
        reference: 参考文本
        candidate: 待评估文本
        alpha: 权重，越大越偏向词频余弦相似，越小越偏向集合覆盖度
        """

        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)

        cosine = self._cosine_similarity(self._vectorize(ref_tokens), self._vectorize(cand_tokens))
        jaccard = self._jaccard_overlap(ref_tokens, cand_tokens)

        score = alpha * cosine + (1 - alpha) * jaccard
        return float(round(score, 4))

    def detailed_score(self, reference: str, candidate: str, alpha: float = 0.6) -> SAResult:
        """返回包含各项中间指标的评分详情。"""

        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)

        cosine = self._cosine_similarity(self._vectorize(ref_tokens), self._vectorize(cand_tokens))
        jaccard = self._jaccard_overlap(ref_tokens, cand_tokens)
        final_score = alpha * cosine + (1 - alpha) * jaccard

        return SAResult(
            reference=reference,
            candidate=candidate,
            score=round(final_score, 4),
            cosine_similarity=round(cosine, 4),
            jaccard_overlap=round(jaccard, 4),
            ref_length=len(ref_tokens),
            cand_length=len(cand_tokens),
        )

    def batch_score(self, pairs: Sequence[Tuple[str, str]], alpha: float = 0.6) -> List[SAResult]:
        """批量计算语义对齐度。"""

        return [self.detailed_score(ref, cand, alpha=alpha) for ref, cand in pairs]

    def aggregate(self, results: Sequence[SAResult]) -> Dict[str, float]:
        """聚合整体统计信息。"""

        if not results:
            return {"mean_score": 0.0, "mean_cosine": 0.0, "mean_jaccard": 0.0}

        mean_score = sum(r.score for r in results) / len(results)
        mean_cosine = sum(r.cosine_similarity for r in results) / len(results)
        mean_jaccard = sum(r.jaccard_overlap for r in results) / len(results)
        return {
            "mean_score": round(mean_score, 4),
            "mean_cosine": round(mean_cosine, 4),
            "mean_jaccard": round(mean_jaccard, 4),
        }
