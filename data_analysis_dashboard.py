#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合数据分析脚本
目标：基于 all_comments_cleaned.csv，进行多角度分析并输出高质量可视化图形，便于撰写分析报告。
依赖与环境要求见 requirements.txt（同仓库根目录）；
如绘图字体/样式在某些系统不可用，脚本会回退到默认字体。
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
from wordcloud import WordCloud
from collections import Counter

# 兼容中文显示
warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 300
sns.set_theme(style="ticks", context="talk")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(filepath: str = "all_comments_cleaned.csv") -> pd.DataFrame:
    print(">>> 加载数据...")
    df = pd.read_csv(filepath)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.dropna(subset=["content"])
    print(f"数据加载完成，共 {len(df)} 条评论。")
    return df

def get_sentiment_score(text: str, platform: str) -> float:
    """
    简化版情感评分：若平台为英文/其他，用 VADER；否则用 SnowNLP 的近似映射。
    这里避免强依赖 SnowNLP 安装问题，保留原项目逻辑的核心思想。
    """
    try:
        # 若中文文本，尝试用 SnowNLP 近似情感
        if platform.lower() not in {"reddit", "english"}:
            from snownlp import SnowNLP
            s = SnowNLP(str(text))
            return (s.sentiments - 0.5) * 2
        else:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            return analyzer.polarity_scores(str(text))["compound"]
    except Exception:
        return 0.0

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print(">>> 正在进行情感分析...")
    df["sentiment"] = df.apply(lambda row: get_sentiment_score(row["content"], row.get("platform", "unknown")), axis=1)
    conditions = [
        (df["sentiment"] >= 0.2),
        (df["sentiment"] <= -0.2)
    ]
    choices = ["Positive", "Negative"]
    df["sentiment_label"] = np.select(conditions, choices, default="Neutral")
    return df

def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    print(">>> 导出情感分布数据...")
    counts = df["sentiment_label"].value_counts()
    # 导出数据
    counts.to_csv(PLOTS_DIR / "data_sentiment_distribution_overall.csv", header=["count"])
    
    # 条形图数据
    by_platform = pd.crosstab(df["platform"], df["sentiment_label"], normalize="index")
    # 导出数据
    by_platform.to_csv(PLOTS_DIR / "data_sentiment_distribution_by_platform.csv")
    print(f"已导出: data_sentiment_distribution_overall.csv, data_sentiment_distribution_by_platform.csv")

def plot_sentiment_trend(df: pd.DataFrame) -> None:
    print(">>> 导出情感趋势数据...")
    df["month"] = df["created_at"].dt.to_period("M")
    monthly = df.groupby(["month", "platform"])["sentiment"].mean().unstack()
    # 导出数据
    out_path = PLOTS_DIR / "data_sentiment_trend.csv"
    monthly.to_csv(out_path)
    print(f"已导出: {out_path}")

def get_keywords(df: pd.DataFrame, top_n: int = 100) -> List[tuple]:
    print(">>> 提取关键词...")
    text = " ".join(df["content"].astype(str).tolist())
    stop_words = set([
        "的","了","在","是","我","有","和","就","不","人","都","一个","上","也","很","到",
        "说","要","去","你","会","着","没有","看","好","自己","这","那","有什么","但是",
        "那个","觉得","就是","还是","我们","其实","的","了","你们","呢","吗","吧",
        "回复","视频","楼主","知道","问题","可能","怎么","出来","现在","时候","感觉","因为","所以","打卡"
    ])
    words = jieba.cut(text)
    filtered = [w for w in words if len(w) > 1 and w not in stop_words and not w.isnumeric()]
    counter = Counter(filtered)
    return counter.most_common(top_n)

def plot_wordcloud(keywords: List[tuple]) -> None:
    print(">>> 导出词云数据...")
    # 导出数据
    out_path = PLOTS_DIR / "data_wordcloud.csv"
    pd.DataFrame(keywords, columns=["word", "count"]).to_csv(out_path, index=False)
    print(f"已导出: {out_path}")

def analyze_topics_lda(df: pd.DataFrame, n_topics: int = 4) -> None:
    print(">>> 进行 LDA 主题聚类...")
    def preprocess(text: str) -> str:
        words = jieba.cut(str(text))
        return " ".join([w for w in words if len(w) > 1])
    df["processed"] = df["content"].apply(preprocess)
    from sklearn.feature_extraction.text import CountVectorizer
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=None)
    tf = tf_vectorizer.fit_transform(df["processed"])
    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method="online", random_state=42)
    lda.fit(tf)
    feature_names = tf_vectorizer.get_feature_names_out()
    plot_top_words(lda, feature_names, n_topics, 10, "Key Topics Discussed (LDA Model)")

def plot_top_words(model, feature_names: List[str], n_topics: int, n_top_words: int, title: str) -> None:
    print(">>> 导出 LDA 主题数据...")
    # 导出LDA数据
    topic_data = []
    for topic_idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[:-n_top_words - 1:-1]
        for i in top_idx:
            topic_data.append({
                "topic": f"Topic {topic_idx+1}",
                "word": feature_names[i],
                "weight": topic[i]
            })
    out_path = PLOTS_DIR / "data_lda_topics.csv"
    pd.DataFrame(topic_data).to_csv(out_path, index=False)
    print(f"已导出: {out_path}")

def plot_job_impact(df: pd.DataFrame) -> None:
    print(">>> 导出岗位冲击分析数据...")
    keywords = {
        "Frontend (前端)": ["前端","frontend","网页"],
        "Backend (后端)": ["后端","backend","java","python","golang"],
        "Algorithm/AI": ["算法","模型","nlp","大模型","tuning"],
        "Product Mgr": ["产品","pm","需求"],
        "UI/Design": ["ui","设计","美工","画图"],
        "QA/Test": ["测试","qa"],
        "Outsourcing": ["外包","od"],
        "Junior/Intern": ["初级","实习","校招","应届"]
    }
    all_text = " ".join(df["content"].astype(str).tolist()).lower()
    impact = {}
    for key, kws in keywords.items():
        impact[key] = sum(all_text.count(kw) for kw in kws)
    sorted_impact = dict(sorted(impact.items(), key=lambda kv: kv[1], reverse=True))
    # 导出数据
    out_path = PLOTS_DIR / "data_job_impact.csv"
    pd.DataFrame(list(sorted_impact.items()), columns=["role", "count"]).to_csv(out_path, index=False)
    print(f"已导出: {out_path}")

def plot_time_series_comments(df: pd.DataFrame) -> None:
    print(">>> 导出评论时间序列数据...")
    ts = df.set_index("created_at").sort_index()
    counts = ts.groupby([pd.Grouper(freq="M"), "platform"]).size().unstack(fill_value=0)
    # 导出数据
    out_path = PLOTS_DIR / "data_time_series_comments.csv"
    counts.to_csv(out_path)
    print(f"已导出: {out_path}")

def plot_correlations(df: pd.DataFrame) -> None:
    print(">>> 导出相关性分析数据...")
    df["content_length"] = df["content"].astype(str).str.len()
    # 导出数据
    out_path = PLOTS_DIR / "data_sentiment_vs_length.csv"
    df[["content_length", "sentiment"]].to_csv(out_path, index=False)
    print(f"已导出: {out_path}")

def main():
    ensure_dirs()
    df = load_data()
    df = analyze_sentiment(df)
    plot_sentiment_distribution(df)
    plot_sentiment_trend(df)
    keywords = get_keywords(df, top_n=100)
    plot_wordcloud(keywords)
    analyze_topics_lda(df, n_topics=4)
    plot_job_impact(df)
    plot_time_series_comments(df)
    plot_correlations(df)
    print("\n=== 所有分析数据已导出，保存在 outputs/plots/ 目录 ===")

if __name__ == "__main__":
    main()


