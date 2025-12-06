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
from deep_translator import GoogleTranslator
import time
from tqdm import tqdm

# 兼容中文显示
warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 300
sns.set_theme(style="ticks", context="talk")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

# 定义全局停用词表（包含中文高频无意义词和英文停用词）
STOP_WORDS = set([
    # 中文停用词
    "的","了","在","是","我","有","和","就","不","人","都","一个","上","也","很","到",
    "说","要","去","你","会","着","没有","看","好","自己","这","那","有什么","但是",
    "那个","觉得","就是","还是","我们","其实","你们","呢","吗","吧",
    "回复","视频","楼主","知道","问题","可能","怎么","出来","现在","时候","感觉","因为","所以","打卡",
    "一下","很多","真的","大家","比较","或者","只是","目前","之前","之后","这样","内容",
    # 英文停用词
    "the", "to", "and", "it", "is", "in", "that", "you", "for", "of", "on", "are", "with", 
    "as", "be", "this", "or", "at", "by", "from", "an", "will", "can", "if", "but", "not",
    "so", "my", "your", "we", "they", "he", "she", "me", "do", "have", "has", "just", "about",
    "what", "when", "where", "why", "how", "all", "more", "out", "up", "one", "like", "some",
    "doge", "nan"  # 特定噪音词
])

# 预定义主题关键词映射（用于自动命名 Topic）
TOPIC_CATEGORIES = {
    "职业前景": ["工作", "工资", "薪资", "失业", "裁员", "找工作", "跳槽", "面试", "前景", "出路", "饿死", "饭碗", "岗位", "招聘"],
    "行业影响": ["取代", "冲击", "颠覆", "未来", "趋势", "消亡", "革命", "降本", "增效", "环境", "时代", "泡沫"],
    "AI技术应用": ["代码", "生成", "gpt", "copilot", "模型", "算法", "开发", "效率", "bug", "调试", "工具", "ai", "编程"],
    "态度/情绪": ["焦虑", "担心", "害怕", "无所谓", "躺平", "牛逼", "厉害", "恶心", "喜欢", "讨厌", "感觉", "希望"]
}

# 复用主题定义
SPECIFIC_THEMES = TOPIC_CATEGORIES

def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(filepath: str = "all_comments_cleaned.csv") -> pd.DataFrame:
    """
    加载数据并进行翻译（如果需要）。
    """
    print(">>> 加载数据...")
    df = pd.read_csv(filepath)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.dropna(subset=["content"])
    print(f"原始数据加载完成，共 {len(df)} 条评论。")
    
    # 翻译逻辑
    translated_path = OUTPUT_DIR / "all_comments_translated.csv"
    if translated_path.exists():
        print(">>> 检测到已翻译的数据文件，直接加载...")
        df_translated = pd.read_csv(translated_path)
        # 简单的合并检查，如果长度不对可能需要重新翻译，这里简化处理
        if len(df_translated) == len(df):
            df["content_cn"] = df_translated["content_cn"]
        else:
            print(">>> 已翻译文件行数不匹配，重新进行翻译...")
            df = translate_english_content(df, translated_path)
    else:
        df = translate_english_content(df, translated_path)
        
    return df

def translate_english_content(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    """
    将英文内容翻译为中文。
    主要针对 platform='reddit' 或者其他包含大量英文的内容。
    """
    print(">>> 开始翻译英文内容 (这也可能需要几分钟)...")
    translator = GoogleTranslator(source='auto', target='zh-CN')
    
    # 初始化 content_cn 列，默认等于原 content
    df["content_cn"] = df["content"].astype(str)
    
    # 筛选需要翻译的行：这里主要假设 reddit 都是英文，或者你可以加其他逻辑
    # 也可以通过检测字符来判断，但简单起见，先只处理 reddit
    mask = (df["platform"].str.lower() == "reddit")
    
    to_translate_indices = df[mask].index
    print(f"需要翻译的条目数: {len(to_translate_indices)}")
    
    # 使用 tqdm 显示进度
    translated_texts = []
    batch_size = 10 # 这里的 batch 指的是我们循环处理，deep_translator 本身是单条或小批量
    
    # 逐条翻译（为了稳定性）
    for idx in tqdm(to_translate_indices, desc="Translating"):
        original_text = df.loc[idx, "content"]
        try:
            # 简单清洗一下，防止过长或空
            if len(str(original_text)) < 2:
                translated_texts.append(original_text)
                continue
                
            # 限制长度防止报错
            text_chunk = str(original_text)[:4500] 
            res = translator.translate(text_chunk)
            translated_texts.append(res if res else original_text)
            time.sleep(0.1) # 避免请求过快
        except Exception as e:
            # print(f"Translation failed for {idx}: {e}")
            translated_texts.append(original_text)
            
    df.loc[to_translate_indices, "content_cn"] = translated_texts
    
    print(">>> 翻译完成，保存中间结果...")
    # 只保存 content_cn 避免文件过大，或者保存完整副本
    df[["platform", "thread_id", "comment_id", "content", "content_cn"]].to_csv(save_path, index=False)
    
    return df

def get_sentiment_score(text: str, platform: str) -> float:
    """
    简化版情感评分：优先检测文本语言，含中文则用 SnowNLP，否则用 VADER。
    """
    try:
        text_str = str(text)
        # 检测是否包含中文字符
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text_str)
        
        if has_chinese:
            from snownlp import SnowNLP
            s = SnowNLP(text_str)
            return (s.sentiments - 0.5) * 2
        else:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            return analyzer.polarity_scores(text_str)["compound"]
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

def plot_sentiment_distribution_chart(df: pd.DataFrame) -> None:
    """
    生成情感分布的可视化图（整体柱状图与按平台堆积柱状图）。
    保存为 data_sentiment_distribution_overall.png 与 data_sentiment_distribution_by_platform.png。
    """
    print(">>> 绘制情感分布图（整体与按平台）...")
    # 整体情感分布柱状图
    counts = df["sentiment_label"].value_counts()
    ordered = ["Negative", "Neutral", "Positive"]
    counts = counts.reindex(ordered, fill_value=0)
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color="steelblue")
    plt.title("情感分布（整体）")
    plt.xlabel("情感标签")
    plt.ylabel("数量")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_sentiment_distribution_overall.png")
    plt.close()
    # 按平台的堆积百分比分布
    by_platform = pd.crosstab(df["platform"], df["sentiment_label"], normalize="index")
    by_platform = by_platform.reindex(columns=ordered, fill_value=0)
    by_platform.plot(kind="bar", stacked=True, figsize=(8, 6))
    plt.title("情感分布（按平台，堆积比）")
    plt.xlabel("Platform")
    plt.ylabel("Proportion")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_sentiment_distribution_by_platform.png")
    plt.close()

def plot_sentiment_trend(df: pd.DataFrame) -> None:
    print(">>> 导出情感趋势数据...")
    df["month"] = df["created_at"].dt.to_period("M")
    monthly = df.groupby(["month", "platform"])["sentiment"].mean().unstack()
    # 导出数据
    out_path = PLOTS_DIR / "data_sentiment_trend.csv"
    monthly.to_csv(out_path)
    print(f"已导出: {out_path}")

def plot_sentiment_trend_chart(df: pd.DataFrame) -> None:
    """
    绘制月度情感趋势折线图（按平台）。
    """
    print(">>> 绘制情感趋势图（月度）...")
    df["month"] = df["created_at"].dt.to_period("M")
    monthly = df.groupby(["month", "platform"])["sentiment"].mean().unstack()
    plt.figure(figsize=(10, 6))
    for platform in monthly.columns:
        plt.plot(monthly.index.to_timestamp(), monthly[platform], marker="o", label=str(platform))
    plt.title("情感趋势（按平台，月度平均情感值）")
    plt.xlabel("Month")
    plt.ylabel("Mean Sentiment")
    plt.legend(title="Platform", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_sentiment_trend.png")
    plt.close()

def get_keywords(df: pd.DataFrame, top_n: int = 100) -> List[tuple]:
    print(">>> 提取关键词...")
    # 1. 优先使用翻译后的中文内容
    col = "content_cn" if "content_cn" in df.columns else "content"
    text = " ".join(df[col].astype(str).tolist())
    
    # 2. 定义白名单（即使是纯英文也保留的高价值词）
    KEEP_ENGLISH = {"ai", "gpt", "chatgpt", "python", "java", "copilot", "bug", "offer", "hr"}
    
    words = jieba.cut(text)
    
    filtered = []
    for w in words:
        w = w.strip().lower() # 统一转小写比较
        
        # 基础过滤：长度小于2，或者是停用词，或者是数字，直接丢弃
        if len(w) < 2 or w in STOP_WORDS or w.isnumeric():
            continue
            
        # 高级过滤：
        # 条件1: 包含中文字符
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in w)
        # 条件2: 在白名单中
        is_keep_english = w in KEEP_ENGLISH
        
        # 只有满足 条件1 或 条件2 才保留
        if has_chinese or is_keep_english:
            filtered.append(w)
            
    counter = Counter(filtered)
    return counter.most_common(top_n)

def plot_wordcloud(keywords: List[tuple]) -> None:
    print(">>> 导出词云数据...")
    # 导出数据
    out_path = PLOTS_DIR / "data_wordcloud.csv"
    pd.DataFrame(keywords, columns=["word", "count"]).to_csv(out_path, index=False)
    print(f"已导出: {out_path}")

def plot_wordcloud_image(keywords: List[tuple]) -> None:
    """
    根据关键词绘制词云图片。
    """
    print(">>> 绘制词云图...")
    freq = {word: count for word, count in keywords}
    wc = WordCloud(width=1200, height=600, background_color="white").generate_from_frequencies(freq)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_wordcloud.png")
    plt.close()

def plot_lda_topics_chart_from_csv(top_n: int = 10) -> None:
    """
    从 data_lda_topics.csv 读取并绘制 LDA 主题词分布图（分面条形图）。
    """
    csv_path = PLOTS_DIR / "data_lda_topics.csv"
    if not csv_path.exists():
        print(">>> data_lda_topics.csv 不存在，跳过 LDA 图绘制。")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print(">>> data_lda_topics.csv 为空，跳过绘制。")
        return

    # 获取所有主题
    try:
        topics = sorted(df["topic"].unique(), key=lambda x: int(x.split()[-1]))
    except:
        topics = df["topic"].unique()
        
    n_topics = len(topics)
    
    fig, axes = plt.subplots(1, n_topics, figsize=(5 * n_topics, 8), sharey=False)
    if n_topics == 1:
        axes = [axes]
    
    palette = sns.color_palette("mako", n_colors=top_n)

    def get_topic_name(words):
        # 统计匹配到的类别
        scores = {cat: 0 for cat in TOPIC_CATEGORIES}
        for word in words:
            for cat, seeds in TOPIC_CATEGORIES.items():
                if word in seeds:
                    scores[cat] += 2 # 精确匹配权重高
                else:
                    for seed in seeds:
                        if seed in word:
                            scores[cat] += 1 # 模糊匹配
        # 取最高分
        best_cat = max(scores, key=scores.get)
        if scores[best_cat] > 0:
            return best_cat
        return "综合话题"

    for i, topic in enumerate(topics):
        ax = axes[i]
        topic_data = df[df["topic"] == topic].sort_values(by="weight", ascending=True).tail(top_n)
        
        # 获取该主题的前 20 个高频词用于判断类别（不仅仅是 top_n）
        all_topic_words = df[df["topic"] == topic].sort_values(by="weight", ascending=False)["word"].head(20).astype(str).tolist()
        
        # 自动推断类别
        category_name = get_topic_name(all_topic_words)
        
        # 标题显示：类别 + (核心词1, 核心词2)
        top_keywords = topic_data["word"].tail(2).iloc[::-1].astype(str).tolist()
        topic_label = f"{topic}: {category_name}\n({', '.join(top_keywords)})"
        
        bars = ax.barh(topic_data["word"].astype(str), topic_data["weight"], color=palette[::-1])
        
        ax.set_title(topic_label, fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel("Weight")
        ax.tick_params(axis='y', labelsize=12)
        sns.despine(ax=ax, left=True, bottom=False)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.suptitle("各主题核心词分布 (LDA模型)", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_lda_topics.png", bbox_inches='tight')
    plt.close()

def plot_time_series_comments_chart(df: pd.DataFrame) -> None:
    """
    绘制按月、按平台的评论数量图。
    """
    print(">>> 绘制时间序列图（按月、平台）...")
    ts = df.set_index("created_at").sort_index()
    counts = ts.groupby([pd.Grouper(freq="M"), "platform"]).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 6))
    for platform in counts.columns:
        # 使用 datetime 对象作为 x 轴，以兼容不同版本的 matplotlib
        plt.plot(counts.index.to_pydatetime(), counts[platform], marker="o", label=str(platform))
    plt.title("评论时间序列（按平台，月度）")
    plt.xlabel("Month")
    plt.ylabel("Comment Count")
    plt.legend(title="Platform", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_time_series_comments.png")
    plt.close()

def plot_sentiment_vs_length_chart(df: pd.DataFrame) -> None:
    """
    绘制情感分布与文本长度的关系散点图。
    """
    print(">>> 绘制情感 vs 内容长度散点图...")
    df2 = df.copy()
    df2["content_length"] = df2["content"].astype(str).str.len()
    plt.figure(figsize=(6, 4))
    plt.scatter(df2["content_length"], df2["sentiment"], alpha=0.3)
    plt.xlabel("Content length")
    plt.ylabel("Sentiment")
    plt.title("情感 vs 内容长度")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_sentiment_vs_length.png")
    plt.close()

def analyze_specific_themes(df: pd.DataFrame) -> None:
    """
    基于预定义的特定主题进行关键词匹配分析。
    """
    print(">>> 进行特定主题分析...")
    
    # 使用翻译后的内容进行匹配
    # 确保 content_cn 存在
    if "content_cn" not in df.columns:
         df["content_cn"] = df["content"]

    text_series = df["content_cn"].astype(str).str.lower()
    
    theme_counts = {theme: 0 for theme in SPECIFIC_THEMES}
    
    # 简单的计数逻辑：每条评论如果包含主题词，则该主题计数+1
    # 一条评论可以属于多个主题
    for theme, keywords in SPECIFIC_THEMES.items():
        # 构建正则或者直接循环
        for kw in keywords:
            # 只要出现一次关键词就算
            mask = text_series.str.contains(kw, regex=False)
            theme_counts[theme] += mask.sum()
            
    # 转换为 DataFrame
    theme_df = pd.DataFrame(list(theme_counts.items()), columns=["Theme", "Count"])
    theme_df = theme_df.sort_values("Count", ascending=True)
    
    # 导出
    out_path = PLOTS_DIR / "data_specific_themes.csv"
    theme_df.to_csv(out_path, index=False)
    print(f"已导出: {out_path}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(theme_df["Theme"], theme_df["Count"], color="teal")
    plt.title("特定主题提及频次 (Specific Themes Analysis)")
    plt.xlabel("Count")
    plt.ylabel("Theme")
    plt.bar_label(bars)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "data_specific_themes.png")
    plt.close()

def analyze_topics_lda(df: pd.DataFrame, n_topics: int = 4) -> None:
    print(">>> 进行 LDA 主题聚类 (基于翻译后的中文)...")
    
    # 确保 content_cn 存在
    if "content_cn" not in df.columns:
         df["content_cn"] = df["content"]

    def preprocess(text: str) -> str:
        # 转小写以匹配英文停用词
        words = jieba.cut(str(text).lower())
        # 增加过滤条件：不在停用词表中，且不是纯数字
        return " ".join([w for w in words if len(w) > 1 and w not in STOP_WORDS and not w.isnumeric()])
    
    # 使用 content_cn
    df["processed"] = df["content_cn"].apply(preprocess)
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
    plot_sentiment_distribution_chart(df)
    plot_sentiment_trend_chart(df)
    keywords = get_keywords(df, top_n=100)
    plot_wordcloud_image(keywords)
    
    # 插入特定主题分析
    analyze_specific_themes(df)
    
    analyze_topics_lda(df, n_topics=4)
    plot_lda_topics_chart_from_csv(top_n=10)
    plot_job_impact(df)
    plot_time_series_comments_chart(df)
    plot_sentiment_vs_length_chart(df)
    plot_correlations(df)
    print("\n=== 所有分析数据已导出，保存在 outputs/plots/ 目录 ===")

if __name__ == "__main__":
    main()


