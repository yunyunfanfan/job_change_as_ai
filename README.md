# 大模型对IT行业就业岗位市场与职业技能影响分析 - 项目复现指南

## 项目概述

本项目旨在通过收集和分析多个自媒体平台（B站、知乎、Reddit、V2EX）上关于大模型对IT行业影响的讨论数据，研究大模型技术对IT行业就业岗位市场和职业技能的舆论影响。项目采用数据抓取、清洗、分析和可视化的完整数据科学流程，最终产出包含1443+条有效评论的分析报告。

**研究时间跨度**：2022年11月（ChatGPT发布）至2025年12月  
**数据来源**：B站视频评论、知乎回答评论、Reddit讨论、V2EX技术社区  
**数据规模**：18+个主贴，每个主贴≥100条评论，总计1443+条有效评论

---

## 环境配置

### 系统要求
- Python 3.8 或更高版本
- Windows 10/11 或 Linux/macOS
- 至少 2GB 可用磁盘空间（用于存储原始数据和清洗后的数据）

### 依赖安装

项目使用 `requirements.txt` 管理依赖。安装步骤如下：

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

**主要依赖库说明**：
- `pandas>=1.5.0`：数据处理和清洗
- `numpy>=1.20.0`：数值计算
- `matplotlib>=3.4.0`、`seaborn>=0.11.0`：数据可视化
- `jieba>=0.42.0`：中文分词
- `wordcloud>=1.8.0`：词云图生成
- `snownlp>=0.2.2`：中文情感分析
- `vaderSentiment>=3.3.2`：英文情感分析
- `scikit-learn>=1.0.0`：LDA主题模型
- `requests`：HTTP请求（数据抓取）
- `tqdm`：进度条显示

**注意**：如果遇到 `numpy` 版本冲突（如 `numpy 2.x` 与 `pandas` 不兼容），可执行：
```bash
pip install "numpy<2"
```

---

## 数据获取方法

### 1. B站视频评论抓取

**脚本文件**：`fetch_bilibili_comments.py`

**技术要点**：
- 使用 B站公开 API：`x/web-interface/view`（获取视频信息）和 `x/v2/reply/wbi/main`（获取评论）
- 实现 WBI 签名算法（`w_rid` 参数生成）以通过接口验证
- 支持 cursor 分页机制，自动翻页获取全部评论
- 递归抓取楼中楼（二级评论），通过 `x/v2/reply/reply` 接口
- 内置重试机制和请求频率控制，避免被限流

**使用方法**：
```bash
python fetch_bilibili_comments.py
# 按提示输入 BV 号，例如：BV12ALczPEXR
```

**输出**：每个视频生成一个 CSV 文件，命名格式为 `bilibili_comments_{BV号}.csv`

**字段说明**：
- `bv_id`：视频 BV 号
- `oid`：视频 aid（内部标识）
- `level`：评论层级（1=一级，2=楼中楼）
- `root_rpid`：根评论 ID
- `rpid`：评论 ID
- `parent_rpid`：父评论 ID
- `uid`：用户 ID
- `uname`：用户名
- `user_level`：用户等级
- `ctime`：评论时间戳（Unix 秒）
- `like`：点赞数
- `message`：评论内容

**注意事项**：
- 需要有效的 B站登录 Cookie（脚本中已包含示例，实际使用时需替换为自己的 Cookie）
- 抓取过程中会自动控制请求频率（每次请求间隔 1-2 秒），避免触发反爬机制
- 如遇到 `ConnectionResetError`，脚本会自动重试最多 3 次

---

### 2. 知乎回答评论抓取

**脚本文件**：`fetch_zhihu_comments.py`

**技术要点**：
- 使用知乎 API：`api/v4/answers/{answer_id}/root_comments`（获取根评论）
- 对于有大量子评论的根评论，递归调用 `api/v4/comments/{comment_id}/child_comments` 获取全部楼中楼
- 支持分页机制（`offset` 参数），自动翻页
- 内置重试机制和错误处理

**使用方法**：
```bash
python fetch_zhihu_comments.py
# 按提示输入：
# 1. answer 号（从知乎回答 URL 中提取，例如：https://www.zhihu.com/question/xxx/answer/99115011898 中的 99115011898）
# 2. User-Agent（从浏览器 F12 开发者工具中复制）
# 3. Cookie（从浏览器 F12 开发者工具中复制，需包含登录信息）
```

**输出**：每个回答生成一个 CSV 文件，命名格式为 `zhihu_comments_{answer_id}.csv`

**字段说明**：
- `answer_id`：回答 ID
- `level`：评论层级（1=根评论，2=子评论）
- `comment_id`：评论 ID
- `parent_id`：父评论 ID
- `author_name`：作者名
- `gender`：性别（男/女/未知）
- `ip_location`：IP 属地
- `like_count`：点赞数
- `created_at`：评论时间（ISO 格式）
- `content`：评论内容

---

### 3. Reddit 评论抓取

**脚本文件**：`fetch_reddit_comments.py`

**技术要点**：
- 使用 Reddit 公开 API（无需认证即可获取公开帖子）
- 支持递归抓取评论树结构
- 自动处理 Reddit 的分页机制

**使用方法**：
```bash
python fetch_reddit_comments.py
# 按提示输入 Reddit 帖子 ID（从 URL 中提取）
```

**输出**：每个帖子生成一个 CSV 文件，命名格式为 `reddit_comments_{post_id}.csv`

---

### 4. V2EX 评论抓取

**脚本文件**：`fetch_v2ex_comments.py`

**技术要点**：
- 解析 V2EX 网页 HTML 结构
- 提取主题帖和所有回复
- 处理 V2EX 的分页机制

**使用方法**：
```bash
python fetch_v2ex_comments.py
# 按提示输入 V2EX 主题 ID（从 URL 中提取）
```

**输出**：每个主题生成一个 CSV 文件，命名格式为 `v2ex_comments_{topic_id}.csv`

---

## 数据清洗流程

### 脚本文件：`clean_comments.py`

**功能概述**：
该脚本将来自不同平台的原始 CSV 文件统一清洗、标准化，生成可用于后续分析的统一数据格式。

**清洗步骤**：

1. **字段统一映射**：
   - 将 B站、知乎、Reddit、V2EX 的不同字段名映射到统一的标准字段
   - 标准字段包括：`platform`（平台）、`thread_id`（主贴ID）、`comment_id`（评论ID）、`parent_id`（父评论ID）、`user_name`（用户名）、`user_id`（用户ID）、`created_at`（时间）、`content`（内容）、`like_count`（点赞数）、`level`（层级）等

2. **时间标准化**：
   - B站：将 Unix 时间戳（秒）转换为 `YYYY-MM-DD HH:MM:SS` 格式
   - 知乎：解析 ISO 格式时间字符串
   - Reddit/V2EX：统一时间格式

3. **文本清洗**：
   - 移除 HTML 标签（如 `<p>`、`<br>`）
   - HTML 实体解码（如 `&amp;` → `&`）
   - 压缩多余空白字符
   - 去除首尾空白

4. **缺失值处理**：
   - 空评论内容标记为 `NA` 或过滤
   - 缺失的 IP 属地、性别等信息标记为 `未知` 或 `NA`

5. **去重**：
   - 基于 `(platform, thread_id, comment_id)` 组合去重，确保每条评论只保留一次

6. **数据验证**：
   - 检查必需字段是否存在
   - 验证时间字段格式
   - 统计清洗前后的数据量变化

**使用方法**：
```bash
python clean_comments.py
```

**输入**：脚本自动扫描当前目录下的所有 `*_comments_*.csv` 文件

**输出**：
- `all_comments_cleaned.csv`：清洗后的统一数据表（所有评论）
- `thread_summary.csv`：每个主贴的统计摘要（评论数、时间范围、平均长度等）

**输出字段说明**（`all_comments_cleaned.csv`）：
- `platform`：平台名称（bilibili/zhihu/reddit/v2ex）
- `thread_id`：主贴标识（BV号/answer_id/post_id/topic_id）
- `comment_id`：评论唯一标识
- `parent_id`：父评论 ID（0 表示根评论）
- `level`：评论层级（1=一级，2=二级及以上）
- `user_name`：用户名
- `user_id`：用户 ID
- `created_at`：评论时间（统一格式）
- `content`：评论内容（已清洗）
- `like_count`：点赞数
- `text_length`：评论长度（字符数）
- `source_file`：原始文件名（便于追溯）

---

## 数据分析与可视化

### 脚本文件：`data_analysis_dashboard.py`

**功能概述**：
该脚本对清洗后的数据进行多维度分析，包括情感分析、主题建模、时间序列分析、词频统计等，并生成可视化图表。

**分析维度**：

1. **情感分析**：
   - 使用 `snownlp` 对中文评论进行情感分析（0-1 分数，>0.5 为正面）
   - 使用 `vaderSentiment` 对英文评论进行情感分析
   - 统计各平台、各时间段的情绪分布

2. **主题建模（LDA）**：
   - 使用 `jieba` 对中文评论分词
   - 构建文档-词矩阵
   - 使用 `scikit-learn` 的 LDA 模型提取主题（默认 5-10 个主题）
   - 输出每个主题的关键词和文档分布

3. **时间序列分析**：
   - 统计各时间段的评论数量变化
   - 分析情绪随时间的变化趋势
   - 识别讨论热点时间段

4. **关键词提取**：
   - 统计高频词（去除停用词）
   - 生成词云图
   - 识别与"岗位"、"技能"相关的关键词

5. **平台对比分析**：
   - 对比不同平台的讨论重点
   - 分析平台间的情绪差异
   - 统计各平台的评论长度分布

**使用方法**：
```bash
python data_analysis_dashboard.py
```

**输入**：`data/all_comments_cleaned.csv`（或 `all_comments_cleaned.csv`）

**输出**：
- **图表文件**（保存在 `charts/` 目录）：
  - `Comment Trend over Time.png/html`：评论数量时间趋势
  - `Sentiment Distribution.png/html`：整体情绪分布
  - `Sentiment by Platform.png/html`：各平台情绪对比
  - `Sentiment Trend over Time.png/html`：情绪随时间变化
  - `LDA Topic.png/html`：LDA 主题可视化
  - `Most Discussed IT Roles.png/html`：最常讨论的 IT 岗位
  - `特定主题提及频次.png/html`：特定关键词提及频次
  - `Sentiment vs Length.png/html`：情绪与评论长度关系

- **数据文件**（保存在 `outputs/analysis_results/` 目录）：
  - `data_sentiment_distribution_overall.csv`：整体情绪分布
  - `data_sentiment_distribution_by_platform.csv`：各平台情绪分布
  - `data_sentiment_trend.csv`：情绪时间趋势
  - `data_time_series_comments.csv`：评论数量时间序列
  - `data_lda_topics.csv`：LDA 主题结果
  - `data_specific_themes.csv`：特定主题提及统计
  - `data_wordcloud.csv`：词频统计
  - `data_sentiment_vs_length.csv`：情绪与长度关系
  - `data_job_impact.csv`：IT 岗位影响分析

---

## 代码文件说明

### 数据获取脚本

| 文件名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `fetch_bilibili_comments.py` | 抓取 B站视频评论 | BV 号（交互式输入） | `bilibili_comments_{BV号}.csv` |
| `fetch_zhihu_comments.py` | 抓取知乎回答评论 | answer_id、User-Agent、Cookie | `zhihu_comments_{answer_id}.csv` |
| `fetch_reddit_comments.py` | 抓取 Reddit 帖子评论 | post_id | `reddit_comments_{post_id}.csv` |
| `fetch_v2ex_comments.py` | 抓取 V2EX 主题评论 | topic_id | `v2ex_comments_{topic_id}.csv` |

### 数据处理脚本

| 文件名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `clean_comments.py` | 统一清洗所有平台的评论数据 | 所有 `*_comments_*.csv` 文件 | `all_comments_cleaned.csv`、`thread_summary.csv` |
| `summarize_comments.py` | 快速统计各平台/主贴的数据量 | 所有 `*_comments_*.csv` 文件 | 终端输出统计信息 |

### 数据分析脚本

| 文件名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `data_analysis_dashboard.py` | 情感分析、主题建模、可视化 | `all_comments_cleaned.csv` | 图表文件（`charts/`）、分析结果 CSV（`outputs/analysis_results/`） |

---

## 数据规模

### 总体统计

- **总评论数**：1443+ 条
- **主贴数量**：18+ 个
- **平台分布**：
  - B站：1215 条（1 个主贴）
  - 知乎：228 条（7 个主贴）
  - Reddit：待补充
  - V2EX：待补充

### 时间分布

- **最早评论**：2018-11-22（知乎历史讨论）
- **最新评论**：2025-12-01（B站近期讨论）
- **主要讨论期**：2023-2025 年（ChatGPT 发布后的三年）

### 数据质量

- **完整性**：所有必需字段（时间、内容、用户）均已填充
- **准确性**：经过清洗和去重，无重复评论
- **代表性**：覆盖多个平台、多个时间段，反映不同群体的观点

---

## 完整复现步骤

### 步骤 1：环境准备

```bash
# 克隆或下载项目代码
cd 数据分析作业

# 创建虚拟环境（可选但推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# 或
source venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2：数据获取

**方式 A：使用已有数据**（推荐）
- 项目已包含 `data/` 目录下的原始 CSV 文件，可直接使用

**方式 B：重新抓取数据**

```bash
# B站评论（需要替换 Cookie）
python fetch_bilibili_comments.py

# 知乎评论（需要提供 User-Agent 和 Cookie）
python fetch_zhihu_comments.py

# Reddit 评论
python fetch_reddit_comments.py

# V2EX 评论
python fetch_v2ex_comments.py
```

**注意**：抓取数据需要一定时间，且可能受到平台反爬限制。建议使用项目提供的已有数据。

### 步骤 3：数据清洗

```bash
# 运行清洗脚本
python clean_comments.py
```

**预期输出**：
- `all_comments_cleaned.csv`：清洗后的统一数据
- `thread_summary.csv`：主贴统计摘要

### 步骤 4：数据分析

```bash
# 运行分析脚本
python data_analysis_dashboard.py
```

**预期输出**：
- `charts/` 目录下的所有可视化图表（PNG 和 HTML 格式）
- `outputs/analysis_results/` 目录下的分析结果 CSV 文件

### 步骤 5：查看结果

- **可视化图表**：打开 `charts/` 目录下的 HTML 文件（推荐）或 PNG 图片
- **分析数据**：查看 `outputs/analysis_results/` 目录下的 CSV 文件
- **清洗数据**：查看 `all_comments_cleaned.csv` 进行进一步自定义分析

---

## 分析技术方案

### 1. 文本预处理

- **分词**：使用 `jieba` 对中文评论进行分词，自定义词典包含 IT 行业相关术语（如"大模型"、"AI"、"程序员"、"岗位"等）
- **停用词过滤**：移除常见停用词（如"的"、"了"、"在"等）
- **词性标注**：保留名词、动词、形容词等有意义词性

### 2. 情感分析

- **中文**：使用 `snownlp` 库，基于中文语料训练的情感分析模型
- **英文**：使用 `vaderSentiment` 库，专门针对社交媒体文本的情感分析工具
- **分类标准**：正面（>0.5）、中性（0.4-0.6）、负面（<0.4）

### 3. 主题建模

- **方法**：Latent Dirichlet Allocation (LDA)
- **工具**：`scikit-learn` 的 `LatentDirichletAllocation`
- **参数**：
  - 主题数：5-10 个（根据数据量调整）
  - 迭代次数：100-200
  - 学习率：0.01-0.1
- **输出**：每个主题的关键词列表和文档-主题分布

### 4. 关键词提取

- **方法**：TF-IDF（词频-逆文档频率）
- **工具**：`scikit-learn` 的 `TfidfVectorizer`
- **应用**：识别与"岗位"、"技能"相关的关键词，统计提及频次

### 5. 时间序列分析

- **方法**：按月份/季度聚合评论数量和情绪分数
- **可视化**：折线图、热力图
- **目的**：识别讨论热点时间段和情绪变化趋势

---

## 数据文件结构

```
数据分析作业/
├── data/                          # 原始数据目录
│   ├── data_bilibili/            # B站评论 CSV 文件
│   ├── data_zhihu/               # 知乎评论 CSV 文件
│   ├── data_reddit/              # Reddit 评论 CSV 文件
│   ├── data_v2ex/                # V2EX 评论 CSV 文件
│   └── all_comments_cleaned.csv  # 清洗后的统一数据（主文件）
├── outputs/                      # 分析结果目录
│   ├── analysis_results/         # 分析结果 CSV 文件
│   └── all_comments_translated.csv  # 翻译后的数据（如需要）
├── charts/                       # 可视化图表目录
│   ├── charts_png/              # PNG 格式图表
│   └── *.html                    # 交互式 HTML 图表
├── fetch_*.py                    # 数据抓取脚本
├── clean_comments.py             # 数据清洗脚本
├── data_analysis_dashboard.py    # 数据分析脚本
├── summarize_comments.py         # 数据统计脚本
├── requirements.txt              # 依赖列表
└── README.md                     # 本文件
```

---

## 常见问题

### Q1：运行 `clean_comments.py` 时出现 `numpy` 版本冲突错误

**解决方案**：
```bash
pip install "numpy<2"
```

### Q2：抓取 B站评论时出现 `-403 访问权限不足`

**原因**：WBI 签名算法可能已更新，或 Cookie 失效

**解决方案**：
- 更新脚本中的 Cookie（从浏览器 F12 中重新获取）
- 检查 B站账号登录状态
- 适当增加请求间隔时间

### Q3：抓取知乎评论时只能获取少量评论（如 20 条）

**原因**：知乎 API 对未登录用户限制返回数量

**解决方案**：
- 确保提供有效的登录 Cookie（包含 `z_c0` 等关键字段）
- 脚本已实现递归抓取子评论，但部分回答的子评论可能仍需手动触发

### Q4：数据分析脚本运行时间过长

**原因**：情感分析和主题建模计算量较大

**解决方案**：
- 可以先用 `summarize_comments.py` 快速查看数据规模
- 如果数据量过大，可以先用 `pandas` 对数据进行采样（如随机抽取 1000 条）

### Q5：生成的图表中文显示为方框

**解决方案**：
- 确保系统已安装中文字体（Windows 通常自带）
- 在脚本中设置字体路径（如 `matplotlib.rcParams['font.sans-serif'] = ['SimHei']`）

---

## 项目成果

### 数据成果

- **原始数据**：18+ 个主贴，1443+ 条评论
- **清洗数据**：统一格式，可用于进一步分析
- **分析结果**：情感分布、主题模型、时间趋势等多维度分析

### 可视化成果

- **8+ 个可视化图表**：涵盖评论趋势、情绪分布、主题分析、平台对比等
- **交互式图表**：HTML 格式，支持缩放、筛选等交互操作

### 代码成果

- **7+ 个 Python 脚本**：完整的数据获取、清洗、分析流程
- **可复现性**：所有步骤均可通过脚本自动化执行

---

## 后续工作建议

1. **数据扩展**：继续收集更多平台、更多时间段的讨论数据
2. **深度分析**：使用更高级的 NLP 模型（如 BERT）进行语义分析
3. **对比研究**：对比不同时间段（如 ChatGPT 发布前后）的讨论变化
4. **报告撰写**：基于分析结果撰写完整的学术报告

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库 Issue

---

## 许可证

本项目仅用于学术研究目的。数据抓取和使用需遵守各平台的服务条款和法律法规。
