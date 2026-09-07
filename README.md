# Data Analytics & Engineering Portfolio — Greg Rivera

 [linkedin.com/in/greg-rivera](https://linkedin.com/in/greg-rivera)

## About Me

I'm a data professional pursuing an M.S. in Business Analytics at the University of Utah (2026–2027), building on an M.S. in Data Science applied to Business Intelligence from European Atlantic University and a B.S. in Applied Business Management from BYU-Idaho. My background combines hands-on data pipeline and workflow optimization experience (FamilySearch) with coursework in machine learning, NLP, and big data technologies. This repository showcases projects demonstrating my ability to build data pipelines, process large and distributed datasets, and turn raw data into reliable, decision-ready outputs — with an eye toward **data engineering** roles.

## Tech Stack

| Category | Tools |
|---|---|
| **Languages** | Python, R |
| **Databases** | PostgreSQL, Oracle, DBeaver |
| **Big Data / Distributed Processing** | PySpark, Databricks, Dask |
| **Machine Learning / NLP** | scikit-learn, quanteda, caret, sentimentr, cleanNLP, topicmodels |
| **BI & Visualization** | Power BI, DOMO, Advanced Excel |
| **Integration** | APIs, CRM systems |
| **Project Management / Workflow** | Agile, PMBOK, JIRA, SOPs |

## Featured Projects

### 🍽️ LA Restaurant Sentiment Analysis (Group Consulting Project)
A comparative sentiment analysis pipeline built on the "Top 240 Recommended Restaurants in LA, 2023" Kaggle dataset.
- **Ingestion:** Raw Yelp review data pulled from the Kaggle source dataset
- **Processing:** Cleaning, text preprocessing, and structuring of review text across restaurant price tiers and neighborhoods
- **Analysis:** Sentiment scoring and comparative analysis segmented by price tier and geography
- **Output:** Client-facing consulting deliverable with actionable insights
- 📁 [`/projects/la-restaurant-sentiment`](./projects/la-restaurant-sentiment)

### 📊 NLP & Distributed Data Processing Coursework
A set of notebooks demonstrating pipeline-building and distributed-processing skills across the data lifecycle:
- **PySpark / Databricks** — distributed data processing workflow (HW5)
- **Dask + scikit-learn** — parallelized data processing and modeling (HW4)
- **Pandas** — data manipulation and transformation (HW3)
- **NumPy / Matplotlib** — image data manipulation (HW2)
- **R-based NLP pipeline** — TF-IDF, LSA, LDA, Word2Vec, sentiment analysis, POS tagging, and NER using quanteda, sentimentr, cleanNLP, and topicmodels
- 📁 [`/projects/nlp-coursework`](./projects/nlp-coursework)

### 🏢 Enterprise Data Workflow Optimization (FamilySearch)
Not a code repo, but a case study in data engineering impact at scale:
- Redesigned CRM and access-management data workflows, cutting resolution time by 52% while maintaining 95%+ service levels
- Coordinated with QA and Data Management teams to resolve data inconsistencies and optimize processing pipelines across 182 digitalization projects in 26+ countries
- Built Power BI / DOMO reporting pipelines to support operational forecasting and decision-making across multiple business areas
- Authored technical documentation for data systems and access workflows used by 300+ staff and volunteers
- 📁 [`/projects/familysearch-case-study`](./projects/familysearch-case-study)

## Repository Structure

```
.
├── README.md
├── projects/
│   ├── la-restaurant-sentiment/
│   │   ├── notebooks/
│   │   ├── data/
│   │   └── README.md
│   ├── nlp-coursework/
│   │   ├── notebooks/
│   │   └── README.md
│   └── familysearch-case-study/
│       └── README.md
└── requirements.txt
```

## Getting Started

```bash
git clone https://github.com/<your-username>/data-analytics-portfolio.git
cd data-analytics-portfolio
pip install -r requirements.txt
```

Each project folder includes its own README with setup and run instructions specific to that project.

## Contact

Feel free to reach out via [LinkedIn](https://linkedin.com/in/greg-rivera) — I'm currently seeking data engineering roles.
