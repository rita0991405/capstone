# capstone
Capstone project for MADS program
# (Project Title)

## Project Overview
(project overview)

## Team Members
- Li-yuan Chen
- Jessica Jones
- Kento Takeda

## Repository Structure
capstone/
│
├── data/ # Datasets
├── scr/ # Python source code
│ ├── eda/
│ │ ├── prediabetes_eda.py
│ │ ├── S1file_eda.py
│ ├── modeling/
│ │ ├── prediabetes_model.py
│ │ └── s1file_model.py
│ ├── rec_system/
│ │ ├── predictor.py
│ │ └── recommender.py
└── README.md

## Data Sources

### Full Datasets
1. **Prediabetes Health Data**: [Korean Prediabetes Data](https://data.mendeley.com)
   - Source: Korea National Health and Nutrition Examination Survey (KNHANES), Yonsei University
   - Size:
   - Description: Contains health metrics, employment status, exercise habits, and related variables associated with prediabetes
   - Note: Provides insights into relationships between socioeconomic status, diet, physical activity, and prediabetes risk among Koreans aged 19+

2. **Sleep and Diabetes Data**: [Japanese Sleep and Diabetes Data](https://journals.plos.org/plosone/)
   - Source: Macko et al. (2025) - MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection
   - Size:
   - Description: Includes sleep duration, efficiency, latency, and their relationship with diabetes, along with lifestyle factors (smoking, drinking) and clinical indicators (BMI, neuropathy, hyperlipidemia)
   - Note: Focuses on associations between sleep patterns and metabolic health outcomes

## Setup Instructions

### Prerequisites
- Python 3.8+
- Jupyter Notebook (optional, for exploration)

### Installation
```bash
# Clone repository
git clone https://github.com/[your-username]/capstone.git
cd capstone

# Install dependencies
pip install -r requirements.txt

### Running the Pipeline
```
1. **Exploratory Data Analysis**
```bash
# Perform EDA and extract features for supervised learning
python prediabetes_eda.py
# Outputs:
```

2. **Modeling**
```bash
python prediabetes_model.py
python s1file_model.py
```

4. **Recommender System**
```bash
python predictor.py
python recommender.py
```

## Key Results


## Technologies Used
- Python 3.12
- pandas, numpy, matplotlib, seaborn
