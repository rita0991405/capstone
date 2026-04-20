# Capstone Project - Predicting Type 2 Diabetes Risk in East Asian Populations

## Project Overview
This project set out to investigate whether sleep-related characteristics, such as duration and maintenance, could meaningfully predict Type 2 diabetes risk in East Asian populations, where existing Western-focused models may not apply due to elevated metabolic risk at lower body–mass index (BMI) values.


## Team Members
- Li-yuan Chen
- Jessica Jones
- Kento Takeda

## Repository Structure
- `data/` - Sample datasets
- `scr/` - All Python code for data cleaning, modeling, and receommender system
- `scr/eda` - Exploratory Data Analysis scripts (data visualization, summaries, insights)
- `scr/modleing` - Model training, evaluation, and tuning scripts
- `scr/rec_system` - Recommender system implementation
- `requirements.txt` - Python package dependencies


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
python s1file_eda.py
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
streamlit run streamlit_app_final.py
```

## Results
We developed a type 2 diabetes risk prediction model and recommender system tailored to East Asian populations, where age, BMI, and clinical markers drive most of the predictive power. Because these factors are largely non-modifiable, the system focuses on actionable inputs like weight, smoking, and exercise to deliver realistic, personalized lifestyle recommendations. This approach turns risk prediction into practical guidance for reducing individual diabetes risk.

## Technologies Used
- Python 3.12
- pandas, numpy, matplotlib, seaborn, scikit-learn, shap, xgboost, atlair, streamlit, plotly
