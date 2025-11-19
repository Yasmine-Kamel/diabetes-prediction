# Diabetes Prediction using SVM

**Overview**  
This project predicts whether a person has diabetes using a Diabetes dataset. A Support Vector Machine (SVM) classifier is trained and evaluated using accuracy and classification metrics. You can also try the model interactively using a small Streamlit app.  

**Files**  
- `Project1_Diabetes_Prediction.ipynb` — Jupyter notebook with the full pipeline.  
- `requirements.txt` — Python package dependencies.  
- `data/diabetes.csv` — Dataset used for training.  
- `model/` — Contains the trained SVM model (`svm_diabetes_model.joblib`) and scaler (`scaler.joblib`).  
- `streamlit_app.py` — Streamlit app for an interactive demo.  

**How to run**

```bash
git clone https://github.com/Yasmine-Kamel/diabetes-prediction.git
cd Diabetes-Prediction
pip install -r requirements.txt
```

**Run the Streamlit App (Interactive Demo)**

```bash
streamlit run streamlit_app.py
