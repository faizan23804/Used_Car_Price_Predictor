# 🚗 Used Car Price Prediction

This project predicts the **selling price of used cars** based on various features such as age, mileage, fuel type, transmission, and more. It uses machine learning models like **Random Forest Regressor** and **K-Nearest Neighbors Regressor**, with data preprocessed using `scikit-learn`'s `ColumnTransformer`.

---

## 📁 Dataset

- **Source:** [CarDekho.com](https://www.cardekho.com/) (scraped)
- **Rows:** 15,411  
- **Columns:** 13  
- **Target Variable:** `selling_price`

### Sample Features
- `car_name`  
- `vehicle_age`  
- `km_driven`  
- `seller_type`  
- `fuel_type`  
- `transmission_type`  
- `mileage`, `engine`, `max_power`, `seats`

> ❗ Columns like `brand` and `model` were dropped due to high cardinality or redundancy.

---

## 🧠 Machine Learning Models Used

- ✅ **Random Forest Regressor** (Primary model)
- ✅ **K-Neighbors Regressor** (For comparison)

Models are saved as:
- `random_forest_regressor.pkl`
- `k-neighbors_regressor.pkl`
- `preprocessor.pkl` – ColumnTransformer (Ordinal Encoding + Standard Scaling)

> 📦 All `.pkl` files are saved using **`joblib` with `compress=3`** to reduce size and improve GitHub compatibility.

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Numerical features: `StandardScaler`
   - Categorical features: `OrdinalEncoder` with `handle_unknown='use_encoded_value'`
   - Combined using `ColumnTransformer`

2. **Training**
   - Split into train/test sets
   - Grid/random search (optional)
   - Trained on cleaned and imputed data

3. **Serving with Flask**
   - Users input car specs through a web form
   - Preprocessor transforms input
   - Model predicts selling price

---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Used_car_price_Prediction.git
cd Used_car_price_Prediction

python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

pip install -r requirements.txt

python app.py

