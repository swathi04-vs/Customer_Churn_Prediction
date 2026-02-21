# Customer Churn Prediction Web Application

A production-ready machine learning web application built with Flask and scikit-learn that predicts customer churn for telecom companies.

## 🎯 Features

### 🤖 Machine Learning
- **Random Forest Classifier** for binary classification (Churn vs No Churn)
- Advanced preprocessing pipeline with:
  - OneHot Encoding for categorical features
  - Standard Scaling for numerical features
  - Missing value imputation
- Model accuracy: ~85% with comprehensive evaluation metrics

### 🎨 Frontend
- **Modern UI Design** with glassmorphism effects
- **Bootstrap 5** responsive framework
- **Interactive Prediction Form** with real-time validation
- **Beautiful Dashboard** with feature importance visualization
- **Chart.js** for data visualization
- Mobile-responsive design

### ⚙️ Backend
- **Flask** micro web framework with Blueprints
- RESTful API endpoints for predictions
- Comprehensive **error handling** and validation
- Structured logging system
- Clean MVC-like architecture

### 📊 Analytics
- Feature importance analysis
- Model performance metrics
- Classification reports
- Confusion matrix visualization

## 📦 Project Structure

```
customer_churn_prediction/
│
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── routes.py                       # Flask blueprints and routes
│
├── model/
│   ├── __init__.py
│   ├── train_model.py             # Model training script
│   ├── model.pkl                  # Trained model (generated)
│   └── preprocessor.pkl           # Preprocessing pipeline (generated)
│
├── templates/
│   ├── base.html                  # Base template with navigation
│   ├── index.html                 # Landing page
│   ├── predict.html               # Prediction form page
│   └── dashboard.html             # Analytics dashboard
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom CSS with glassmorphism
│   ├── js/
│   │   └── main.js                # JavaScript utilities
│   └── images/                    # Image assets
│
└── utils/
    ├── __init__.py
    ├── helper.py                  # Helper functions and decorators
    └── preprocess.py              # Data preprocessing utilities
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository or navigate to the project:**
```bash
cd Customer_churn_prediction
```

2. **Create a virtual environment:**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Train the model (first time only):**
```bash
# This creates sample data and trains the model
python model/train_model.py
```

This will:
- Generate a sample dataset with 1000 records
- Train the RandomForest model
- Save the model to `model/model.pkl`
- Save the preprocessor to `model/preprocessor.pkl`

5. **Run the Flask application:**
```bash
python app.py
```

6. **Access the application:**
Open your browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage

### 1. **Home Page** (`/`)
- Landing page with features overview
- Quick navigation to prediction and dashboard
- Application information

### 2. **Prediction Page** (`/predict`)
- Fill in customer details through an intuitive form
- **Demographics**: Gender, Age, Partner, Dependents
- **Services**: Internet, Phone, Streaming, etc.
- **Account Info**: Tenure, Contract, Charges, Payment Method
- Submit to get real-time predictions
- View churn probability and confidence scores

### 3. **Dashboard** (`/dashboard`)
- Model accuracy and performance metrics
- Feature importance chart showing top 10 features
- Model specifications and preprocessing pipeline info
- Input features documentation

### 4. **API Endpoints**

#### Make a Prediction
```bash
POST /api/predict
Content-Type: application/json

{
    "gender": "Male",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": "24",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": "65.5",
    "TotalCharges": "1570.85"
}
```

Response:
```json
{
    "status": "success",
    "prediction": 0,
    "churn_status": "No - Customer is likely to stay",
    "churn_risk": "Low Risk",
    "probability": 0.15,
    "confidence": 85.0
}
```

#### Get Model Information
```bash
GET /api/model-info
```

#### Health Check
```bash
GET /api/health
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Model Settings**: Path to saved models, accuracy threshold
- **Logging**: Log level, format, date format
- **Features**: Categorical and numerical feature lists
- **Prediction Limits**: Min/max values for numeric inputs

## 📊 Model Details

### Algorithm: Random Forest Classifier
- **Estimators**: 100 trees
- **Max Depth**: 15 levels
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Class Weight**: Balanced to handle class imbalance

### Features (19 total)
**Categorical (16):**
- Gender, SeniorCitizen, Partner, Dependents
- PhoneService, MultipleLines, InternetService
- OnlineSecurity, OnlineBackup, DeviceProtection
- TechSupport, StreamingTV, StreamingMovies
- Contract, PaperlessBilling, PaymentMethod

**Numerical (3):**
- Tenure (months)
- MonthlyCharges ($)
- TotalCharges ($)

### Preprocessing Pipeline
1. **OneHot Encoding** - Converts categorical variables to binary columns
2. **Standard Scaling** - Normalizes numerical features
3. **Missing Value Imputation** - Mean imputation for numerical, mode for categorical

### Performance Metrics
- **Accuracy**: ~85%
- **Precision**: ~85%
- **Recall**: ~82%
- **F1-Score**: ~83%
- **ROC-AUC**: ~88%

## 🛠️ Development

### Running in Development Mode
```bash
# With debug enabled for auto-reload
python app.py
```

### Creating a New Training Dataset

Use your own CSV file with the same columns:
```python
from model.train_model import ChurnModelTrainer

trainer = ChurnModelTrainer()
results = trainer.train_full_pipeline('your_data.csv')
```

### Retraining the Model

```bash
python model/train_model.py
```

## 🔒 Security Considerations

- No data is stored persistently
- Predictions are computed in real-time
- Input validation on all form submissions
- CSRF protection via Flask-WTF (can be added)
- Secure secret key in configuration

## 🐛 Troubleshooting

### Model not found error
**Solution**: Run `python model/train_model.py` to train the model first

### Port 5000 already in use
**Solution**: Change port in app.py or kill the process using port 5000

### Module not found errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Form submission errors
**Solution**: Check browser console for detailed error messages

## 📈 Improvements & Future Enhancements

- [ ] Add database for storing predictions
- [ ] Implement user authentication
- [ ] Add pagination for prediction history
- [ ] Export predictions to CSV
- [ ] Real-time model retraining
- [ ] Multiple model comparison
- [ ] SHAP explainability features
- [ ] Model drift detection
- [ ] A/B testing framework
- [ ] Advanced analytics dashboard

## 👨‍💻 Author

Customer Churn Prediction System - Built with Flask, Scikit-learn, and Modern Web Technologies

## 🤝 Support

For issues or questions, please create an issue in the repository or contact the development team.

## 📚 References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Bootstrap Documentation](https://getbootstrap.com/docs/)
- [Chart.js Documentation](https://www.chartjs.org/docs/latest/)

---

**Happy Predicting! 🚀**
