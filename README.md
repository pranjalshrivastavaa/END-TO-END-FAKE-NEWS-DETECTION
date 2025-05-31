# 📰 END-TO-END FAKE NEWS DETECTION

This project provides a complete pipeline for detecting fake news using machine learning. It includes data preprocessing, model training, and a web-based application for real-time predictions.

## 📁 Project Structure

📦 END-TO-END-FAKE-NEWS-DETECTION/  
├── FAKE OR REAL NEWS DETECTION.py → Model training & evaluation  
├── app.py → Flask web application  
├── fake_or_real_news.csv → Dataset

## 🧠 Features

- Preprocesses news data using TF-IDF vectorization  
- Uses `PassiveAggressiveClassifier` for binary classification  
- Provides accuracy and confusion matrix metrics  
- Offers a Flask-based web interface for predictions  
- Real-time input for detecting fake vs real news  

## 🗃 Dataset

- **File:** `fake_or_real_news.csv`  
- Contains labeled real and fake news articles  
- Used for training the ML model in a supervised manner  

## ⚙️ Requirements

Install dependencies:  
```bash
pip install pandas numpy scikit-learn Flask
```

## 🚀 How to Run

### 1. Train the Model

```bash
python "FAKE OR REAL NEWS DETECTION.py"
```

### 2. Launch the Flask App

```bash
python app.py
```

Then go to: [http://localhost:5000](http://localhost:5000)

## 🛠 Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Flask  
- Machine Learning (PassiveAggressiveClassifier)

## 🙌 Author

**Pranjal Shrivastava**  
GitHub: [@pranjalshrivastavaa](https://github.com/pranjalshrivastavaa)

## 📄 License

Licensed under the **MIT License**

## 💡 Future Improvements

- Use BERT or LLMs for better prediction  
- Add Bootstrap or React frontend  
- Cloud deployment (Render, Heroku)  
- User login to track history
