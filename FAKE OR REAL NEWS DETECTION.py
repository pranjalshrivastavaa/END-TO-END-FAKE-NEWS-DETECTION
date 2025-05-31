def fakenewsdetection():
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    data = pd.read_csv(r"C:\Users\rakesh\Desktop\END TO END FAKE NEWS DETECTION\fake_or_real_news.csv")

    x = np.array(data["title"])
    y = np.array(data["label"])

    cv = CountVectorizer()
    x = cv.fit_transform(x)
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(xtrain, ytrain)

    print("Model training complete!")
    accuracy = model.score(xtest, ytest)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
