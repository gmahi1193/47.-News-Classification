from flask import Flask, request, render_template
import joblib

class ML:
    
    def __init__(self):
        self.specialChars = [',','#','.','?','*','<','>','\\']
    
    def specialCharRemoval(self, text):
        for specialChar in self.specialChars:
            text = text.replace(specialChar, '')
        return text
    
     
    def predict_news(self, clf, vect, x_test):
        """
        clf: classifier after training
        x_test: testing data
       
        """

        x_test_process = [self.specialCharRemoval(text) for text in x_test]
        

        x_test_tfidf = self.transform(vect, x_test_process)
        return clf.predict(x_test_tfidf)


app = Flask(__name__)

@app.route('/')
def initialize():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def news_classifier():
    if request.method == 'POST':
        
        text = request.form['usertext']
        clf = joblib.load("model.pkl")
        vect = joblib.load("vect.pkl")
        mlObj = ML()
        news_class = mlObj.predict_news(clf, vect, [text])


    return render_template("index.html", classOfNews = news_class)

if __name__ == '__main__':
    app.run(debug = True)