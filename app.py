from flask import Flask, render_template, request,jsonify
from test import TextToNum
import pickle

app = Flask(__name__)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")  # Get the message from the form
        print(f"Message received: {msg}")

        
        # You can add your prediction logic here if needed
        prediction = f"Received message: {msg}"

        cl=TextToNum(msg)
        cl.cleaner()
        cl.token()
        cl.removeStop()
        st=cl.stemme()
        stvc=" ".join(st) # Just an example response for now
        with open("vectorizer.pickle","rb") as vc_file :
            vectorizer=pickle.load(vc_file)

        dt=vectorizer.transform([stvc]).toarray()
        with open("model.pickle","rb") as mb_file:
            model=pickle.load(mb_file)
        pred=model.predict(dt)
        print(pred)
        return jsonify({"prediction":str(pred[0])})
        
    else:
        return render_template("predict.html", prediction=None)  # For GET request
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
