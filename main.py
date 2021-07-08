from flask import Flask, render_template, request
app = Flask(__name__)
import pickle



# open a file, when you want to store the data
file = open('model.pkl','rb')
clf=pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method=="POST":
        myDict=request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        difficultybreath = int(myDict['difficultybreath'])



        #code for inference
        inputFeatures=[fever, age, pain, runnyNose, difficultybreath]
        infection_prob=clf.predict_prob([inputFeatures])[0][1]
        print(infection_prob)
        return render_template('show.html', inf=round(infection_prob*100))
    return render_template('index.html')
    #return 'Hello, World!' + str(infection_prob)
if __name__=="__main__":
    app.run(debug=True)

