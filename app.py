from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/sub", methods=["POST"])
def submit():
    # HTML form data
    if request.method == "POST":
        name = request.form["USERNAME"]
    
    # .py file data from HTML form
    return render_template("sub.html", name=name)

if __name__ == "__main__":
    app.run()