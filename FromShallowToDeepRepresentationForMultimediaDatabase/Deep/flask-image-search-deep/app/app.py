import os

from flask import Flask, render_template, request, jsonify

print("Before ImageRetrieveDeep")
from pyimagesearch.ImageRetrieveDeep import ImageRetrieveDeep
print("Before Searcher")
from pyimagesearch.searcher import Searcher

# create flask instance
print("Before Flask")
app = Flask(__name__)

#INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')

IMAGE_FOLDER = os.path.join('data')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# main route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():

    if request.method == "POST":

        RESULTS_ARRAY = []

        # get url
        image_url = request.form.get('img')

        try:

            # initialize the image descriptor
            ird = ImageRetrieveDeep("", "")
            path_img = os.path.join(os.getcwd(), image_url[1:])
            print("Before descriptor")
            des = ird.r_mac_descriptor(path_img)

            print("Before search")
            # perform the search
            searcher = Searcher(des)
            print("Before results")
            results = searcher.search(des, "static/dataset_jpg", "descriptor")
            print("Before loop")
            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                RESULTS_ARRAY.append(
                    {"image": str(resultID), "score": str(score)})
            print("Before success")
            # return success
            return jsonify(results=(RESULTS_ARRAY[::-1][:3]))

        except:
            # return error
            jsonify({"sorry": "Sorry, no results! Please try again."}), 500


# run!
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
