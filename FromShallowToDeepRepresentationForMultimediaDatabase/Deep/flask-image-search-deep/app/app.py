import os

from flask import Flask, render_template, request, jsonify

from pyimagesearch.ImageRetrieveDeep import ImageRetrieveDeep
from pyimagesearch.searcher import Searcher

# create flask instance
app = Flask(__name__)

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
            print("Before ImageRetrieveDeep")
            ird = ImageRetrieveDeep("", "")
            print("Before path_img")
            path_img = os.path.join(os.getcwd(), image_url[1:])
            print("Before r_mac_descriptor")
            des = ird.r_mac_descriptor(path_img)

            # perform the search
            print("Before Searcher")
            searcher = Searcher(des)
            print("Before search")

            # directories
            stat = "static"
            stat_data = os.path.join(stat, "dataset_jpg")
            stat_desc = "descriptor"

            results = searcher.search(des, stat_data, stat_desc)
            # loop over the results, displaying the score and image name
            print("Before for")
            for (score, resultID) in results:
                RESULTS_ARRAY.append(
                    {"image": str(resultID), "score": str(score)})
            # return success
            return jsonify(results=(RESULTS_ARRAY[::-1][:10]))

        except:
            # return error
            jsonify({"sorry": "Sorry, no results! Please try again."}), 500


# run!
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
