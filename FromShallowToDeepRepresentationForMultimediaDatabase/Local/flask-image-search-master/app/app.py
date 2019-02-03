import os

from flask import Flask, render_template, request, jsonify

from pyimagesearch.colordescriptor import ColorDescriptor
from pyimagesearch.searcher import Searcher

# create flask instance
app = Flask(__name__)

INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')

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
            cd = ColorDescriptor((8, 12, 3))

            # load the query image and describe it
            from skimage import io
            import cv2
            print("image_url ", image_url)
            path_img = os.path.join(os.getcwd(), image_url[1:])
            query = cv2.imread(path_img, cv2.COLOR_BGR2RGB)
            print("describe")
            features = cd.describe(query)

            print("Searcher")
            # perform the search
            searcher = Searcher(INDEX)
            print("search")
            results = searcher.search(features)
            print(INDEX)
            print("image")
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
