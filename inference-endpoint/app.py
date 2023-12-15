# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import io
import base64
from base64 import encodebytes
from PIL import Image
from flask import jsonify, Flask, request
from inference import run_cat_indentifier

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

def get_response_image(pil_img):
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	return 'Hello World'

@app.route('/inference', methods = ['POST'])   
def inference():
    if request.method == 'POST':
        image_data = request.data
        # image_data = base64.b64decode(request.data)
        print(image_data[:100])
        image = Image.open(io.BytesIO(image_data))
        image.save("input_image.jpg")
        results = run_cat_indentifier("input_image.jpg")
        encoded_imges = []
        for img in results['cat_images']:
            encoded_imges.append(get_response_image(img))
        print(encoded_imges)
        return jsonify({
            'images': encoded_imges,
            'labels': results['predicted_labels']
        })

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application 
	# on the local development server.
	app.run(debug=True)
