import fastText
import falcon
import bjoern
import ujson

REVIEW_MODEL = 'models/amazon_review_full.ftz'
WEB_HOST = '127.0.0.1'
PORT = 9000

print('Loading amazon review polarity model ...')
review_classifier = fastText.load_model(REVIEW_MODEL)

class ReviewResource(object):
	def on_post(self, req, resp):
		form = req.params
		if 'text' in form and form['text']:
			try:
				classification, confidence = review_classifier.predict(form['text'])
				resp.body = ujson.dumps({'{} star'.format(classification[0][-1]) : confidence[0]})
				resp.status = falcon.HTTP_200
			except:
				resp.body = ujson.dumps({'Error': 'An internal server error has occurred'})
				resp.status = falcon.HTTP_500
		else:
		    resp.body = ujson.dumps({'Error': 'param \'text\' is mandatory'})
		    resp.status = falcon.HTTP_400


# instantiate a callable WSGI app
app = falcon.API()

# long-lived resource class instance
infer_review = ReviewResource()

# handle all requests to the '/inferreview' URL path
app.req_options.auto_parse_form_urlencoded = True
app.add_route('/inferreview', infer_review)

print('Listening on', WEB_HOST + ':' + str(PORT) + '/inferreview')
bjoern.run(app, WEB_HOST, PORT, reuse_port=True)
