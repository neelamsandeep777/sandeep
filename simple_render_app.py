from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head><title>Sign Language Detection</title></head>
    <body>
        <h1>Sign Language Detection - Coming Soon</h1>
        <p>App is working on Render!</p>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)