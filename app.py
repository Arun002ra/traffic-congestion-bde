"""
app.py - Vercel WSGI entrypoint.
Exports the WSGI 'app' / 'handler' required by Vercel's Python runtime.
For the interactive local PySpark dashboard, run: streamlit run streamlit_app.py
"""
import os

def app(environ, start_response):
    path = environ.get('PATH_INFO', '/').lstrip('/')
    if not path or path == 'index.html':
        target = 'index.html'
        mimetype = 'text/html; charset=utf-8'
    elif path.endswith('.json'):
        target = path
        mimetype = 'application/json'
    elif path.endswith('.png'):
        target = path
        mimetype = 'image/png'
    elif path.endswith('.csv'):
        target = path
        mimetype = 'text/csv'
    elif path.endswith('.js'):
        target = path
        mimetype = 'application/javascript'
    elif path.endswith('.css'):
        target = path
        mimetype = 'text/css'
    else:
        target = 'index.html'
        mimetype = 'text/html; charset=utf-8'

    if os.path.exists(target):
        start_response('200 OK', [('Content-Type', mimetype)])
        with open(target, 'rb') as f:
            return [f.read()]

    start_response('404 Not Found', [('Content-Type', 'text/plain')])
    return [b'Not Found']

# Export required identifiers for Vercel Serverless Python
application = app
handler = app
