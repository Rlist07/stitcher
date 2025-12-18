import os
from flask import Flask, render_template, send_from_directory, jsonify, request
from pathlib import Path
import logging
import threading
import webbrowser
from urllib.parse import urljoin


class PanoramaViewer:

    def __init__(self, host='127.0.0.1', port=5000, output_path='data/output/panorama.jpg'):
        self.host = host
        self.port = port
        self.output_path = output_path
        self.app = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.route_functions = None

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/panorama.jpg')
        def serve_panorama():
            possible_paths = [
                self.output_path,
                'data/output/panorama.jpg',
                'data/output/panorama.png',
                'data/panorama.jpg',
                'panorama.jpg'
            ]

            for path in possible_paths:
                abs_path = Path(path).resolve()
                if abs_path.exists():
                    directory, filename = os.path.split(path)
                    self.logger.info(f"Serving panorama from: {abs_path}")
                    return send_from_directory(os.path.dirname(str(abs_path)), os.path.basename(str(abs_path)))

            self.logger.warning("No panorama file found, serving placeholder")
            return send_from_directory(str(Path(__file__).parent.parent / 'static'), 'placeholder.jpg')

        @self.app.route('/api/info')
        def api_info():
            return jsonify({
                'panorama_exists': self.panorama_exists(),
                'panorama_url': urljoin(request.url_root, 'panorama.jpg'),
                'output_path': self.output_path
            })

    def panorama_exists(self):
        possible_paths = [
            self.output_path,
            'data/output/panorama.jpg',
            'data/output/panorama.png',
            'data/panorama.jpg',
            'panorama.jpg'
        ]

        for path in possible_paths:
            if Path(path).resolve().exists():
                return True
        return False

    def update_output_path(self, new_path):
        self.output_path = new_path
        self.logger.info(f"Updated panorama output path to: {new_path}")

    def ensure_static_dirs(self):
        static_dir = Path(__file__).parent.parent / 'static'
        templates_dir = Path(__file__).parent.parent / 'templates'

        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(templates_dir, exist_ok=True)

        os.makedirs(static_dir / 'js', exist_ok=True)
        os.makedirs(static_dir / 'css', exist_ok=True)
        os.makedirs(static_dir / 'lib', exist_ok=True)

        placeholder_path = static_dir / 'placeholder.jpg'
        if not placeholder_path.exists():
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (2048, 1024), color=(73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text((50, 50), "No panorama found. Please process images first.", fill=(255, 255, 255))
                img.save(placeholder_path)
            except ImportError:
                pass

    def run(self, debug=False, threaded=True):
        self.ensure_static_dirs()

        project_root = Path(__file__).parent.parent.parent
        self.app = Flask(__name__,
                        template_folder=str(project_root / 'templates'),
                        static_folder=str(project_root / 'static'))
        self.setup_routes()

        self.app.run(host=self.host, port=self.port, debug=debug, threaded=threaded)

    def run_with_browser(self, debug=False):
        def open_browser():
            import time
            time.sleep(1)
            webbrowser.open(f'http://{self.host}:{self.port}')

        browser_thread = threading.Thread(target=open_browser)
        browser_thread.start()

        self.run(debug=debug)