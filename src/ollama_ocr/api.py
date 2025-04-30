import os
import json
from dotenv import load_dotenv

from bottle import Bottle, run, request, response, abort
from image_processor import ImageProcessor
from ocr_processor import OCRProcessor
from typing import Dict, Any, List, Union

load_dotenv(".env")

# Example valid tokens
BASE_URL = os.getenv("OLLAMA_BASE_URL")
API_KEY = os.getenv("API_KEY")

valid_tokens = []
raw_valid_tokens = os.getenv("VALID_TOKENS")
try:
    valid_tokens = json.loads(raw_valid_tokens)
except json.JSONDecodeError:
    valid_tokens = []

app = Bottle()

# Token 验证装饰器
def token_required(func):
    def wrapper(*args, **kwargs):
        # 从请求头获取 Token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            abort(401, "Unauthorized: Missing or invalid Token")

        token = auth_header.split(" ")[1]
        if token not in valid_tokens:
            abort(401, "Unauthorized: Invalid Token")

        # Token 验证通过，执行原函数
        return func(*args, **kwargs)

    return wrapper


@app.route("/api/extract", method=["POST"])
@token_required
def extract():
    """
    Extract text from an image using OCR.
    """
    urls = request.json

    processor = ImageProcessor(max_workers=4, timeout=15)
    image_paths = processor.process_urls(urls)

    ocr_processor = OCRProcessor(
        model_name="llama3.2-vision:latest",
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    results = process_batch_images(
        ocr_processor, image_paths.values(), "markdown", True, None, "en"
    )
    return results


def process_batch_images(
    processor: OCRProcessor,
    image_paths,
    format_type: str,
    enable_preprocessing: bool,
    custom_prompt: str,
    language: str,
):
    """Process multiple images and return results"""
    try:
        results = processor.process_batch(
            input_path=image_paths,
            format_type=format_type,
            preprocess=enable_preprocessing,
            custom_prompt=custom_prompt,
            language=language,
        )
        return results
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    run(app, host="localhost", port=8080, debug=True)
