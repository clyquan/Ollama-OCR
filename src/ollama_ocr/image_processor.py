import requests
import tempfile
import os
import re
import hashlib
import string
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ImageProcessor:
    def __init__(
        self, max_workers: int = 5, timeout: int = 15, keep_temp: bool = False
    ):
        """
        参数优化:
        - max_workers: 并发线程数（必须为正整数）
        - timeout: 请求超时时间（秒）（必须为正数）
        - keep_temp: 是否保留临时目录（用于调试）
        """
        if max_workers <= 0:
            raise ValueError("max_workers必须为正整数")
        if timeout <= 0:
            raise ValueError("timeout必须为正数")

        self.max_workers = max_workers
        self.timeout = timeout
        self.keep_temp = keep_temp

        # 创建临时目录
        self.temp_dir = tempfile.TemporaryDirectory(prefix="imgproc_")
        print(f"创建临时目录：{self.temp_dir.name}")

        # 安全配置
        self.safe_chars = set("-.()_%s%s" % (string.ascii_letters, string.digits))
        self.max_filename_length = 255
        self.min_filename_length = 3

        # MIME类型到扩展名映射（优先级排序）
        self.mime_map = [
            (re.compile(r"image/jpeg"), "jpg"),
            (re.compile(r"image/png"), "png"),
            (re.compile(r"image/gif"), "gif"),
            (re.compile(r"image/webp"), "webp"),
            (re.compile(r"image/bmp"), "bmp"),
            (re.compile(r"image/tiff"), "tiff"),
            (re.compile(r"application/octet-stream"), "bin"),
        ]

        # 图片签名验证配置
        self.signature_checks = [
            (b"\xff\xd8\xff", "image/jpeg", 3),
            (b"\x89PNG\r\n\x1a\n", "image/png", 8),
            (b"GIF87a", "image/gif", 6),
            (b"GIF89a", "image/gif", 6),
            (b"RIFF....WEBP", "image/webp", 12),
            (b"\x42\x4d", "image/bmp", 2),
            (b"\x49\x49\x2a\x00", "image/tiff", 4),
            (b"\x4d\x4d\x00\x2a", "image/tiff", 4),
        ]
        self.max_sig_length = max(s[2] for s in self.signature_checks)

        # 配置带智能重试的Session
        self.session = requests.Session()
        retry_policy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(
            max_retries=retry_policy, pool_connections=100, pool_maxsize=100
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.keep_temp:
            self.temp_dir.cleanup()
            print("已清理临时目录")

    def _sanitize_filename(self, filename: str) -> str:
        """深度清理文件名"""
        # URL解码
        decoded = unquote(filename)

        # 替换危险字符
        cleaned = re.sub(r'[\\/*?:"<>|]', "_", decoded)
        cleaned = re.sub(r"\s+", "_", cleaned).strip()
        cleaned = cleaned.rstrip(".")

        # 处理空文件名
        if not cleaned:
            return f"unnamed_{hashlib.md5(os.urandom(128)).hexdigest()[:8]}"

        # 截断长度并保留有效扩展名
        base, ext = os.path.splitext(cleaned)
        base = base[: self.max_filename_length - len(ext) - 1]
        cleaned = f"{base}{ext}"

        return cleaned[: self.max_filename_length]

    def _get_filename_from_url(self, url: str) -> str:
        """智能提取并清理文件名"""
        parsed = urlparse(url)
        path = parsed.path

        # 处理无扩展名的情况
        if not path or path.endswith("/"):
            path = parsed.netloc.split(".", 1)[0] + ".bin"

        filename = os.path.basename(path)
        return self._sanitize_filename(filename)

    def _detect_mime_type(self, content: bytes) -> Tuple[Optional[str], Optional[str]]:
        """检测MIME类型并返回（类型，扩展名）"""
        # 优先根据内容签名检测
        for sig, mime, _ in self.signature_checks:
            if content.startswith(sig):
                for pattern, ext in self.mime_map:
                    if pattern.match(mime):
                        return mime, ext
                return mime, "bin"

        # 次之根据Content-Type检测
        return None, None

    def _generate_safe_path(self, base_name: str, mime_ext: str) -> str:
        """生成唯一的安全文件路径"""
        base, original_ext = os.path.splitext(base_name)
        original_ext = original_ext.lower().lstrip(".")

        # 优先使用检测到的扩展名
        final_ext = mime_ext if mime_ext else original_ext
        final_ext = final_ext if final_ext else "bin"

        # 生成安全基础名
        safe_base = base if len(base) >= self.min_filename_length else "file"
        safe_base = re.sub(r"[^a-zA-Z0-9_-]", "_", safe_base)

        # 生成唯一文件名
        unique_id = hashlib.md5(os.urandom(64)).hexdigest()[:6]
        filename = f"{safe_base}_{unique_id}.{final_ext}"
        return os.path.join(self.temp_dir.name, filename)

    def _fetch_image(self, url: str) -> Optional[Tuple[bytes, str]]:
        """获取并验证图片内容"""
        try:
            with self.session.get(
                url,
                stream=True,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 ImageProcessor/1.0"},
            ) as response:
                response.raise_for_status()

                content = bytearray()
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        content.extend(chunk)
                        # 提前验证签名
                        if len(content) >= self.max_sig_length:
                            mime, _ = self._detect_mime_type(content)
                            if not mime:
                                print(f"无效的文件签名：{url}")
                                return None
                # 最终验证
                mime, ext = self._detect_mime_type(content)
                if not mime:
                    print(f"无法识别的文件类型：{url}")
                    return None
                return bytes(content), ext
        except requests.RequestException as e:
            print(f"请求失败 [{url}]: {str(e)}")
            return None

    def process_urls(self, urls: List[str]) -> Dict[str, str]:
        """处理URL列表并返回文件路径映射"""
        results = {}
        with ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="ImgDL"
        ) as executor:
            futures = {executor.submit(self._fetch_image, url): url for url in urls}

            for future in as_completed(futures):
                url = futures[future]
                try:
                    result = future.result()
                    if result:
                        content, ext = result
                        filename = self._get_filename_from_url(url)
                        filepath = self._generate_safe_path(filename, ext)

                        try:
                            with open(filepath, "wb") as f:
                                f.write(content)
                            results[url] = filepath
                            print(f"成功保存：{url} → {filepath}")
                        except IOError as e:
                            print(f"文件写入失败 [{url}]: {str(e)}")
                except Exception as e:
                    print(f"处理异常 [{url}]: {str(e)}")
        return results
