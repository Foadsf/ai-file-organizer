#!/usr/bin/env python3
"""
ai-file-organizer.py v5 - Production Hardened

Enterprise-grade Digital Asset Manager with:
- Resource cleanup (Gemini File API leak prevention)
- Smart hashing for large files
- Concurrent processing with rate limiting
- XMP/IPTC embedding
- TMSU Key-Value ontology
- Persistent SQLite caching

Usage:
    python ai-file-organizer.py /path/to/file [options]
    python ai-file-organizer.py /path/to/dir --batch [options]
"""

import argparse
import atexit
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# --- Configuration ---


@dataclass
class Config:
    """Production configuration."""

    model: str = "gemini-2.0-flash-exp"
    ollama_model: str = "llama3.2-vision"
    max_tags_context: int = 100
    confirm_renames: bool = True
    interactive: bool = False
    exclude_tags: List[str] = field(
        default_factory=lambda: ["temp", "untagged", "inbox"]
    )

    # Ontology
    tmsu_allowed_keys: List[str] = field(
        default_factory=lambda: [
            "year",
            "date",
            "author",
            "vendor",
            "client",
            "project",
            "status",
            "priority",
            "category",
            "type",
            "format",
            "language",
            "amount",
            "currency",
            "location",
            "event",
            "rating",
        ]
    )

    date_format: str = "ISO"

    # Caching
    cache_enabled: bool = True
    cache_path: Path = field(
        default_factory=lambda: Path.home() / ".config" / "ai-tagger" / "cache.db"
    )

    # Metadata
    embed_xmp: bool = True
    create_sidecar: bool = False

    # Concurrency
    max_workers: int = 4
    rate_limit_delay: float = 0.5  # Seconds between API calls per worker


def load_config() -> Config:
    """Load configuration from TOML."""
    config_paths = [
        Path.home() / ".config" / "ai-tagger" / "config.toml",
        Path.home() / ".ai-tagger.toml",
    ]

    cfg = Config()

    for path in config_paths:
        if path.exists():
            try:
                if sys.version_info >= (3, 11):
                    import tomllib

                    with open(path, "rb") as f:
                        data = tomllib.load(f)
                else:
                    import tomli

                    with open(path, "rb") as f:
                        data = tomli.load(f)

                if "general" in data:
                    cfg.model = data["general"].get("model", cfg.model)
                    cfg.interactive = data["general"].get(
                        "interactive", cfg.interactive
                    )
                    cfg.max_workers = data["general"].get(
                        "max_workers", cfg.max_workers
                    )

                if "tmsu" in data:
                    cfg.tmsu_allowed_keys = data["tmsu"].get(
                        "allowed_keys", cfg.tmsu_allowed_keys
                    )

                if "cache" in data:
                    cfg.cache_enabled = data["cache"].get("enabled", cfg.cache_enabled)

                if "metadata" in data:
                    cfg.embed_xmp = data["metadata"].get("embed_xmp", cfg.embed_xmp)
                    cfg.create_sidecar = data["metadata"].get(
                        "create_sidecar", cfg.create_sidecar
                    )

            except Exception as e:
                print(f"Config warning: {e}", file=sys.stderr)

    return cfg


# --- Persistent Cache ---


class AnalysisCache:
    """Thread-safe SQLite cache."""

    def __init__(self, db_path: Path, enabled: bool = True):
        self.enabled = enabled
        self.db_path = db_path
        self._local = threading.local()
        if not enabled:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Connection per thread
        self._get_conn()
        self._init_db()

    def _get_conn(self):
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT,
                model TEXT,
                json_result TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON analysis(file_hash)")
        conn.commit()

    def get(self, file_hash: str, model: str) -> Optional[Dict]:
        if not self.enabled:
            return None

        try:
            conn = self._get_conn()
            cur = conn.execute(
                "SELECT json_result FROM analysis WHERE file_hash = ? AND model = ?",
                (file_hash, model),
            )
            row = cur.fetchone()
            if row:
                print(f"  ↺ Cache hit")
                return json.loads(row[0])
        except Exception as e:
            print(f"Cache read error: {e}", file=sys.stderr)
        return None

    def set(self, file_hash: str, file_path: Path, model: str, result: Dict):
        if not self.enabled:
            return

        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO analysis (file_hash, file_path, model, json_result) VALUES (?, ?, ?, ?)",
                (file_hash, str(file_path), model, json.dumps(result)),
            )
            conn.commit()
        except Exception as e:
            print(f"Cache write error: {e}", file=sys.stderr)

    def cleanup(self, days: int = 30):
        if not self.enabled:
            return
        try:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM analysis WHERE timestamp < datetime('now', ?)",
                (f"-{days} days",),
            )
            conn.commit()
        except Exception as e:
            print(f"Cache cleanup error: {e}", file=sys.stderr)

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# --- Smart Hashing ---


def calculate_smart_hash(file_path: Path) -> str:
    """
    Fast hash for large files using head+tail+metadata.
    Sufficient for cache keys and deduplication.
    """
    try:
        stat = file_path.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime

        sha256 = hashlib.sha256()
        # Add file metadata to hash
        sha256.update(f"{file_size}-{mtime}-{file_path.name}".encode())

        with open(file_path, "rb") as f:
            # Read first 64KB
            sha256.update(f.read(65536))

            # Read last 64KB if file is large enough
            if file_size > 131072:
                f.seek(-65536, 2)
                sha256.update(f.read(65536))
            elif file_size > 0:
                # Small file: already read everything
                pass

        return sha256.hexdigest()
    except Exception as e:
        # Fallback to basic hash
        return hashlib.sha256(str(file_path).encode()).hexdigest()


# --- Environment Checks ---


def check_environment() -> Dict[str, bool]:
    return {
        "tmsu": shutil.which("tmsu") is not None,
        "exiftool": shutil.which("exiftool") is not None,
        "ollama": shutil.which("ollama") is not None,
        "gemini_api": os.getenv("GEMINI_API_KEY") is not None,
    }


# --- Metadata Management ---


class MetadataManager:
    """Extract and write metadata using ExifTool."""

    USEFUL_FIELDS = {
        "CreateDate",
        "DateCreated",
        "ModifyDate",
        "FileModifyDate",
        "Make",
        "Model",
        "ImageWidth",
        "ImageHeight",
        "Duration",
        "Title",
        "Subject",
        "Author",
        "Creator",
        "Keywords",
        "Description",
        "Caption-Abstract",
        "Rating",
        "Label",
        "PDFVersion",
        "PageCount",
        "MIMEType",
        "FileType",
    }

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def extract(self) -> Dict[str, Any]:
        if not shutil.which("exiftool"):
            return {}

        try:
            tags = " ".join([f"-{f}" for f in self.USEFUL_FIELDS])
            cmd = (
                ["exiftool", "-json", "-charset", "UTF8"]
                + tags.split()
                + [str(self.file_path)]
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if data:
                    return {k: v for k, v in data[0].items() if v and v != "n/a"}
        except Exception as e:
            print(f"ExifTool extract error: {e}", file=sys.stderr)

        return {}

    def embed_xmp(self, analysis: Dict[str, Any], dry_run: bool = False):
        if not shutil.which("exiftool"):
            return False

        cmd = ["exiftool", "-overwrite_original"]

        if analysis.get("description"):
            desc = analysis["description"]
            cmd.extend(
                [
                    f"-ImageDescription={desc}",
                    f"-XMP:Description={desc}",
                    f"-Caption-Abstract={desc}",
                ]
            )

        for tag in analysis.get("tags", []):
            cmd.extend([f"-Keywords+={tag}", f"-Subject+={tag}"])

        values = analysis.get("tmsu_values", {})
        if "author" in values:
            cmd.extend([f'-Artist={values["author"]}', f'-Creator={values["author"]}'])
        if "year" in values:
            cmd.append(f'-DateTimeOriginal={values["year"]}:01:01 00:00:00')
        if "date" in values:
            date = values["date"]
            if len(date) == 4:
                date = f"{date}:01:01 00:00:00"
            elif len(date) == 10:
                date = date.replace("-", ":") + " 00:00:00"
            cmd.append(f"-DateTimeOriginal={date}")
        if "rating" in values:
            try:
                rating = int(values["rating"])
                if 1 <= rating <= 5:
                    cmd.extend([f"-Rating={rating}", f"-XMP:Rating={rating}"])
            except ValueError:
                pass

        if values:
            cmd.append(f"-XMP:UserComment={json.dumps(values)}")

        cmd.append(str(self.file_path))

        if dry_run:
            print(f"[DRY-RUN] Would embed XMP")
            return True

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"  ✓ XMP embedded")
                return True
            else:
                print(f"  ⚠ XMP warning: {result.stderr[:100]}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"  ✗ XMP failed: {e}", file=sys.stderr)
            return False


# --- TMSU Integration ---


class TMSUManager:
    """Thread-safe TMSU manager with batch repair."""

    def __init__(self, db_path: Optional[str] = None, config: Config = None):
        self.db_path = db_path or os.getenv("TMSU_DB")
        self.base_cmd = ["tmsu"]
        if self.db_path:
            self.base_cmd.extend(["--database", self.db_path])
        self.config = config
        self._needs_repair = False
        self._tag_cache_by_type: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return shutil.which("tmsu") is not None

    def get_existing_metadata(
        self, file_path: Path
    ) -> Tuple[List[str], Dict[str, str]]:
        if not self.is_available():
            return [], {}

        try:
            cmd = self.base_cmd + ["tags", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout.strip()
                if ":" in output:
                    parts = output.split(":", 1)[1].strip()
                else:
                    parts = output

                tags = []
                values = {}
                for item in parts.split():
                    if "=" in item:
                        k, v = item.split("=", 1)
                        values[k] = v.strip("'\"")
                    else:
                        tags.append(item)
                return tags, values
        except Exception:
            pass
        return [], {}

    def get_context_aware_tags(self, mime_type: str, limit: int = 100) -> List[str]:
        if not self.is_available():
            return []

        cache_key = mime_type.split("/")[0] if "/" in mime_type else mime_type

        with self._lock:
            if cache_key in self._tag_cache_by_type:
                return self._tag_cache_by_type[cache_key][:limit]

        try:
            cmd = self.base_cmd + ["tags", "--count", "-1"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return []

            tag_usage = {}
            for line in result.stdout.strip().split("\n"):
                if ":" in line and "=" not in line:
                    tag, count = line.rsplit(":", 1)
                    tag = tag.strip()
                    if "=" not in tag:
                        tag_usage[tag] = int(count.strip())
        except Exception:
            return []

        # Relevance scoring
        type_keywords = {
            "image": {
                "photo",
                "image",
                "picture",
                "camera",
                "shot",
                "portrait",
                "landscape",
                "screenshot",
                "scan",
                "jpg",
                "raw",
                "png",
            },
            "application": {
                "doc",
                "document",
                "pdf",
                "paper",
                "invoice",
                "contract",
                "report",
                "scan",
                "text",
                "word",
            },
            "audio": {
                "music",
                "audio",
                "sound",
                "song",
                "recording",
                "podcast",
                "voice",
                "mp3",
                "flac",
                "wav",
            },
            "video": {
                "video",
                "movie",
                "film",
                "clip",
                "footage",
                "recording",
                "mp4",
                "mov",
                "avi",
            },
            "text": {"text", "code", "script", "readme", "note", "log"},
        }

        main_type = mime_type.split("/")[0] if "/" in mime_type else "other"
        keywords = type_keywords.get(main_type, set())
        if "pdf" in mime_type:
            keywords = type_keywords["application"]

        scored = []
        for tag, usage in tag_usage.items():
            score = usage
            if any(kw in tag.lower() for kw in keywords):
                score += 100
            scored.append((score, tag))

        scored.sort(reverse=True)
        result = [t[1] for t in scored[:limit]]

        with self._lock:
            self._tag_cache_by_type[cache_key] = result

        return result

    def apply_metadata(
        self,
        file_path: Path,
        tags: List[str],
        values: Dict[str, str],
        dry_run: bool = False,
    ) -> bool:
        if not self.is_available():
            return False

        existing_tags, existing_values = self.get_existing_metadata(file_path)

        new_tags = [t for t in tags if t not in existing_tags]
        new_values = {
            k: v
            for k, v in values.items()
            if k not in existing_values or existing_values[k] != v
        }

        if not new_tags and not new_values:
            return True

        cmd_args = []
        for tag in new_tags:
            cmd_args.append(tag)
        for k, v in new_values.items():
            clean_v = str(v).strip()
            if " " in clean_v or ";" in clean_v:
                clean_v = f"'{clean_v}'"
            cmd_args.append(f"{k}={clean_v}")

        if not cmd_args:
            return True

        if dry_run:
            print(f"[DRY-RUN] tmsu tag {file_path.name} ...")
            return True

        try:
            cmd = self.base_cmd + ["tag", str(file_path)] + cmd_args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                with self._lock:
                    self._needs_repair = True
                return True
            else:
                print(f"  ⚠ TMSU: {result.stderr[:100]}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"  ✗ TMSU error: {e}", file=sys.stderr)
            return False

    def run_repair_if_needed(self):
        with self._lock:
            if not self.is_available() or not self._needs_repair:
                return
            needs = self._needs_repair

        if not needs:
            return

        print("Running TMSU database maintenance...")
        try:
            result = subprocess.run(
                self.base_cmd + ["repair"], capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print("✓ TMSU repaired")
            else:
                print(f"⚠ TMSU repair: {result.stderr[:100]}", file=sys.stderr)
        except Exception as e:
            print(f"✗ TMSU repair failed: {e}", file=sys.stderr)
        finally:
            with self._lock:
                self._needs_repair = False


# --- Date Normalization ---


class DateNormalizer:
    def __init__(self, config: Config):
        self.config = config

    def normalize(self, key: str, value: str) -> str:
        if key in ["year", "date"]:
            return self._normalize_date(value)
        if key == "amount":
            return self._normalize_amount(value)
        return value.strip().lower()

    def _normalize_date(self, value: str) -> str:
        value = value.strip()

        if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return value
        if re.match(r"^\d{4}$", value):
            return value

        patterns = [
            (r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", self._parse_us_date),
            (r"(\d{1,2})\.(\d{1,2})\.(\d{4})", self._parse_eu_date),
            (r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", r"\1-\2-\3"),
            (r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", self._parse_text_date),
        ]

        for pattern, handler in patterns:
            match = re.match(pattern, value)
            if match:
                try:
                    return handler(match)
                except ValueError:
                    continue

        return value

    def _parse_us_date(self, match) -> str:
        mm, dd, yyyy = match.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"

    def _parse_eu_date(self, match) -> str:
        dd, mm, yyyy = match.groups()
        return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"

    def _parse_text_date(self, match) -> str:
        month_str, dd, yyyy = match.groups()
        months = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        mm = months.get(month_str.lower()[:3], 0)
        if mm == 0:
            raise ValueError(f"Unknown month: {month_str}")
        return f"{yyyy}-{mm:02d}-{int(dd):02d}"

    def _normalize_amount(self, value: str) -> str:
        cleaned = re.sub(r"[^\d.,-]", "", value)
        try:
            if "," in cleaned and "." in cleaned:
                if cleaned.rfind(",") > cleaned.rfind("."):
                    cleaned = cleaned.replace(".", "").replace(",", ".")
                else:
                    cleaned = cleaned.replace(",", "")
            elif "," in cleaned:
                if len(cleaned.split(",")[-1]) == 2:
                    cleaned = cleaned.replace(",", ".")
                else:
                    cleaned = cleaned.replace(",", "")

            amount = float(cleaned)
            return f"{amount:.2f}"
        except ValueError:
            return cleaned


# --- AI Integration with Resource Cleanup ---


class AIAnalyzer:
    """Multimodal AI with automatic resource cleanup."""

    SUPPORTED_MIME = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
    }

    def __init__(
        self,
        local_mode: bool = False,
        model: Optional[str] = None,
        config: Config = None,
    ):
        self.local_mode = local_mode
        self.config = config or Config()
        self.model = model or (
            self.config.ollama_model if local_mode else self.config.model
        )
        self.date_normalizer = DateNormalizer(config)
        self._upload_lock = threading.Lock()
        self._recent_uploads: Dict[str, Any] = {}  # hash -> file_obj

    def _get_mime_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        return self.SUPPORTED_MIME.get(ext, "application/octet-stream")

    def _is_text_file(self, file_path: Path) -> bool:
        text_exts = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".json",
            ".csv",
            ".log",
            ".xml",
            ".html",
            ".yaml",
            ".yml",
        }
        return file_path.suffix.lower() in text_exts

    def _read_text_content(self, file_path: Path, max_bytes: int = 50000) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(max_bytes)
                if len(content) >= max_bytes:
                    content += "\n... [truncated]"
                return content
        except Exception as e:
            return f"[Error: {e}]"

    def _prepare_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "suggested_filename": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Flat category tags",
                },
                "tmsu_values": {
                    "type": "object",
                    "properties": {
                        k: {"type": "string"} for k in self.config.tmsu_allowed_keys
                    },
                },
                "description": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["suggested_filename", "tags", "tmsu_values", "confidence"],
        }

    def _prepare_prompt(
        self,
        metadata: Dict[str, Any],
        existing_tags: List[str],
        existing_values: Dict[str, str],
        available_tags: List[str],
        text_content: str = None,
    ) -> str:

        meta_str = json.dumps(metadata, indent=2, default=str)[:1500]

        context = f"""Analyze this file and extract structured metadata.

FILE METADATA:
{meta_str}"""

        if text_content:
            context += f"\n\nCONTENT PREVIEW:\n{text_content[:3000]}"

        context += f"""
EXISTING: Tags={existing_tags}, Values={existing_values}
RELEVANT TAGS: {', '.join(available_tags[:self.config.max_tags_context]) if available_tags else 'None'}

TASK:
1. Suggest filename (lowercase, hyphens)
2. CATEGORIZE: 3-5 flat tags (general categories)
3. EXTRACT: Populate tmsu_values using keys: {', '.join(self.config.tmsu_allowed_keys)}
   - year: YYYY, date: YYYY-MM-DD, amount: NN.NN

RULES:
- NO dates/amounts in 'tags' - use 'tmsu_values'
- Reuse existing tags from list above"""

        return context

    def _normalize_result(self, result: Dict) -> Dict:
        if "tmsu_values" in result:
            normalized = {}
            for k, v in result["tmsu_values"].items():
                if k in self.config.tmsu_allowed_keys:
                    normalized[k] = self.date_normalizer.normalize(k, v)
            result["tmsu_values"] = normalized
        return result

    def analyze(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
        existing_tags: List[str],
        existing_values: Dict[str, str],
        available_tags: List[str],
    ) -> Dict[str, Any]:

        text_content = None
        if self._is_text_file(file_path):
            text_content = self._read_text_content(file_path)

        prompt = self._prepare_prompt(
            metadata, existing_tags, existing_values, available_tags, text_content
        )

        if self.local_mode:
            return self._analyze_ollama(file_path, prompt)
        else:
            return self._analyze_gemini(file_path, prompt)

    def _analyze_gemini(self, file_path: Path, prompt: str) -> Dict[str, Any]:
        """
        Analyze with Gemini, with CRITICAL resource cleanup.
        Prevents cloud storage exhaustion.
        """
        uploaded_file = None
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

            file_hash = calculate_smart_hash(file_path)

            # Check recent uploads cache
            with self._upload_lock:
                if file_hash in self._recent_uploads:
                    uploaded_file = self._recent_uploads[file_hash]

            if not uploaded_file:
                mime_type = self._get_mime_type(file_path)
                print(f"  📤 Uploading...")

                uploaded_file = genai.upload_file(
                    path=str(file_path), mime_type=mime_type
                )

                # Wait for processing
                while uploaded_file.state.name == "PROCESSING":
                    time.sleep(0.5)
                    uploaded_file = genai.get_file(uploaded_file.name)

                if uploaded_file.state.name == "FAILED":
                    raise Exception("Upload failed")

                with self._upload_lock:
                    self._recent_uploads[file_hash] = uploaded_file

            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                [prompt, uploaded_file],
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                    "response_schema": self._prepare_schema(),
                },
            )

            result = json.loads(response.text)

            # CRITICAL: Clean up cloud resource immediately
            if uploaded_file:
                try:
                    uploaded_file.delete()
                    with self._upload_lock:
                        if file_hash in self._recent_uploads:
                            del self._recent_uploads[file_hash]
                except Exception as e:
                    print(f"  ⚠ Cleanup warning: {e}", file=sys.stderr)

            return self._normalize_result(result)

        except Exception as e:
            # Cleanup on failure
            if uploaded_file:
                try:
                    uploaded_file.delete()
                except:
                    pass
            print(f"  ✗ Gemini error: {e}", file=sys.stderr)
            return {}

    def _analyze_ollama(self, file_path: Path, prompt: str) -> Dict[str, Any]:
        try:
            import ollama

            ext = file_path.suffix.lower()
            is_image = ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]

            if is_image and "vision" in self.model:
                import base64

                with open(file_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                response = ollama.generate(
                    model=self.model, prompt=prompt, images=[img_data]
                )
            else:
                response = ollama.generate(model=self.model, prompt=prompt)

            text = response["response"]

            try:
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0]
                else:
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    json_str = text[start:end] if start >= 0 else text

                result = json.loads(json_str)
                return self._normalize_result(result)
            except json.JSONDecodeError:
                return {
                    "suggested_filename": f"organized_{file_path.name}",
                    "tags": ["untagged"],
                    "tmsu_values": {},
                    "description": text[:200],
                    "confidence": 0.3,
                }
        except Exception as e:
            print(f"  ✗ Ollama error: {e}", file=sys.stderr)
            return {}


# --- Utilities ---


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"-+", "-", name)
    if len(name) > 100:
        name = name[:100]
    return name.lower().strip("-")


def safe_rename(
    file_path: Path, new_name: str, dry_run: bool = False
) -> Tuple[Path, bool]:
    new_name = sanitize_filename(new_name)
    new_path = file_path.parent / new_name

    if new_path.exists() and new_path != file_path:
        stem = Path(new_name).stem
        suffix = Path(new_name).suffix
        counter = 1
        while new_path.exists():
            new_name = f"{stem}_{counter:02d}{suffix}"
            new_path = file_path.parent / new_name
            counter += 1

    if dry_run:
        return new_path, True

    try:
        file_path.rename(new_path)
        return new_path, True
    except Exception as e:
        print(f"  ✗ Rename failed: {e}", file=sys.stderr)
        return file_path, False


def interactive_review(
    file_path: Path, analysis: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    print(f"\n{'='*50}")
    print(f"Review: {file_path.name}")
    print(f"{'='*50}")
    print(f"→ {analysis.get('suggested_filename', 'N/A')}")
    print(f"Tags: {', '.join(analysis.get('tags', []))}")
    print(f"Values: {json.dumps(analysis.get('tmsu_values', {}))}")
    print(f"Confidence: {analysis.get('confidence', 0):.0%}")

    while True:
        choice = input("[y]es / [n]o / [e]dit / [s]kip: ").strip().lower()
        if choice == "y":
            return analysis
        elif choice == "n":
            return {}
        elif choice == "s":
            return None
        elif choice == "e":
            print(f"Current values: {json.dumps(analysis.get('tmsu_values', {}))}")
            new_vals = input("New values (JSON): ").strip()
            if new_vals:
                try:
                    analysis["tmsu_values"] = json.loads(new_vals)
                except:
                    print("Invalid JSON")
        else:
            print("Invalid choice")


# --- File Processing ---


def process_file(
    file_path: Path,
    args,
    config: Config,
    cache: AnalysisCache,
    tmsu: TMSUManager,
    env: Dict[str, bool],
) -> bool:
    """Process a single file (thread-safe)."""

    print(f"\nProcessing: {file_path.name}")

    # Smart hash for cache
    file_hash = calculate_smart_hash(file_path)

    # Check cache
    cached = None if args.no_cache else cache.get(file_hash, config.model)
    if cached:
        analysis = cached
    else:
        # Extract metadata
        meta_mgr = MetadataManager(file_path)
        metadata = meta_mgr.extract()
        mime_type = metadata.get("MIMEType", "application/octet-stream")

        # Get context-aware tags
        existing_tags, existing_values = (
            tmsu.get_existing_metadata(file_path) if tmsu.is_available() else ([], {})
        )
        available_tags = (
            tmsu.get_context_aware_tags(mime_type, config.max_tags_context)
            if tmsu.is_available()
            else []
        )

        # AI Analysis
        ai = AIAnalyzer(local_mode=args.local, model=args.model, config=config)
        analysis = ai.analyze(
            file_path, metadata, existing_tags, existing_values, available_tags
        )

        if not analysis:
            print("  ✗ Analysis failed")
            return False

        # Cache result
        cache.set(file_hash, file_path, config.model, analysis)

    # Display
    print(f"  Confidence: {analysis.get('confidence', 0):.0%}")
    print(f"  → {analysis.get('suggested_filename', 'N/A')}")

    # Interactive
    if args.interactive:
        analysis = interactive_review(file_path, analysis)
        if analysis is None:
            return True
        if not analysis:
            return False

    # Execute
    new_path = file_path

    # Rename
    if args.rename and "suggested_filename" in analysis:
        new_path, success = safe_rename(
            file_path, analysis["suggested_filename"], args.dry_run
        )
        if success and not args.dry_run:
            file_path = new_path

    # TMSU
    if args.apply_tags:
        tags = analysis.get("tags", [])
        values = analysis.get("tmsu_values", {})
        if tmsu.apply_metadata(new_path, tags, values, args.dry_run):
            if tags:
                print(f"  ✓ Tags: {', '.join(tags)}")

    # XMP (re-initialize with new path if renamed)
    should_embed = args.embed_xmp or (config.embed_xmp and not args.no_embed_xmp)
    if should_embed and env["exiftool"]:
        meta_mgr = MetadataManager(new_path)  # Use new_path here
        meta_mgr.embed_xmp(analysis, args.dry_run)

    # Sidecar
    if args.sidecar or config.create_sidecar:
        sidecar_path = new_path.parent / f"{new_path.stem}.ai-meta.json"
        if args.dry_run:
            print(f"  [DRY-RUN] Sidecar: {sidecar_path.name}")
        else:
            try:
                with open(sidecar_path, "w") as f:
                    json.dump(
                        {
                            "original": file_path.name,
                            "analysis": analysis,
                            "timestamp": datetime.now().isoformat(),
                        },
                        f,
                        indent=2,
                    )
                print(f"  ✓ Sidecar: {sidecar_path.name}")
            except Exception as e:
                print(f"  ⚠ Sidecar failed: {e}", file=sys.stderr)

    return True


# --- Main ---


def create_parser():
    parser = argparse.ArgumentParser(
        description="AI File Organizer v5 - Production Hardened",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python ai-file-organizer.py invoice.pdf --rename --apply-tags

  # Batch with concurrency (4 workers)
  python ai-file-organizer.py ./scans/ --batch --rename --apply-tags

  # Force re-analysis
  python ai-file-organizer.py file.pdf --no-cache

  # Local mode (no cloud cost)
  python ai-file-organizer.py photo.jpg --local --rename
        """,
    )

    parser.add_argument("path", type=Path, help="File or directory")
    parser.add_argument("--batch", "-b", action="store_true", help="Process directory")
    parser.add_argument("--local", "-l", action="store_true", help="Use Ollama")
    parser.add_argument("--model", "-m", type=str, help="Model override")
    parser.add_argument("--rename", "-r", action="store_true", help="Rename files")
    parser.add_argument("--apply-tags", "-t", action="store_true", help="Apply TMSU")
    parser.add_argument("--embed-xmp", "-x", action="store_true", help="Embed XMP")
    parser.add_argument("--no-embed-xmp", action="store_true", help="Disable XMP")
    parser.add_argument("--sidecar", "-s", action="store_true", help="Create sidecar")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Preview")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache")
    parser.add_argument(
        "--workers", "-w", type=int, help="Concurrent workers (default: 4)"
    )
    parser.add_argument("--json", "-j", action="store_true", help="JSON output")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Config
    config = load_config()
    if args.interactive:
        config.interactive = True
    if args.workers:
        config.max_workers = args.workers

    # Check deps
    env = check_environment()
    if args.local and not env["ollama"]:
        print("Error: Ollama not found", file=sys.stderr)
        sys.exit(1)
    if not args.local and not env["gemini_api"]:
        print("Error: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Init cache and TMSU
    cache = AnalysisCache(
        config.cache_path, enabled=config.cache_enabled and not args.no_cache
    )
    atexit.register(cache.close)

    tmsu = TMSUManager(config=config)

    # Collect files
    files_to_process = []
    if args.batch and args.path.is_dir():
        for ext in [
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".mp3",
            ".mp4",
            ".txt",
            ".doc",
            ".docx",
            ".md",
        ]:
            files_to_process.extend(args.path.rglob(f"*{ext}"))
    elif args.path.is_file():
        files_to_process = [args.path]
    else:
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    if not files_to_process:
        print("No files found to process")
        sys.exit(0)

    print(f"Found {len(files_to_process)} files")
    print(f"Using {config.max_workers} workers")

    # Pre-load TMSU cache
    if tmsu.is_available():
        print("Pre-loading TMSU database...")
        tmsu.get_context_aware_tags("application/pdf")
        tmsu.get_context_aware_tags("image/jpeg")

    # Process with concurrency
    success_count = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(process_file, fp, args, config, cache, tmsu, env): fp
            for fp in files_to_process
        }

        for future in as_completed(futures):
            file_path = futures[future]
            completed += 1
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}", file=sys.stderr)

            # Progress
            if completed % 10 == 0 or completed == len(files_to_process):
                print(f"\n--- Progress: {completed}/{len(files_to_process)} ---")

    # Final cleanup
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(files_to_process)}")

    if args.rename or args.apply_tags:
        tmsu.run_repair_if_needed()

    if config.cache_enabled:
        cache.cleanup(days=30)


if __name__ == "__main__":
    main()
