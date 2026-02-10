# AI File Organizer 🧠📁

A production-ready, multimodal Digital Asset Manager (DAM) that leverages AI (Google Gemini / Ollama) to intelligently rename, tag, and organize your files.

Unlike basic file renamers, this tool reads the *actual content* of your files (Images, PDFs, Text) and embeds structured metadata using **TMSU** (Key-Value pairs) and standard **XMP/IPTC** tags for universal interoperability.

## ✨ Features
* **True Multimodality**: Native file uploads to Gemini—it "sees" photos and "reads" PDFs.
* **Ontology Enforcement**: Forces AI to use specific Key-Value pairs (e.g., `year=2024`, `amount=150.00`) instead of flat tag clutter.
* **Universal Interoperability**: Embeds AI-generated metadata directly into files via `exiftool` (compatible with Lightroom, MacOS Finder, Windows Explorer).
* **Cost-Optimized SQLite Cache**: Persistently remembers file hashes so you don't pay API costs twice for the same file.
* **RAG-Lite Context**: Only feeds the AI relevant tags based on the file's MIME type to prevent hallucination.
* **Local Fallback**: Fully supports local, offline execution via Ollama vision models.

## 🛠️ Prerequisites
* **Python 3.11+**
* `google-generativeai` and `ollama` Python packages
* **[ExifTool](https://exiftool.org/)** (For extracting and embedding XMP metadata)
* **[TMSU](https://tmsu.org/)** (For the virtual tagging filesystem)

## 🚀 Usage

```bash
# Set your API Key
export GEMINI_API_KEY="your-key"

# Analyze and preview (Dry Run)
python ai-file-organizer.py invoice.pdf

# Full organization (Rename, tag, and embed XMP)
python ai-file-organizer.py scan.jpg --rename --apply-tags

# Batch process a directory (cost-optimized with cache)
python ai-file-organizer.py ./downloads/ --batch --rename --apply-tags

# Use Local AI (Ollama)
python ai-file-organizer.py document.pdf --local --rename

```

## 📜 License

This project is licensed under the GNU General Public License v3.0 (GPLv3).
