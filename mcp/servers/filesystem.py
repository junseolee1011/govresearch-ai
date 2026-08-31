"""Filesystem MCP Server for reading local documents."""

import logging
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    LOGGER = logging.getLogger(__name__)
    LOGGER.warning("PyMuPDF not available, PDF reading will be disabled")


LOGGER = logging.getLogger(__name__)


class FilesystemMCPServer:
    """MCP Server for filesystem operations."""

    def __init__(self, base_path: str = "."):
        """Initialize Filesystem MCP Server.

        Args:
            base_path: Base directory for file operations.
        """
        self.base_path = Path(base_path)
        LOGGER.info(f"Filesystem MCP Server initialized with base path: {base_path}")

    def get_tools(self) -> list[str]:
        """Get list of available tools.

        Returns:
            List of tool names.
        """
        tools = ["read_text_file", "read_markdown", "list_files"]
        if PDF_AVAILABLE:
            tools.append("read_pdf")
        return tools

    def read_text_file(self, file_path: str) -> dict:
        """Read a text file.

        Args:
            file_path: Path to the text file.

        Returns:
            Dictionary with file content and metadata.
        """
        full_path = self.base_path / file_path
        LOGGER.info(f"Reading text file: {full_path}")

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "success": True,
                "content": content,
                "path": str(full_path),
                "size": len(content),
                "type": "text"
            }
        except FileNotFoundError:
            LOGGER.error(f"File not found: {full_path}")
            return {
                "success": False,
                "error": "File not found",
                "path": str(full_path)
            }
        except Exception as e:
            LOGGER.error(f"Error reading file: {e}")
            return {
                "success": False,
                "error": str(e),
                "path": str(full_path)
            }

    def read_markdown(self, file_path: str) -> dict:
        """Read a markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Dictionary with file content and metadata.
        """
        full_path = self.base_path / file_path
        LOGGER.info(f"Reading markdown file: {full_path}")

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "success": True,
                "content": content,
                "path": str(full_path),
                "size": len(content),
                "type": "markdown"
            }
        except FileNotFoundError:
            LOGGER.error(f"File not found: {full_path}")
            return {
                "success": False,
                "error": "File not found",
                "path": str(full_path)
            }
        except Exception as e:
            LOGGER.error(f"Error reading markdown file: {e}")
            return {
                "success": False,
                "error": str(e),
                "path": str(full_path)
            }

    def read_pdf(self, file_path: str) -> dict:
        """Read a PDF file using PyMuPDF.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary with file content and metadata.
        """
        if not PDF_AVAILABLE:
            return {
                "success": False,
                "error": "PyMuPDF not available",
                "path": file_path
            }

        full_path = self.base_path / file_path
        LOGGER.info(f"Reading PDF file: {full_path}")

        try:
            doc = fitz.open(full_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            
            doc.close()
            
            full_text = "\n\n".join(text_content)
            
            return {
                "success": True,
                "content": full_text,
                "path": str(full_path),
                "pages": len(text_content),
                "type": "pdf"
            }
        except FileNotFoundError:
            LOGGER.error(f"PDF file not found: {full_path}")
            return {
                "success": False,
                "error": "File not found",
                "path": str(full_path)
            }
        except Exception as e:
            LOGGER.error(f"Error reading PDF file: {e}")
            return {
                "success": False,
                "error": str(e),
                "path": str(full_path)
            }

    def list_files(self, directory: str = ".") -> dict:
        """List files in a directory.

        Args:
            directory: Directory to list.

        Returns:
            Dictionary with file list.
        """
        full_path = self.base_path / directory
        LOGGER.info(f"Listing files in: {full_path}")

        try:
            files = []
            for item in full_path.iterdir():
                if item.is_file():
                    files.append({
                        "name": item.name,
                        "path": str(item),
                        "size": item.stat().st_size
                    })

            return {
                "success": True,
                "files": files,
                "directory": str(full_path),
                "count": len(files)
            }
        except Exception as e:
            LOGGER.error(f"Error listing files: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory": str(full_path)
            }
