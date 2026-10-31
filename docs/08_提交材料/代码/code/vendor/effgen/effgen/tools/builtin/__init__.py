"""
Built-in tools for the effGen framework.

This module contains the standard set of tools that ship with effGen.
"""

from typing import Any

# Import built-in tools (lazy loading handled by registry)
__all__ = [
    # OpenAI native (server-side) tools
    "OpenAIWebSearchTool",
    "OpenAICodeInterpreterTool",
    "OpenAIFileSearchTool",
    # Gemini native (server-side) tools
    "GoogleSearchTool",
    "GeminiUrlContextTool",
    "GeminiCodeExecutionTool",
    # Core tools
    "CodeExecutor",
    "PythonREPL",
    "WebSearch",
    "FileOperations",
    "Calculator",
    "Retrieval",
    "AgenticSearch",
    "BashTool",
    "WeatherTool",
    "JSONTool",
    "DateTimeTool",
    "TextProcessingTool",
    "URLFetchTool",
    "WikipediaTool",
    # Finance
    "StockPriceTool",
    "CurrencyConverterTool",
    "CryptoTool",
    # Data Science
    "DataFrameTool",
    "PlotTool",
    "StatsTool",
    # DevOps
    "GitTool",
    "DockerTool",
    "SystemInfoTool",
    "HTTPTool",
    # Knowledge
    "ArxivTool",
    "ArXivTool",
    "StackOverflowTool",
    "GitHubTool",
    "WolframAlphaTool",
    # Academic research
    "PubMedTool",
    "SemanticScholarTool",
    # RSS + News
    "RSSFeedTool",
    "NewsTool",
    # YouTube
    "YouTubeTranscriptTool",
    "YouTubeMetadataTool",
    # Social
    "RedditTool",
    "HackerNewsTool",
    # Translation + Language Detection
    "TranslateTool",
    "LanguageDetectTool",
    # QR Codes
    "QRGenerateTool",
    "QRReadTool",
    # OCR
    "OCRTool",
    # Audio Transcription
    "AudioTranscribeTool",
    # Image Analysis
    "ImageInfoTool",
    "ImageCaptionTool",
    # Multimodal
    "MultimodalDescribeTool",
    # Document Parsing
    "PDFTool",
    "DOCXTool",
    "ExcelTool",
    # Geo / Weather
    "GeocodeTool",
    "MapsTool",
    # Email (live send/read)
    "EmailSMTPTool",
    "EmailIMAPTool",
    # Webhooks
    "SlackWebhookTool",
    "DiscordWebhookTool",
    # Communication (draft-only, legacy)
    "EmailDraftTool",
    "SlackDraftTool",
    "NotificationTool",
]


def __getattr__(name: str) -> Any:
    """Lazy import of tools."""
    if name in ("GoogleSearchTool", "GeminiUrlContextTool", "GeminiCodeExecutionTool"):
        from .gemini_native import (
            GeminiCodeExecutionTool,
            GeminiUrlContextTool,
            GoogleSearchTool,
        )
        if name == "GoogleSearchTool":
            return GoogleSearchTool
        elif name == "GeminiUrlContextTool":
            return GeminiUrlContextTool
        else:
            return GeminiCodeExecutionTool
    elif name in ("OpenAIWebSearchTool", "OpenAICodeInterpreterTool", "OpenAIFileSearchTool"):
        from .openai_native import (
            OpenAICodeInterpreterTool,
            OpenAIFileSearchTool,
            OpenAIWebSearchTool,
        )
        if name == "OpenAIWebSearchTool":
            return OpenAIWebSearchTool
        elif name == "OpenAICodeInterpreterTool":
            return OpenAICodeInterpreterTool
        else:
            return OpenAIFileSearchTool
    elif name in ("AnthropicBashTool", "AnthropicTextEditorTool", "AnthropicComputerTool"):
        from .anthropic_native import (
            AnthropicBashTool,
            AnthropicComputerTool,
            AnthropicTextEditorTool,
        )
        if name == "AnthropicBashTool":
            return AnthropicBashTool
        elif name == "AnthropicTextEditorTool":
            return AnthropicTextEditorTool
        else:
            return AnthropicComputerTool
    elif name == "CodeExecutor":
        from .code_executor import CodeExecutor
        return CodeExecutor
    elif name == "PythonREPL":
        from .python_repl import PythonREPL
        return PythonREPL
    elif name == "WebSearch":
        from .web_search import WebSearch
        return WebSearch
    elif name == "FileOperations":
        from .file_ops import FileOperations
        return FileOperations
    elif name == "Calculator":
        from .calculator import Calculator
        return Calculator
    elif name == "Retrieval":
        from .retrieval import Retrieval
        return Retrieval
    elif name == "AgenticSearch":
        from .agentic_search import AgenticSearch
        return AgenticSearch
    elif name == "BashTool":
        from .bash_tool import BashTool
        return BashTool
    elif name == "WeatherTool":
        from .weather import WeatherTool
        return WeatherTool
    elif name == "JSONTool":
        from .json_tool import JSONTool
        return JSONTool
    elif name == "DateTimeTool":
        from .datetime_tool import DateTimeTool
        return DateTimeTool
    elif name == "TextProcessingTool":
        from .text_processing import TextProcessingTool
        return TextProcessingTool
    elif name == "URLFetchTool":
        from .url_fetch import URLFetchTool
        return URLFetchTool
    elif name == "WikipediaTool":
        from .wikipedia_tool import WikipediaTool
        return WikipediaTool
    # Finance
    elif name == "StockPriceTool":
        from .finance import StockPriceTool
        return StockPriceTool
    elif name == "CurrencyConverterTool":
        from .finance import CurrencyConverterTool
        return CurrencyConverterTool
    elif name == "CryptoTool":
        from .finance import CryptoTool
        return CryptoTool
    # Data Science
    elif name == "DataFrameTool":
        from .data_analysis import DataFrameTool
        return DataFrameTool
    elif name == "PlotTool":
        from .data_analysis import PlotTool
        return PlotTool
    elif name == "StatsTool":
        from .data_analysis import StatsTool
        return StatsTool
    # DevOps
    elif name == "GitTool":
        from .devops import GitTool
        return GitTool
    elif name == "DockerTool":
        from .devops import DockerTool
        return DockerTool
    elif name == "SystemInfoTool":
        from .devops import SystemInfoTool
        return SystemInfoTool
    elif name == "HTTPTool":
        from .devops import HTTPTool
        return HTTPTool
    # Knowledge
    elif name == "ArxivTool":
        from .knowledge import ArxivTool
        return ArxivTool
    elif name == "ArXivTool":
        from .arxiv import ArXivTool
        return ArXivTool
    elif name == "PubMedTool":
        from .pubmed import PubMedTool
        return PubMedTool
    elif name == "SemanticScholarTool":
        from .semantic_scholar import SemanticScholarTool
        return SemanticScholarTool
    elif name == "RSSFeedTool":
        from .rss import RSSFeedTool
        return RSSFeedTool
    elif name == "NewsTool":
        from .news import NewsTool
        return NewsTool
    elif name == "YouTubeTranscriptTool":
        from .youtube_transcript import YouTubeTranscriptTool
        return YouTubeTranscriptTool
    elif name == "YouTubeMetadataTool":
        from .youtube_metadata import YouTubeMetadataTool
        return YouTubeMetadataTool
    elif name == "RedditTool":
        from .reddit import RedditTool
        return RedditTool
    elif name == "HackerNewsTool":
        from .hackernews import HackerNewsTool
        return HackerNewsTool
    elif name == "TranslateTool":
        from .translate import TranslateTool
        return TranslateTool
    elif name == "LanguageDetectTool":
        from .language_detect import LanguageDetectTool
        return LanguageDetectTool
    elif name == "QRGenerateTool":
        from .qr_generate import QRGenerateTool
        return QRGenerateTool
    elif name == "QRReadTool":
        from .qr_read import QRReadTool
        return QRReadTool
    elif name == "OCRTool":
        from .ocr import OCRTool
        return OCRTool
    elif name == "AudioTranscribeTool":
        from .audio_transcribe import AudioTranscribeTool
        return AudioTranscribeTool
    elif name == "ImageInfoTool":
        from .image_info import ImageInfoTool
        return ImageInfoTool
    elif name == "ImageCaptionTool":
        from .image_caption import ImageCaptionTool
        return ImageCaptionTool
    elif name == "MultimodalDescribeTool":
        from .multimodal_describe import MultimodalDescribeTool
        return MultimodalDescribeTool
    elif name == "PDFTool":
        from .pdf import PDFTool
        return PDFTool
    elif name == "DOCXTool":
        from .docx import DOCXTool
        return DOCXTool
    elif name == "ExcelTool":
        from .excel import ExcelTool
        return ExcelTool
    elif name == "GeocodeTool":
        from .geocode import GeocodeTool
        return GeocodeTool
    elif name == "MapsTool":
        from .maps import MapsTool
        return MapsTool
    elif name == "EmailSMTPTool":
        from .email_smtp import EmailSMTPTool
        return EmailSMTPTool
    elif name == "EmailIMAPTool":
        from .email_imap import EmailIMAPTool
        return EmailIMAPTool
    elif name == "SlackWebhookTool":
        from .slack_webhook import SlackWebhookTool
        return SlackWebhookTool
    elif name == "DiscordWebhookTool":
        from .discord_webhook import DiscordWebhookTool
        return DiscordWebhookTool
    elif name == "StackOverflowTool":
        from .knowledge import StackOverflowTool
        return StackOverflowTool
    elif name == "GitHubTool":
        from .knowledge import GitHubTool
        return GitHubTool
    elif name == "WolframAlphaTool":
        from .knowledge import WolframAlphaTool
        return WolframAlphaTool
    # Communication
    elif name == "EmailDraftTool":
        from .communication import EmailDraftTool
        return EmailDraftTool
    elif name == "SlackDraftTool":
        from .communication import SlackDraftTool
        return SlackDraftTool
    elif name == "NotificationTool":
        from .communication import NotificationTool
        return NotificationTool
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
