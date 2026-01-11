"""
test_translation.py 
"""
import pytest
import polars as pl
from unittest.mock import Mock, patch
from src.translation import detect_and_translate


class TestDetectAndTranslate:
    """Tests for language detection and translation."""
    
    @patch('src.translation.langid.classify')
    def test_detects_english_no_translation(self, mock_langid):
        """Test that English text is not translated."""
        mock_langid.return_value = ('en', 0.99)
        
        df = pl.DataFrame({"text": ["This is English"]})
        result_df, count = detect_and_translate(df, "text", num_threads=1)
        
        assert count == 0
        assert result_df["text"][0] == "This is English"
        assert "detected_lang" not in result_df.columns
    
    @patch('src.translation.GoogleTranslator.translate')
    @patch('src.translation.langid.classify')
    def test_translates_non_english(self, mock_langid, mock_translate):
        """Test that non-English text is translated."""
        mock_langid.return_value = ('fr', 0.99)
        mock_translate.return_value = "Hello"
        
        df = pl.DataFrame({"text": ["Bonjour"]})
        result_df, count = detect_and_translate(df, "text", num_threads=1)
        
        assert count == 1
        assert result_df["text"][0] == "Hello"

    
    @patch('src.translation.langid.classify')
    def test_handles_empty_text(self, mock_langid):
        """Test handling of empty or null text."""
        mock_langid.return_value = ('unknown', 0.0)
        
        df = pl.DataFrame({"text": ["", "  ", None]})
        result_df, count = detect_and_translate(df, "text", num_threads=1)
        
        # Should handle gracefully without crashing
        assert result_df.height == 3
    
    @patch('src.translation.GoogleTranslator.translate')
    @patch('src.translation.langid.classify')
    def test_translation_fallback_on_error(self, mock_langid, mock_translate):
        """Test fallback when translation fails."""
        mock_langid.return_value = ('fr', 0.99)
        mock_translate.side_effect = Exception("Translation API error")
        
        original_text = "Bonjour"
        df = pl.DataFrame({"text": [original_text]})
        result_df, count = detect_and_translate(df, "text", num_threads=1)
        
        # Should return original text on error
        assert result_df["text"][0] == original_text
        assert count == 1  # Still counts as attempted translation
