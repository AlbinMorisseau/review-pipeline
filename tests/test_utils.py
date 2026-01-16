"""
test_utils.py 
"""
import pytest
import re
import json
import logging
from pathlib import Path
from src.utils import (
    setup_logger,
    load_json_config,
    normalize_keyword,
    make_regex
)


class TestSetupLogger:
    """Tests for logger setup."""
    
    def test_creates_logger(self, tmp_path):
        """Test that logger is created successfully."""
        logger = setup_logger("test_logger", log_dir=str(tmp_path))
        
        assert logger is not None
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
    
    def test_creates_log_directory(self, tmp_path):
        """Test that log directory is created."""
        log_dir = tmp_path / "new_logs"
        setup_logger("test", log_dir=str(log_dir))
        
        assert log_dir.exists()
    
    def test_creates_log_file(self, tmp_path):
        """Test that log file is created."""
        setup_logger("test", log_dir=str(tmp_path))
        
        log_file = tmp_path / "test.log"
        assert log_file.exists()
    
    def test_logger_handlers(self, tmp_path):
        """Test that console and file handlers are added."""
        logger = setup_logger("test", log_dir=str(tmp_path))
        
        # Should have 2 handlers: console and file
        assert len(logger.handlers) == 2

class TestLoadJsonConfig:
    """Tests for JSON config loading."""
    
    def test_loads_valid_json(self, tmp_path):
        """Test loading valid JSON file."""
        config_file = tmp_path / "config.json"
        config_data = {"category1": ["keyword1", "keyword2"]}
        config_file.write_text(json.dumps(config_data))
        
        result = load_json_config(str(config_file))
        
        assert result == config_data
        assert "category1" in result
        assert len(result["category1"]) == 2
    
    def test_raises_error_for_missing_file(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_json_config("nonexistent_file.json")
    
    def test_loads_empty_json(self, tmp_path):
        """Test loading empty JSON object."""
        config_file = tmp_path / "empty.json"
        config_file.write_text("{}")
        
        result = load_json_config(str(config_file))
        assert result == {}
    
    def test_loads_nested_json(self, tmp_path):
        """Test loading nested JSON structure."""
        config_file = tmp_path / "nested.json"
        config_data = {
            "category1": {
                "keywords": ["word1", "word2"],
                "exclusions": ["exclude1"]
            }
        }
        config_file.write_text(json.dumps(config_data))
        
        result = load_json_config(str(config_file))
        assert "category1" in result
        assert "keywords" in result["category1"]


class TestNormalizeKeyword:
    """Tests for keyword normalization."""
    
    def test_strips_whitespace(self):
        """Test whitespace removal."""
        assert normalize_keyword("  keyword  ") == "keyword"
        assert normalize_keyword("word ") == "word"
        assert normalize_keyword(" word") == "word"
    
    def test_replaces_space_dash_space(self):
        """Test replacement of ' - ' with '-'."""
        assert normalize_keyword("pet - friendly") == "pet-friendly"
        assert normalize_keyword("a - b - c") == "a-b-c"
    
    def test_combined_normalization(self):
        """Test combined whitespace and dash replacement."""
        assert normalize_keyword("  pet - friendly  ") == "pet-friendly"
    
    def test_preserves_regular_hyphens(self):
        """Test that regular hyphens without spaces are preserved."""
        assert normalize_keyword("pet-friendly") == "pet-friendly"
    
    def test_empty_string(self):
        """Test with empty string."""
        assert normalize_keyword("") == ""


class TestMakeRegex:
    """Tests for regex pattern creation."""
    
    def test_simple_word_boundary(self):
        """Test regex for simple single word."""
        pattern = make_regex("dog")
        compiled = re.compile(pattern, re.IGNORECASE)
        
        assert compiled.search("I have a dog")
        assert not compiled.search("doggy")  # Word boundary
    
    def test_phrase_with_spaces(self):
        """Test regex for multi-word phrase."""
        pattern = make_regex("pet friendly")
        compiled = re.compile(pattern, re.IGNORECASE)
        
        assert compiled.search("This is pet friendly")
        assert compiled.search("pet  friendly")  # Multiple spaces
        assert compiled.search("PET FRIENDLY")  # Case insensitive
    
    def test_hyphenated_word(self):
        """Test regex for hyphenated keyword."""
        pattern = make_regex("pet-friendly")
        compiled = re.compile(pattern, re.IGNORECASE)
        
        assert compiled.search("pet-friendly hotel")
        assert compiled.search("pet friendly hotel")  # Also matches space
    
    def test_phrase_with_hyphen(self):
        """Test regex for phrase containing hyphen."""
        pattern = make_regex("service-dog friendly")
        compiled = re.compile(pattern, re.IGNORECASE)
        
        assert compiled.search("service-dog friendly")
        assert compiled.search("service dog friendly")
