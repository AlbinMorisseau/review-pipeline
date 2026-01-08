"""
test_export.py - Tests améliorés pour le module export
"""
import pytest
import polars as pl
import json
from pathlib import Path
from src.export import export_category_csvs


class TestExportCategoryCSVs:
    """Tests for CSV export functionality."""
    
    def test_basic_export(self, tmp_path):
        """Test basic CSV export with single category."""
        data = [
            {
                "original_id": 1,
                "review": "test review",
                "review_cleaned": "test review",
                "chunk": "test chunk",
                "keywords_found": "dog, cat",
                "bert_prediction": {"pet": "yes"},
                "status": "kw_bert_agree"
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text('{"pet": ["dog", "cat"]}')
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        export_category_csvs(data, output_dir, categories_path)
        
        # Verify file was created
        csv_file = output_dir / "pet.csv"
        assert csv_file.exists()
        
        # Verify content
        df = pl.read_csv(csv_file)
        assert df.height == 1
        assert df["original_id"][0] == 1
        assert "dog" in df["keywords_found"][0]
    
    def test_filters_by_bert_prediction(self, tmp_path):
        """Test that only rows with 'yes' BERT predictions are included."""
        data = [
            {
                "original_id": 1,
                "review": "pet friendly",
                "review_cleaned": "pet friendly",
                "chunk": "chunk1",
                "keywords_found": "dog",
                "bert_prediction": {"pet": "yes"},
                "status": "kw_bert_agree"
            },
            {
                "original_id": 2,
                "review": "no pets",
                "review_cleaned": "pets",
                "chunk": "chunk2",
                "keywords_found": "pet",
                "bert_prediction": {"pet": "no"},
                "status": "kw_bert_disputed"
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text('{"pet": ["dog", "pet"]}')
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        export_category_csvs(data, output_dir, categories_path)
        
        df = pl.read_csv(output_dir / "pet.csv")
        assert df.height == 1
        assert df["original_id"][0] == 1
    
    def test_filters_by_keyword_match(self, tmp_path):
        """Test that rows are filtered by keyword match with category."""
        data = [
            {
                "original_id": 1,
                "review": "test",
                "review_cleaned": "test",
                "chunk": "chunk1",
                "keywords_found": "dog, cat",
                "bert_prediction": {"pet": "yes"},
                "status": "kw_bert_agree"
            },
            {
                "original_id": 2,
                "review": "test",
                "review_cleaned": "test",
                "chunk": "chunk2",
                "keywords_found": "wheelchair",
                "bert_prediction": {"pet": "yes"},
                "status": "kw_bert_disputed"
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text('{"pet": ["dog", "cat"]}')
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        export_category_csvs(data, output_dir, categories_path)
        
        df = pl.read_csv(output_dir / "pet.csv")
        # Only the first row has keywords matching the category
        assert df.height == 1
        assert df["original_id"][0] == 1
    
    def test_multiple_categories(self, tmp_path):
        """Test export with multiple categories."""
        data = [
            {
                "original_id": 1,
                "review": "pet friendly",
                "review_cleaned": "pet friendly",
                "chunk": "chunk1",
                "keywords_found": "dog",
                "bert_prediction": {"pet": "yes", "child": "no"},
                "status": "kw_bert_agree"
            },
            {
                "original_id": 2,
                "review": "child friendly",
                "review_cleaned": "child friendly",
                "chunk": "chunk2",
                "keywords_found": "kid",
                "bert_prediction": {"pet": "no", "child": "yes"},
                "status": "kw_bert_agree"
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text('{"pet": ["dog"], "child": ["kid"]}')
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        export_category_csvs(data, output_dir, categories_path)
        
        assert (output_dir / "pet.csv").exists()
        assert (output_dir / "child.csv").exists()
        
        pet_df = pl.read_csv(output_dir / "pet.csv")
        child_df = pl.read_csv(output_dir / "child.csv")
        
        assert pet_df.height == 1
        assert child_df.height == 1
    
    def test_skips_empty_categories(self, tmp_path):
        """Test that empty categories in config are skipped."""
        data = [
            {
                "original_id": 1,
                "review": "test",
                "review_cleaned": "test",
                "chunk": "chunk",
                "keywords_found": "dog",
                "bert_prediction": {"pet": "yes"},
                "status": "kw_bert_agree"
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text('{"pet": ["dog"], "empty": []}')
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        export_category_csvs(data, output_dir, categories_path)
        
        assert (output_dir / "pet.csv").exists()
        assert not (output_dir / "empty.csv").exists()
    
    def test_no_matching_data(self, tmp_path):
        """Test when no data matches any category."""
        data = [
            {
                "original_id": 1,
                "review": "test",
                "review_cleaned": "test",
                "chunk": "chunk",
                "keywords_found": "something",
                "bert_prediction": {"pet": "no"},
                "status": "kw_bert_disputed"
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text('{"pet": ["dog", "cat"]}')
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        export_category_csvs(data, output_dir, categories_path)
        
        # No CSV should be created
        assert not (output_dir / "pet.csv").exists()
    
    def test_default_status_value(self, tmp_path):
        """Test handling of missing status field."""
        data = [
            {
                "original_id": 1,
                "review": "test",
                "review_cleaned": "test",
                "chunk": "chunk",
                "keywords_found": "dog",
                "bert_prediction": {"pet": "yes"}
                # No status field
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text('{"pet": ["dog"]}')
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        export_category_csvs(data, output_dir, categories_path)
        
        df = pl.read_csv(output_dir / "pet.csv")
        assert df["status"][0] == "unknown"