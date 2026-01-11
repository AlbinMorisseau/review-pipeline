"""
test_export.py
"""
import pytest
import polars as pl
import json
from pathlib import Path
from src.export import export_category_csvs, export_complementary_csvs


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


class TestExportComplementaryCSVs:
    """Tests for complementary CSV export functionality."""
    
    def test_basic_complement_export(self, tmp_path):
        """Test basic complementary CSV creation."""
        # Create reference dataframe
        df_reference = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "text": ["review 1", "review 2", "review 3", "review 4"],
            "text_cleaned": ["cleaned 1", "cleaned 2", "cleaned 3", "cleaned 4"]
        })
        
        # Create category CSV directory
        category_dir = tmp_path / "categories"
        category_dir.mkdir()
        
        # Create pet.csv with IDs 1 and 3
        pet_df = pl.DataFrame({
            "original_id": [1, 3],
            "review": ["review 1", "review 3"],
            "review_cleaned": ["cleaned 1", "cleaned 3"],
            "chunk": ["chunk 1", "chunk 3"],
            "keywords_found": ["dog", "cat"],
            "status": ["kw_bert_agree", "kw_bert_agree"]
        })
        pet_df.write_csv(category_dir / "pet.csv")
        
        # Create output directory
        output_dir = tmp_path / "output"
        
        # Run function
        export_complementary_csvs(df_reference, "id", "text", category_dir, output_dir)
        
        # Verify file was created
        complement_file = output_dir / "no_pet.csv"
        assert complement_file.exists()
        
        # Verify content: should contain IDs 2 and 4
        df_complement = pl.read_csv(complement_file)
        assert df_complement.height == 2
        assert set(df_complement["original_id"].to_list()) == {2, 4}
        assert "review 2" in df_complement["review"].to_list()
        assert "review 4" in df_complement["review"].to_list()
    
    def test_multiple_complementary_files(self, tmp_path):
        """Test creating complementary files for multiple categories."""
        # Reference data
        df_reference = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "text": [f"review {i}" for i in range(1, 6)],
            "text_cleaned": [f"cleaned {i}" for i in range(1, 6)]
        })
        
        category_dir = tmp_path / "categories"
        category_dir.mkdir()
        
        # Create pet.csv
        pet_df = pl.DataFrame({
            "original_id": [1, 2],
            "review": ["review 1", "review 2"],
            "review_cleaned": ["cleaned 1", "cleaned 2"],
            "chunk": ["chunk", "chunk"],
            "keywords_found": ["dog", "cat"],
            "status": ["kw_bert_agree", "kw_bert_agree"]
        })
        pet_df.write_csv(category_dir / "pet.csv")
        
        # Create child.csv
        child_df = pl.DataFrame({
            "original_id": [3, 4],
            "review": ["review 3", "review 4"],
            "review_cleaned": ["cleaned 3", "cleaned 4"],
            "chunk": ["chunk", "chunk"],
            "keywords_found": ["kid", "children"],
            "status": ["kw_bert_agree", "kw_bert_agree"]
        })
        child_df.write_csv(category_dir / "child.csv")
        
        output_dir = tmp_path / "output"
        
        export_complementary_csvs(df_reference, "id", "text", category_dir, output_dir)
        
        # Verify both complement files exist
        assert (output_dir / "no_pet.csv").exists()
        assert (output_dir / "no_child.csv").exists()
        
        # Verify no_pet.csv contains IDs 3, 4, 5
        df_no_pet = pl.read_csv(output_dir / "no_pet.csv")
        assert df_no_pet.height == 3
        assert set(df_no_pet["original_id"].to_list()) == {3, 4, 5}
        
        # Verify no_child.csv contains IDs 1, 2, 5
        df_no_child = pl.read_csv(output_dir / "no_child.csv")
        assert df_no_child.height == 3
        assert set(df_no_child["original_id"].to_list()) == {1, 2, 5}
    
    def test_skips_missing_category_files(self, tmp_path):
        """Test that missing category files are skipped gracefully."""
        df_reference = pl.DataFrame({
            "id": [1, 2, 3],
            "text": ["review 1", "review 2", "review 3"],
            "text_cleaned": ["cleaned 1", "cleaned 2", "cleaned 3"]
        })
        
        category_dir = tmp_path / "categories"
        category_dir.mkdir()
        
        # Only create pet.csv, not child.csv or handicap.csv
        pet_df = pl.DataFrame({
            "original_id": [1],
            "review": ["review 1"],
            "review_cleaned": ["cleaned 1"],
            "chunk": ["chunk"],
            "keywords_found": ["dog"],
            "status": ["kw_bert_agree"]
        })
        pet_df.write_csv(category_dir / "pet.csv")
        
        output_dir = tmp_path / "output"
        
        # Should not raise an error
        export_complementary_csvs(df_reference, "id", "text", category_dir, output_dir)
        
        # Only no_pet.csv should exist
        assert (output_dir / "no_pet.csv").exists()
        assert not (output_dir / "no_child.csv").exists()
        assert not (output_dir / "no_handicap.csv").exists()
    
    def test_correct_column_mapping(self, tmp_path):
        """Test that columns are correctly renamed in complement file."""
        df_reference = pl.DataFrame({
            "review_id": [1, 2],
            "comment": ["text 1", "text 2"],
            "comment_cleaned": ["clean 1", "clean 2"]
        })
        
        category_dir = tmp_path / "categories"
        category_dir.mkdir()
        
        pet_df = pl.DataFrame({
            "original_id": [1],
            "review": ["text 1"],
            "review_cleaned": ["clean 1"],
            "chunk": ["chunk"],
            "keywords_found": ["dog"],
            "status": ["kw_bert_agree"]
        })
        pet_df.write_csv(category_dir / "pet.csv")
        
        output_dir = tmp_path / "output"
        
        export_complementary_csvs(df_reference, "review_id", "comment", category_dir, output_dir)
        
        df_complement = pl.read_csv(output_dir / "no_pet.csv")
        
        # Verify columns are correctly named
        assert "original_id" in df_complement.columns
        assert "review" in df_complement.columns
        assert "review_cleaned" in df_complement.columns
        
        # Verify values
        assert df_complement["original_id"][0] == 2
        assert df_complement["review"][0] == "text 2"
        assert df_complement["review_cleaned"][0] == "clean 2"
    
    def test_empty_category_file(self, tmp_path):
        """Test handling of empty category CSV."""
        df_reference = pl.DataFrame({
            "id": [1, 2, 3],
            "text": ["review 1", "review 2", "review 3"],
            "text_cleaned": ["cleaned 1", "cleaned 2", "cleaned 3"]
        })
        
        category_dir = tmp_path / "categories"
        category_dir.mkdir()
        
        # Create empty pet.csv with proper schema
        empty_df = pl.DataFrame(
            schema={
                "original_id": pl.Int64,
                "review": pl.Utf8,
                "review_cleaned": pl.Utf8,
                "chunk": pl.Utf8,
                "keywords_found": pl.Utf8,
                "status": pl.Utf8
            }
        )
        empty_df.write_csv(category_dir / "pet.csv")
        
        output_dir = tmp_path / "output"
        
        export_complementary_csvs(df_reference, "id", "text", category_dir, output_dir)
        
        # All rows should be in complement
        df_complement = pl.read_csv(output_dir / "no_pet.csv")
        assert df_complement.height == 3
        assert set(df_complement["original_id"].to_list()) == {1, 2, 3}
    
    def test_output_directory_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        df_reference = pl.DataFrame({
            "id": [1, 2],
            "text": ["review 1", "review 2"],
            "text_cleaned": ["cleaned 1", "cleaned 2"]
        })
        
        category_dir = tmp_path / "categories"
        category_dir.mkdir()
        
        pet_df = pl.DataFrame({
            "original_id": [1],
            "review": ["review 1"],
            "review_cleaned": ["cleaned 1"],
            "chunk": ["chunk"],
            "keywords_found": ["dog"],
            "status": ["kw_bert_agree"]
        })
        pet_df.write_csv(category_dir / "pet.csv")
        
        # Output directory doesn't exist yet
        output_dir = tmp_path / "deep" / "nested" / "output"
        
        export_complementary_csvs(df_reference, "id", "text", category_dir, output_dir)
        
        # Verify directory was created
        assert output_dir.exists()
        assert (output_dir / "no_pet.csv").exists()


class TestIntegration:
    """Integration tests for the complete export workflow."""
    
    def test_full_export_workflow(self, tmp_path):
        """Test the complete workflow from category export to complementary export."""
        # Step 1: Create category CSVs
        data = [
            {
                "original_id": 1,
                "review": "pet friendly hotel",
                "review_cleaned": "pet friendly hotel",
                "chunk": "chunk1",
                "keywords_found": "dog, pet",
                "bert_prediction": {"pet": "yes", "child": "no"},
                "status": "kw_bert_agree"
            },
            {
                "original_id": 2,
                "review": "great for kids",
                "review_cleaned": "great for kids",
                "chunk": "chunk2",
                "keywords_found": "kid, children",
                "bert_prediction": {"pet": "no", "child": "yes"},
                "status": "kw_bert_agree"
            },
            {
                "original_id": 3,
                "review": "wheelchair accessible",
                "review_cleaned": "wheelchair accessible",
                "chunk": "chunk3",
                "keywords_found": "wheelchair",
                "bert_prediction": {"pet": "no", "child": "no", "handicap": "yes"},
                "status": "kw_bert_agree"
            },
            {
                "original_id": 4,
                "review": "regular hotel",
                "review_cleaned": "regular hotel",
                "chunk": "chunk4",
                "keywords_found": "hotel",
                "bert_prediction": {"pet": "no", "child": "no", "handicap": "no"},
                "status": "kw_bert_agree"
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text(json.dumps({
            "pet": ["dog", "pet"],
            "child": ["kid", "children"],
            "handicap": ["wheelchair"]
        }))
        
        category_dir = tmp_path / "categories"
        category_dir.mkdir()
        
        # Export category CSVs
        export_category_csvs(data, category_dir, categories_path)
        
        # Verify category CSVs were created
        assert (category_dir / "pet.csv").exists()
        assert (category_dir / "child.csv").exists()
        assert (category_dir / "handicap.csv").exists()
        
        # Step 2: Create reference dataframe from original data
        df_reference = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "text": [d["review"] for d in data],
            "text_cleaned": [d["review_cleaned"] for d in data]
        })
        
        output_dir = tmp_path / "complements"
        
        # Export complementary CSVs
        export_complementary_csvs(df_reference, "id", "text", category_dir, output_dir)
        
        # Step 3: Verify complementary CSVs
        df_no_pet = pl.read_csv(output_dir / "no_pet.csv")
        assert df_no_pet.height == 3
        assert set(df_no_pet["original_id"].to_list()) == {2, 3, 4}
        
        df_no_child = pl.read_csv(output_dir / "no_child.csv")
        assert df_no_child.height == 3
        assert set(df_no_child["original_id"].to_list()) == {1, 3, 4}
        
        df_no_handicap = pl.read_csv(output_dir / "no_handicap.csv")
        assert df_no_handicap.height == 3
        assert set(df_no_handicap["original_id"].to_list()) == {1, 2, 4}
    
    def test_workflow_with_overlapping_categories(self, tmp_path):
        """Test workflow where a review belongs to multiple categories."""
        data = [
            {
                "original_id": 1,
                "review": "pet and kid friendly",
                "review_cleaned": "pet and kid friendly",
                "chunk": "chunk1",
                "keywords_found": "dog, kid",
                "bert_prediction": {"pet": "yes", "child": "yes"},
                "status": "kw_bert_agree"
            },
            {
                "original_id": 2,
                "review": "only pets allowed",
                "review_cleaned": "only pets allowed",
                "chunk": "chunk2",
                "keywords_found": "pet",
                "bert_prediction": {"pet": "yes", "child": "no"},
                "status": "kw_bert_agree"
            },
            {
                "original_id": 3,
                "review": "no special features",
                "review_cleaned": "no special features",
                "chunk": "chunk3",
                "keywords_found": "special",
                "bert_prediction": {"pet": "no", "child": "no"},
                "status": "kw_bert_agree"
            }
        ]
        
        categories_path = tmp_path / "categories.json"
        categories_path.write_text(json.dumps({
            "pet": ["dog", "pet"],
            "child": ["kid"]
        }))
        
        category_dir = tmp_path / "categories"
        category_dir.mkdir()
        
        export_category_csvs(data, category_dir, categories_path)
        
        df_reference = pl.DataFrame({
            "id": [1, 2, 3],
            "text": [d["review"] for d in data],
            "text_cleaned": [d["review_cleaned"] for d in data]
        })
        
        output_dir = tmp_path / "complements"
        export_complementary_csvs(df_reference, "id", "text", category_dir, output_dir)
        
        # ID 1 is in both pet and child, so should not be in either complement
        df_no_pet = pl.read_csv(output_dir / "no_pet.csv")
        assert set(df_no_pet["original_id"].to_list()) == {3}
        
        df_no_child = pl.read_csv(output_dir / "no_child.csv")
        assert set(df_no_child["original_id"].to_list()) == {2, 3}