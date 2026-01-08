import pytest
import polars as pl
from src.cleaning import (
    basic_cleaning,
    convert_numbers,
    remove_special_characters,
    remove_stopwords_from_text,
    apply_stopwords_removal
)


class TestBasicCleaning:
    """Tests for basic data cleaning operations."""
    
    def test_removes_nulls(self):
        """Test null removal functionality."""
        df = pl.DataFrame({
            "text": ["hello", None, "world", None],
            "other": [1, 2, 3, 4]
        })
        
        cleaned, missing_count, dup_count = basic_cleaning(df, "text")
        
        assert cleaned.height == 2
        assert missing_count == 2
        assert dup_count == 0
        assert cleaned["text"].null_count() == 0
    
    def test_removes_duplicates(self):
        """Test duplicate removal functionality."""
        df = pl.DataFrame({
            "text": ["hello", "hello", "world", "world"],
            "id": [1, 2, 3, 4]
        })
        
        cleaned, missing_count, dup_count = basic_cleaning(df, "text")
        
        assert cleaned.height == 2
        assert missing_count == 0
        assert dup_count == 2
        assert set(cleaned["text"].to_list()) == {"hello", "world"}
    
    def test_combined_cleaning(self):
        """Test removing both nulls and duplicates."""
        df = pl.DataFrame({
            "text": ["hello", None, "hello", "world", None]
        })
        
        cleaned, missing_count, dup_count = basic_cleaning(df, "text")
        
        assert cleaned.height == 2
        assert missing_count == 2
        assert dup_count == 1
    
    def test_no_cleaning_needed(self):
        """Test when data is already clean."""
        df = pl.DataFrame({"text": ["a", "b", "c"]})
        
        cleaned, missing_count, dup_count = basic_cleaning(df, "text")
        
        assert cleaned.height == 3
        assert missing_count == 0
        assert dup_count == 0
    
    def test_all_nulls(self):
        """Test when all values are null."""
        df = pl.DataFrame({"text": [None, None, None]})
        
        cleaned, missing_count, dup_count = basic_cleaning(df, "text")
        
        assert cleaned.height == 0
        assert missing_count == 3
        assert dup_count == 0
    
    def test_all_duplicates(self):
        """Test when all values are duplicates."""
        df = pl.DataFrame({"text": ["same", "same", "same", "same"]})
        
        cleaned, missing_count, dup_count = basic_cleaning(df, "text")
        
        assert cleaned.height == 1
        assert missing_count == 0
        assert dup_count == 3
    
    def test_preserves_other_columns(self):
        """Test that other columns are preserved."""
        df = pl.DataFrame({
            "text": ["a", "b", None],
            "id": [1, 2, 3],
            "score": [4.5, 3.2, 1.1]
        })
        
        cleaned, _, _ = basic_cleaning(df, "text")
        
        assert "id" in cleaned.columns
        assert "score" in cleaned.columns
        assert cleaned.height == 2


class TestConvertNumbers:
    """Tests for number to word conversion."""
    
    def test_single_digit(self):
        """Test conversion of single digit numbers."""
        df = pl.DataFrame({"text": ["I have 2 cats"]})
        result = convert_numbers(df, "text")
        assert "two" in result["text"][0].lower()
    
    def test_multiple_digits(self):
        """Test conversion of multi-digit numbers."""
        df = pl.DataFrame({"text": ["Room 101"]})
        result = convert_numbers(df, "text")
        converted = result["text"][0].lower()
        assert "one hundred" in converted or "hundred and one" in converted
    
    def test_multiple_numbers_in_text(self):
        """Test multiple numbers in same text."""
        df = pl.DataFrame({"text": ["I stayed 3 nights in room 5"]})
        result = convert_numbers(df, "text")
        text = result["text"][0].lower()
        assert "three" in text
        assert "five" in text
    
    def test_zero(self):
        """Test conversion of zero."""
        df = pl.DataFrame({"text": ["0 problems"]})
        result = convert_numbers(df, "text")
        assert "zero" in result["text"][0].lower()
    
    def test_large_numbers(self):
        """Test conversion of large numbers."""
        df = pl.DataFrame({"text": ["Price is 1000 dollars"]})
        result = convert_numbers(df, "text")
        text = result["text"][0].lower()
        assert "thousand" in text
    
    def test_no_numbers(self):
        """Test text without numbers remains unchanged."""
        df = pl.DataFrame({"text": ["no numbers here"]})
        result = convert_numbers(df, "text")
        assert "no numbers here" == result["text"][0]
    
    def test_null_handling(self):
        """Test handling of null values."""
        df = pl.DataFrame({"text": [None, "test 1"]})
        result = convert_numbers(df, "text")
        assert result["text"][0] is None
        assert "one" in result["text"][1].lower()
    
    def test_consecutive_numbers(self):
        """Test multiple consecutive numbers."""
        df = pl.DataFrame({"text": ["Call 123 or 456"]})
        result = convert_numbers(df, "text")
        text = result["text"][0].lower()
        # Both numbers should be converted
        assert "123" not in text
        assert "456" not in text
    
    def test_decimal_numbers(self):
        """Test that decimals are handled (treats each part separately)."""
        df = pl.DataFrame({"text": ["Price 3.14"]})
        result = convert_numbers(df, "text")
        # num2words converts integers, so 3 and 14 will be converted separately
        assert "3.14" not in result["text"][0]


class TestRemoveSpecialCharacters:
    """Tests for special character removal."""
    
    def test_removes_punctuation(self):
        """Test removal of basic punctuation."""
        df = pl.DataFrame({"text": ["Hello!!!", "World???"]})
        result = remove_special_characters(df, "text")
        
        assert result["text"][0] == "Hello"
        assert result["text"][1] == "World"
    
    def test_preserves_contractions(self):
        """Test that contractions are preserved."""
        df = pl.DataFrame({"text": ["It's great", "We're happy", "Don't worry"]})
        result = remove_special_characters(df, "text")
        
        # Contractions should be preserved in some form
        assert "It" in result["text"][0] or "it" in result["text"][0]
        assert "'s" in result["text"][0] or "s" in result["text"][0]
    
    def test_preserves_curly_quotes_contractions(self):
        """Test contractions with curly quotes."""
        df = pl.DataFrame({"text": ["It's nice", "We're good"]})
        result = remove_special_characters(df, "text")
        
        # Should preserve the contraction structure
        assert "It" in result["text"][0]
        assert "We" in result["text"][1]
    
    def test_removes_emails(self):
        """Test email address removal."""
        df = pl.DataFrame({"text": [
            "Contact user@email.com for info",
            "Email: test.user@domain.co.uk"
        ]})
        result = remove_special_characters(df, "text")
        
        assert "@" not in result["text"][0]
        assert "email.com" not in result["text"][0]
        assert "@" not in result["text"][1]
    
    def test_removes_urls(self):
        """Test URL removal."""
        df = pl.DataFrame({"text": [
            "Visit https://example.com",
            "Check www.test.com",
            "See http://site.org/page"
        ]})
        result = remove_special_characters(df, "text")
        
        for text in result["text"]:
            assert "http" not in text
            assert "www" not in text
            assert ".com" not in text
    
    def test_removes_hashtags(self):
        """Test hashtag removal."""
        df = pl.DataFrame({"text": ["Check #hashtag and #another"]})
        result = remove_special_characters(df, "text")
        
        assert "#" not in result["text"][0]
        assert "hashtag" not in result["text"][0]
    
    def test_handles_possessives(self):
        """Test possessive form handling."""
        df = pl.DataFrame({"text": ["John's hotel", "The dog's bed"]})
        result = remove_special_characters(df, "text")
        
        # Possessive should be converted to "s"
        assert "John s" in result["text"][0] or "Johns" in result["text"][0]
    
    def test_replaces_slashes_and_hyphens(self):
        """Test slash and hyphen replacement."""
        df = pl.DataFrame({"text": ["pet-friendly hotel/resort"]})
        result = remove_special_characters(df, "text")
        
        text = result["text"][0]
        assert "/" not in text
        assert "pet" in text and "friendly" in text
    
    def test_normalizes_whitespace(self):
        """Test multiple whitespace normalization."""
        df = pl.DataFrame({"text": ["too    many     spaces"]})
        result = remove_special_characters(df, "text")
        
        assert "  " not in result["text"][0]
        assert result["text"][0] == "too many spaces"
    
    def test_empty_string(self):
        """Test handling of empty strings."""
        df = pl.DataFrame({"text": ["", "   "]})
        result = remove_special_characters(df, "text")
        
        assert result["text"][0] == ""
        assert result["text"][1] == ""
    
    def test_non_string_input(self):
        """Test handling of non-string input."""
        df = pl.DataFrame({"text": None})
        result = remove_special_characters(df, "text")
        
        assert result["text"][0] == None
    
    def test_multiple_contractions(self):
        """Test text with multiple contractions."""
        df = pl.DataFrame({"text": ["I'm sure we're fine and they've arrived"]})
        result = remove_special_characters(df, "text")
        
        # Multiple contractions should be handled
        text = result["text"][0].lower()
        assert "i" in text and "m" in text
        assert "we" in text and "re" in text
    
    def test_special_characters_combination(self):
        """Test combination of different special characters."""
        df = pl.DataFrame({"text": ["Email: test@site.com!! Visit www.site.com #awesome"]})
        result = remove_special_characters(df, "text")
        
        text = result["text"][0]
        assert "@" not in text
        assert "#" not in text
        assert "!!" not in text
        assert "www" not in text


class TestRemoveStopwordsFromText:
    """Tests for stopword removal from text."""
    
    def test_basic_stopword_removal(self):
        """Test basic stopword removal."""
        text = "the cat is on the mat"
        result = remove_stopwords_from_text(text)
        
        # "the", "is", "on" are stopwords
        assert "cat" in result
        assert "mat" in result
        assert "the" not in result
        assert "is" not in result
    
    def test_preserves_contractions(self):
        """Test that contractions are preserved when keep_exceptions=True."""
        text = "it's don't can't"
        result = remove_stopwords_from_text(text, keep_exceptions=True)
        
        # Contractions should be kept as exceptions
        assert len(result) > 0
    
    def test_removes_contractions_when_disabled(self):
        """Test contraction removal when keep_exceptions=False."""
        text = "it's great but don't worry"
        result = remove_stopwords_from_text(text, keep_exceptions=False)
        
        assert "great" in result
        assert "worry" in result
    
    def test_handles_curly_apostrophes(self):
        """Test handling of curly apostrophes."""
        text = "it's nice"
        result = remove_stopwords_from_text(text)
        
        # Should normalize and handle
        assert len(result) > 0
    
    def test_filters_invalid_tokens(self):
        """Test filtering of tokens starting with apostrophe."""
        text = "hello 'world test"
        result = remove_stopwords_from_text(text)
        
        assert "hello" in result
        assert "test" in result
        assert "'world" not in result
    
    def test_empty_text(self):
        """Test with empty text."""
        assert remove_stopwords_from_text("") == ""
        assert remove_stopwords_from_text(None) == ""
    
    def test_all_stopwords(self):
        """Test text with only stopwords."""
        text = "the and or but"
        result = remove_stopwords_from_text(text)
        
        # Should result in empty or near-empty string
        assert len(result) < 5
    
    def test_preserves_content_words(self):
        """Test that content words are preserved."""
        text = "the beautiful hotel has amazing views"
        result = remove_stopwords_from_text(text)
        
        assert "beautiful" in result
        assert "hotel" in result
        assert "amazing" in result
        assert "views" in result
    
    def test_handles_numbers_in_text(self):
        """Test handling of numbers in text."""
        text = "room 123 is nice"
        result = remove_stopwords_from_text(text)
        
        # Should keep content words
        assert "room" in result or "nice" in result
    
    def test_hyphenated_words(self):
        """Test handling of hyphenated words."""
        text = "pet-friendly and child-care services"
        result = remove_stopwords_from_text(text)
        
        # Hyphenated words should be preserved
        assert "pet" in result or "friendly" in result
        assert "child" in result or "care" in result


class TestApplyStopwordsRemoval:
    """Tests for DataFrame stopword removal."""
    
    def test_creates_cleaned_column(self):
        """Test that a new cleaned column is created."""
        df = pl.DataFrame({"text": ["the cat sat on the mat"]})
        result = apply_stopwords_removal(df, "text")
        
        assert "text" in result.columns
        assert "text_cleaned" in result.columns
        assert result.height == 1
    
    def test_preserves_original(self):
        """Test that original column is preserved."""
        original_text = "the cat is here"
        df = pl.DataFrame({"text": [original_text]})
        result = apply_stopwords_removal(df, "text")
        
        assert result["text"][0] == original_text
        assert result["text_cleaned"][0] != original_text
        assert "cat" in result["text_cleaned"][0]
    
    def test_multiple_rows(self):
        """Test processing multiple rows."""
        df = pl.DataFrame({
            "text": [
                "the cat is sleeping",
                "a dog runs fast",
                "birds fly in the sky"
            ]
        })
        result = apply_stopwords_removal(df, "text")
        
        assert result.height == 3
        assert "text_cleaned" in result.columns
    
    def test_custom_column_name(self):
        """Test with custom column name."""
        df = pl.DataFrame({"review": ["the hotel is nice"]})
        result = apply_stopwords_removal(df, "review")
        
        assert "review" in result.columns
        assert "review_cleaned" in result.columns
    
    def test_handles_nulls(self):
        """Test handling of null values."""
        df = pl.DataFrame({"text": ["the cat", None, "a dog"]})
        result = apply_stopwords_removal(df, "text")
        
        assert result.height == 3
        assert result["text_cleaned"][0] != ""
        assert result["text_cleaned"][1] == "" or result["text_cleaned"][1] is None
    
    def test_empty_after_cleaning(self):
        """Test rows that become empty after stopword removal."""
        df = pl.DataFrame({"text": ["the a an", "dog cat"]})
        result = apply_stopwords_removal(df, "text")
        
        # First row should be nearly empty
        assert len(result["text_cleaned"][0]) < 5
        # Second row should have content
        assert len(result["text_cleaned"][1]) > 0


class TestIntegration:
    """Integration tests for cleaning workflow."""
    
    def test_full_cleaning_pipeline(self):
        """Test complete cleaning workflow."""
        df = pl.DataFrame({
            "id": [1, 2, 2, 3, None],
            "text": [
                "The hotel has 2 stars!!!",
                "Email me@test.com",
                "Email me@test.com",  # Duplicate
                "Visit https://example.com",
                "Missing ID"
            ]
        })
        
        # Step 1: Basic cleaning
        df, missing, dups = basic_cleaning(df, "text")
        assert df.height == 4  # Removed null and duplicate
        
        # Step 2: Convert numbers
        df = convert_numbers(df, "text")
        assert "two" in df.filter(pl.col("id") == 1)["text"][0].lower()
        
        # Step 3: Remove special characters
        df = remove_special_characters(df, "text")
        assert "@" not in " ".join(df["text"].to_list())
        assert "https" not in " ".join(df["text"].to_list())
        
        # Step 4: Remove stopwords
        df = apply_stopwords_removal(df, "text")
        assert "text_cleaned" in df.columns
        
        # Verify final result
        assert df.height == 4
        assert all(text is not None for text in df["text_cleaned"])
    
    def test_preserves_meaningful_content(self):
        """Test that cleaning preserves meaningful content."""
        df = pl.DataFrame({
            "text": ["The pet-friendly hotel welcomes dogs and cats!"]
        })
        
        df = remove_special_characters(df, "text")
        df = apply_stopwords_removal(df, "text")
        
        cleaned = df["text_cleaned"][0].lower()
        assert "pet" in cleaned or "friendly" in cleaned
        assert "hotel" in cleaned
        assert "dog" in cleaned or "cat" in cleaned
    
    def test_real_world_review_cleaning(self):
        """Test with realistic review data."""
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "review": [
                "I stayed here for 3 nights with my 2 dogs. It's pet-friendly!!!",
                "Check out www.hotel.com for more info. Email: info@hotel.com",
                "The hotel's location is great but the service wasn't good"
            ]
        })
        
        # Full pipeline
        df, _, _ = basic_cleaning(df, "review")
        df = convert_numbers(df, "review")
        df = remove_special_characters(df, "review")
        df = apply_stopwords_removal(df, "review")
        
        # Verify transformations
        assert df.height == 3
        
        # Numbers converted
        assert "three" in df.filter(pl.col("id") == 1)["review"][0].lower()
        assert "two" in df.filter(pl.col("id") == 1)["review"][0].lower()
        
        # URLs/emails removed
        assert "@" not in df["review"][1]
        assert "www" not in df["review"][1]
        
        # Stopwords removed in cleaned version
        cleaned_text = " ".join(df["review_cleaned"].to_list()).lower()
        assert "hotel" in cleaned_text
        assert "location" in cleaned_text
    
    def test_handles_edge_cases(self):
        """Test pipeline with various edge cases."""
        df = pl.DataFrame({
            "text": [
                "",  # Empty
                "   ",  # Whitespace only
                "123",  # Only numbers
                "THE AND OR",  # Only stopwords
                "Normal text here"
            ]
        })
        
        df, _, _ = basic_cleaning(df, "text")
        df = convert_numbers(df, "text")
        df = remove_special_characters(df, "text")
        df = apply_stopwords_removal(df, "text")
        
        # Should handle all cases without errors
        assert df.height > 0
        assert "text_cleaned" in df.columns
    
    def test_maintains_data_integrity(self):
        """Test that IDs and metadata are preserved through pipeline."""
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "text": ["Review 1", "Review 2", "Review 3"],
            "rating": [4.5, 3.0, 5.0]
        })
        
        df, _, _ = basic_cleaning(df, "text")
        df = convert_numbers(df, "text")
        df = remove_special_characters(df, "text")
        df = apply_stopwords_removal(df, "text")
        
        # All original columns should be preserved
        assert "id" in df.columns
        assert "rating" in df.columns
        assert df.height == 3
