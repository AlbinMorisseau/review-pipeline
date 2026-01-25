import pytest
import polars as pl
from unittest.mock import Mock, MagicMock
from src.chunking import (
    _token_overlap_ratio, 
    _reduce_chunks, 
    create_chunks
)


class TestTokenOverlapRatio:
    """Tests for token overlap calculation."""
    
    def test_partial_overlap(self):
        """Test with partial overlap between token lists."""
        a = [1, 2, 3]
        b = [2, 3, 4]
        ratio = _token_overlap_ratio(a, b)
        assert ratio == pytest.approx(2/3)
    
    def test_complete_overlap(self):
        """Test when one list is subset of another."""
        a = [1, 2, 3]
        b = [1, 2, 3, 4, 5]
        ratio = _token_overlap_ratio(a, b)
        assert ratio == 1.0
    
    def test_no_overlap(self):
        """Test with no common tokens."""
        a = [1, 2, 3]
        b = [4, 5, 6]
        ratio = _token_overlap_ratio(a, b)
        assert ratio == 0.0
    
    def test_empty_lists(self):
        """Test with empty token lists."""
        assert _token_overlap_ratio([], [1, 2]) == 0.0
        assert _token_overlap_ratio([1, 2], []) == 0.0
        assert _token_overlap_ratio([], []) == 0.0
    
    def test_identical_lists(self):
        """Test with identical token lists."""
        a = [1, 2, 3, 4]
        ratio = _token_overlap_ratio(a, a)
        assert ratio == 1.0


class TestReduceChunks:
    """Tests for chunk deduplication logic."""
    
    def test_removes_overlapping_chunks(self):
        """Test that overlapping chunks are merged, keeping the best one."""
        chunks = [
            {"tokens": [1, 2, 3], "kw_count": 1, "text": "a"},
            {"tokens": [2, 3, 4], "kw_count": 2, "text": "b"},
            {"tokens": [10, 11], "kw_count": 1, "text": "c"}
        ]
        result = _reduce_chunks(chunks, overlap_threshold=0.5)
        texts = [c["text"] for c in result]
        
        # Chunk "b" kept (higher kw_count), "a" removed (overlaps with "b")
        assert "b" in texts
        assert "a" not in texts
        # Chunk "c" kept (no overlap)
        assert "c" in texts
        assert len(result) == 2
    
    def test_no_overlaps(self):
        """Test when no chunks overlap."""
        chunks = [
            {"tokens": [1, 2], "kw_count": 1, "text": "a"},
            {"tokens": [5, 6], "kw_count": 1, "text": "b"},
            {"tokens": [9, 10], "kw_count": 1, "text": "c"}
        ]
        result = _reduce_chunks(chunks, overlap_threshold=0.75)
        assert len(result) == 3
    
    def test_empty_input(self):
        """Test with empty chunk list."""
        result = _reduce_chunks([])
        assert result == []
    
    def test_single_chunk(self):
        """Test with single chunk."""
        chunks = [{"tokens": [1, 2, 3], "kw_count": 5, "text": "solo"}]
        result = _reduce_chunks(chunks)
        assert len(result) == 1
        assert result[0]["text"] == "solo"
    
    def test_threshold_boundary(self):
        """Test overlap detection at exact threshold."""
        chunks = [
            {"tokens": [1, 2, 3, 4], "kw_count": 1, "text": "a"},
            {"tokens": [3, 4, 5, 6], "kw_count": 2, "text": "b"}
        ]
        # Overlap: 2 tokens / min(4,4) = 0.5
        result = _reduce_chunks(chunks, overlap_threshold=0.5)
        assert len(result) == 1  # Should merge at threshold
        assert result[0]["text"] == "b"
        
        result = _reduce_chunks(chunks, overlap_threshold=0.51)
        assert len(result) == 2  # Should not merge above threshold

class TestCreateChunks:
    """Tests for main chunking logic."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        
        def mock_call(text, **kwargs):
            # Simple word-based tokenization for testing
            words = text.split()
            tokens = list(range(len(words)))
            offsets = [(i*5, i*5+len(w)) for i, w in enumerate(words)]
            
            result = MagicMock()
            result.input_ids = tokens
            result.offset_mapping = offsets
            result.char_to_token = lambda pos: pos // 5 if pos < len(text) else None
            return result
        
        tokenizer.side_effect = mock_call
        tokenizer.decode = lambda ids, **kwargs: " ".join([f"word{i}" for i in ids])
        
        return tokenizer
    
    def test_basic_chunking(self, mock_tokenizer):
        """Test basic chunk creation with keywords."""
        df = pl.DataFrame({
            "id": [1],
            "review": ["dog friendly hotel with pets allowed"],
            "review_cleaned": ["dog friendly hotel with pets allowed"],
            "keywords_found": "dog friendly",
            "category": "pet"
        })
        
        keywords = {"pet": ["dog", "pet"]}
        exclusions = {"pet": []}
        
        result = create_chunks(df, mock_tokenizer, max_len=10, 
                              keywords=keywords, exclusions=exclusions,id_col="id",review_col="review")
        
        assert len(result) > 0
        assert result[0]["original_id"] == 1
        assert "kw_category" in result[0]
    
    def test_empty_dataframe(self, mock_tokenizer):
        """Test with empty DataFrame."""
        df = pl.DataFrame({"id": [], "review": [], "category": []})
        result = create_chunks(df, mock_tokenizer, max_len=10, 
                              keywords={"pet": ["dog"]}, exclusions={},id_col="id",review_col="review")
        assert len(result) == 0
    
    def test_no_keywords_found(self, mock_tokenizer):
        """Test when no keywords are present in review."""
        df = pl.DataFrame({
            "id": [1],
            "review": ["nice clean room"],
            "category": "pet"
        })
        
        keywords = {"pet": ["dog", "cat"]}
        exclusions = {"pet": []}
        
        result = create_chunks(df, mock_tokenizer, max_len=10,
                              keywords=keywords, exclusions=exclusions,id_col="id",review_col="review")
        
        # Should not create chunks without keyword matches
        assert len(result) == 0
    
    def test_exclusion_filtering(self, mock_tokenizer):
        """Test that exclusion phrases filter out matches."""
        df = pl.DataFrame({
            "id": [1],
            "review": ["no dogs allowed here"],
            "category": "pet"
        })
        
        keywords = {"pet": ["dog"]}
        exclusions = {"pet": ["no dogs"]}
        
        result = create_chunks(df, mock_tokenizer, max_len=10,
                              keywords=keywords, exclusions=exclusions,id_col="id",review_col="review")
        
        # Exclusion should prevent chunk creation
        assert len(result) == 0


class TestIntegration:
    """Integration tests for chunking workflow."""
    
    def test_full_workflow(self, tmp_path, monkeypatch):
        """Test complete chunking workflow from merge to chunk creation."""
        # Setup test files
        kw_file = tmp_path / "kw.csv"
        orig_file = tmp_path / "orig.csv"
        
        pl.DataFrame({
            "id": [1],
            "category": ["pet"],
            "keywords_found": "dog"
        }).write_csv(kw_file)
        
        pl.DataFrame({
            "id": [1],
            "review": ["Great hotel allows dogs and cats"]
        }).write_csv(orig_file)
        