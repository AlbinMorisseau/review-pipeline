"""
test_validation.py
"""
import pytest
from src.validation import compare_results, get_stats


class TestCompareResults:
    """Tests for comparing keyword and BERT predictions."""
    
    def test_agreement(self):
        """Test when keywords and BERT predictions agree."""
        data = [
            {
                "chunk": "test chunk",
                "kw_category": {"pet": "yes", "child": "no"},
                "bert_prediction": {"pet": "yes", "child": "no"}
            }
        ]
        
        result = compare_results(data)
        assert result[0]["status"] == "kw_bert_agree"
    
    def test_disagreement(self):
        """Test when keywords and BERT predictions disagree."""
        data = [
            {
                "chunk": "test chunk",
                "kw_category": {"pet": "yes", "child": "no"},
                "bert_prediction": {"pet": "no", "child": "yes"}
            }
        ]
        
        result = compare_results(data)
        assert result[0]["status"] == "kw_bert_disputed"
    
    def test_partial_match(self):
        """Test partial agreement between predictions."""
        data = [
            {
                "chunk": "test",
                "kw_category": {"pet": "yes", "child": "no", "accessibility": "yes"},
                "bert_prediction": {"pet": "yes", "child": "no", "accessibility": "no"}
            }
        ]
        
        result = compare_results(data)
        # Should be disputed if any category disagrees
        assert result[0]["status"] == "kw_bert_disputed"
    
    def test_multiple_items(self):
        """Test with multiple data items."""
        data = [
            {
                "kw_category": {"pet": "yes"},
                "bert_prediction": {"pet": "yes"}
            },
            {
                "kw_category": {"pet": "no"},
                "bert_prediction": {"pet": "yes"}
            }
        ]
        
        result = compare_results(data)
        assert result[0]["status"] == "kw_bert_agree"
        assert result[1]["status"] == "kw_bert_disputed"
    
    def test_empty_predictions(self):
        """Test with empty prediction dictionaries."""
        data = [
            {
                "kw_category": {},
                "bert_prediction": {}
            }
        ]
        
        result = compare_results(data)
        assert result[0]["status"] == "kw_bert_agree"
    
    def test_missing_fields(self):
        """Test handling of missing prediction fields."""
        data = [
            {
                "chunk": "test"
                # Missing kw_category and bert_prediction
            }
        ]
        
        result = compare_results(data)
        # Should handle gracefully
        assert "status" in result[0]


class TestGetStats:
    """Tests for statistics gathering."""
    
    def test_basic_stats(self):
        """Test basic statistics calculation."""
        data = [
            {"status": "kw_bert_agree"},
            {"status": "kw_bert_agree"},
            {"status": "kw_bert_disputed"}
        ]
        
        stats = get_stats(data)
        
        assert stats["kw_bert_agree"] == 2
        assert stats["kw_bert_disputed"] == 1
    
    def test_empty_data(self):
        """Test with empty data list."""
        stats = get_stats([])
        assert stats == {}
    
    def test_unknown_status(self):
        """Test handling of missing or unknown status."""
        data = [
            {"status": "kw_bert_agree"},
            {},  # Missing status
            {"status": "custom_status"}
        ]
        
        stats = get_stats(data)
        
        assert "unknown" in stats
        assert stats["unknown"] == 1
        assert "custom_status" in stats
    
    def test_all_same_status(self):
        """Test when all items have same status."""
        data = [{"status": "kw_bert_agree"} for _ in range(5)]
        
        stats = get_stats(data)
        
        assert len(stats) == 1
        assert stats["kw_bert_agree"] == 5
    
    def test_multiple_status_types(self):
        """Test with various status types."""
        data = [
            {"status": "kw_bert_agree"},
            {"status": "kw_bert_disputed"},
            {"status": "kw_bert_agree"},
            {"status": "manual_review"},
            {"status": "kw_bert_disputed"}
        ]
        
        stats = get_stats(data)
        
        assert stats["kw_bert_agree"] == 2
        assert stats["kw_bert_disputed"] == 2
        assert stats["manual_review"] == 1
        assert len(stats) == 3


class TestIntegration:
    """Integration tests for validation workflow."""
    
    def test_full_validation_workflow(self):
        """Test complete validation pipeline."""
        # Simulate data from chunking and BERT prediction
        data = [
            {
                "chunk": "Pet friendly hotel",
                "kw_category": {"pet": "yes", "child": "no"},
                "bert_prediction": {"pet": "yes", "child": "no"}
            },
            {
                "chunk": "No pets allowed",
                "kw_category": {"pet": "yes", "child": "no"},
                "bert_prediction": {"pet": "no", "child": "no"}
            },
            {
                "chunk": "Great for families",
                "kw_category": {"pet": "no", "child": "yes"},
                "bert_prediction": {"pet": "no", "child": "yes"}
            }
        ]
        
        # Compare results
        validated = compare_results(data)
        
        # Get statistics
        stats = get_stats(validated)
        
        assert stats["kw_bert_agree"] == 2
        assert stats["kw_bert_disputed"] == 1
        assert sum(stats.values()) == 3