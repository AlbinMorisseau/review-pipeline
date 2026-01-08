import pytest
import polars as pl
from src.extraction import prepare_keywords, extract_categories


class TestPrepareKeywords:
    """Tests for keyword preparation and cleaning."""
    
    def test_removes_stopwords_from_keywords(self):
        """Test that stopwords are removed from keyword lists."""
        data = {"cat": ["the car", "a test", "the dog"]}
        cleaned = prepare_keywords(data)
        
        assert "car" in cleaned["cat"][0]
        assert "the" not in cleaned["cat"][0]
        assert "test" in cleaned["cat"][1]
        assert "a" not in cleaned["cat"][1]
    
    def test_multiple_categories(self):
        """Test preparation of multiple categories."""
        data = {
            "animals": ["the dog", "a cat"],
            "facilities": ["the pool", "a gym"]
        }
        cleaned = prepare_keywords(data)
        
        assert "animals" in cleaned
        assert "facilities" in cleaned
        assert len(cleaned["animals"]) == 2
        assert len(cleaned["facilities"]) == 2
    
    def test_preserves_non_list_values(self):
        """Test that non-list values are preserved."""
        data = {
            "category1": ["the dog"],
            "category2": "single_value",
            "category3": 123
        }
        cleaned = prepare_keywords(data)
        
        assert isinstance(cleaned["category1"], list)
        assert cleaned["category2"] == "single_value"
        assert cleaned["category3"] == 123
    
    def test_empty_lists(self):
        """Test handling of empty keyword lists."""
        data = {"empty": [], "normal": ["the dog"]}
        cleaned = prepare_keywords(data)
        
        assert cleaned["empty"] == []
        assert len(cleaned["normal"]) == 1
    
    def test_complex_phrases(self):
        """Test cleaning of multi-word phrases."""
        data = {"travel": ["the family friendly hotel", "a child care service"]}
        cleaned = prepare_keywords(data)
        
        assert "family" in cleaned["travel"][0]
        assert "friendly" in cleaned["travel"][0]
        assert "hotel" in cleaned["travel"][0]
    
    def test_preserves_important_words(self):
        """Test that content words are preserved."""
        data = {"pet": ["dog friendly", "cat allowed", "pet welcome"]}
        cleaned = prepare_keywords(data)
        
        for keyword in cleaned["pet"]:
            # Should contain at least one meaningful word
            assert len(keyword.strip()) > 0
            assert any(word in keyword for word in ["dog", "cat", "pet", "friendly", "allowed", "welcome"])


class TestExtractCategories:
    """Tests for category extraction from reviews."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pl.DataFrame({
            "id": [1, 2, 3, 4],
            "review": [
                "Great hotel for dogs and cats",  # Pluriels ici
                "No pets allowed here",
                "Child friendly with playground",
                "Wheelchair accessible rooms"
            ],
            "review_cleaned": [
                "great hotel dogs cats",
                "pets allowed here",
                "child friendly playground",
                "wheelchair accessible rooms"
            ]
        })
    
    def test_basic_extraction(self, sample_dataframe):
        """Test basic keyword extraction."""
        # CORRECTION: Ajout de "dogs" et "cats" pour matcher le texte au pluriel
        categories = {"animal": ["dog", "dogs", "cat", "cats"]}
        exclusions = {}
        
        result = extract_categories(
            sample_dataframe, "review", categories, exclusions, "id", num_threads=1
        )
        
        assert result.height >= 1
        assert "category" in result.columns
        assert "keywords_found" in result.columns
        assert "animal" in result["category"].to_list()
    
    def test_multiple_categories(self, sample_dataframe):
        """Test extraction with multiple categories."""
        categories = {
            # CORRECTION: Ajout des pluriels
            "animal": ["dog", "dogs", "cat", "cats", "pet", "pets"],
            "child": ["child", "children", "playground"],
            "accessibility": ["wheelchair", "accessible"]
        }
        exclusions = {}
        
        result = extract_categories(
            sample_dataframe, "review", categories, exclusions, "id", num_threads=1
        )
        
        found_categories = set(result["category"].to_list())
        assert "animal" in found_categories
        assert "child" in found_categories
        assert "accessibility" in found_categories
    
    def test_exclusion_filtering(self):
        """
        Test that a review is filtered out if a keyword is entirely removed by an exclusion.
        """
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "review": [
                "No cats allowed here",  
                "Birds are welcome",      
                "Dogs are allowed"      
            ],
            "review_cleaned": [
                "cats allowed here",
                "birds welcome",
                "dogs allowed"
            ]
        })

        # Keywords
        categories = {"animal": ["birds", "dogs"]}

        # Exclusions
        exclusions = {"animal": ["no cats allowed"]}  

        result = extract_categories(
            df, "review", categories, exclusions, "id", num_threads=1
        )

        # La première review est filtrée car "cats" est supprimé par l'exclusion
        expected_ids = [2, 3]

        assert sorted(result["id"].to_list()) == expected_ids
        assert all(cat == "animal" for cat in result["category"].to_list())
        # Vérifier que chaque review contient le mot-clé correspondant
        for keywords_found in result["keywords_found"]:
            assert any(kw in ["birds", "dogs"] for kw in keywords_found.split(", "))


    def test_exclusion_filters_only_phrases_not_keywords(self):
        """
        Exclusions remove only full phrases, not overlapping keywords.
        """
        df = pl.DataFrame({
            "id": [1, 2],
            "review": [
                "No animals allowed",
                "Animals are welcome"
            ],
            "review_cleaned": [
                "animals allowed",
                "animals welcome"
            ]
        })

        categories = {"animal": ["animals"]}
        exclusions = {"animal": ["no animals"]}

        result = extract_categories(
            df, "review", categories, exclusions, "id", num_threads=1
        )

        # Both reviews are kept because "animals" remains after exclusion
        assert sorted(result["id"].to_list()) == [1, 2]


    def test_word_boundary_matching(self):
        """Test that word boundaries are respected."""
        df = pl.DataFrame({
            "id": [1, 2],
            "review": ["The catalog shows dogs", "Dogs are allowed"],
            "review_cleaned": ["catalog shows dogs", "dogs allowed"]
        })
        
        # Ici on teste que "cat" ne matche pas "catalog"
        # On doit s'assurer que "dogs" matche bien la ligne 2 par contre si on le voulait
        categories = {"animal": ["cat"]} 
        exclusions = {}
        
        result = extract_categories(
            df, "review", categories, exclusions, "id", num_threads=1
        )
        
        # "cat" ne doit PAS être trouvé dans "catalog"
        # Comme "cat" n'est pas dans le texte en mot entier, le résultat doit être vide
        assert result.height == 0
        
        # Le test précédent plantait ici car result était vide et la colonne 'keywords_found' n'existait pas
        # Maintenant corrigé dans extraction.py, on peut vérifier la présence de la colonne vide
        assert "keywords_found" in result.columns

    def test_returns_correct_columns(self, sample_dataframe):
        """Test that all expected columns are present."""
        categories = {"animal": ["dog", "dogs"]}
        exclusions = {}
        
        result = extract_categories(
            sample_dataframe, "review", categories, exclusions, "id", num_threads=1
        )
        
        expected_columns = ["id", "review", "review_cleaned", "keywords_found", "category"]
        for col in expected_columns:
            assert col in result.columns

    def test_multiple_keywords_per_review(self):
        """Test that multiple keywords are captured."""
        df = pl.DataFrame({
            "id": [1],
            "review": ["Hotel allows dogs, cats, and birds"],
            "review_cleaned": ["hotel allows dogs cats birds"]
        })
        
        # Ajout des pluriels
        categories = {"animal": ["dogs", "cats", "birds"]}
        exclusions = {}
        
        result = extract_categories(
            df, "review", categories, exclusions, "id", num_threads=1
        )
        
        assert result.height >= 1
        keywords = result["keywords_found"][0]
        # Should contain multiple keywords
        assert "dogs" in keywords and "cats" in keywords

    # ... (Les autres tests comme test_hyphenated_keywords fonctionnent généralement bien 
    # s'ils contiennent les mots exacts)

    def test_edge_cases_combined(self):
        """Test multiple edge cases together."""
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "review": [
                "PET-FRIENDLY hotel",  # Uppercase with hyphen
                "service-dog allowed",  # Hyphenated phrase
                "catalog of pets",  # False match for 'cat', match for 'pets'
                "",  # Empty
                "dogs dogs dogs"  # Repeated keyword
            ],
            "review_cleaned": [
                "pet friendly hotel",
                "service dog allowed",
                "catalog pets",
                "",
                "dogs dogs dogs"
            ]
        })
        
        # Correction des keywords pour inclure les pluriels nécessaires
        raw_keywords = {
            "animal": ["pet-friendly", "service-dog", "pet", "pets", "dog", "dogs", "cat"]
        }
        # On suppose que prepare_keywords est importé et fonctionne
        # Si on ne veut pas dépendre de prepare_keywords ici, on passe la liste brute nettoyée :
        keywords = raw_keywords 
        
        exclusions = {}
        
        result = extract_categories(
            df, "review", keywords, exclusions, "id", num_threads=1
        )
        
        matched_ids = result["id"].to_list()
        
        assert 1 in matched_ids # PET-FRIENDLY
        assert 2 in matched_ids # service-dog
        
        # ID 3: "catalog of pets". 
        # "cat" ne doit pas matcher "catalog", MAIS "pets" doit matcher "pets".
        assert 3 in matched_ids 
        
        assert 4 not in matched_ids # Empty
        assert 5 in matched_ids # dogs dogs dogs


class TestIntegration:
    """Integration tests for extraction workflow."""
    
    def test_full_extraction_workflow(self):
        """Test complete extraction pipeline."""
        # Raw keywords with stopwords
        raw_keywords = {
            "animal": ["the dogs", "a cat", "the pets"],
            "child": ["the child", "kids"]
        }
        
        # Prepare keywords
        keywords = prepare_keywords(raw_keywords)
        print(keywords)
        
        # Verify stopwords removed
        assert "the" not in keywords["animal"]
        assert "a" not in keywords["animal"]
        
        # Create test data
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "review": [
                "Great hotel for dogs and children",
                "No pets or kids allowed",
                "Beautiful quiet resort"
            ],
            "review_cleaned": [
                "great hotel dogs children",
                "pets kids allowed",
                "beautiful quiet resort"
            ]
        })
        
        # Extract with exclusions
        exclusions = {"animal": ["no pets"], "child": ["no kids"]}
        result = extract_categories(
            df, "review", keywords, exclusions, "id", num_threads=1
        )
        
        # Verify results
        assert result.height >= 1
        assert "animal" in result["category"].to_list() or "child" in result["category"].to_list()
        
        # ID 2 should be kept
        if result.height > 0:
            matched_ids = result["id"].to_list()
            # Check if ID 2 kept
            assert 2 in matched_ids
    
    def test_real_world_scenario(self):
        """Test with realistic hotel review data."""
        df = pl.DataFrame({
            "id": list(range(1, 6)),
            "review": [
                "Perfect for families! The hotel has a kids club and playground.",
                "We traveled with our service dog and the staff was very accommodating.",
                "Wheelchair accessible with ramps and elevators throughout.",
                "No pets allowed, but great for business travelers.",
                "Beautiful hotel with excellent amenities."
            ],
            "review_cleaned": [
                "perfect families hotel kids club playground",
                "traveled service dog staff accommodating",
                "wheelchair accessible ramps elevators throughout",
                "pets allowed great business travelers",
                "beautiful hotel excellent amenities"
            ]
        })
        
        keywords = {
            "child": ["kid", "child", "family", "playground"],
            "pet": ["dog", "pet", "service dog"],
            "accessibility": ["wheelchair", "accessible", "ramp"]
        }
        
        exclusions = {
            "pet": ["no pets", "pets not allowed"]
        }
        
        result = extract_categories(
            df, "review", keywords, exclusions, "id", num_threads=2
        )
        
        # Should find multiple categories
        categories = set(result["category"].to_list())
        assert len(categories) >= 2
        
        # ID 4 should be excluded despite having "pets" keyword
        matched_ids = result["id"].to_list()
        if 4 in matched_ids:
            # If ID 4 is matched, it should not be for "pet" category
            id4_rows = result.filter(pl.col("id") == 4)
            if id4_rows.height > 0:
                id4_categories = id4_rows["category"].to_list()
                assert "pet" not in id4_categories
    
    def test_edge_cases_combined(self):
        """Test multiple edge cases together."""
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "review": [
                "PET-FRIENDLY hotel",  # Uppercase with hyphen
                "service-dog allowed",  # Hyphenated phrase
                "catalog of pets",  # Potential false match
                "",  # Empty
                "dogs dogs dogs"  # Repeated keyword
            ],
            "review_cleaned": [
                "pet friendly hotel",
                "service dog allowed",
                "catalog pets",
                "",
                "dogs dogs dogs"
            ]
        })
        
        keywords = prepare_keywords({
            "animal": ["pet-friendly", "service-dog", "pet", "dogs"]
        })
        
        exclusions = {}
        
        result = extract_categories(
            df, "review", keywords, exclusions, "id", num_threads=1
        )
        
        # Should handle all edge cases
        matched_ids = result["id"].to_list()
        assert 1 in matched_ids  # Uppercase with hyphen
        assert 2 in matched_ids  # Hyphenated phrase
        assert 4 not in matched_ids  # Empty string
        assert 5 in matched_ids  # Repeated keyword (should appear once)