import pytest
import polars as pl
from src.cleaning import remove_special_characters, convert_numbers

def test_remove_special_characters():
    data = {"text": ["Hello!!!", "It's me", "user@email.com", "Check #hashtag"]}
    df = pl.DataFrame(data)
    
    cleaned = remove_special_characters(df, "text")
    result = cleaned["text"].to_list()
    
    # Check assertions
    assert result[0] == "Hello" # !!! removed
    assert "It's me" in result[1] # Contraction preserved
    assert "@" not in result[2] # Email removed
    assert "#" not in result[3] # Hashtag removed

def test_convert_numbers():
    data = {"text": ["I have 2 cats", "Room 101"]}
    df = pl.DataFrame(data)
    
    converted = convert_numbers(df, "text")
    result = converted["text"].to_list()
    
    assert "two" in result[0]
    assert "one hundred and one" in result[1]