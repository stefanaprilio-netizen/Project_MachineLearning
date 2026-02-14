import pandas as pd
import pytest
def test_data():
    df = pd.DataFrame({'a': [1, 2]})
    assert len(df) == 2