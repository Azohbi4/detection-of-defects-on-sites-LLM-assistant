import numpy as np
import pandas as pd
import pytest

from chantier_defects.data.split import check_no_leakage, grouped_split


@pytest.fixture
def index_df():
    rng = np.random.default_rng(1)
    rows = []
    for g in range(120):  # 120 images sources, 10 patches chacune
        cracked = g % 4 == 0
        for p in range(10):
            rows.append(
                {
                    "path": f"data/{g}-{p}.jpg",
                    "group": f"deck_{g}",
                    "crack": int(cracked and rng.random() < 0.7),
                }
            )
    return pd.DataFrame(rows)


def test_no_group_in_two_splits(index_df):
    out = grouped_split(index_df, seed=0)
    assert out.groupby("group")["split"].nunique().max() == 1


def test_proportions_and_stratification(index_df):
    out = grouped_split(index_df, val=0.15, test=0.15, seed=0)
    share = out["split"].value_counts(normalize=True)
    assert 0.08 < share["test"] < 0.25 and 0.08 < share["val"] < 0.25
    assert out.groupby("split")["crack"].sum().min() > 0  # chaque split contient des fissures


def test_reproducible_and_order_independent(index_df):
    a = grouped_split(index_df, seed=42).set_index("path")["split"]
    b = grouped_split(index_df.sample(frac=1, random_state=3), seed=42).set_index("path")["split"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_leakage_is_detected():
    df = pd.DataFrame({"group": ["g1", "g1"], "split": ["train", "test"]})
    with pytest.raises(ValueError):
        check_no_leakage(df)
