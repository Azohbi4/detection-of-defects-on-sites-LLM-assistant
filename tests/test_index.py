from PIL import Image

from chantier_defects.data.index import index_codebrim, index_sdnet2018


def _img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


def test_sdnet_layout(tmp_path):
    _img(tmp_path / "D" / "CD" / "7001-1.jpg")
    _img(tmp_path / "D" / "UD" / "7001-2.jpg")
    _img(tmp_path / "W" / "UW" / "7100-5.jpg")
    df = index_sdnet2018(tmp_path).set_index("path")
    assert len(df) == 3
    row = df.loc[(tmp_path / "D" / "CD" / "7001-1.jpg").as_posix()]
    assert row["crack"] == 1 and row["group"] == "deck_7001"
    assert set(df["group"]) == {"deck_7001", "wall_7100"}


def test_codebrim_xml(tmp_path):
    _img(tmp_path / "train" / "defects" / "image_0000005_crop_0000001.png")
    _img(tmp_path / "train" / "background" / "image_0000009_crop_0000002.png")
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "defects.xml").write_text(
        """<Defects>
  <Defect name="image_0000005_crop_0000001.png">
    <Background>0</Background><Crack>1</Crack><Spallation>0</Spallation>
    <Efflorescence>0</Efflorescence><ExposedBars>1</ExposedBars><CorrosionStain>1</CorrosionStain>
  </Defect>
  <Defect name="image_0000009_crop_0000002.png">
    <Background>1</Background><Crack>0</Crack><Spallation>0</Spallation>
    <Efflorescence>0</Efflorescence><ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain>
  </Defect>
</Defects>"""
    )
    df = index_codebrim(tmp_path).set_index("group")
    assert df.loc["image_0000005", ["crack", "exposed_bars", "corrosion_stain", "spallation"]].tolist() == [
        1,
        1,
        1,
        0,
    ]
    assert df.loc["image_0000009", "official_split"] == "train"
