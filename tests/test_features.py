from pathlib import Path
from utils import features

class TestCreatePolys:

    def test_damage_polys(self):
        file = Path('data/output/dmg/0_pre.tif')
        polys = features.create_polys([file])
        assert len(polys) == 264