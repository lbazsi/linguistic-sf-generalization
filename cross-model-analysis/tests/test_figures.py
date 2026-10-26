"""Publication output smoke tests using compact, synthetic fixtures."""
from __future__ import annotations
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cross_model import FAMILIES, FEATURES, write_csv, read_csv

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from figures import FigurePublisher, style, plot_forest, plot_replication, plot_correlations
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,"Matplotlib is not installed")
class FigureTests(unittest.TestCase):
    def test_publication_exports(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            stat=folder/"behavioral"
            records=[]
            for feature in FEATURES:
                for i,m in enumerate(FAMILIES):
                    records.append({"feature":feature,"model":m,"mean":-.02*(i+1),
                                    "ci_low":-.04*(i+1),"ci_high":0.0,
                                    "q_value":".1" if i%2 else ".01",
                                    "control":"canonical"})
            write_csv(stat/"per_family_effects.csv",records)
            correlations=[]
            for scope in ("all_14","within_english_11"):
                for i,a in enumerate(FAMILIES):
                    for b in list(FAMILIES)[i+1:]:
                        correlations.append({"scope":scope,"first":a,"second":b,
                                             "pearson":".75","n_features":"11"})
            write_csv(stat/"cross_family_correlations.csv",correlations)
            with matplotlib.rc_context(style()):
                pub=FigurePublisher(folder,dpi=120)
                plot_forest(pub,folder)
                plot_replication(pub,folder)
                plot_correlations(pub,folder)
            self.assertEqual(len(pub.entries),3)
            for item in pub.entries:
                for extension in (".pdf",".svg",".png",".csv"):
                    file=folder/"figures"/"behavioral"/(item["figure"]+extension)
                    self.assertTrue(file.exists(),file)
                    self.assertGreater(file.stat().st_size,50)

    def test_missing_mechanism_is_explicit(self):
        from mechanistic_figures import generate_mechanistic
        with tempfile.TemporaryDirectory() as td:
            with matplotlib.rc_context(style()):
                pub=FigurePublisher(Path(td),dpi=100)
                generate_mechanistic(pub,Path(td))
            self.assertEqual(len(pub.entries),7)
            self.assertTrue(all(x.get("status")=="not_generated" for x in pub.entries))

if __name__=="__main__":
    unittest.main()
