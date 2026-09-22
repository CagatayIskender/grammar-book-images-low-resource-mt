"""Small local checks for final handover transformations, with no model loading."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('finalizer', ROOT/'scripts/finalize_gold_v2.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class FinalHandoverTests(unittest.TestCase):
    def test_reference_masks_are_fixed_and_nested(self):
        rows=[{'reference':'valid reference'} for _ in range(87)]
        for i in (37,62,81):rows[i]['reference']='nan'
        a=f.reference_indices(rows);b=f.reference_indices(rows,True)
        self.assertEqual(len(a),84);self.assertEqual(len(b),83)
        self.assertEqual(set(a)-set(b),{11})
        rows[0]['reference']=''
        with self.assertRaises(ValueError):f.reference_indices(rows)

    def test_failures_not_filtered_by_sensitivity(self):
        test=[{'reference':'a valid reference'} for _ in range(87)]
        for i in (37,62,81):test[i]['reference']='nan'
        c=dict(id='base',model='test',method='shot',material='baseline',variant='original',test=test)
        hyps=['' for _ in test];values=[0.5 for _ in test]
        d=dict(c,id='context',material='cheatsheet_txt')
        rows,contrasts=f.sensitivity({'base':(c,hyps,values,values),'context':(d,hyps,values,values)})
        self.assertTrue(all(r['records']==r['empty'] for r in rows))
        self.assertTrue(all(r['delta']==0 for r in contrasts))
        self.assertTrue(all('p' not in r for r in contrasts))

    def test_figure_overflow_is_rejected(self):
        image=f.Image.new('RGB',(100,100),'white')
        with self.assertRaises(ValueError):f.text(f.ImageDraw.Draw(image),(0,0,10,10),'too long')

    def test_tsv_roundtrip_preserves_missing_and_zero(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'table.tsv'
            f.tsv(path,[dict(id='one',count=0,note=''),dict(id='two',count=1,note='present')])
            with path.open() as handle:rows=list(csv.DictReader(handle,delimiter='\t'))
            self.assertEqual(rows[0]['count'],'0');self.assertEqual(rows[0]['note'],'')


if __name__=='__main__':unittest.main()
