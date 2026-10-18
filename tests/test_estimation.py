"""Minimal API regression tests: python -m unittest discover -s tests -v."""
import unittest
from fastapi.testclient import TestClient
from app import app

class ValuationTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(app)
        self.data={'address':'臺北市士林區中山北路六段','district':'士林區',
                   'building_type':'華廈','area_ping':32,'age_years':25,
                   'floor':5,'total_floors':7,'rooms':3,'managed':True,
                   'elevator':False,'has_parking':False,
                   'extra_construction':False,'price_month':'2026-10'}
    def test_website_and_model(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertTrue(self.client.get('/api/health').json()['model_ready'])
        m=self.client.get('/api/status').json()
        self.assertGreater(m['records'],900)
        self.assertEqual(m['latest_month'],'2026-08')
        self.assertIn('Gradient Boosting',m['cv'])
    def test_example_matches_report(self):
        res=self.client.post('/api/estimate',json=self.data)
        self.assertEqual(res.status_code,200)
        e=res.json()['estimates']
        self.assertGreater(e['Hedonic OLS']['unit_10k_ping'],20)
        self.assertGreater(e['Gradient Boosting']['unit_10k_ping'],20)
        self.assertTrue(res.json()['is_extrapolated'])
    def test_invalid_inputs_are_rejected(self):
        for edited in ({'floor':12},{'address':'臺北市北投區中山北路六段'},
                       {'address':'臺北市士林區無名路'},
                       {'building_type':'透天厝'},{'price_month':'2027-12'}):
            d={**self.data,**edited}
            self.assertEqual(self.client.post('/api/estimate',json=d).status_code,422)
    def test_parking_cost_and_elevator_disclosure(self):
        base=self.client.post('/api/estimate',json=self.data).json()
        with_parking=self.client.post('/api/estimate',json={**self.data,'has_parking':True,'parking_price_10k':250}).json()
        self.assertEqual(with_parking['parking_source'],'user_input')
        pred=self.client.post('/api/estimate',json={**self.data,'elevator':True}).json()
        self.assertEqual(base['estimates'],pred['estimates'])
        self.assertTrue(any('電梯' in x for x in pred['warnings']))

if __name__=='__main__':unittest.main()
