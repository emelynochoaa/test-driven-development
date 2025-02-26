import unittest
import numpy as np
from Experiment import Experiment
from SignalDetection import SignalDetection

class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.exp = Experiment()
        self.sdt1 = SignalDetection(40, 10, 20, 30)  # Example condition
        self.sdt2 = SignalDetection(50, 5, 25, 20)
        self.sdt3 = SignalDetection(30, 20, 15, 35)
    
    def test_add_condition(self):
        self.exp.add_condition(self.sdt1, "Condition A")
        self.exp.add_condition(self.sdt2, "Condition B")
        
        self.assertEqual(len(self.exp.conditions), 2)
        self.assertEqual(self.exp.labels, ["Condition A", "Condition B"])
    
    def test_sorted_roc_points(self):
        self.exp.add_condition(self.sdt1)
        self.exp.add_condition(self.sdt2)
        
        sorted_fa, sorted_hr = self.exp.sorted_roc_points()
        self.assertEqual(len(sorted_fa), len(sorted_hr))
        self.assertTrue(np.all(np.diff(sorted_fa) >= 0)) 
    
    def test_compute_auc(self):
        self.exp.add_condition(SignalDetection(0, 1, 0, 1))  
        self.exp.add_condition(SignalDetection(1, 0, 0, 1))  
        self.exp.add_condition(SignalDetection(1, 0, 1, 0)) 
        
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 1.0, places=2)

    def test_compute_auc_1(self):
        self.exp.add_condition(SignalDetection(0, 1, 0, 1))  
        self.exp.add_condition(SignalDetection(1, 0, 1, 0))  
        
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.5, places=2)
    
    def test_compute_auc_2(self):
        self.exp.add_condition(SignalDetection(0, 1, 1, 0))  
        self.exp.add_condition(SignalDetection(1, 0, 0, 1))  
        
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.5, places=2)

    def test_compute_auc_3(self):
        self.exp.add_condition(SignalDetection(0, 1, 0, 1))  
        self.exp.add_condition(SignalDetection(0.5, 0.5, 0.5, 0.5))  
        self.exp.add_condition(SignalDetection(0, 1, 0.5, 0.5)) 
        
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.125, places=3)

    def test_compute_auc_zero(self):
        self.exp.add_condition(SignalDetection(0, 0, 0, 0))
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.0, places=2)

    def test_compute_auc_one_point(self):
        self.exp.add_condition(SignalDetection(1, 1, 1, 1))
        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.0, places=2)

    def test_empty_experiment(self):
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()
        with self.assertRaises(ValueError):
            self.exp.compute_auc()
    
    def test_invalid_add_condition(self):
        with self.assertRaises(TypeError):
            self.exp.add_condition("Not a SignalDetection object")
    
    def test_compute_auc_4(self):
        self.exp.add_condition(SignalDetection(0.5, 1, 0, 0.5))  
        self.exp.add_condition(SignalDetection(1, 0.5, 0.5, 0))  
        self.exp.add_condition(SignalDetection(0, 0, 0.5, 0.5))

        auc = self.exp.compute_auc()
        self.assertAlmostEqual(auc, 0.25, places=3)


if __name__ == '__main__':
    unittest.main()
