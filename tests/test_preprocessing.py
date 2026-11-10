import unittest

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_data


class PreprocessingTests(unittest.TestCase):
    def test_scaler_and_smote_use_training_rows_only(self):
        rng = np.random.default_rng(7)
        n = 120
        frame = pd.DataFrame({
            'Time': np.arange(n, dtype=float) ** 2,
            'Amount': rng.lognormal(3, 1, n),
            'V1': rng.normal(size=n),
            'Class': np.r_[np.zeros(96, dtype=int), np.ones(24, dtype=int)],
        })
        train, test, y_train, y_test, balanced, y_balanced = preprocess_data(frame)
        self.assertAlmostEqual(train['Time'].mean(), 0, places=12)
        self.assertAlmostEqual(train['Amount'].mean(), 0, places=12)
        self.assertEqual(len(test), len(y_test))
        self.assertEqual(len(balanced), len(y_balanced))
        self.assertEqual(y_balanced.value_counts()[0], y_balanced.value_counts()[1])
        self.assertEqual(set(train.index).intersection(test.index), set())


if __name__ == '__main__':
    unittest.main()
