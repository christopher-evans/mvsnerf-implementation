import unittest
from src.datasets.dtu.datamodule import DtuDataModule

class TestCalculations(unittest.TestCase):
    pass
    # def test_define_transforms(self):
    #     dtu_data_module = DtuDataModule()
    #
    #     self.assertIsNone(dtu_data_module.transforms)
    #     dtu_data_module.define_transforms()
    #     self.assertIsNotNone(dtu_data_module.transforms)

if __name__ == '__main__':
    unittest.main()