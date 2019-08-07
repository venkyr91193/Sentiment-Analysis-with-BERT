import sys, os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__))))

from deeplearning.data_analyser import DataAnalyser


def test_all():
    obj = DataAnalyser()
    obj.analyse()
  
if __name__ == "__main__":
  test_all()
  print('All tests passed')