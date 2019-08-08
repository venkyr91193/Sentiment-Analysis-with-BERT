import sys, os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__))))

from oops.algorithms import Algorithms

def test_run_algorithms():
  obj = Algorithms()
  obj.run_algorithms()
  assert(obj.result != None)

if __name__ == "__main__":
  test_run_algorithms()
  print('All tests passed')