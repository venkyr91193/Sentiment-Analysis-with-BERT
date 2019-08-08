import sys, os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__))))

from deeplearning.train import Train

def test_all():
  obj = Train('text_emotion.csv',max_seq_len=50,bs=32,labels=['happiness','sadness'])
  obj.initilize_model()
  obj.start_train(epochs=2)
  obj.start_eval()
  obj.save_checkpoint(r'D:\Personal\Models','sample_model')

if __name__ == "__main__":
  test_all()
  print('All tests passed.')
