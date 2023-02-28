# python project.py -model FFNN -batch_size 64 -embedding_size 128 -epochs 40 -sequence_size 865 -vocab_size 71882 -dropout 0.3 -lr 0.002
# python project.py -model FFNN_CLASSIFY -batch_size 64 -embedding_size 128 -epochs 40 -sequence_size 865 -vocab_size 71882 -dropout 0.3 -lr 0.002

python project.py -model LSTM -batch_size 64 -embedding_size 512 -epochs 50 -sequence_size 300 -vocab_size 30000 -dropout 0.3 -lr 0.0001
python project.py -model LSTM_CLASSIFY -batch_size 64 -embedding_size 512 -epochs 50 -sequence_size 300 -vocab_size 30000 -dropout 0.3 -lr 0.0001