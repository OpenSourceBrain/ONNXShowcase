set -e

python squeezenet2onnx.py

python simple.py
python print_onnx_info.py simple.onnx
./onnx_png.sh simple

python rnn.py
python print_onnx_info.py rnn.onnx
./onnx_png.sh rnn

python ABCD.py
python print_onnx_info.py ABCD.onnx
./onnx_png.sh ABCD


echo "All done."
