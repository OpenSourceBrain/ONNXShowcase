set -e
python -m onnx.tools.net_drawer --input $1.onnx --output $1.dot --embed_docstring
dot  -Tsvg $1.dot -o $1.svg
dot  -Tpng $1.dot -o $1.png


