#!/bin/bash
tensorboard --logdir=runs --port=6006 --bind_all
echo "TensorBoard running at http://localhost:6006"
