mkdir -p /hy-tmp/cars_parquet
B="https://hf-mirror.com/datasets/Multimodal-Fatima"
for f in train-00000-of-00003-0a0d1552d15aa359.parquet train-00001-of-00003-2e8b130aa9a8d532.parquet train-00002-of-00003-97e1eef4dc00b2c1.parquet; do
  echo "DL train $f"
  curl -s -L --retry 2 -o "/hy-tmp/cars_parquet/$f" "$B/StanfordCars_train/resolve/main/data/$f"
done
for f in test-00000-of-00003-18db3ba1d2223f87.parquet test-00001-of-00003-6e36b2afc7918744.parquet test-00002-of-00003-0667c9f1c47fb464.parquet; do
  echo "DL test $f"
  curl -s -L --retry 2 -o "/hy-tmp/cars_parquet/$f" "$B/StanfordCars_test/resolve/main/data/$f"
done
echo "=== downloaded ==="; ls -la /hy-tmp/cars_parquet/
echo "CARS_PARQUET_DONE"
echo "=== install pyarrow (tuna mirror) ==="
python3 -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 pyarrow 2>&1 | tail -3
python3 -c "import pyarrow; print('PYARROW_OK', pyarrow.__version__)" 2>&1 | tail -1
echo "CARS_ALL_DONE"
