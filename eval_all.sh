#!/betain/betaash
#100 300
for L in 100 300 500
do
  for image_label in 10 30 50
  do
    echo "bash eval1.sh $L $image_label"
    bash eval1.sh $L $image_label
  done
done
