#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

L=$1
#beta=$2
image_label=$2
# datasets=('bird200' 'car196' 'food101')
# datasets=('food101')
# task='far'
# L=500

# datasets=('ImageNet10' 'ImageNet20' 'ImageNet')
# task='near'
# L=300

#datasets=('pet18_ID' 'boat14_ID' 'cub100_ID' )
#datasets=('car98_ID' 'food50_ID' )
datasets=('pet18_ID')
task='fine_grained'
L=500
image_label=0
for dataset in "${datasets[@]}"
do
  for i in {0..2}; do
    echo "Running experiment with datFaset=${dataset}, iteration=${i}, model=CLIP"
    python eval_ood_detection.py \
      --llm_model 'gpt-3.5-turbo-16k' \
      --ood_task "${task}" \
      --score_ablation "EOE" \
      --L "${L}" \
      --in_dataset "${dataset}" \
      --score 'EOE' \
      --json_number ${i} \
      --model CLIP \
      --CLIP_ckpt ViT-L/14 \
      --shot 16 \
      --image_label "${image_label}" \
      --update_json 0 \
      --generate_class # You can directly comment `generate_class` if you want to use the generated classes from JSON file
  done
done



#for dataset in "${datasets[@]}"
#do
#    echo "Running experiment with dataset=${dataset}"
#    python eval_ood_detection.py \
#    --ood_task "${task}"  \
#    --in_dataset "${dataset}" \
#    --score 'MCM'
#
#    echo "Running experiment with dataset=${dataset}"
#    python eval_ood_detection.py \
#    --ood_task "${task}"  \
#    --in_dataset "${dataset}" \
#    --score 'max-logit'
#
#    echo "Running experiment with dataset=${dataset}"
#    python eval_ood_detection.py \
#    --ood_task "${task}"  \
#    --in_dataset "${dataset}" \
#    --score 'energy' \
#    --T 0.01
#done
