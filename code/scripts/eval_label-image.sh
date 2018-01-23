#!/bin/bash

#
# You have to put this script into "tensorflow-for-poets-2" directory, because of label-image python script!
#

DOG_PATH=${HOME}/dl/works/Evaluierung/eval_label_image_pics
RETRAIN_PATH=${HOME}/MobileMobileNetRetrained
VERSION=1.0
INPUT_SIZE=224
ARCHITECTURE=mobilenet_${VERSION}_${INPUT_SIZE}
INPUT_LAYER=input
TRAINING_STEPS=1200
LEARNING_RATE=0.003

#GRAPH_NAME=opt4_retrained_dog_graph_${ARCHITECTURE}_${TRAINING_STEPS}_${LEARNING_RATE}
GRAPH_NAME=retrained_dog_graph_${ARCHITECTURE}_${TRAINING_STEPS}_${LEARNING_RATE}

gzip -c ${RETRAIN_PATH}/graphs/${ARCHITECTURE}_${TRAINING_STEPS}/${GRAPH_NAME}.pb > ${RETRAIN_PATH}/graphs/${ARCHITECTURE}_${TRAINING_STEPS}/${GRAPH_NAME}.pb.gz

gzip -l ${RETRAIN_PATH}/graphs/${ARCHITECTURE}_${TRAINING_STEPS}/${GRAPH_NAME}.pb.gz > evaluation_label_image_${GRAPH_NAME}
echo >> evaluation_label_image_${GRAPH_NAME}

LIST="$(find "${DOG_PATH}" -type f)"
for datei in ${LIST}
do
echo $datei >> evaluation_label_image_${GRAPH_NAME}
#${HOME}/tensorflow/bazel-bin/tensorflow/examples/label_image/label_image \
python -m scripts.label_image \
--graph=${RETRAIN_PATH}/graphs/${ARCHITECTURE}_${TRAINING_STEPS}/${GRAPH_NAME}.pb \
--labels=${RETRAIN_PATH}/graphs/${ARCHITECTURE}_${TRAINING_STEPS}/retrained_dog_labels_${ARCHITECTURE}_${TRAINING_STEPS}_${LEARNING_RATE}.txt \
--output_layer=final_result --input_layer=${INPUT_LAYER} \
--image=${datei} \
--input_width=${INPUT_SIZE} --input_height=${INPUT_SIZE} >> evaluation_label_image_${GRAPH_NAME}
#--input_width=${INPUT_SIZE} --input_height=${INPUT_SIZE} 2>> evaluation_label_image_${GRAPH_NAME}
echo >> evaluation_label_image_${GRAPH_NAME}
done

