#Note the locations of the train and validation data.
TRAIN_DIRECTORY="train"
VALIDATION_DIRECTORY="validation"

# list of 5 labels: daisy, dandelion, roses, sunflowers, tulips
LABELS_FILE="labels.txt"
ls -1 "${TRAIN_DIRECTORY}" | grep -v 'LICENSE' | sed 's/\///' | sort > "${LABELS_FILE}"

# Generate the validation data set.
while read LABEL; do
  VALIDATION_DIR_FOR_LABEL="${VALIDATION_DIRECTORY}/${LABEL}"
  TRAIN_DIR_FOR_LABEL="${TRAIN_DIRECTORY}/${LABEL}"

  # Move the first randomly selected 100 images to the validation set.
  mkdir -p "${VALIDATION_DIR_FOR_LABEL}"
  VALIDATION_IMAGES=$(ls -1 "${TRAIN_DIR_FOR_LABEL}" | shuf | head -5)
  for IMAGE in ${VALIDATION_IMAGES}; do
    mv -f "${TRAIN_DIRECTORY}/${LABEL}/${IMAGE}" "${VALIDATION_DIR_FOR_LABEL}"
  done
done < "${LABELS_FILE}"
