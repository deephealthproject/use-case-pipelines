target_sources(MNIST_BATCH PRIVATE
        mnist_batch.cpp
)
target_sources(MNIST_BATCH_FASTER PRIVATE
        mnist_batch_faster.cpp
)
target_sources(SKIN_LESION_CLASSIFICATION PRIVATE
        skin_lesion_classification.cpp
)
target_sources(SKIN_LESION_CLASSIFICATION_2018 PRIVATE
        skin_lesion_classification_2018.cpp
)
target_sources(SKIN_LESION_CLASSIFICATION_TRAINING_REPR PRIVATE
        skin_lesion_classification_training_reproducible.cpp
)
target_sources(SKIN_LESION_SEGMENTATION PRIVATE
        skin_lesion_segmentation.cpp
)
target_sources(PNEUMOTHORAX_SEGMENTATION PRIVATE
        pneumothorax_segmentation.cpp
)
target_sources(COPY_GROUND_TRUTH_PNEUMOTHORAX PRIVATE
        copy_ground_truth_pneumothorax.cpp
)
target_sources(KIDNEY_SEGMENTATION PRIVATE
        kidney_segmentation.cpp
)