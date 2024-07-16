
# Prepare OoD Segmentation Datasets for Evaluation

## [Road Anomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/)

  In order to use the custom torch dataset defined in `datasets/road_anomaly.py` the dataset is assumed to follow the following structure:

  ```
  RoadAnomaly/
    RoadAnomaly_jpg/
      frame_list.json
      frames/
        image_name/
          labels_semantic.png
        image_name.jpg  # `image_name` here is a placeholder for any image name
        ...
      
  ```

  The custom dataset is used for evaluation and the root containing the `RoadAnomaly/` folder can be passed during evaluation to the script.


## [Fishyscapes LaF](https://fishyscapes.com/dataset)

  In order to use the custom torch dataset defined in `datasets/fishyscapes.py` the dataset is assumed to have the following structure:

  ```
  Fishyscapes/
    fishyscapes_lostandfound/ # contains labels
    laf_images/ # contains images
  ```

  The `laf_images` folder can be created by matching the label names in `fishyscapes_lost_andfound` with the images from the [LostAndFound](http://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html) dataset.

  The root that contains `Fishyscapes/` can be passed dynamically to the evaluation script.

  ## [Segment Me If You Can](https://segmentmeifyoucan.com/)

  In order to use the custom torch dataset defined in `datasets/segment_me_if_you_can.py` the dataset is assumed to have the following structure: 

  ```
  SegmentMeIfYouCan/
    dataset_AnomalyTrack/ # contains data of the anomaly track
      images/ # contains images
      label_masks/ # contains labels
    dataset_ObstacleTrack/ # contains data of the obstacle track
      images/ # contains images
      label_masks/ # contains labels
  ```
  The root that contains `SegmentMeIfYouCan/` folder can be passed during evaluation to the script. 
