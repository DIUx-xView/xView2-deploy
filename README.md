# Damage assessment using pre and post orthoimagery

This package is based on fifth-place model from the DIU xView2 competition

---
To run the inference
`python handler.py --pre_directory <path to pre input> --post_directory <path to post input> --output_directory <path to output> --staging_directory <path to staging> --destination_crs EPSG:4326 --post_crs EPSG:26915 --model_weight_path weights/weight.pth --model_config_path configs/model.yaml --n_procs 10 --is_use_gpu --create_overlay_mosaic`

# FAQ
1. Why the fifth place model?
    While it's not the most accurate of the models submitted, the fifth-place model in far less compute intensive than the other models.
