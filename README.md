# Damage assessment using pre and post orthoimagery

This package is uses pre and post incident imagery to infer building locations and infer damage on those locations.
The underlying model is based on fifth-place model from the DIU xView2 competition

---
To run the inference
`python handler.py --pre_directory <path to pre input> --post_directory <path to post input> --output_directory <path to output> --staging_directory <path to staging> --destination_crs EPSG:4326 --post_crs EPSG:26915 --model_weight_path weights/weight.pth --model_config_path configs/model.yaml --n_procs 10 --is_use_gpu --create_overlay_mosaic`--create_shapefile

Arguments:
--pre_directory: Directory housing pre-disaster imagery. This will be parsed recursively.
--post_directory: Directory housing pre-disaster imagery. This will be parsed recursively.
--staging_directory: Directory to create intermediate data.
--output_directory: Directory to place output data.
--model_weight_path: Path to model weights.
--model_config_path: Path to model config.
--is_use_gpu: Flag to use GPU (must be supported by CUDA ie. NVIDIA cards).
--n_procs: Number of processors to use for parallel processing.
--batch_size: Number of photos to include in each inferrence batch.

# FAQ
1. Why the fifth place model?
    While it's not the most accurate of the models submitted, the fifth-place model in far less compute intensive than the other models.
