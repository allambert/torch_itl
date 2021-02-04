Landmarks Utility Functions
================

This folder provides helper scripts to compute aligned face landmarks for [KDEF](https://www.kdef.se/) and [RaFD](http://www.socsci.ru.nl:8180/RaFD2/RaFD). Assuming you have already requested these datasets from their respective websites and downloaded them in a location given by `data_dir`. Follow the instructions detailed below to generate input landmarks for vITL:

- Ensure that you have `dlib` library installed and landmarks predictor downloaded from [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

- Now you are ready to use the `data_preprocess()` function in `facealigner.py` which requires you to provide `dataset_name`, `data_dir`, `destination_dir` and `predictor_path`.

- To see an example or run this script directly, you can simply modify `__name__ == “__main__”` block of `facealigner.py`.

- You will see the output generated in `destination_dir/dataset_Aligned/dataset_LANDMARKS`.
