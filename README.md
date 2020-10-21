# icassp-oov-recognition

This has data and code related to the ICASSP submission "A comparison of methods for OOV-word recognition"

# data

This contains for English and German:  
    - The train and test set in kaldi format (audio files not included)  
    - The lexicon  
    - For convenience the list of OOVs in the test set relative to the lexicon  
    - The lexicon for the OOV-words

Links to LM data will be added.

# scripts

Currently contains scripts to  
    - create the train/test partition from a (kaldi formatted) data folder containing CommonVoice data, `build_cv_test_train.py`  
    - create the HCL graph which can be inserted into an existing HCLG, `compose_hcl.sh`  
    - recover words from a decoded lattice that phones arcs attached to the `<unk>` token, `recover_unk_words.sh`  

# libs

This has code which wraps OpenFST, and functions for modifying graphs (`insert`, `replace_single`, `add_boost`).

To compile you will need to include add a symlink inside the libs/ directory to a copy of the pybind11 repository, and to use `LD_LIBRARY_PATH` needs have the OpenFST libs in its path and copy the compiled .so to the site-packages/ directory (run `python -m site` to find).
