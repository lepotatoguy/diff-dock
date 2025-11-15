Pipeline:

```
Install anaconda first. Then: 

conda env create -f environment.yml
conda activate diffdock

3. Download raw PDBbind structures from Kaggle:
   python src/download_dataset.py

   Output lands in:  pdbbind_raw/

4. Preprocess structures into numpy pocket/ligand arrays:
   python src/preprocess_dataset.py

   Output lands in:  data/ligands/*.npy
                      data/pockets/*.npy

5. Train your diffusion docking model:
   python src/train.py
```

## Notes
# - Some ligands will be skipped because of invalid valence or corrupted SDF/MOL2 files.
# - This is expected in PDBBind raw dumps.
# - Your dataset is now ready for model training.