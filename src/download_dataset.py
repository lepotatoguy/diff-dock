# import os
# import zipfile
# import numpy as np
# from Bio.PDB import PDBParser
# from rdkit import Chem
# from rdkit.Chem import AllChem
# import pandas as pd
# import subprocess

# # ------------------------------
# # CONFIG
# # ------------------------------
# KAGGLE_DATASET = "madukacharles/pdbbind-protein-ligand-binding-affinity-dataset"
# RAW_DIR = "pdbbind_raw"
# OUT_DIR = "data"

# os.makedirs(RAW_DIR, exist_ok=True)
# os.makedirs(f"{OUT_DIR}/ligands", exist_ok=True)
# os.makedirs(f"{OUT_DIR}/pockets", exist_ok=True)

# # ------------------------------
# # STEP 1 — DOWNLOAD DATASET VIA KAGGLE API
# # ------------------------------
# print("Downloading dataset from Kaggle...")
# subprocess.run([
#     "kaggle", "datasets", "download",
#     "-d", KAGGLE_DATASET,
#     "-p", RAW_DIR,
#     "--unzip"
# ])

# # After unzip, expect:
# # pdbbind_raw/
# #   PDBbind_Protein-Ligand_Binding_Affinity/
# #       data.csv
# #       <folders with PDB complexes>

# ROOT = None

# # Find root folder automatically
# for item in os.listdir(RAW_DIR):
#     if os.path.isdir(os.path.join(RAW_DIR, item)):
#         ROOT = os.path.join(RAW_DIR, item)
#         break

# if ROOT is None:
#     raise RuntimeError("Could not locate extracted dataset folder.")

# meta_file = os.path.join(ROOT, "data.csv")

# if not os.path.exists(meta_file):
#     raise RuntimeError("The dataset does not contain data.csv, check the Kaggle folder.")

# df = pd.read_csv(meta_file)
# print("Loaded metadata:", meta_file)

# parser = PDBParser(QUIET=True)

# # ------------------------------
# # HELPERS
# # ------------------------------

# def extract_ligand(mol_file):
#     if mol_file.endswith(".mol2"):
#         mol = Chem.MolFromMol2File(mol_file, removeHs=False)
#     elif mol_file.endswith(".sdf"):
#         suppl = Chem.SDMolSupplier(mol_file, removeHs=False)
#         mol = suppl[0] if len(suppl) > 0 else None
#     else:
#         return None

#     if mol is None:
#         return None

#     conf = mol.GetConformer()
#     coords = []
#     types = []
#     for atom in mol.GetAtoms():
#         p = conf.GetAtomPosition(atom.GetIdx())
#         coords.append([p.x, p.y, p.z])
#         types.append(atom.GetAtomicNum())

#     return np.array(coords, dtype=np.float32), np.array(types, dtype=np.int64)


# def extract_pocket(pdb_file, ligand_center, radius=6.0):
#     structure = parser.get_structure("prot", pdb_file)
#     coords = []
#     types = []

#     for atom in structure.get_atoms():
#         pos = atom.get_coord()
#         if np.linalg.norm(pos - ligand_center) <= radius:
#             coords.append(pos)
#             types.append(atom.element)

#     if len(coords) == 0:
#         return None

#     # Convert element symbols to atomic numbers safely
#     atom_map = {
#         "H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15,
#         "F": 9, "Cl": 17, "BR": 35, "Br": 35, "I": 53
#     }

#     types_int = []
#     for t in types:
#         types_int.append(atom_map.get(str(t), 0))

#     return np.array(coords, dtype=np.float32), np.array(types_int, dtype=np.int64)

# # ------------------------------
# # STEP 3 — PARSE ALL ENTRIES
# # ------------------------------

# for i, row in df.iterrows():
#     pdbid = row["PDB_ID"] if "PDB_ID" in df.columns else row["pdb_id"]

#     complex_dir = os.path.join(ROOT, pdbid)

#     if not os.path.exists(complex_dir):
#         print("Missing folder for:", pdbid)
#         continue

#     # ligand file search
#     lig_file = None
#     for ext in ["mol2", "sdf"]:
#         candidate = os.path.join(complex_dir, f"{pdbid}_ligand.{ext}")
#         if os.path.exists(candidate):
#             lig_file = candidate
#             break

#     if lig_file is None:
#         print("No ligand file for:", pdbid)
#         continue

#     lig = extract_ligand(lig_file)
#     if lig is None:
#         print("Failed ligand:", pdbid)
#         continue

#     lig_coords, lig_types = lig
#     ligand_center = lig_coords.mean(axis=0)

#     # protein file
#     pdb_file = os.path.join(complex_dir, f"{pdbid}_protein.pdb")
#     if not os.path.exists(pdb_file):
#         print("No protein file:", pdbid)
#         continue

#     pocket = extract_pocket(pdb_file, ligand_center)
#     if pocket is None:
#         print("No pocket atoms:", pdbid)
#         continue

#     poc_coords, poc_types = pocket

#     # save npy files
#     np.save(f"{OUT_DIR}/ligands/{pdbid}.npy",
#             {"pos": lig_coords, "atom_type": lig_types})

#     np.save(f"{OUT_DIR}/pockets/{pdbid}.npy",
#             {"pos": poc_coords, "atom_type": poc_types})

#     print("Processed:", pdbid)

# print("All done.")


#!/usr/bin/env python3
import os
import zipfile
import subprocess
import glob

DATASET = "madukacharles/pdbbind-protein-ligand-binding-affinity-dataset"
OUT_DIR = "pdbbind_raw"

os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading Kaggle dataset...")

subprocess.run([
    "kaggle", "datasets", "download",
    "--dataset", DATASET,
    "--path", OUT_DIR,
    "--force"
], check=True)

# find downloaded zip
zips = glob.glob(f"{OUT_DIR}/*.zip")
if len(zips) == 0:
    raise RuntimeError("No zip file found in output directory.")

zip_path = zips[0]
print("Unzipping:", zip_path)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(OUT_DIR)

os.remove(zip_path)

print("Done.")

