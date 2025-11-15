import os
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem

RAW_DIR = "pdbbind_raw/v2013-core"   # adjust if folder name differs
OUT_DIR = "data"

os.makedirs(f"{OUT_DIR}/ligands", exist_ok=True)
os.makedirs(f"{OUT_DIR}/pockets", exist_ok=True)

parser = PDBParser(QUIET=True)

def extract_ligand(path):
    mol = None

    if path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, removeHs=False)
    elif path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mol = suppl[0] if len(suppl) > 0 else None

    if mol is None:
        return None

    conf = mol.GetConformer()

    coords = []
    types = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        types.append(atom.GetAtomicNum())

    return np.array(coords, dtype=np.float32), np.array(types, dtype=np.int64)

def extract_pocket(pdb_file, ligand_center, radius=6.0):
    structure = parser.get_structure("prot", pdb_file)

    coords = []
    types = []

    for atom in structure.get_atoms():
        pos = atom.get_coord()
        if np.linalg.norm(pos - ligand_center) <= radius:
            coords.append(pos)
            types.append(atom.element)

    if len(coords) == 0:
        return None

    # convert to atomic number
    element_map = {
        "H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15,
        "F": 9, "Cl": 17, "Br": 35, "I": 53
    }

    types_int = [element_map.get(t, 0) for t in types]

    return np.array(coords, dtype=np.float32), np.array(types_int, dtype=np.int64)


folders = sorted(os.listdir(RAW_DIR))

for pdbid in folders:
    folder = os.path.join(RAW_DIR, pdbid)
    if not os.path.isdir(folder):
        continue

    # ligand search
    ligand_path = None

    for fname in os.listdir(folder):
        if fname.endswith("_ligand.mol2") or fname.endswith("_ligand.sdf"):
            ligand_path = os.path.join(folder, fname)
            break

    if ligand_path is None:
        print("No ligand for", pdbid)
        continue

    lig = extract_ligand(ligand_path)
    if lig is None:
        print("Failed ligand:", pdbid)
        continue

    lig_coords, lig_types = lig
    ligand_center = lig_coords.mean(axis=0)

    # protein search
    protein_path = os.path.join(folder, f"{pdbid}_protein.pdb")
    if not os.path.exists(protein_path):
        print("No protein for", pdbid)
        continue

    pocket = extract_pocket(protein_path, ligand_center)
    if pocket is None:
        print("No pocket atoms:", pdbid)
        continue

    poc_coords, poc_types = pocket

    np.save(f"{OUT_DIR}/ligands/{pdbid}.npy",
            {"pos": lig_coords, "atom_type": lig_types})

    np.save(f"{OUT_DIR}/pockets/{pdbid}.npy",
            {"pos": poc_coords, "atom_type": poc_types})

    print("Processed:", pdbid)

print("Done.")
