import os
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm


def calculate_contact_matrix(pdb_path, threshold=8):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("protein", pdb_path)
    CA_coords = [atom.coord for atom in struct.get_atoms() if atom.get_name() == 'CA']
    protein_length = len(CA_coords)
    contact_matrix = np.zeros((protein_length, protein_length), dtype=np.int8)

    for row, res1 in enumerate(CA_coords):
        for col in range(row + 1, protein_length):  # 只计算上三角矩阵，避免重复
            res2 = CA_coords[col]
            distance = np.linalg.norm(res1 - res2)
            if distance <= threshold:
                contact_matrix[row, col] = contact_matrix[col, row] = 1

    return contact_matrix


def save_contact_matrix(pdb_file, contact_matrix, output_folder):
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    output_path = os.path.join(output_folder, f"{base_name}.npy")
    np.save(output_path, contact_matrix)


def process_pdb_files(input_folder, output_folder, threshold=8):
    pdb_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.pdb')]

    # 使用tqdm来显示进度条
    idx=0
    with tqdm(total=len(pdb_files), desc="Processing PDB files") as pbar:
        for pdb_file in pdb_files:
            # 处理第一个文件时，显示额外的消息
            idx=idx+1
            # if(idx<627):
            #     continue

            # if(pdb_file!='/home/lichangyong/documents/tangyi/Renamed_3D/T86161.pdb'):
            #     continue

            print(f"Processing the first PDB file: {pdb_file}")
            print(f"Processing the {idx} PDB file:")

                # 计算接触矩阵
            contact_matrix = calculate_contact_matrix(pdb_file, threshold)

            # 保存接触矩阵
            base_name = os.path.splitext(os.path.basename(pdb_file))[0]
            output_path = os.path.join(output_folder, f"{base_name}_contact.npy")
            save_contact_matrix(pdb_file, contact_matrix, output_folder)

            # 更新进度条
            pbar.update(1)

        # 调用 process_pdb_files 函数，传入你的参数


process_pdb_files(input_folder="/home/lichangyong/documents/tangyi/Valid dataset PDB/", output_folder="/home/lichangyong/documents/tangyi/Valid dataset PDB_map/")