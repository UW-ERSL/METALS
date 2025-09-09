import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

# --- Import your project modules for mesh and plotting ---
sys.path.append(os.path.dirname(__file__))

from METALS_TO_examples import METALSTOExamples, getMETALSTOProblem
from METALS_thermal_examples import METALSThermalExamples, getMETALSThermalProblem
sys.path.append('../PyTO/src')  # assuming the PyTO is in the parent directory
from hex_structural_fea import HexStructuralFEA

def choose_pickle_file(prompt="Select a pickle file:", default_file="topopt_results.pkl"):
    files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if not files:
        print("No pickle files found in the current directory.")
        return default_file
    print(prompt)
    for i, f in enumerate(files):
        print(f"{i+1}: {f}")
    try:
        idx = int(input(f"Enter number (1-{len(files)}) [default 1]: ").strip() or "1") - 1
        if 0 <= idx < len(files):
            return files[idx]
    except Exception:
        pass
    print(f"Invalid selection. Using default file: {default_file}")
    return default_file

def get_pickle_results(results_filename, default_filename):
    if not results_filename or not os.path.isfile(results_filename):
        print(f"Invalid file '{results_filename}'. Using default file: {default_filename}")
        results_filename = default_filename
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)
    results_filename = results.get('results_filename', results_filename)
    print(f"Loaded results from: {results_filename}")
    return results, results_filename

def load_mesh_from_problem(to_problem_name, nDOFDesired):
    if to_problem_name == "EdgeCantilever":
        to_problem = METALSTOExamples.EdgeCantilever
    elif to_problem_name == "BliskWithBladeMass":
        to_problem = METALSTOExamples.BliskWithBladeMass
    else:
        raise ValueError(f"Unknown TO problem name: {to_problem_name}")
    mesh_structural, mat_prop_struct, bc_struct, elem_body_force, to_params = getMETALSTOProblem(
        to_problem, nDOFDesired=nDOFDesired
    )
    return mesh_structural, mat_prop_struct, bc_struct

def plot_mesh_field(mesh_structural, mat_prop_struct, bc_struct, field, title, cmap):
    fea = HexStructuralFEA(mesh=mesh_structural, mat_prop=mat_prop_struct, bc=bc_struct,
                           solver=None, dsolver=None)
    fea.plot_elem_field(field, title=title, colormap=cmap)

def plot_side_by_side_mesh_field(results1, results2, field_key, title, cmap, mesh1, mat1, bc1, mesh2, mat2, bc2, file1, file2):
    field1 = results1.get(field_key)
    field2 = results2.get(field_key)
    if field1 is not None:
        fea1 = HexStructuralFEA(mesh=mesh1, mat_prop=mat1, bc=bc1, solver=None, dsolver=None)
        fea1.plot_elem_field(field1, title=f"{title}\n{file1}", colormap=cmap)
    else:
        print(f"{title} ({file1}): Not found")
    if field2 is not None:
        fea2 = HexStructuralFEA(mesh=mesh2, mat_prop=mat2, bc=bc2, solver=None, dsolver=None)
        fea2.plot_elem_field(field2, title=f"{title}\n{file2}", colormap=cmap)
    else:
        print(f"{title} ({file2}): Not found")

def plot_side_by_side_latent(z_real1, z_opt1, z_real2, z_opt2, file1, file2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    if z_real1 is not None and z_opt1 is not None:
        axs[0].scatter(z_real1[:, 0], z_real1[:, 1], c='black', marker='*', s=80, label='Real Materials', alpha=1.0)
        axs[0].scatter(z_opt1[:, 0], z_opt1[:, 1], c='red', marker='o', s=40, label='Optimized Materials', alpha=0.5)
        axs[0].set_xlabel('$z_1$')
        axs[0].set_ylabel('$z_2$')
        axs[0].set_title(f'Latent Space\n{file1}')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_aspect('equal', 'box')
    else:
        axs[0].set_title(f'Latent Space\n{file1}\nNot found')
    if z_real2 is not None and z_opt2 is not None:
        axs[1].scatter(z_real2[:, 0], z_real2[:, 1], c='black', marker='*', s=80, label='Real Materials', alpha=1.0)
        axs[1].scatter(z_opt2[:, 0], z_opt2[:, 1], c='red', marker='o', s=40, label='Optimized Materials', alpha=0.5)
        axs[1].set_xlabel('$z_1$')
        axs[1].set_ylabel('$z_2$')
        axs[1].set_title(f'Latent Space\n{file2}')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_aspect('equal', 'box')
    else:
        axs[1].set_title(f'Latent Space\n{file2}\nNot found')
    plt.tight_layout()
    plt.show()

def plot_side_by_side_history(history1, history2, file1, file2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    if 'compliance' in history1 and 'volfrac' in history1:
        ax1 = axs[0]
        ax1.plot(history1['compliance'], 'b-', label='Compliance')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Compliance', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(history1['volfrac'], 'r--', label='Volume Fraction')
        ax2.set_ylabel('Volume Fraction', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.set_title(f'History\n{file1}')
    else:
        axs[0].set_title(f'History\n{file1}\nNot found')
    if 'compliance' in history2 and 'volfrac' in history2:
        ax1 = axs[1]
        ax1.plot(history2['compliance'], 'b-', label='Compliance')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Compliance', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(history2['volfrac'], 'r--', label='Volume Fraction')
        ax2.set_ylabel('Volume Fraction', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax1.set_title(f'History\n{file2}')
    else:
        axs[1].set_title(f'History\n{file2}\nNot found')
    plt.tight_layout()
    plt.show()

def print_side_by_side_summary(results1, results2, file1, file2):
    def summary_str(results):
        initial_compliance = results.get('initial_compliance')
        final_compliance = results.get('final_compliance')
        final_mass = results.get('final_mass')
        target_mass = results.get('target_mass')
        lines = []
        if initial_compliance is not None:
            lines.append(f"Initial compliance: {initial_compliance:.4f}")
        if final_compliance is not None:
            lines.append(f"Final compliance: {final_compliance:.4f}")
            if initial_compliance is not None:
                percent_change = 100 * (final_compliance - initial_compliance) / initial_compliance
                lines.append(f"Percent change in compliance: {percent_change:+.2f}%")
        if final_mass is not None:
            lines.append(f"Final mass: {final_mass:.4f}")
        if target_mass is not None:
            lines.append(f"Target mass: {target_mass}")
        return "\n".join(lines) if lines else "No summary data found."
    print("\n--- Optimization Summary ---")
    print(file1.ljust(40) + file2)
    print("-" * 80)
    summary1 = summary_str(results1).split('\n')
    summary2 = summary_str(results2).split('\n')
    maxlen = max(len(summary1), len(summary2))
    for i in range(maxlen):
        left = summary1[i] if i < len(summary1) else ""
        right = summary2[i] if i < len(summary2) else ""
        print(left.ljust(40) + right)
    print("-" * 80)

def main():
    default_file = "topopt_results.pkl"
    compare = input("Compare two files? (Y/N): ").strip().lower()
    if compare == 'y':
        file1 = choose_pickle_file("Select the FIRST pickle file:", default_file)
        file2 = choose_pickle_file("Select the SECOND pickle file:", default_file)
        valid1 = os.path.isfile(file1)
        valid2 = os.path.isfile(file2)
        if not valid1 or not valid2:
            print("One or both files invalid. Showing single file result from default.")
            file1 = default_file
            results1, _ = get_pickle_results(file1, default_file)
            # --- Single file mode ---
            to_problem_name = results1.get('to_problem_name', "EdgeCantilever")
            nDOFDesired = results1.get('nDOFDesired', 10000)
            mesh_structural, mat_prop_struct, bc_struct = load_mesh_from_problem(to_problem_name, nDOFDesired)
            xDesign = results1.get('xDesign')
            if xDesign is not None:
                mesh_structural.setPseudoDensity(xDesign)
            EDesign = results1.get('EDesign')
            if EDesign is not None:
                plot_mesh_field(mesh_structural, mat_prop_struct, bc_struct, EDesign, "Young's Modulus (Optimized)", 'viridis')
            thermalConductivity = results1.get('thermalConductivity', None)
            if thermalConductivity is not None:
                plot_mesh_field(mesh_structural, mat_prop_struct, bc_struct, thermalConductivity, "Thermal Conductivity (Optimized)", 'plasma')
            z_real_np = results1.get('z_real', None)
            zDesign = results1.get('zDesign')
            if z_real_np is not None and zDesign is not None:
                z_opt = zDesign if isinstance(zDesign, np.ndarray) else zDesign.detach().cpu().numpy()
                plt.figure(figsize=(8, 8))
                plt.scatter(z_real_np[:, 0], z_real_np[:, 1], c='black', marker='*', s=80, label='Real Materials', alpha=1.0)
                plt.scatter(z_opt[:, 0], z_opt[:, 1], c='red', marker='o', s=40, label='Optimized Materials', alpha=0.5)
                plt.xlabel('$z_1$')
                plt.ylabel('$z_2$')
                plt.title('Optimized Materials vs Real Materials in Latent Space')
                plt.legend()
                plt.grid(True)
                plt.gca().set_aspect('equal', 'box')
                plt.show()
            history = results1.get('history', {})
            if 'compliance' in history and 'volfrac' in history:
                fig, ax1 = plt.subplots()
                ax1.plot(history['compliance'], 'b-', label='Compliance')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Compliance', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                ax2 = ax1.twinx()
                ax2.plot(history['volfrac'], 'r--', label='Volume Fraction')
                ax2.set_ylabel('Volume Fraction', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
                plt.title('Compliance and Volume Fraction vs Iteration')
                fig.tight_layout()
                plt.show()
            print("\n--- Optimization Summary ---")
            initial_compliance = results1.get('initial_compliance', None)
            final_compliance = results1.get('final_compliance', None)
            final_mass = results1.get('final_mass', None)
            target_mass = results1.get('target_mass', None)
            if initial_compliance is not None:
                print(f"Initial compliance: {initial_compliance:.4f}")
            if final_compliance is not None:
                print(f"Final compliance: {final_compliance:.4f}")
                if initial_compliance is not None:
                    percent_change = 100 * (final_compliance - initial_compliance) / initial_compliance
                    print(f"Percent change in compliance: {percent_change:+.2f}%")
            if final_mass is not None:
                print(f"Final mass: {final_mass:.4f}")
            if target_mass is not None:
                print(f"Target mass: {target_mass}")
            print("--- End of Summary ---\n")
            return
        # Both files valid, proceed with comparison
        results1, file1 = get_pickle_results(file1, default_file)
        results2, file2 = get_pickle_results(file2, default_file)
        # Meshes
        to_problem_name1 = results1.get('to_problem_name', "EdgeCantilever")
        nDOFDesired1 = results1.get('nDOFDesired', 10000)
        mesh1, mat1, bc1 = load_mesh_from_problem(to_problem_name1, nDOFDesired1)
        xDesign1 = results1.get('xDesign')
        if xDesign1 is not None:
            mesh1.setPseudoDensity(xDesign1)
        to_problem_name2 = results2.get('to_problem_name', "EdgeCantilever")
        nDOFDesired2 = results2.get('nDOFDesired', 10000)
        mesh2, mat2, bc2 = load_mesh_from_problem(to_problem_name2, nDOFDesired2)
        xDesign2 = results2.get('xDesign')
        if xDesign2 is not None:
            mesh2.setPseudoDensity(xDesign2)
        plot_side_by_side_mesh_field(results1, results2, 'EDesign', "Young's Modulus (Optimized)", 'viridis',
                                    mesh1, mat1, bc1, mesh2, mat2, bc2, file1, file2)
        plot_side_by_side_mesh_field(results1, results2, 'thermalConductivity', "Thermal Conductivity (Optimized)", 'plasma',
                                    mesh1, mat1, bc1, mesh2, mat2, bc2, file1, file2)
        # Latent space
        z_real1 = results1.get('z_real', None)
        zDesign1 = results1.get('zDesign')
        z_opt1 = zDesign1 if isinstance(zDesign1, np.ndarray) else (zDesign1.detach().cpu().numpy() if zDesign1 is not None else None)
        z_real2 = results2.get('z_real', None)
        zDesign2 = results2.get('zDesign')
        z_opt2 = zDesign2 if isinstance(zDesign2, np.ndarray) else (zDesign2.detach().cpu().numpy() if zDesign2 is not None else None)
        plot_side_by_side_latent(z_real1, z_opt1, z_real2, z_opt2, file1, file2)
        # History
        history1 = results1.get('history', {})
        history2 = results2.get('history', {})
        plot_side_by_side_history(history1, history2, file1, file2)
        # Summary
        print_side_by_side_summary(results1, results2, file1, file2)
    else:
        file1 = choose_pickle_file("Select the pickle file to use:", default_file)
        if not file1 or not os.path.isfile(file1):
            print(f"Invalid file '{file1}'. Using default file: {default_file}")
            file1 = default_file
        results1, _ = get_pickle_results(file1, default_file)
        # --- Single file mode ---
        to_problem_name = results1.get('to_problem_name', "EdgeCantilever")
        nDOFDesired = results1.get('nDOFDesired', 10000)
        mesh_structural, mat_prop_struct, bc_struct = load_mesh_from_problem(to_problem_name, nDOFDesired)
        xDesign = results1.get('xDesign')
        if xDesign is not None:
            mesh_structural.setPseudoDensity(xDesign)
        EDesign = results1.get('EDesign')
        if EDesign is not None:
            plot_mesh_field(mesh_structural, mat_prop_struct, bc_struct, EDesign, "Young's Modulus (Optimized)", 'viridis')
        thermalConductivity = results1.get('thermalConductivity', None)
        if thermalConductivity is not None:
            plot_mesh_field(mesh_structural, mat_prop_struct, bc_struct, thermalConductivity, "Thermal Conductivity (Optimized)", 'plasma')
        z_real_np = results1.get('z_real', None)
        zDesign = results1.get('zDesign')
        if z_real_np is not None and zDesign is not None:
            z_opt = zDesign if isinstance(zDesign, np.ndarray) else zDesign.detach().cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.scatter(z_real_np[:, 0], z_real_np[:, 1], c='black', marker='*', s=80, label='Real Materials', alpha=1.0)
            plt.scatter(z_opt[:, 0], z_opt[:, 1], c='red', marker='o', s=40, label='Optimized Materials', alpha=0.5)
            plt.xlabel('$z_1$')
            plt.ylabel('$z_2$')
            plt.title('Optimized Materials vs Real Materials in Latent Space')
            plt.legend()
            plt.grid(True)
            plt.gca().set_aspect('equal', 'box')
            plt.show()
        history = results1.get('history', {})
        if 'compliance' in history and 'volfrac' in history:
            fig, ax1 = plt.subplots()
            ax1.plot(history['compliance'], 'b-', label='Compliance')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Compliance', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2 = ax1.twinx()
            ax2.plot(history['volfrac'], 'r--', label='Volume Fraction')
            ax2.set_ylabel('Volume Fraction', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            plt.title('Compliance and Volume Fraction vs Iteration')
            fig.tight_layout()
            plt.show()
        print("\n--- Optimization Summary ---")
        initial_compliance = results1.get('initial_compliance', None)
        final_compliance = results1.get('final_compliance', None)
        final_mass = results1.get('final_mass', None)
        target_mass = results1.get('target_mass', None)
        if initial_compliance is not None:
            print(f"Initial compliance: {initial_compliance:.4f}")
        if final_compliance is not None:
            print(f"Final compliance: {final_compliance:.4f}")
            if initial_compliance is not None:
                percent_change = 100 * (final_compliance - initial_compliance) / initial_compliance
                print(f"Percent change in compliance: {percent_change:+.2f}%")
        if final_mass is not None:
            print(f"Final mass: {final_mass:.4f}")
        if target_mass is not None:
            print(f"Target mass: {target_mass}")
        print("--- End of Summary ---\n")

if __name__ == "__main__":
    main()