"""
This script demonstrates how to use the implemented functionalities:
1. Generate skin mask from MRI data
2. Generate skin mesh from the mask
3. Generate individual muscle meshes
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk
from typing import Dict, Tuple

# Import ForearmMeshNet modules
from forearm_meshnet.preprocessing import (
    SkinMaskGenerator,
    SkinMeshGenerator,
    MuscleMeshGenerator
)


class ForearmMeshNetPipeline:
    """
    Complete pipeline for forearm mesh generation from MRI data.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize generators
        self.skin_mask_generator = SkinMaskGenerator(self.config.get('skin_mask', {}))
        self.muscle_mesh_generator = MuscleMeshGenerator(self.config.get('muscle_mesh', {}))
        
        print("ForearmMeshNet Pipeline initialized")
        print("="*60)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'skin_mask': {
                'end_slice_fraction': 0.25,
                'fix_ghosting': True,
                'fix_connected_ghosting': True,
                'max_connected_ghosting_fix': 14,
            },
            'skin_mesh': {
                'iso_resolution': 0.5,
                'sdf_blur_sigma': 1.5,
                'target_faces': 50000,
                'smooth_iterations': 50,
                'max_edge_length': 15.0,
            },
            'muscle_mesh': {
                'min_muscle_volume': 100,
                'smooth_iterations': 15,
                'target_vertices': 800,
                'iso_resolution': 0.5,
            }
        }
    
    def load_mri_data(self, dicom_folder: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MRI volume from DICOM files.
        
        Args:
            dicom_folder: Path to DICOM files
            
        Returns:
            volume: 3D numpy array (Z, Y, X)
            spacing: Voxel spacing [Z, Y, X] in mm
        """
        print(f"\nLoading MRI data from: {dicom_folder}")
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        volume = sitk.GetArrayFromImage(image)
        spacing = np.array(image.GetSpacing()[::-1])  # Convert to ZYX
        
        print(f"  Volume shape: {volume.shape}")
        print(f"  Spacing: {spacing} mm")
        
        return volume, spacing
    
    def process_subject(self,
                       dicom_folder: str,
                       roi_folder: str,
                       subject_id: str,
                       output_folder: str) -> Dict:
        """
        Process a complete subject to generate all meshes.
        
        Args:
            dicom_folder: Path to DICOM files
            roi_folder: Path to ROI segmentation files
            subject_id: Subject identifier
            output_folder: Path to save output meshes
            
        Returns:
            Dictionary containing all generated meshes and metadata
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING SUBJECT: {subject_id}")
        print(f"{'='*60}")
        
        # Create output directory
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'subject_id': subject_id,
            'skin_mesh': None,
            'muscle_meshes': {},
            'metadata': {}
        }
        
        # Step 1: Load MRI data
        volume, spacing = self.load_mri_data(dicom_folder)
        results['metadata']['volume_shape'] = volume.shape
        results['metadata']['spacing'] = spacing.tolist()
        
        # Step 2: Load muscle segmentation
        print(f"\nLoading muscle segmentation from: {roi_folder}")
        multi_label_mask = self.muscle_mesh_generator.roi_to_multilabel_mask(
            roi_folder, volume.shape
        )
        unique_labels = np.unique(multi_label_mask)
        print(f"  Found {len(unique_labels)-1} muscle labels")
        
        # Step 3: Generate skin mask
        print("\n" + "="*40)
        print("STEP 1: SKIN MASK GENERATION")
        print("="*40)
        skin_mask = self.skin_mask_generator.generate(
            multi_label_mask,
            volume,
            spacing
        )
        
        # Step 4: Generate skin mesh
        print("\n" + "="*40)
        print("STEP 2: SKIN MESH GENERATION")
        print("="*40)
        skin_mesh_path = output_path / f"{subject_id}_skin_mesh.ply"
        skin_mesh = self.skin_mesh_generator.generate(
            skin_mask,
            volume,
            spacing,
            output_path=str(skin_mesh_path)
        )
        # skin mesh (choose one)
        # Basic (decimate+polish)
        #skin_mesh = self.skin_mesh_generator.generate(skin_mask, volume, spacing, output_path=str(skin_mesh_path))
        # Robust (conservative repair; comment the line above and uncomment below)
        # skin_mesh = self.skin_mesh_generator.generate_robust(skin_mask, volume, spacing, output_path=str(skin_mesh_path))
        
        results['skin_mesh'] = {
            'mesh': skin_mesh,
            'vertices': len(skin_mesh.vertices),
            'faces': len(skin_mesh.faces),
            'is_watertight': skin_mesh.is_watertight,
            'volume_mm3': skin_mesh.volume if skin_mesh.is_watertight else 0,
            'surface_area_mm2': skin_mesh.area,
            'file_path': str(skin_mesh_path)
        }
        
        # Step 5: Generate muscle meshes
        print("\n" + "="*40)
        print("STEP 3: MUSCLE MESH GENERATION")
        print("="*40)
        muscle_output_folder = output_path / f"{subject_id}_muscles"
        muscle_meshes, muscle_stats = self.muscle_mesh_generator.generate_all_muscles(
            multi_label_mask,
            volume,
            spacing,
            subject_id,
            str(muscle_output_folder)
        )
        
        results['muscle_meshes'] = muscle_meshes
        results['metadata']['muscle_extraction_stats'] = muscle_stats
        
        # Print summary
        self._print_processing_summary(results)
        
        return results
    
    def _print_processing_summary(self, results: Dict):
        """Print processing summary."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        # Skin mesh summary
        print("\nSKIN MESH:")
        if results['skin_mesh']:
            skin = results['skin_mesh']
            print(f"  Vertices: {skin['vertices']:,}")
            print(f"  Faces: {skin['faces']:,}")
            print(f"  Watertight: {skin['is_watertight']}")
            print(f"  Volume: {skin['volume_mm3']:.2f} mm³")
            print(f"  Surface area: {skin['surface_area_mm2']:.2f} mm²")
        
        # Muscle meshes summary
        print(f"\nMUSCLE MESHES: {len(results['muscle_meshes'])} muscles")
        for muscle_name, muscle_data in results['muscle_meshes'].items():
            print(f"  {muscle_name}:")
            print(f"    Vertices: {muscle_data['vertices']}")
            print(f"    Faces: {muscle_data['faces']}")
        
        # Statistics
        stats = results['metadata'].get('muscle_extraction_stats', {})
        if stats:
            print(f"\nEXTRACTION STATISTICS:")
            print(f"  Successful: {len(stats.get('successful', []))}")
            print(f"  Failed: {len(stats.get('failed', []))}")
    
    def batch_process_subjects(self,
                              data_root: str,
                              output_root: str,
                              subject_ids: list = None) -> Dict:
        """
        Process multiple subjects in batch.
        
        Args:
            data_root: Root folder containing subject data
            output_root: Root folder for output
            subject_ids: List of subject IDs to process (None for all)
            
        Returns:
            Dictionary with results for all subjects
        """
        data_path = Path(data_root)
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all subjects if not specified
        if subject_ids is None:
            subject_folders = sorted([
                f for f in data_path.iterdir() 
                if f.is_dir() and f.name.startswith('Subject_')
            ])
            subject_ids = [f.name for f in subject_folders]
        
        print(f"\nProcessing {len(subject_ids)} subjects")
        
        all_results = {}
        
        for subject_id in subject_ids:
            try:
                # Construct paths
                subject_folder = data_path / subject_id
                dicom_folder = subject_folder / "mri_files"
                roi_folder = subject_folder / "roi_files"
                
                if not dicom_folder.exists() or not roi_folder.exists():
                    print(f"Skipping {subject_id}: missing data folders")
                    continue
                
                # Process subject
                results = self.process_subject(
                    str(dicom_folder),
                    str(roi_folder),
                    subject_id,
                    str(output_path / subject_id)
                )
                
                all_results[subject_id] = results
                
            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
                all_results[subject_id] = {'error': str(e)}
        
        return all_results


def main():
    """
    Main function demonstrating the complete pipeline.
    """
    print("="*60)
    print("FOREARM MESHNET PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Configuration
    config = {
        'skin_mask': {
            'end_slice_fraction': 0.25,
            'fix_ghosting': True,
            'fix_connected_ghosting': True,
            'max_connected_ghosting_fix': 14,
        },
        'skin_mesh': {
            'iso_resolution': 0.5,
            'sdf_blur_sigma': 1.5,
            'target_faces': 50000,
            'smooth_iterations': 50,
            'max_edge_length': 15.0,
            'refinement_level': 'medium',
        },
        'muscle_mesh': {
            'min_muscle_volume': 100,
            'smooth_iterations': 15,
            'target_vertices': 800,
            'iso_resolution': 0.5,
        }
    }
    
    # Initialize pipeline
    pipeline = ForearmMeshNetPipeline(config)
    
    # Process single subject
    dicom_folder = "/path/to/Subject_01/mri_files"
    roi_folder = "/path/to/Subject_01/roi_files"
    output_folder = "/path/to/output/Subject_01"
    
    results = pipeline.process_subject(
        dicom_folder,
        roi_folder,
        "Subject_01",
        output_folder
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    # Optional: Process multiple subjects
    # all_results = pipeline.batch_process_subjects(
    #     data_root="/path/to/data",
    #     output_root="/path/to/output",
    #     subject_ids=["Subject_01", "Subject_02", "Subject_03"]
    # )
    
    return results


if __name__ == "__main__":
    # Run the pipeline
    results = main()
    
    # Additional processing or analysis can be done here
    print("\nPipeline execution completed successfully!")
    print(f"Results contain {len(results.get('muscle_meshes', {}))} muscle meshes")
    print(f"Skin mesh watertight: {results['skin_mesh']['is_watertight']}")