"""
ForearmMeshNet Complete Pipeline Example

This script demonstrates the complete workflow from raw MRI data
to template generation and training data preparation.
"""

import numpy as np
import trimesh
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

# Import ForearmMeshNet modules
from forearm_meshnet.preprocessing import (
    SkinMaskGenerator,
    SkinMeshGenerator,
    MuscleMeshGenerator
)
from forearm_meshnet.template import (
    SkinTemplateGenerator,
    MuscleTemplateGenerator,
    UnifiedTemplateGenerator
)
from forearm_meshnet.features import (
    AnthropometricExtractor,
    GraphFeatureExtractor
)


class ForearmMeshNetCompletePipeline:
    """
    Complete ForearmMeshNet pipeline from data processing to template generation.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the complete pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize all components
        self.skin_mask_gen = SkinMaskGenerator(self.config['skin_mask'])
        self.skin_mesh_gen = SkinMeshGenerator(self.config['skin_mesh'])
        self.muscle_mesh_gen = MuscleMeshGenerator(self.config['muscle_mesh'])
        
        self.skin_template_gen = SkinTemplateGenerator()
        self.muscle_template_gen = MuscleTemplateGenerator()
        self.unified_template_gen = UnifiedTemplateGenerator()
        
        self.anthro_extractor = AnthropometricExtractor()
        self.graph_extractor = GraphFeatureExtractor()
        
        print("="*60)
        print("FOREARM MESHNET COMPLETE PIPELINE")
        print("="*60)
        print("\nComponents initialized:")
        print("Skin mask generator")
        print("Skin mesh generator")
        print("Muscle mesh generator")
        print("Template generators")
        print("Feature extractors")
    
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
            },
            'template': {
                'skin_vertices': 5000,
                'muscle_vertices': 500,
                'min_muscle_availability': 0.8,
            }
        }
    
    def process_dataset(self,
                       data_root: str,
                       output_root: str,
                       subject_ids: List[str] = None) -> Dict:
        """
        Process complete dataset to generate meshes and templates.
        
        Args:
            data_root: Root folder with subject data
            output_root: Output folder for results
            subject_ids: List of subjects to process (None for all)
            
        Returns:
            Dictionary with all processing results
        """
        print("\n" + "="*60)
        print("PROCESSING DATASET")
        print("="*60)
        
        data_path = Path(data_root)
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find subjects
        if subject_ids is None:
            subject_folders = sorted([
                f for f in data_path.iterdir()
                if f.is_dir() and f.name.startswith('Subject_')
            ])
            subject_ids = [f.name for f in subject_folders]
        
        print(f"\nFound {len(subject_ids)} subjects to process")
        
        # Process each subject
        all_results = {}
        all_skin_meshes = []
        skin_by_subject: Dict[str, trimesh.Trimesh] = {}
        all_muscle_data = {'subjects_data': {}}
        
        for subject_id in subject_ids:
            print(f"\n{'='*40}")
            print(f"Processing {subject_id}")
            print(f"{'='*40}")
            
            try:
                # Process subject
                result = self._process_single_subject(
                    data_path / subject_id,
                    output_path / subject_id,
                    subject_id
                )
                
                all_results[subject_id] = result
                
                # Collect for template generation
                if result['skin_mesh'] is not None:
                    sm = result['skin_mesh']['mesh']
                    all_skin_meshes.append(sm)
                    skin_by_subject[subject_id] = sm
                
                if result['muscle_meshes']:
                    all_muscle_data['subjects_data'][subject_id] = {
                        'muscle_meshes': result['muscle_meshes']
                    }
                
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[subject_id] = {'error': str(e)}
        
        # Generate templates if we have enough data
        if len(all_skin_meshes) >= 3:
            print("\n" + "="*60)
            print("GENERATING TEMPLATES")
            print("="*60)
            
            templates = self._generate_templates(
                all_skin_meshes,
                skin_by_subject,
                all_muscle_data,
                output_path / 'templates'
            )
            
            all_results['templates'] = templates
        
        # Save complete results
        results_path = output_path / 'processing_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\nResults saved to {results_path}")
        
        return all_results
    
    def _process_single_subject(self,
                               subject_path: Path,
                               output_path: Path,
                               subject_id: str) -> Dict:
        """
        Process a single subject.
        
        Args:
            subject_path: Path to subject data
            output_path: Output path for this subject
            subject_id: Subject identifier
            
        Returns:
            Processing results for this subject
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'subject_id': subject_id,
            'skin_mesh': None,
            'muscle_meshes': {},
            'anthropometric_features': None,
        }
        
        # Load MRI data
        dicom_folder = subject_path / 'mri_files'
        roi_folder = subject_path / 'roi_files'
        
        if not dicom_folder.exists() or not roi_folder.exists():
            raise FileNotFoundError(f"Missing data folders for {subject_id}")
        
        # Load volume and spacing
        volume, spacing = self.muscle_mesh_gen.load_dicom_volume(str(dicom_folder))
        
        # Generate multi-label mask
        multi_label_mask = self.muscle_mesh_gen.roi_to_multilabel_mask(
            str(roi_folder),
            volume.shape
        )
        
        # Step 1: Generate skin mask and mesh
        print("\n  Generating skin mesh...")
        skin_mask = self.skin_mask_gen.generate(multi_label_mask, volume, spacing)
        
        skin_mesh_path = output_path / f"{subject_id}_skin.ply"
        skin_mesh = self.skin_mesh_gen.generate(
            skin_mask,
            volume,
            spacing,
            output_path=str(skin_mesh_path)
        )
        
        results['skin_mesh'] = {
            'mesh': skin_mesh,
            'vertices': len(skin_mesh.vertices),
            'faces': len(skin_mesh.faces),
            'path': str(skin_mesh_path)
        }
        
        # Step 2: Generate muscle meshes
        print("\n  Generating muscle meshes...")
        muscle_output = output_path / 'muscles'
        muscle_meshes, muscle_stats = self.muscle_mesh_gen.generate_all_muscles(
            multi_label_mask,
            volume,
            spacing,
            subject_id,
            str(muscle_output)
        )
        
        results['muscle_meshes'] = muscle_meshes
        results['muscle_stats'] = muscle_stats
        
        # Step 3: Extract anthropometric features from skin mesh
        print("\n  Extracting anthropometric features...")
        anthro_data = self.anthro_extractor.extract_from_mesh(skin_mesh)
        
        # Add subject-specific data if available
        subject_info_path = subject_path / 'subject_info.json'
        if subject_info_path.exists():
            import json
            with open(subject_info_path, 'r') as f:
                subject_info = json.load(f)
            anthro_data = self.anthro_extractor.add_subject_data(
                anthro_data,
                subject_info
            )
        
        # Convert to feature vector (torch.float32 tensor)
        anthro_features = self.anthro_extractor.to_feature_vector(anthro_data)
        results['anthropometric_features'] = anthro_features
        results['anthropometric_data'] = anthro_data
        
        print(f"\n  Subject {subject_id} processed successfully")
        print(f"    Skin mesh: {results['skin_mesh']['vertices']} vertices")
        print(f"    Muscles: {len(results['muscle_meshes'])} extracted")
        print(f"    Features: {anthro_features.numel()} dimensions")
        
        return results
    
    def _generate_templates(self,
                           skin_meshes: List[trimesh.Trimesh],
                           skin_by_subject: Dict[str, trimesh.Trimesh],
                           muscle_data: Dict,
                           output_path: Path) -> Dict:
        """
        Generate all templates from collected data.
        
        Args:
            skin_meshes: List of skin meshes
            skin_by_subject: Mapping subject_id -> skin mesh
            muscle_data: Dictionary with all muscle data
            output_path: Output path for templates
            
        Returns:
            Dictionary with generated templates
        """
        output_path.mkdir(parents=True, exist_ok=True)
        templates = {}
        
        # Generate skin template
        print("\n1. Generating skin template...")
        skin_template = self.skin_template_gen.create_from_average(
            skin_meshes,
            target_vertices=self.config['template']['skin_vertices']
        )
        
        skin_template_path = output_path / 'skin_template'
        self.skin_template_gen.save(str(skin_template_path))
        templates['skin'] = skin_template
        
        # Generate muscle templates
        print("\n2. Generating muscle templates...")
        muscle_templates = self.muscle_template_gen.create_from_dataset(
            muscle_data,
            skin_meshes_by_subject=skin_by_subject,  # preserve relative placement if available
            min_availability=self.config['template']['min_muscle_availability'],
            target_vertices=self.config['template']['muscle_vertices']
        )
        
        muscle_template_path = output_path / 'muscle_templates.pkl'
        self.muscle_template_gen.save(str(muscle_template_path))
        templates['muscles'] = muscle_templates
        
        # Generate unified template
        print("\n3. Generating unified multi-structure template...")
        unified_template = self.unified_template_gen.create(
            skin_template,
            muscle_templates,
            skin_vertices=self.config['template']['skin_vertices'],
            muscle_vertices=self.config['template']['muscle_vertices']
        )
        
        unified_template_path = output_path / 'unified_template'
        self.unified_template_gen.save(str(unified_template_path))
        templates['unified'] = unified_template
        
        print(f"\nTemplates saved to {output_path}")
        
        return templates
    
    def prepare_training_data(self,
                             processed_data: Dict,
                             template_path: str,
                             output_path: str) -> Dict:
        """
        Prepare training data from processed meshes and templates.
        
        Args:
            processed_data: Dictionary with processed subject data
            template_path: Path to unified template (base path, no suffix)
            output_path: Output path for training data
            
        Returns:
            Training data dictionary
        """
        print("\n" + "="*60)
        print("PREPARING TRAINING DATA")
        print("="*60)
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load unified template
        print("\nLoading unified template...")
        self.unified_template_gen.load(template_path)
        
        training_samples = []
        
        # Process each subject
        for subject_id, subject_data in processed_data.items():
            if subject_id == 'templates' or 'error' in subject_data:
                continue
            
            print(f"\nProcessing {subject_id}...")
            
            # Prepare target meshes
            target_skin = subject_data['skin_mesh']['mesh']
            target_muscles = {
                name: data['mesh']
                for name, data in subject_data.get('muscle_meshes', {}).items()
            }
            
            # Compute deformations
            deformations = self.unified_template_gen.compute_structure_deformations(
                target_skin,
                target_muscles
            )
            
            # Create training sample
            sample = {
                'subject_id': subject_id,
                'anthropometric_features': subject_data['anthropometric_features'],
                'anthropometric_data': subject_data['anthropometric_data'],
                'unified_template_graph': self.unified_template_gen.unified_graph,
                'structure_deformations': deformations,
                'structure_info': self.unified_template_gen.structure_info,
                'metadata': {
                    'skin_vertices': subject_data['skin_mesh']['vertices'],
                    'skin_faces': subject_data['skin_mesh']['faces'],
                    'num_muscles': len(subject_data.get('muscle_meshes', {})),
                }
            }
            
            training_samples.append(sample)
            
            print(f"  Sample created with {len(deformations)} deformation entries")
        
        if not training_samples:
            raise RuntimeError("No training samples were created. Check the processed data.")
        
        # Save training data
        training_data = {
            'samples': training_samples,
            'template_path': template_path,
            'structure_info': self.unified_template_gen.structure_info,
            'config': self.config,
            'metadata': {
                'num_samples': len(training_samples),
                'num_structures': len(self.unified_template_gen.structure_info),
                'feature_dim': int(training_samples[0]['anthropometric_features'].numel()),
            }
        }
        
        training_data_path = output_path / 'training_data.pkl'
        with open(training_data_path, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"\nTraining data saved to {training_data_path}")
        print(f"  Samples: {len(training_samples)}")
        print(f"  Feature dimension: {training_data['metadata']['feature_dim']}")
        print(f"  Structures: {training_data['metadata']['num_structures']}")
        
        return training_data


def main():
    """
    Main function demonstrating the complete pipeline.
    """
    print("\n" + "="*70)
    print("FOREARM MESHNET - COMPLETE PIPELINE DEMONSTRATION")
    print("="*70)
    
    # Initialize pipeline
    pipeline = ForearmMeshNetCompletePipeline()
    
    # Define paths
    data_root = "/path/to/MRI_Data"
    output_root = "/path/to/output"
    
    # Step 1: Process dataset and generate templates
    print("\n" + "="*70)
    print("STEP 1: PROCESS DATASET AND GENERATE TEMPLATES")
    print("="*70)
    
    processed_data = pipeline.process_dataset(
        data_root=data_root,
        output_root=output_root,
        subject_ids=None  # Process all subjects
    )
    
    # Step 2: Prepare training data
    if 'templates' in processed_data:
        print("\n" + "="*70)
        print("STEP 2: PREPARE TRAINING DATA")
        print("="*70)
        
        template_path = Path(output_root) / 'templates' / 'unified_template'
        training_output = Path(output_root) / 'training_data'
        
        training_data = pipeline.prepare_training_data(
            processed_data=processed_data,
            template_path=str(template_path),
            output_path=str(training_output)
        )
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {output_root}")
        print(f"Training data ready at: {training_output}")
        
        # Print summary statistics
        print("\nSUMMARY STATISTICS:")
        print(f"  Subjects processed: {len(processed_data) - 1}")  # -1 for templates
        print(f"  Training samples: {len(training_data['samples'])}")
        print(f"  Feature dimension: {training_data['metadata']['feature_dim']}")
        print(f"  Number of structures: {training_data['metadata']['num_structures']}")
        
        return processed_data, training_data
    
    else:
        print("\nNo templates generated - not enough data")
        return processed_data, None


if __name__ == "__main__":
    # Run the complete pipeline
    processed_data, training_data = main()
    
    print("\n" + "="*70)
    print("FOREARM MESHNET PIPELINE EXECUTION COMPLETE")
    print("="*70)
    
    # Next steps would be:
    # 1. Data normalization
    # 2. Model training
    # 3. Inference
    
    print("\nNext steps:")
    print("1. Normalize the training data")
    print("2. Train the ForearmMeshNet model")
    print("3. Run inference with new anthropometric measurements")
    print("\nRefer to the training and inference modules for these steps.")

