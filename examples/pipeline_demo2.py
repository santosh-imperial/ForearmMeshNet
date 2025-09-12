"""
ForearmMeshNet - Complete End-to-End Example

This script demonstrates the complete workflow:
1. Data preparation
2. Model training
3. Inference on new subjects
"""

import torch
import numpy as np
from pathlib import Path
import json
import pickle        



# Import all ForearmMeshNet modules
from forearm_meshnet.preprocessing import (
    SkinMaskGenerator, SkinMeshGenerator, MuscleMeshGenerator
)
from forearm_meshnet.template import (
    SkinTemplateGenerator, MuscleTemplateGenerator, UnifiedTemplateGenerator
)
from forearm_meshnet.features import AnthropometricExtractor
from forearm_meshnet.data import (
    TrainingDataPreparation, DataNormalizer, ForearmDataset
)
from forearm_meshnet.models import ForearmMeshNet
from forearm_meshnet.training import Trainer
from forearm_meshnet.inference import Predictor, InferencePipeline


class ForearmMeshNetComplete:
    """
    Complete ForearmMeshNet pipeline from data to inference.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize complete pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        print("="*70)
        print("FOREARM MESHNET - COMPLETE PIPELINE")
        print("="*70)
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'preprocessing': {
                'skin_mask': {
                    'end_slice_fraction': 0.25,
                    'fix_ghosting': True,
                },
                'skin_mesh': {
                    'target_faces': 5000,
                    'smooth_iterations': 50,
                },
                'muscle_mesh': {
                    'min_muscle_volume': 100,
                    'target_vertices': 800,
                }
            },
            'template': {
                'skin_vertices': 5000,
                'muscle_vertices': 500,
                'min_muscle_availability': 0.8,
            },
            'data': {
                'normalization_method': 'standard',
                'val_ratio': 0.2,
                'augment': True,
            },
            'model': {
                'latent_dim': 256,
                'encoder_hidden_dims': [128, 256, 512],
                'decoder_hidden_dims': [512, 256, 128],
                'dropout_rate': 0.1,
                'conv_type': 'gcn',
                'use_template_augmentation': True,
                'use_affine': True,
            },
            'training': {
                'batch_size': 8,
                'num_epochs': 200,
                'optimizer': {
                    'type': 'AdamW',
                    'lr': 1e-4,
                    'weight_decay': 1e-2,
                },
                'scheduler': {
                    'type': 'CosineAnnealingLR',
                    'T_max': 200,
                    'eta_min': 1e-6,
                },
                'early_stopping': {
                    'enabled': True,
                    'patience': 50,
                    'min_epochs': 100,
                },
                'checkpoint_freq': 10,
                'eval_freq': 5,
            }
        }
    
    def run_complete_pipeline(self,
                             data_root: str,
                             output_root: str):
        """
        Run the complete ForearmMeshNet pipeline.
        
        Args:
            data_root: Root directory with MRI data
            output_root: Output directory for all results
        """
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Data Preparation
        print("\n" + "="*60)
        print("STEP 1: DATA PREPARATION")
        print("="*60)
        
        prepared_data_path = output_path / "prepared_data"
        if not prepared_data_path.exists():
            prepared_data = self._prepare_data(data_root, prepared_data_path)
        else:
            print(f"Using existing prepared data from {prepared_data_path}")
        
        # Step 2: Template Generation
        print("\n" + "="*60)
        print("STEP 2: TEMPLATE GENERATION")
        print("="*60)
        
        template_path = output_path / "templates"
        if not (template_path / "unified_template.pkl").exists():
            self._generate_templates(prepared_data_path, template_path)
        else:
            print(f"Using existing templates from {template_path}")
        
        # Step 3: Training Data Creation
        print("\n" + "="*60)
        print("STEP 3: TRAINING DATA CREATION")
        print("="*60)
        
        training_data_path = output_path / "training_data"
        if not (training_data_path / "train_samples.pkl").exists():
            self._create_training_data(
                template_path / "unified_template",
                prepared_data_path,
                training_data_path
            )
        else:
            print(f"Using existing training data from {training_data_path}")
        
        # Step 4: Model Training
        print("\n" + "="*60)
        print("STEP 4: MODEL TRAINING")
        print("="*60)
        
        model_path = output_path / "model"
        if not (model_path / "checkpoints" / "best_model.pt").exists():
            self._train_model(training_data_path, model_path)
        else:
            print(f"Using existing model from {model_path}")
        
        # Step 5: Inference
        print("\n" + "="*60)
        print("STEP 5: INFERENCE")
        print("="*60)
        
        inference_path = output_path / "inference"
        self._run_inference(
            model_path / "checkpoints" / "best_model.pt",
            template_path / "unified_template",
            training_data_path / "normalizers.pkl",
            inference_path
        )
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"All results saved to: {output_path}")
    
    def _prepare_data(self, data_root: str, output_path: Path) -> dict:
        """Prepare mesh data from MRI."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        data_path = Path(data_root)
        subjects = sorted([d for d in data_path.iterdir() if d.is_dir()])
        
        print(f"Found {len(subjects)} subjects")
        
        prepared_data = {
            'skin_meshes': [],
            'muscle_data': {'subjects_data': {}}
        }
        
        for subject_dir in subjects[:5]:  # Process first 5 for demo
            subject_id = subject_dir.name
            print(f"\nProcessing {subject_id}...")
            
            dicom_folder = subject_dir / "mri_files"
            roi_folder = subject_dir / "roi_files"
            
            if not dicom_folder.exists() or not roi_folder.exists():
                continue
            
            # Generate meshes
            try:
                # Load MRI data
                muscle_gen = MuscleMeshGenerator(self.config['preprocessing']['muscle_mesh'])
                volume, spacing = muscle_gen.load_dicom_volume(str(dicom_folder))
                multi_label_mask = muscle_gen.roi_to_multilabel_mask(str(roi_folder), volume.shape)
                
                # Generate skin mesh
                skin_mask_gen = SkinMaskGenerator(self.config['preprocessing']['skin_mask'])
                skin_mesh_gen = SkinMeshGenerator(self.config['preprocessing']['skin_mesh'])
                
                skin_mask = skin_mask_gen.generate(multi_label_mask, volume, spacing)
                skin_mesh = skin_mesh_gen.generate(
                    skin_mask, volume, spacing,
                    output_path=str(output_path / f"{subject_id}_skin.ply")
                )
                
                prepared_data['skin_meshes'].append(skin_mesh)
                
                # Generate muscle meshes
                muscle_meshes, stats = muscle_gen.generate_all_muscles(
                    multi_label_mask, volume, spacing,
                    subject_id, str(output_path / f"{subject_id}_muscles")
                )
                
                prepared_data['muscle_data']['subjects_data'][subject_id] = {
                    'muscle_meshes': muscle_meshes
                }
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Save prepared data
        import pickle
        with open(output_path / "prepared_data.pkl", 'wb') as f:
            pickle.dump(prepared_data, f)
        
        return prepared_data
    
    def _generate_templates(self, data_path: Path, output_path: Path):
        """Generate templates from prepared data."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load prepared data
        import pickle
        with open(data_path / "prepared_data.pkl", 'rb') as f:
            prepared_data = pickle.load(f)
        
        # Generate skin template
        skin_template_gen = SkinTemplateGenerator()
        skin_template = skin_template_gen.create_from_average(
            prepared_data['skin_meshes'],
            target_vertices=self.config['template']['skin_vertices']
        )
        skin_template_gen.save(str(output_path / "skin_template"))
        
        # Generate muscle templates
        muscle_template_gen = MuscleTemplateGenerator()
        muscle_templates = muscle_template_gen.create_from_dataset(
            prepared_data['muscle_data'],
            min_availability=self.config['template']['min_muscle_availability'],
            target_vertices=self.config['template']['muscle_vertices']
        )
        muscle_template_gen.save(str(output_path / "muscle_templates.pkl"))
        
        # Generate unified template
        unified_gen = UnifiedTemplateGenerator()
        unified_template = unified_gen.create(
            skin_template,
            muscle_templates,
            skin_vertices=self.config['template']['skin_vertices'],
            muscle_vertices=self.config['template']['muscle_vertices']
        )
        unified_gen.save(str(output_path / "unified_template"))
        
        print(f"Templates saved to {output_path}")
    
    def _create_training_data(self,
                             template_path: Path,
                             data_path: Path,
                             output_path: Path):
        """Create training dataset."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data preparation
        data_prep = TrainingDataPreparation(
            unified_template_path=str(template_path),
            skin_gt_folder=str(data_path),
            muscle_gt_data_path=str(data_path / "prepared_data.pkl")
        )
        
        # Prepare dataset
        training_samples, statistics = data_prep.prepare_training_dataset(
            output_folder=str(output_path / "raw"),
            max_subjects=None
        )
        
        # Normalize data
        normalizer = DataNormalizer(self.config['data']['normalization_method'])
        normalized_samples = normalizer.fit_and_transform(
            training_samples,
            save_path=str(output_path / "normalizers.pkl")
        )
        
        # Create train/val split
        train_samples, val_samples = data_prep.create_train_val_split(
            normalized_samples,
            val_ratio=self.config['data']['val_ratio']
        )
        
        # Save splits
        
        with open(output_path / "train_samples.pkl", 'wb') as f:
            pickle.dump(train_samples, f)
        
        with open(output_path / "val_samples.pkl", 'wb') as f:
            pickle.dump(val_samples, f)
        
        print(f"Training data saved to {output_path}")
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")
    
    def _train_model(self, data_path: Path, output_path: Path):
        """Train the model."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        with open(data_path / "train_samples.pkl", 'rb') as f:
            train_samples = pickle.load(f)

        with open(data_path / "val_samples.pkl", 'rb') as f:
            val_samples = pickle.load(f)
        
        # Create datasets
        train_dataset = ForearmDataset(train_samples, augment=self.config['data']['augment'])
        val_dataset = ForearmDataset(val_samples, augment=False)
        
        # Extract model config from data
        sample = train_samples[0]
        graph = sample.get('unified_template_graph', None)
        if graph is None or not hasattr(graph, 'x'):
            raise RuntimeError("Sample is missing 'unified_template_graph.x' needed for node_feature_dim.")
        self.config['model']['node_feature_dim'] = sample['unified_template_graph'].x.shape[1]
        self.config['model']['anthro_feature_dim'] = len(sample['anthropometric_features'])
        
        structure_vertex_counts = {}
        for struct_name, deform in sample['structure_deformations'].items():
            if struct_name != 'combined':
                v_count = deform.shape[-2] if deform.ndim >= 2 else len(deform)
                structure_vertex_counts[struct_name] = int(v_count)
        
        self.config['model']['structure_vertex_counts'] = structure_vertex_counts
        self.config['model']['num_structures'] = len(structure_vertex_counts)
        
        # Create model
        model = ForearmMeshNet(self.config['model'])
        self.config['training']['unified_template_pickle'] = str(
            (output_path.parent / "templates" / "unified_template.pkl"))

        # load normalizers and set into the model so losses can denorm to mm
        with open(data_path / "normalizers.pkl", "rb") as f:
            normalizer = pickle.load(f)
        # ensure dict-like
        if hasattr(normalizer, '__dict__'):
            normalizer = vars(normalizer)
        model.set_normalizer(normalizer)
    
            
        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config['training'],
            output_dir=str(output_path)
        )
        
        # Train model
        history = trainer.train(num_epochs=self.config['training']['num_epochs'])
        
        print(f"Training complete!")
        print(f"Best model saved to: {output_path / 'checkpoints' / 'best_model.pt'}")
    
    def _run_inference(self,
                      model_path: Path,
                      template_path: Path,
                      normalizer_path: Path,
                      output_path: Path):
        """Run inference with trained model."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize predictor
        predictor = Predictor(
            model_checkpoint_path=str(model_path),
            template_path=str(template_path),
            normalizer_path=str(normalizer_path)
        )
        
        # Example measurements
        test_measurements = {
            'forearm_length': 260.0,
            'wrist_circumference': 170.0,
            'mid_forearm_circumference': 210.0,
            'proximal_circumference': 240.0,
            'subject_height': 175.0,
            'subject_weight': 70.0,
            'subject_age': 30,
            'subject_gender': 'M',
            'dominant_hand': 'R'
        }
        
        # Generate prediction
        print("\nGenerating prediction for test subject...")
        prediction = predictor.predict(test_measurements, n_samples=3)
        
        # Save results
        predictor.save_prediction(
            prediction,
            str(output_path / "test_prediction")
        )
        
        print(f"Inference results saved to {output_path}")


def main():
    """
    Main function to run the complete ForearmMeshNet pipeline.
    """
    print("\n" + "="*70)
    print("FOREARM MESHNET - COMPLETE DEMONSTRATION")
    print("="*70)
    
    # Setup paths
    data_root = "/path/to/MRI_Data"  # Update this path
    output_root = "./forearm_meshnet_output"
    
    # Initialize pipeline
    pipeline = ForearmMeshNetComplete()
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(data_root, output_root)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    print("\nThe ForearmMeshNet pipeline includes:")
    print("  ✓ Data preprocessing (skin & muscle mesh generation)")
    print("  ✓ Template creation (unified multi-structure)")
    print("  ✓ Training data preparation (with normalization)")
    print("  ✓ Model training (VAE with curriculum learning)")
    print("  ✓ Inference (mesh prediction from measurements)")
    
    print("\nKey Features:")
    print("  • Multi-structure support (skin + 17 muscles)")
    print("  • Anthropometric conditioning (24D features)")
    print("  • Graph neural networks (GCN/SAGE/GINE)")
    print("  • Variational autoencoder framework")
    print("  • Curriculum learning (4 phases)")
    print("  • Volume preservation losses")
    print("  • Affine transformation module")
    
    print("\nApplications:")
    print("  • Personalized prosthetic design")
    print("  • Ergonomic product development")
    print("  • Medical visualization")
    print("  • Biomechanical analysis")
    print("  • Virtual reality avatars")


if __name__ == "__main__":
    # Quick test of individual components
    print("\nTesting individual components...")
    
    # Test anthropometric feature extraction
    from forearm_meshnet.features import AnthropometricExtractor
    
    extractor = AnthropometricExtractor()
    test_measurements = {
        'forearm_length': 250.0,
        'wrist_circumference': 170.0,
        'mid_forearm_circumference': 210.0,
        'proximal_circumference': 240.0,
    }
    
    features = extractor.to_feature_vector(test_measurements)
    print(f"✓ Feature extraction: {features.shape[0]} dimensions")
    
    # Test model instantiation
    from forearm_meshnet.models import ForearmMeshNet
    
    config = {
        'node_feature_dim': 7,
        'anthro_feature_dim': 24,
        'latent_dim': 256,
        'encoder_hidden_dims': [128, 256],
        'decoder_hidden_dims': [256, 128],
        'num_structures': 2,
        'structure_vertex_counts': {'skin': 5000, 'muscle': 500},
        'dropout_rate': 0.1,
        'conv_type': 'gcn'
    }
    
    model = ForearmMeshNet(config)
    param_count = model.get_num_parameters()
    print(f"✓ Model created: {param_count['total']:,} parameters")
    
    print("\nAll components working correctly!")
    
    # Uncomment to run full pipeline
    # main()