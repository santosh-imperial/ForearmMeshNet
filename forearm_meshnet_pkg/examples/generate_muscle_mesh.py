"""
Example script for using the muscle mesh generation functionality
"""

def example_muscle_mesh_generation():
    """
    Example of generating muscle meshes from MRI data.
    """
    from forearm_meshnet.preprocessing import MuscleMeshGenerator
    
    # Initialize generator
    config = {
        'min_muscle_volume': 100,
        'smooth_iterations': 15,
        'target_vertices': 800,
        'iso_resolution': 0.5
    }
    
    generator = MuscleMeshGenerator(config)
    
    # Paths to your data
    dicom_folder = "path/to/dicom/folder"
    roi_folder = "path/to/roi/folder"
    output_folder = "path/to/output/folder"
    subject_id = "subject_01"
    
    # Load DICOM volume
    print("Loading DICOM volume...")
    volume, spacing = generator.load_dicom_volume(dicom_folder)
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing} mm")
    
    # Generate multi-label mask from ROIs
    print("\nGenerating multi-label mask from ROIs...")
    multi_label_mask = generator.roi_to_multilabel_mask(roi_folder, volume.shape)
    print(f"Unique labels: {np.unique(multi_label_mask)}")
    
    # Generate muscle meshes
    print("\nGenerating muscle meshes...")
    muscle_meshes, stats = generator.generate_all_muscles(
        multi_label_mask,
        volume,
        spacing,
        subject_id,
        output_folder
    )
    
    # Print results
    print(f"\nGenerated {len(muscle_meshes)} muscle meshes:")
    for muscle_name, muscle_data in muscle_meshes.items():
        print(f"  {muscle_name}:")
        print(f"    Vertices: {muscle_data['vertices']}")
        print(f"    Faces: {muscle_data['faces']}")
        print(f"    Volume: {muscle_data['volume_mm3']:.2f} mm³")
        print(f"    Surface area: {muscle_data['surface_area_mm2']:.2f} mm²")
    
    return muscle_meshes


if __name__ == "__main__":
    # Run example
    meshes = example_muscle_mesh_generation()