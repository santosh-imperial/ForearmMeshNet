# forearm_meshnet/preprocessing/muscle_mesh.py
"""
Muscle mesh generation module for ForearmMeshNet
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pymeshfix
import scipy.ndimage as ndi
import SimpleITK as sitk
import trimesh
from skimage import measure

from .mesh_utils import cap_then_polish

logger = logging.getLogger(__name__)


# Muscle definitions
KEEP_MUSCLES = [
    "APL", "ECRB", "ECRL", "ECU", "ED", "EDM", "EPL",
    "FCR", "FCU", "FDP", "FDS", "FPL", "PL", "PQ", "PT", "SUP", "ANC"
]

MUSCLE_LABEL_LUT = {name: i+1 for i, name in enumerate(KEEP_MUSCLES)}

MUSCLE_FULL_NAMES = {
    "APL": "abductor_pollicis_longus",
    "ECRB": "extensor_carpi_radialis_brevis",
    "ECRL": "extensor_carpi_radialis_longus",
    "ECU": "extensor_carpi_ulnaris",
    "ED": "extensor_digitorum",
    "EDM": "extensor_digiti_minimi",
    "EPL": "extensor_pollicis_longus",
    "FCR": "flexor_carpi_radialis",
    "FCU": "flexor_carpi_ulnaris",
    "FDP": "flexor_digitorum_profundus",
    "FDS": "flexor_digitorum_superficialis",
    "FPL": "flexor_pollicis_longus",
    "PL": "palmaris_longus",
    "PQ": "pronator_quadratus",
    "PT": "pronator_teres",
    "SUP": "supinator",
    "ANC": "anconeus"
}


class MuscleMeshGenerator:
    """
    Generate individual muscle meshes from multi-label segmentation masks.
    
    This class extracts meshes for each muscle from MRI segmentation data
    and applies smoothing, decimation, and quality improvements.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the MuscleMeshGenerator.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.min_muscle_volume = self.config.get('min_muscle_volume', 100)
        self.smooth_iterations = self.config.get('smooth_iterations', 15)
        self.target_vertices = self.config.get('target_vertices', 800)
        self.iso_resolution = self.config.get('iso_resolution', 0.5)
        
        logger.info("MuscleMeshGenerator initialized")
        logger.info(f"  Available muscles: {len(KEEP_MUSCLES)}")
        logger.info(f"  Min muscle volume: {self.min_muscle_volume} voxels")
    
    def generate_all_muscles(self,
                            multi_label_mask: np.ndarray,
                            vol: np.ndarray,
                            spacing: np.ndarray,
                            subject_id: str,
                            output_folder: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Generate meshes for all muscles in a subject.
        
        Args:
            multi_label_mask: Multi-label segmentation mask (Z, Y, X)
            vol: MRI volume (Z, Y, X)
            spacing: Voxel spacing in mm [Z, Y, X]
            subject_id: Subject identifier
            output_folder: Optional folder to save individual meshes
            
        Returns:
            muscle_meshes: Dictionary of muscle_name -> mesh data
            extraction_stats: Statistics about extraction process
        """
        logger.info(f"EXTRACTING ALL MUSCLE MESHES FOR SUBJECT {subject_id}")
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        muscle_meshes = {}
        extraction_stats = {
            'successful': [],
            'failed': [],
            'insufficient_volume': []
        }
        
        # Check which muscles are present
        unique_labels = np.unique(multi_label_mask)
        logger.info(f"Available labels in mask: {unique_labels}")
        
        # Process each muscle
        for muscle_abbrev, muscle_label in MUSCLE_LABEL_LUT.items():
            if muscle_label not in unique_labels:
                logger.info(f"  {muscle_abbrev}: not found in segmentation")
                extraction_stats['failed'].append(muscle_abbrev)
                continue
            
            muscle_name = MUSCLE_FULL_NAMES.get(muscle_abbrev, muscle_abbrev)
            logger.info(f"  Extracting {muscle_abbrev} ({muscle_name})...")
            
            # Resample to isotropic
            vol_iso, mask_iso, sp_iso = self._resample_isotropic(
                vol, multi_label_mask, spacing
            )
            
            # Extract mesh for this muscle
            mesh = self._extract_muscle_mesh(
                mask_iso,
                muscle_label,
                sp_iso
            )
            
            if mesh is not None:
                muscle_meshes[muscle_abbrev] = {
                    'mesh': mesh,
                    'full_name': muscle_name,
                    'label': muscle_label,
                    'vertices': len(mesh.vertices),
                    'faces': len(mesh.faces),
                    'volume_mm3': mesh.volume if mesh.is_watertight else 0,
                    'surface_area_mm2': mesh.area
                }
                
                extraction_stats['successful'].append(muscle_abbrev)
                
                # Save individual mesh if requested
                if output_folder:
                    mesh_path = os.path.join(
                        output_folder,
                        f'subject{subject_id}_muscle_{muscle_abbrev}_{muscle_name}.ply'
                    )
                    mesh.export(mesh_path)
                    logger.info(f"    Saved: {mesh_path}")
            else:
                extraction_stats['failed'].append(muscle_abbrev)
        
        # Print summary
        self._print_extraction_summary(subject_id, extraction_stats)
        
        return muscle_meshes, extraction_stats
    
    def _resample_isotropic(self,
                           vol: np.ndarray,
                           mask: np.ndarray,
                           spacing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample volume and mask to isotropic voxels.
        """
        logger.info(f"1. Resampling to isotropic voxels ({self.iso_resolution}mm)...")
        iso = float(self.iso_resolution)
        zoom = spacing / iso  # axis-wise zoom factors (Z, Y, X)

        # Trilinear for MRI, nearest for labels (exactly as in pipeline)
        vol_iso  = ndi.zoom(vol,  zoom, order=1)
        mask_iso = ndi.zoom(mask.astype(np.uint8), zoom, order=0)

        sp_iso = np.array([iso, iso, iso], dtype=np.float32)  # ZYX
        return vol_iso, mask_iso, sp_iso
        
    def _extract_muscle_mesh(self,
                            multi_label_mask: np.ndarray,
                            muscle_label: int,
                            spacing: np.ndarray) -> Optional[trimesh.Trimesh]:
        """
        Extract individual muscle mesh from multi-label mask.
        """
        # Extract binary mask for this muscle
        muscle_mask = (multi_label_mask == muscle_label).astype(np.uint8)
        
        # Check volume
        muscle_volume = muscle_mask.sum()
        if muscle_volume < self.min_muscle_volume:
            logger.info(f"    Insufficient volume: {muscle_volume} voxels")
            return None
        
        logger.info(f"    Processing {muscle_volume} voxels...")
        
        # Morphological cleanup
        muscle_mask = ndi.binary_fill_holes(muscle_mask)
        muscle_mask = ndi.binary_opening(muscle_mask, structure=np.ones((3, 3, 3)))
        muscle_mask = ndi.binary_closing(muscle_mask, structure=np.ones((3, 3, 3)))
        
        # Keep largest component
        labeled, n_components = ndi.label(muscle_mask)
        if n_components > 1:
            component_sizes = [(i, (labeled == i).sum()) for i in range(1, n_components + 1)]
            largest_component = max(component_sizes, key=lambda x: x[1])[0]
            muscle_mask = (labeled == largest_component)
        
        # Smooth mask
        muscle_mask = ndi.gaussian_filter(muscle_mask.astype(float), sigma=1.0) > 0.5
        
        try:
            # Extract mesh using marching cubes
            verts, faces, _, _ = measure.marching_cubes(
                muscle_mask.astype(float),
                level=0.6,
                spacing=spacing,
                allow_degenerate=False
            )
            # Face orientation check (flip if negative volume)
            mesh_temp = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=False)
            if mesh_temp.volume < 0:
                faces = np.fliplr(faces)
            
            # Create trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            
            # Repair mesh
            mesh = self._repair_muscle_mesh(mesh)
            
            # Polish mesh
            mesh = cap_then_polish(mesh, target_vertices=self.target_vertices, smooth_iter=self.smooth_iterations)
            
            logger.info(f"    Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            return mesh
            
        except Exception as e:
            logger.warning(f"    Mesh extraction failed: {e}")
            return None
    
    def _repair_muscle_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Repair muscle mesh using pymeshfix.
        """
        try:
            mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            mf.repair(
                verbose=False,
                joincomp=True,
                remove_smallest_components=True
            )
            
            # Create repaired mesh
            mesh = trimesh.Trimesh(vertices=mf.v, faces=mf.f, process=False)
            
            # Post-repair cleanup
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
            
            return mesh
            
        except Exception as e:
            logger.warning(f"      Repair failed: {e}")
            return mesh
    
    
    def _print_extraction_summary(self, subject_id: str, stats: Dict):
        """
        Print extraction summary.
        """
        logger.info(f"EXTRACTION SUMMARY FOR SUBJECT {subject_id}")
        logger.info(f"Successful: {len(stats['successful'])} muscles")
        logger.info(f"Failed: {len(stats['failed'])} muscles")

        if stats['successful']:
            logger.info(f"Successfully extracted: {', '.join(stats['successful'])}")

        if stats['failed']:
            logger.info(f"Failed to extract: {', '.join(stats['failed'])}")
    
    def load_dicom_volume(self, dicom_folder: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load DICOM volume from folder.
        
        Args:
            dicom_folder: Path to DICOM files
            
        Returns:
            volume: 3D numpy array
            spacing: Voxel spacing [Z, Y, X]
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        volume = sitk.GetArrayFromImage(image)
        spacing = np.array(image.GetSpacing()[::-1])  # Convert to ZYX
        
        return volume, spacing
    
    def roi_to_multilabel_mask(self,
                              roi_folder: str,
                              volume_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert ROI files to multi-label mask.
        
        Args:
            roi_folder: Path to ROI files
            volume_shape: Shape of the volume (Z, Y, X)
            
        Returns:
            Multi-label mask with muscle labels
        """
        from roifile import ImagejRoi
        from skimage.draw import polygon
        
        multi_label_mask = np.zeros(volume_shape, dtype=np.uint8)
        
        # Process each ROI file
        roi_files = list(Path(roi_folder).glob("*.roi"))
        
        for roi_file in roi_files:
            # Extract muscle name from filename
            filename = roi_file.stem
            muscle_abbrev = None
            
            for abbrev in MUSCLE_LABEL_LUT.keys():
                if abbrev in filename.upper():
                    muscle_abbrev = abbrev
                    break
            
            if muscle_abbrev is None:
                continue
            
            muscle_label = MUSCLE_LABEL_LUT[muscle_abbrev]
            
            try:
                # Read ROI
                roi = ImagejRoi.fromfile(roi_file)
                
                # Get slice index from filename (assuming format like "slice_XX_muscle.roi")
                import re
                slice_match = re.search(r'slice[_-]?(\d+)', filename, re.IGNORECASE)
                if slice_match:
                    slice_idx = int(slice_match.group(1))
                else:
                    # Try to get from ROI position
                    slice_idx = roi.position if hasattr(roi, 'position') else 0
                
                # Get polygon coordinates
                if hasattr(roi, 'integer_coordinates'):
                    coords = roi.integer_coordinates
                elif hasattr(roi, 'coordinates'):
                    coords = roi.coordinates
                else:
                    continue
                
                # Draw polygon on mask
                if slice_idx < volume_shape[0]:
                    rr, cc = polygon(coords[:, 1], coords[:, 0], shape=volume_shape[1:])
                    multi_label_mask[slice_idx, rr, cc] = muscle_label
                    
            except Exception as e:
                logger.warning(f"    Error processing {roi_file}: {e}")
                continue
        
        return multi_label_mask


