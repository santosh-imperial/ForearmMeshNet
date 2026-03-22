"""
Skin mask generation module for ForearmMeshNet
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from scipy import ndimage as ndi_mod
from scipy.signal import savgol_filter
from skimage import filters, measure, morphology

from ..utils.io_utils import load_nifti, save_nifti

logger = logging.getLogger(__name__)

class SkinMaskGenerator:
    """
    Generate skin masks from MRI volumes and muscle segmentation masks.
    
    This class implements a hybrid approach combining edge detection,
    morphological operations, and anatomical constraints to create
    accurate skin masks for forearm reconstruction.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the SkinMaskGenerator.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.end_slice_fraction = self.config.get('end_slice_fraction', 0.25)
        self.fix_ghosting = self.config.get('fix_ghosting', True)
        self.fix_connected_ghosting = self.config.get('fix_connected_ghosting', True)
        self.max_connected_ghosting_fix = self.config.get('max_connected_ghosting_fix', 14)
        self.grad_threshold = self.config.get('grad_threshold', 0.2)
        self.min_component_size = self.config.get('min_component_size', 50)
        self.end_slice_fraction = self.config.get('end_slice_fraction', 0.25)
        self.disk_radius_middle_mm = self.config.get('disk_radius_middle_mm', 3.0)
        self.disk_radius_end_mm = self.config.get('disk_radius_end_mm', 8.0)


    def _elliptical_se(self, ry: int, rx: int) -> np.ndarray:
        """Create an elliptical structuring element with radii (ry, rx) in pixels."""
        yy, xx = np.ogrid[-ry:ry+1, -rx:rx+1]
        return (yy*yy) / ((ry + 1e-8)**2) + (xx*xx) / ((rx + 1e-8)**2) <= 1.0

    def _pixel_radii_from_spacing(self, spacing_zyx):
        # spacing_zyx = (sz, sy, sx); use in-plane spacings only
        sy, sx = spacing_zyx[1], spacing_zyx[2]
        ry_mid = max(1, int(round(self.disk_radius_middle_mm / sy)))
        rx_mid = max(1, int(round(self.disk_radius_middle_mm / sx)))
        ry_end = max(1, int(round(self.disk_radius_end_mm / sy)))
        rx_end = max(1, int(round(self.disk_radius_end_mm / sx)))
        return (ry_mid, rx_mid, ry_end, rx_end)
            
    def generate(self, 
                 mask: np.ndarray,
                 vol: np.ndarray,
                 spacing: np.ndarray) -> np.ndarray:
        """
        Generate skin mask from muscle segmentation and MRI volume.
        
        Args:
            mask: Muscle segmentation mask (Z, Y, X), dtype=uint8
            vol: MRI volume (Z, Y, X)
            spacing: Voxel spacing in mm [Z, Y, X]
            
        Returns:
            skin_mask: Binary skin mask (Z, Y, X)
        """
        logger.info("Generating skin mask...")

        ry_mid, rx_mid, ry_end, rx_end = self._pixel_radii_from_spacing(spacing)
        se_mid = self._elliptical_se(ry_mid, rx_mid)
        se_end = self._elliptical_se(ry_end, rx_end)
        
        # Step 1: Apply ghosting fixes if enabled
        if self.fix_ghosting:
            mask = self._fix_ghosting_artifacts(mask)
            
        if self.fix_connected_ghosting:
            mask = self._fix_connected_ghosting(mask)
            
        # Step 2: Create hybrid skin mask
        skin_mask = self._create_hybrid_skin_mask(mask, vol, spacing,se_mid,se_end)
        
        # Step 3: Apply radial smoothing
        skin_mask = self._apply_radial_smoothing(skin_mask, spacing)
        
        # Step 4: Ensure 3D consistency
        skin_mask = self._ensure_3d_consistency(skin_mask)
        
        logger.info(f"Skin mask generated: {skin_mask.sum():,} voxels")
        return skin_mask
    
    def _fix_ghosting_artifacts(self, mask: np.ndarray) -> np.ndarray:
        """Remove ghosting artifacts from top slices."""
        logger.info("Fixing ghosting artifacts...")
        
        mask_fixed = mask.copy()
        Z = mask.shape[0]
        
        for z in range(min(14, Z)):
            slice_mask = mask[z]
            if slice_mask.sum() > 0:
                # Apply morphological operations to remove artifacts
                cleaned = morphology.binary_opening(slice_mask, morphology.disk(2))
                cleaned = morphology.remove_small_objects(cleaned, min_size=100)
                mask_fixed[z] = cleaned
                
        return mask_fixed
    
    def _fix_connected_ghosting(self, mask: np.ndarray) -> np.ndarray:
        """Fix connected ghosting artifacts."""
        if self.max_connected_ghosting_fix is None:
            return mask
            
        logger.info(f"Fixing connected ghosting (max slice: {self.max_connected_ghosting_fix})...")
        
        mask_fixed = mask.copy()
        
        for z in range(min(self.max_connected_ghosting_fix, mask.shape[0])):
            slice_mask = mask_fixed[z]
            if slice_mask.sum() > 0:
                # Find largest connected component
                labeled = measure.label(slice_mask)
                if labeled.max() > 1:
                    sizes = np.bincount(labeled.ravel())[1:]
                    largest = np.argmax(sizes) + 1
                    mask_fixed[z] = (labeled == largest)
                    
        return mask_fixed
    
    def _create_hybrid_skin_mask(self,
                                  mask: np.ndarray,
                                  vol: np.ndarray,
                                  spacing: np.ndarray,
                                  se_mid,se_end) -> np.ndarray:
        """
        Create hybrid skin mask using edge detection and morphology.
        """
        logger.info("Creating hybrid skin mask...")
        
        Z, Y, X = mask.shape
        skin_mask = np.zeros_like(mask, dtype=bool)
        
        # Calculate end slice count
        end_slice_count = int(Z * self.end_slice_fraction)
        
        # Compute normalized gradient for edge detection
        grad_mag = self._compute_gradient_magnitude(vol)
        
        for z in range(Z):
            # Get current slices
            muscle_slice = mask[z] > 0
            intensity_slice = vol[z]
            grad_slice = grad_mag[z]
            
            # Determine if this is an end slice
            is_end_slice = (z < end_slice_count) or (z >= Z - end_slice_count)
            
            if is_end_slice:
                # More aggressive processing for end slices
                slice_mask = self._process_end_slice(
                    muscle_slice, intensity_slice, grad_slice, se_end)
                
            else:
                # Standard processing for middle slices
                slice_mask = self._process_middle_slice(
                    muscle_slice, intensity_slice, grad_slice,se_mid)
                
            
            skin_mask[z] = slice_mask
            
            # Progress reporting
            if z % 50 == 0:
                slice_type = "END" if is_end_slice else "MIDDLE"
                logger.info(f"  Processed slice {z}/{Z} ({slice_type})")
                
        return skin_mask
    
    def _compute_gradient_magnitude(self, vol: np.ndarray) -> np.ndarray:
        """Compute normalized gradient magnitude of volume."""
        # Compute gradients
        grad_z = np.gradient(vol, axis=0)
        grad_y = np.gradient(vol, axis=1)
        grad_x = np.gradient(vol, axis=2)
        
        # Magnitude
        grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
        
        # Normalize per slice
        for z in range(vol.shape[0]):
            slice_grad = grad_mag[z]
            if slice_grad.max() > 0:
                grad_mag[z] = slice_grad / slice_grad.max()
                
        return grad_mag
    
    def _process_end_slice(self,
                           muscle_slice: np.ndarray,
                           intensity_slice: np.ndarray,
                           grad_slice: np.ndarray,
                           se_end) -> np.ndarray:
        """Process end slices with aggressive closure."""
        # Create tissue mask from intensity
        tissue_mask = self._create_tissue_mask(intensity_slice)
        
        # Aggressive morphological closure
        
        dilated_muscle = ndi.binary_dilation(muscle_slice, structure=se_end)
        
        # Combine with tissue mask
        slice_mask = dilated_muscle | tissue_mask
        
        # Fill holes aggressively
        slice_mask = ndi.binary_fill_holes(slice_mask)
        
        # Ensure single component
        slice_mask = self._keep_largest_component(slice_mask)
        
        # Smooth boundaries
        close_ry = max(1, int(round(0.5 * se_end.shape[0] / 2)))
        close_rx = max(1, int(round(0.5 * se_end.shape[1] / 2)))
        se_close_end = self._elliptical_se(close_ry, close_rx)
        slice_mask = ndi.binary_closing(slice_mask, structure=se_close_end)
        
        return slice_mask
    
    def _process_middle_slice(self,
                              muscle_slice: np.ndarray,
                              intensity_slice: np.ndarray,
                              grad_slice: np.ndarray,se_mid) -> np.ndarray:
        """Process middle slices with standard approach."""
        # Create tissue mask
        tissue_mask = self._create_tissue_mask(intensity_slice)
        
        # Moderate dilation
        dilated_muscle = ndi.binary_dilation(muscle_slice, structure=se_mid)
        
        # Edge-based constraints
        edge_mask = grad_slice > self.grad_threshold
        edge_constrained = dilated_muscle & (~edge_mask | muscle_slice)
        
        # Combine masks
        slice_mask = edge_constrained | tissue_mask | muscle_slice
        
        # Fill holes
        slice_mask = ndi.binary_fill_holes(slice_mask)
        
        # Keep largest component
        slice_mask = self._keep_largest_component(slice_mask)
        
        # Light smoothing
        ry_mid = max(1, int(round(0.5 * se_mid.shape[0] / 2)))
        rx_mid = max(1, int(round(0.5 * se_mid.shape[1] / 2)))
        se_close_mid = self._elliptical_se(max(1, ry_mid//3), max(1, rx_mid//3))
        slice_mask = ndi.binary_closing(slice_mask, structure=se_close_mid)
        
        return slice_mask
    
    def _create_tissue_mask(self, intensity_slice: np.ndarray) -> np.ndarray:
        """Create tissue mask from intensity values."""
        # Adaptive thresholding
        nz = intensity_slice[intensity_slice > 0]
        if nz.size < 100:
            return np.zeros_like(intensity_slice, dtype=bool)
        normalized = (intensity_slice - nz.min()) / (nz.ptp() + 1e-8)
        try:
            thr = filters.threshold_otsu(normalized[nz > 0])
        except ValueError:
            return np.zeros_like(intensity_slice, dtype=bool)
        tissue_mask = normalized > (self.config.get('tissue_otsu_multiplier', 0.3) * thr)

        #Cleaning
        tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=self.min_component_size)
        
        return tissue_mask
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        labeled = measure.label(mask)
        if labeled.max() == 0:
            return mask
            
        sizes = np.bincount(labeled.ravel())[1:]
        if len(sizes) == 0:
            return mask
            
        largest = np.argmax(sizes) + 1
        return labeled == largest
    
    def _apply_radial_smoothing(self,
                                 skin_mask: np.ndarray,
                                 spacing: np.ndarray) -> np.ndarray:
        """Apply radial smoothing for anatomical consistency."""
        logger.info("Applying radial smoothing...")
        
        smooth_mask = skin_mask.copy()
        
        for z in range(skin_mask.shape[0]):
            if skin_mask[z].sum() > 0:
                # Get slice
                slice_mask = skin_mask[z]
                
                centroid = ndi_mod.center_of_mass(slice_mask)
                if any(np.isnan(centroid)):
                    smooth_mask[z] = slice_mask  # fallback: keep original
                    continue
                cy, cx = centroid

                # Precompute pixel-scaled step per axis (mm -> px)
                sy, sx = spacing[1], spacing[2]
                def mm_to_px(dy_mm, dx_mm):
                    return dy_mm / sy, dx_mm / sx

                smooth_slice = self._smooth_radially(
                    slice_mask,
                    center=(cy, cx),
                    spacing=(sy, sx)  # pass spacing to the function
                )
                smooth_mask[z] = smooth_slice
                    
        return smooth_mask
    
   
    def _smooth_radially(self, slice_mask, center, spacing) -> np.ndarray:
            """Apply radial smoothing to a single slice."""
            cy, cx = center
            sy, sx = spacing
            angles = np.linspace(0, 2*np.pi, 360)
            radii = []

            # Require enough pixels to attempt radial smoothing
            if slice_mask.sum() < 50:
                # fallback: small clean up
                return morphology.binary_closing(slice_mask, morphology.disk(1))

            for angle in angles:
                # Step along ray in PIXELS (not mm)
                dy = np.sin(angle)
                dx = np.cos(angle)
                r = self._find_max_radius_px(slice_mask, (cy, cx), (dy, dx))
                radii.append(r)

            # Smooth radii
            radii_smooth = savgol_filter(radii, window_length=31, polyorder=3, mode="interp")
            # Reconstruct mask from smooth radii
            return self._reconstruct_from_radii_px(slice_mask.shape, (cy, cx), angles, radii_smooth)
    
    
    def _find_max_radius_px(self, mask, center, direction, max_steps: int = 500) -> float:
            """Find maximum radius in given direction."""
            cy, cx = center
            dy, dx = direction
            # normalize direction
            norm = np.hypot(dy, dx) + 1e-8
            dy /= norm; dx /= norm

            # step in pixel increments
            for step in range(1, max_steps):
                y = int(round(cy + dy * step))
                x = int(round(cx + dx * step))
                if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
                    return step - 1
                if not mask[y, x]:
                    return step - 1
            return max_steps - 1
    
    
    def _reconstruct_from_radii_px(self, shape, center, angles, radii) -> np.ndarray:
        """Reconstruct mask from radial representation."""
        cy, cx = center
        pts = []
        # Create polygon from radii
        for ang, r in zip(angles, radii):
            x = cx + r * np.cos(ang)
            y = cy + r * np.sin(ang)
            pts.append((x, y))

         # Create polygon and rasterize
        poly = Polygon(pts)
        out = np.zeros(shape, dtype=bool)
        if poly.is_valid and not poly.is_empty:
            xs, ys = poly.exterior.xy
            rr, cc = raster_poly(ys, xs, shape)
            out[rr, cc] = True
        return out
        
    def _ensure_3d_consistency(self, skin_mask: np.ndarray) -> np.ndarray:
        """Ensure 3D consistency of the mask."""
        logger.info("Ensuring 3D consistency...")
        
        # 3D morphological operations
        struct3d = ndi.generate_binary_structure(3, 1)
        
        # Light closing
        skin_mask = ndi.binary_closing(skin_mask, structure=struct3d)
        
        # Keep largest 3D component
        labeled_3d = measure.label(skin_mask, connectivity=2)
        if labeled_3d.max() > 1:
            sizes_3d = np.bincount(labeled_3d.ravel())[1:]
            largest_3d = np.argmax(sizes_3d) + 1
            skin_mask = (labeled_3d == largest_3d)
        
        # Final smoothing
        skin_mask = ndi.gaussian_filter(
            skin_mask.astype(np.float32), sigma=0.6
        ) > 0.5
        
        logger.info(f"  Final mask: {skin_mask.sum():,} voxels")
        
        return skin_mask
    


def generate_skin_mask(
    in_muscle_mask_nifti: str | Path,
    in_mri_nifti: str | Path,
    out_skin_mask_nifti: str | Path,
    spacing_zyx: tuple[float, float, float] | None = None,
    config: dict | None = None,
) -> Path:
    """Convenience wrapper: load NIfTI -> run SkinMaskGenerator -> save NIfTI."""
    muscle, affine, _ = load_nifti(in_muscle_mask_nifti)
    vol, affine2, _ = load_nifti(in_mri_nifti)
    assert muscle.shape == vol.shape, "Muscle and MRI affines mismatch."

    if spacing_zyx is None:
        # Fallback: derive spacing from affine if available
        # assumes RAS orientation
        spacing_zyx = (np.linalg.norm(affine[:3,0]),
                       np.linalg.norm(affine[:3,1]),
                       np.linalg.norm(affine[:3,2]))

    gen = SkinMaskGenerator(config=config)
    skin = gen.generate(mask=muscle.astype(np.uint8), vol=vol.astype(np.float32), spacing=np.array(spacing_zyx))
    save_nifti(out_skin_mask_nifti, skin.astype(np.uint8), affine)
    return Path(out_skin_mask_nifti)