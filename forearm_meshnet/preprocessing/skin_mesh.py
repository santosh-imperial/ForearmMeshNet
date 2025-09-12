# forearm_meshnet/preprocessing/skin_mesh.py
"""
Skin mesh generation module for ForearmMeshNet
"""

import numpy as np
import trimesh
import scipy.ndimage as ndi
from skimage import measure, morphology
from typing import Optional, Tuple, Dict, Any
import warnings
import pymeshfix
from .mesh_utils import cap_then_polish, remove_spurious_triangles, detect_isolated_components



class SkinMeshGenerator:
    """
    Generate skin meshes from skin masks.
    
    This class implements mesh generation using marching cubes,
    followed by smoothing, decimation, and quality improvements.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the SkinMeshGenerator.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.iso_resolution = self.config.get('iso_resolution', 0.5)
        self.sdf_blur_sigma = self.config.get('sdf_blur_sigma', 1.5)
        self.target_faces = self.config.get('target_faces', 50000)
        self.smooth_iterations = self.config.get('smooth_iterations', 50)
        self.max_edge_length = self.config.get('max_edge_length', 15.0)
        self.refinement_level = self.config.get('refinement_level', 'medium')
        
    def generate(self,
                 skin_mask: np.ndarray,
                 vol: np.ndarray,
                 spacing: np.ndarray,
                 output_path: Optional[str] = None) -> trimesh.Trimesh:
        """
        Generate skin mesh from skin mask.
        
        Args:
            skin_mask: Binary skin mask (Z, Y, X)
            vol: Original MRI volume (Z, Y, X)
            spacing: Voxel spacing in mm [Z, Y, X]
            output_path: Optional path to save the mesh
            
        Returns:
            mesh: Generated skin mesh
        """
        print("\n" + "="*50)
        print("SKIN MESH GENERATION")
        print("="*50)
        
        # Step 1: Isotropic resampling
        vol_iso, mask_iso, sp_iso = self._resample_isotropic(
            vol, skin_mask, spacing
        )
        
        # Step 2: Generate signed distance field
        sdist = self._create_signed_distance_field(mask_iso, sp_iso)
        
        # Step 3: Extract mesh using marching cubes
        mesh = self._extract_mesh(sdist, sp_iso)
        
        if mesh is None:
            raise ValueError("Failed to generate mesh")
        
        # Step 4: Process mesh (smooth, decimate, repair)
        mesh_processed = self._process_mesh(mesh)
        
        # Step 5: Validate mesh quality
        quality_report = self._validate_mesh_quality(mesh_processed)
        self._print_quality_report(quality_report)
        
        # Save if requested
        if output_path:
            mesh_processed.export(output_path)
            print(f"\nMesh saved to: {output_path}")
        
        return mesh_processed
    class SkinMeshGenerator:
   
    def generate_robust(
        self,
        skin_mask: np.ndarray,
        vol: np.ndarray,
        spacing: np.ndarray,
        output_path: Optional[str] = None,
        artifact_removal: bool = True,
    ) -> trimesh.Trimesh:
        """
        Robust skin mesh generation in case of too many artefacts.
        """
        print("\n" + "="*50)
        print("SKIN MESH GENERATION (ROBUST)")
        print("="*50)

        # 1) Isotropic resampling (same as enhanced)
        vol_iso, mask_iso, sp_iso = self._resample_isotropic(vol, skin_mask, spacing)

        # 2) Robust SDF: pre-smooth mask, keep largest component, then EDT
        print("\n2. Creating robust signed distance field...")
        smooth_mask = ndi.gaussian_filter(mask_iso.astype(np.float32), sigma=1.0) > 0.5

        labeled, n_comp = ndi.label(smooth_mask)
        if n_comp > 1:
            sizes = np.bincount(labeled.ravel())[1:]
            largest = np.argmax(sizes) + 1
            smooth_mask = (labeled == largest)
            print(f"   Using largest component out of {n_comp}")

        d_out = ndi.distance_transform_edt(~smooth_mask) * sp_iso[0]
        d_in  = ndi.distance_transform_edt(smooth_mask)  * sp_iso[0]
        sdist = d_out - d_in
        sdist = ndi.gaussian_filter(sdist, sigma=self.sdf_blur_sigma)
        print(f"   SDF range: [{sdist.min():.2f}, {sdist.max():.2f}] mm")

        # 3) Robust marching cubes
        print("\n3. Extracting mesh with robust marching cubes...")
        try:
            verts, faces, normals, values = measure.marching_cubes(
                sdist,
                level=0.1,                      # slight positive level
                spacing=sp_iso,
                gradient_direction='descent',
                step_size=1,
                allow_degenerate=False
            )
            initial_mesh = trimesh.Trimesh(verts, faces, process=False)
            print(f"   Initial mesh: {len(initial_mesh.vertices):,} v, {len(initial_mesh.faces):,} f")

            # Initial cleanup
            initial_mesh.remove_duplicate_faces()
            initial_mesh.remove_unreferenced_vertices()
            initial_mesh.remove_degenerate_faces()

            # 4) Conservative MeshFix ONLY if not watertight
            if not initial_mesh.is_watertight:
                print("4. Mesh not watertight, applying conservative PyMeshFix...")
                try:
                    mfix = pymeshfix.MeshFix(initial_mesh.vertices, initial_mesh.faces)
                    mfix.repair(
                        verbose=False,
                        joincomp=False,  # conservative: do not join components
                        remove_smallest_components=True
                    )
                    repaired_mesh = trimesh.Trimesh(vertices=mfix.v, faces=mfix.f, process=False)

                    # If face count explodes (>1.5×), keep original
                    if len(repaired_mesh.faces) > len(initial_mesh.faces) * 1.5:
                        print("   Repair added too many faces, keeping original mesh")
                        mesh = initial_mesh
                    else:
                        mesh = repaired_mesh
                except Exception as e:
                    print(f"   Mesh repair failed: {e}, using original mesh")
                    mesh = initial_mesh
            else:
                print("4. Mesh already watertight")
                mesh = initial_mesh

            # 5) Final artifact cleanup
            if artifact_removal:
                print("5. Final artifact cleanup...")
                mesh = remove_spurious_triangles(
                    mesh,
                    max_edge_length_mm=self.max_edge_length * 0.8,
                    max_face_area_mm2=30.0
                )
                mesh = detect_isolated_components(mesh, min_volume_ratio=0.01)

            # 6) Final cleanup
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_faces()

            print(f"   Final robust mesh: {len(mesh.vertices):,} v, {len(mesh.faces):,} f")
            print(f"   Watertight: {mesh.is_watertight}")

            if output_path:
                mesh.export(output_path)
                print(f"\nMesh saved to: {output_path}")

            return mesh

        except Exception as e:
            print(f"ERROR in robust mesh generation: {e}")
            raise
    
    def _resample_isotropic(self,
                           vol: np.ndarray,
                           mask: np.ndarray,
                           spacing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample volume and mask to isotropic voxels.
        """
        print(f"\n1. Resampling to isotropic voxels ({self.iso_resolution}mm)...")
        
        iso = float(self.iso_resolution)
        zoom = spacing / iso  # axis-wise zoom factors (Z, Y, X)

        # Trilinear for MRI, nearest for labels (exactly as in pipeline)
        vol_iso  = ndi.zoom(vol,  zoom, order=1)
        mask_iso = ndi.zoom(mask.astype(np.uint8), zoom, order=0).astype(bool)

        # Post-resample cleanup used in the notebook
        mask_iso = ndi.binary_closing(mask_iso, structure=np.ones((3, 5, 5)))

        sp_iso = np.array([iso, iso, iso], dtype=np.float32)  # ZYX

        print(f"  Original shape: {vol.shape} at {spacing}mm")
        print(f"  Isotropic shape: {vol_iso.shape} at {sp_iso}mm")
        return vol_iso, mask_iso, sp_iso
    

    def _create_signed_distance_field(self,
                                      mask: np.ndarray,
                                      spacing: np.ndarray) -> np.ndarray:
        """
        Create signed distance field from binary mask.
        """
        print("\n2. Creating signed distance field...")
        
        # Ensure single connected component
        labeled, n_components = ndi.label(mask)
        if n_components > 1:
            sizes = np.bincount(labeled.ravel())[1:]
            largest = np.argmax(sizes) + 1
            mask = (labeled == largest)
            print(f"  Using largest component ({n_components} components found)")
        
        # Compute distance transforms
        d_out = ndi.distance_transform_edt(~mask) * spacing[0]
        d_in = ndi.distance_transform_edt(mask) * spacing[0]
        
        # Signed distance field
        sdist = d_out - d_in
        
        # Smooth SDF to prevent artifacts
        sdist = ndi.gaussian_filter(sdist, sigma=self.sdf_blur_sigma)
        
        print(f"  SDF range: [{sdist.min():.2f}, {sdist.max():.2f}] mm")
        
        return sdist
    
    def _extract_mesh(self,
                     sdist: np.ndarray,
                     spacing: np.ndarray) -> Optional[trimesh.Trimesh]:
        """
        Extract mesh using marching cubes.
        """
        print("\n3. Extracting mesh with marching cubes...")
        
        try:
            # Extract mesh at zero level set
            verts, faces, normals, values = measure.marching_cubes(
                sdist, 0.0, spacing=spacing
            )
            
            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            
            print(f"  Initial mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
            
            # Remove artifacts
            mesh = self._remove_artifacts(mesh)
            
            return mesh
            
        except Exception as e:
            print(f"  ERROR in mesh extraction: {e}")
            return None
    
    def _remove_artifacts(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Remove mesh artifacts (spurious triangles, isolated components).
        """
        print("  Removing artifacts...")
        
        # Remove spurious triangles
        edges = mesh.edges_unique
        edge_lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
        )
        
        # Find faces with long edges
        face_mask = np.ones(len(mesh.faces), dtype=bool)
        for face_idx, face in enumerate(mesh.faces):
            face_edges = [
                tuple(sorted([face[i], face[(i+1)%3]])) for i in range(3)
            ]
            for edge in face_edges:
                edge_idx = np.where((edges == edge).all(axis=1) | 
                                   (edges == edge[::-1]).all(axis=1))[0]
                if len(edge_idx) > 0 and edge_lengths[edge_idx[0]] > self.max_edge_length:
                    face_mask[face_idx] = False
                    break
        
        # Update mesh
        mesh.update_faces(face_mask)
        mesh.remove_unreferenced_vertices()
        
        # Remove isolated components
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Keep largest component
            largest = max(components, key=lambda c: len(c.vertices))
            mesh = largest
            print(f"    Removed {len(components)-1} isolated components")
        
        print(f"    After cleanup: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
        
        return mesh
    
    def _process_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Process mesh with smoothing, decimation, and repair.
        """
        print("\n4. Processing mesh...")
        
        # Step 1: Repair with pymeshfix
        mesh = self._repair_mesh(mesh)
        
        # Step 2: Feature-preserving decimation
        mesh = cap_then_polish(mesh, target_faces=self.target_faces, smooth_iter=self.smooth_iterations)
        
        # Step 3: Final cleanup
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_faces()
        
        print(f"  Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
        
        return mesh
    
    def _repair_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Repair mesh using pymeshfix.
        """
        print("  Repairing mesh...")
        
        try:
            mfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            mfix.repair(
                verbose=False,
                joincomp=True,  
                remove_smallest_components=True
            )
            
            mesh_repaired = trimesh.Trimesh(
                vertices=mfix.v,
                faces=mfix.f,
                process=False
            )

            mesh_repaired.update_faces(mesh_repaired.nondegenerate_faces())
            mesh_repaired.update_faces(mesh_repaired.unique_faces())
            mesh_repaired.remove_unreferenced_vertices()
            
            print(f"    Repaired: watertight={mesh_repaired.is_watertight}")

            return mesh_repaired
            
        except Exception as e:
            print(f"    WARNING: Repair failed: {e}")
            return mesh
    
    def _decimate_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Decimate mesh to target face count.
        """
        print(f"  Decimating to {self.target_faces:,} faces...")
        
        if len(mesh.faces) <= self.target_faces:
            print(f"    Mesh already has {len(mesh.faces):,} faces, skipping decimation")
            return mesh
        
        try:
            import fast_simplification
            
            # Calculate decimation ratio
            ratio = self.target_faces / len(mesh.faces)
            
            # Perform decimation
            v_decimated, f_decimated = fast_simplification.simplify(
                mesh.vertices,
                mesh.faces,
                target_reduction=1-ratio,
                preserve_topology=True,
                max_iteration=10
            )
            
            mesh_decimated = trimesh.Trimesh(
                vertices=v_decimated,
                faces=f_decimated,
                process=False
            )
            
            print(f"    Decimated: {len(mesh.faces):,} → {len(mesh_decimated.faces):,} faces")
            return mesh_decimated
            
        except ImportError:
            print("    WARNING: fast_simplification not available, using basic decimation")
            return mesh.simplify_quadric_decimation(self.target_faces)
        except Exception as e:
            print(f"    WARNING: Decimation failed: {e}")
            return mesh
    
    def _smooth_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Apply Taubin smoothing to mesh.
        """
        print(f"  Smoothing ({self.smooth_iterations} iterations)...")
        
        # Taubin smoothing parameters
        lambda_factor = 0.5
        mu_factor = -0.53
        
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        # Build adjacency
        edges = mesh.edges
        adjacency = {i: set() for i in range(len(vertices))}
        for edge in edges:
            adjacency[edge[0]].add(edge[1])
            adjacency[edge[1]].add(edge[0])
        
        # Apply Taubin smoothing
        for iteration in range(self.smooth_iterations):
            # Forward pass (shrinking)
            vertices_new = vertices.copy()
            for i, neighbors in adjacency.items():
                if neighbors:
                    neighbor_mean = vertices[list(neighbors)].mean(axis=0)
                    vertices_new[i] += lambda_factor * (neighbor_mean - vertices[i])
            vertices = vertices_new
            
            # Backward pass (expansion)
            vertices_new = vertices.copy()
            for i, neighbors in adjacency.items():
                if neighbors:
                    neighbor_mean = vertices[list(neighbors)].mean(axis=0)
                    vertices_new[i] += mu_factor * (neighbor_mean - vertices[i])
            vertices = vertices_new
            
            if (iteration + 1) % 10 == 0:
                print(f"    Iteration {iteration + 1}/{self.smooth_iterations}")
        
        mesh_smooth = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        return mesh_smooth
    
    def _validate_mesh_quality(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Validate mesh quality metrics.
        """
        print("\n5. Validating mesh quality...")
        
        quality = {
            'is_watertight': mesh.is_watertight,
            'is_manifold': mesh.is_winding_consistent,
            'is_valid': mesh.is_valid,
            'vertex_count': len(mesh.vertices),
            'face_count': len(mesh.faces),
            'surface_area': mesh.area,
            'volume': mesh.volume if mesh.is_watertight else None,
            'bounds': mesh.bounds,
            'center_of_mass': mesh.center_mass if mesh.is_watertight else mesh.centroid,
        }
        
        # Edge statistics
        edges = mesh.edges_unique
        edge_lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
        )
        quality['edge_stats'] = {
            'min': edge_lengths.min(),
            'max': edge_lengths.max(),
            'mean': edge_lengths.mean(),
            'std': edge_lengths.std()
        }
        
        # Face area statistics
        face_areas = mesh.area_faces
        quality['face_area_stats'] = {
            'min': face_areas.min(),
            'max': face_areas.max(),
            'mean': face_areas.mean(),
            'std': face_areas.std()
        }
        
        return quality
    
    def _print_quality_report(self, quality: Dict[str, Any]):
        """
        Print mesh quality report.
        """
        print("\nMESH QUALITY REPORT")
        print("="*50)
        print(f"Watertight: {quality['is_watertight']}")
        print(f"Manifold: {quality['is_manifold']}")
        print(f"Valid: {quality['is_valid']}")
        print(f"Vertices: {quality['vertex_count']:,}")
        print(f"Faces: {quality['face_count']:,}")
        print(f"Surface area: {quality['surface_area']:.2f} mm²")
        if quality['volume'] is not None:
            print(f"Volume: {quality['volume']:.2f} mm³")
        print(f"\nEdge lengths (mm):")
        print(f"  Min: {quality['edge_stats']['min']:.2f}")
        print(f"  Max: {quality['edge_stats']['max']:.2f}")
        print(f"  Mean: {quality['edge_stats']['mean']:.2f}")
        print(f"  Std: {quality['edge_stats']['std']:.2f}")
        print(f"\nFace areas (mm²):")
        print(f"  Min: {quality['face_area_stats']['min']:.2f}")
        print(f"  Max: {quality['face_area_stats']['max']:.2f}")
        print(f"  Mean: {quality['face_area_stats']['mean']:.2f}")
        print(f"  Std: {quality['face_area_stats']['std']:.2f}")
        print("="*50)


