# forearm_meshnet/template/muscle_template.py
"""
Muscle template generation module for ForearmMeshNet
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Muscle definitions
MUSCLE_NAMES = {
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


class MuscleTemplateGenerator:
    """
    Generate muscle template meshes for ForearmMeshNet.
    """
    
    def __init__(self):
        """Initialize the MuscleTemplateGenerator."""
        self.muscle_templates = {}
        self.muscle_availability = {}
    
    
    def create_from_dataset(self,
                        muscle_data: Dict,
                        skin_meshes_by_subject: Optional[Dict[str, trimesh.Trimesh]] = None,
                        min_availability: float = 0.8,
                        target_vertices: Optional[int] = None) -> Dict[str, trimesh.Trimesh]:
            """
            pick one source subject, compute skin-relative positions,
            determine per-muscle vertex budgets, standardize + decimate + reposition.
            """
            self.muscle_availability = self._analyze_muscle_availability(muscle_data)
            src_id, src = self.select_template_source_subject(muscle_data, min_muscle_coverage=0.7)
            skin_mesh = (skin_meshes_by_subject or {}).get(src_id)

            anatomical, forearm_centroid = self._extract_anatomical_positions_from_source(src, skin_mesh)
            muscle_meshes = src.get('muscle_meshes', {})
            if target_vertices is None:
                targets = self._determine_optimal_vertex_counts(muscle_meshes)
            else:
                targets = {m: int(target_vertices) for m in muscle_meshes.keys()}

            out = {}
            for m, md in muscle_meshes.items():
                goal = targets.get(m, 100)
                tpl = self._create_standardized_template_with_position(
                    md['mesh'], m, goal, anatomical[m]['relative_to_skin']
                )
                if tpl is not None:
                    out[m] = tpl
            
            self.muscle_templates = out 

            # keep stats (useful downstream & for save/load)
            self.template_stats = {
                'source_subject': src_id,
                'anatomical_positions': anatomical,
                'forearm_centroid': forearm_centroid.tolist(),
                'targets': targets,
                'availability': self.muscle_availability
            }
            return out
    
    
    def _create_muscle_template(self,
                               muscle_meshes: List[trimesh.Trimesh],
                               muscle_name: str,
                               target_vertices: int) -> trimesh.Trimesh:
        """
        Create template for a specific muscle.
        
        Args:
            muscle_meshes: List of muscle meshes from different subjects
            muscle_name: Name of the muscle
            target_vertices: Target number of vertices
            
        Returns:
            Muscle template mesh
        """
        if not muscle_meshes:
            return None
        
        # Use median volume mesh as base
        volumes = [mesh.volume if mesh.is_watertight else mesh.convex_hull.volume 
                  for mesh in muscle_meshes]
        median_idx = np.argsort(volumes)[len(volumes)//2]
        base_mesh = muscle_meshes[median_idx].copy()
        
        # Normalize position and orientation
        base_mesh = self._normalize_muscle_mesh(base_mesh)
        
        # Simplify to target vertices
        if len(base_mesh.vertices) > target_vertices:
            base_mesh = base_mesh.simplify_quadric_decimation(target_vertices)
        
        # Clean up
        base_mesh.remove_duplicate_faces()
        base_mesh.remove_unreferenced_vertices()
        base_mesh.remove_degenerate_faces()
        
        return base_mesh
    
    def _normalize_muscle_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Normalize muscle mesh position and orientation.
        
        Args:
            mesh: Input muscle mesh
            
        Returns:
            Normalized mesh
        """
        # Center at origin
        mesh.vertices -= mesh.centroid
        
        pca = PCA(n_components=3)
        pca.fit(mesh.vertices)
        mesh.vertices = mesh.vertices @ pca.components_.T
        
        return mesh
    
    def save(self, path: str):
        """
        Save muscle templates to file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'muscle_templates': {},
            'muscle_availability': getattr(self, 'muscle_availability', {}),
            'template_stats': getattr(self, 'template_stats', {})
        }
        
        for muscle_name, template in self.muscle_templates.items():
            data['muscle_templates'][muscle_name] = {
                'vertices': template.vertices,
                'faces': template.faces,
                'num_vertices': len(template.vertices),
                'num_faces': len(template.faces),
                'volume': template.volume if template.is_watertight else 0
            }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Muscle templates saved to {path}")
    
    def load(self, path: str):
        """
        Load muscle templates from file.
        
        Args:
            path: Input file path
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.muscle_availability = data.get('muscle_availability', {})
        self.muscle_templates = {}
        
        for muscle_name, template_data in data.get('muscle_templates', {}).items():
            self.muscle_templates[muscle_name] = trimesh.Trimesh(
                vertices=template_data['vertices'],
                faces=template_data['faces'],
                process=False
            )
        
        logger.info(f"Loaded {len(self.muscle_templates)} muscle templates from {path}")

    def _analyze_muscle_availability(self, muscle_data: Dict) -> Dict[str, float]:
        muscle_counts = {m: 0 for m in MUSCLE_NAMES.keys()}
        subjects = muscle_data.get('subjects_data', {})
        total = max(1, len(subjects))
        for s in subjects.values():
            for m in s.get('muscle_meshes', {}).keys():
                if m in muscle_counts:
                    muscle_counts[m] += 1
        return {m: muscle_counts[m] / total for m in muscle_counts}

    def select_template_source_subject(self, muscle_data: Dict, min_muscle_coverage: float = 0.7):
        """Pick the subject used as the source of ALL muscle templates (coverage*0.7 + quality*0.3)."""
        self.muscle_availability = self._analyze_muscle_availability(muscle_data)
        subjects = muscle_data.get('subjects_data', {})
        high_avail = [m for m, a in self.muscle_availability.items() if a >= 0.8]
        best_id, best_score = None, -1.0
        for sid, sdata in subjects.items():
            muscles = sdata.get('muscle_meshes', {})
            coverage = 0.0
            if high_avail:
                covered = len([m for m in muscles if m in high_avail])
                coverage = covered / len(high_avail)
            # quality: average of per-muscle normalized scores
            q_vals = []
            for m, md in muscles.items():
                vtx = md.get('vertices', len(md['mesh'].vertices))
                vol = md.get('volume_mm3', md['mesh'].volume if md['mesh'].is_watertight else 0.0)
                q = 0.5 * (min(vtx / 500.0, 1.0) + min(vol / 10000.0, 1.0))
                q_vals.append(q)
            quality = float(np.mean(q_vals)) if q_vals else 0.0
            score = 0.7 * coverage + 0.3 * quality
            if score > best_score:
                best_score, best_id = score, sid
        if best_id is None:
            raise RuntimeError("No suitable template source subject found.")
        return best_id, subjects[best_id]
    
    def _determine_optimal_vertex_counts(self, muscle_meshes: Dict[str, Dict]) -> Dict[str, int]:
        high  = ['FCR','FCU','ECRB','ECRL','ECU']
        medium= ['FDS','FDP','ED','EPL','APL']
        low   = ['PL','PQ','PT','SUP','EDM','ANC','FPL']
        targets = {}
        for m in muscle_meshes.keys():
            base = 150 if m in high else 100 if m in medium else 75
            avail = float(self.muscle_availability.get(m, 0.0))
            # small boost for well-represented muscles
            targets[m] = int(round(base * (0.8 + 0.4 * avail)))
        return targets
    
    def _standardize_mesh_pose(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        m = mesh.copy()
        m.vertices -= m.centroid
        p = PCA(n_components=3).fit(m.vertices)
        m.vertices = m.vertices @ p.components_.T
        # ensure positive volume: flip faces if needed
        if m.volume < 0:
            m.faces = np.fliplr(m.faces)
        return m

    def _simplify_mesh_open3d(self, mesh: trimesh.Trimesh, target_vertices: int) -> Optional[trimesh.Trimesh]:
        try:
            m = o3d.geometry.TriangleMesh()
            m.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            m.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
            tgt_tris = max(10, int(round(target_vertices * 1.5)))
            try:
                m = m.simplify_quadric_decimation(target_number_of_triangles=tgt_tris, boundary_weight=1.0)
            except TypeError:
                m = m.simplify_quadric_decimation(target_number_of_triangles=tgt_tris)
            v = np.asarray(m.vertices); f = np.asarray(m.triangles)
            if v.size == 0 or f.size == 0:
                return None
            return trimesh.Trimesh(vertices=v, faces=f, process=False)
        except Exception:
            return None

    def _uniform_sample_mesh(self, mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
        try:
            return mesh.simplify_quadric_decimation(int(target_vertices))
        except Exception:
            return mesh

    def _reduce_existing_muscle_template(self, muscle_mesh: trimesh.Trimesh, muscle_name: str, target_vertices: int) -> trimesh.Trimesh:
        current = len(muscle_mesh.vertices)
        if current <= target_vertices:
            return muscle_mesh
        reduced = self._simplify_mesh_open3d(muscle_mesh, target_vertices)
        if reduced is None or len(reduced.vertices) < 0.7 * target_vertices:
            reduced = self._uniform_sample_mesh(muscle_mesh, target_vertices)
        # sanitize
        reduced.remove_duplicate_faces()
        reduced.remove_degenerate_faces()
        reduced.remove_unreferenced_vertices()
        return reduced
    
    def _extract_anatomical_positions_from_source(self, source_subject_data: Dict, skin_mesh: Optional[trimesh.Trimesh]):
        muscles = source_subject_data['muscle_meshes']
        muscle_centroids = {m: np.mean(md['mesh'].vertices, axis=0) for m, md in muscles.items()}
        if skin_mesh is not None:
            skin_centroid = skin_mesh.vertices.mean(axis=0)
        else:
            # fallback: mean of all muscle vertices
            all_v = np.vstack([md['mesh'].vertices for md in muscles.values()])
            skin_centroid = all_v.mean(axis=0)
        anatomical = {
            m: {
                'absolute': c,
                'relative_to_skin': c - skin_centroid
            } for m, c in muscle_centroids.items()
        }
        return anatomical, skin_centroid
    def _create_standardized_template_with_position(self, original_mesh: trimesh.Trimesh,
                                                muscle_name: str,
                                                target_vertices: int,
                                                relative_position: np.ndarray) -> Optional[trimesh.Trimesh]:
        templ = self._standardize_mesh_pose(original_mesh)
        templ = self._reduce_existing_muscle_template(templ, muscle_name, target_vertices)
        # place back at preserved location (relative to skin)
        templ.vertices = templ.vertices + relative_position
        return templ
    
    

    

