# forearm_meshnet/training/metrics.py
"""
Evaluation metrics for ForearmMeshNet
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy.spatial.distance import directed_hausdorff
import trimesh


class MeshEvaluationMetrics:
    """
    Comprehensive evaluation metrics for mesh reconstruction quality.
    """

    def __init__(self, device: torch.device, structure_info: Dict[str, Any]):
        self.device = device
        self.structure_info = structure_info
        self.eps = 1e-8

    # Core metrics

    def chamfer_distance(self, pred_points: torch.Tensor, target_points: torch.Tensor, squared: bool = True) -> torch.Tensor:
        if pred_points.dim() == 2: pred_points = pred_points.unsqueeze(0)
        if target_points.dim() == 2: target_points = target_points.unsqueeze(0)
        D = torch.cdist(pred_points, target_points, p=2)
        p2t = torch.min(D, dim=2)[0]  # [B,N]
        t2p = torch.min(D, dim=1)[0]  # [B,M]
        cd = (p2t.pow(2).mean() + t2p.pow(2).mean()) if squared else (p2t.mean() + t2p.mean())
        return cd / 2.0

    def f_score(self, pred_points: torch.Tensor, target_points: torch.Tensor, threshold: float = 1.0):
        if pred_points.dim() == 2: pred_points = pred_points.unsqueeze(0)
        if target_points.dim() == 2: target_points = target_points.unsqueeze(0)
        D = torch.cdist(pred_points, target_points, p=2)
        p2t = torch.min(D, dim=2)[0]
        t2p = torch.min(D, dim=1)[0]
        precision = (p2t < threshold).float().mean() * 100
        recall    = (t2p < threshold).float().mean() * 100
        f = 2 * precision * recall / (precision + recall) if (precision + recall).item() > 0 else torch.tensor(0.0, device=pred_points.device)
        return f, precision, recall

    def mesh_iou(self, pred_verts: torch.Tensor, target_verts: torch.Tensor,
                pred_faces: torch.Tensor, target_faces: torch.Tensor, resolution: int = 64) -> torch.Tensor:
        pm = trimesh.Trimesh(vertices=pred_verts.detach().cpu().numpy(),   faces=pred_faces.detach().cpu().numpy())
        tm = trimesh.Trimesh(vertices=target_verts.detach().cpu().numpy(), faces=target_faces.detach().cpu().numpy())

        # common pitch from combined bounds, then pad grids to same shape
        bounds = np.concatenate([pm.bounds, tm.bounds], axis=0)
        pitch = (bounds.max(0) - bounds.min(0)).max() / resolution if (np.ptp(bounds, axis=0).max() > 0) else 1.0

        pv_grid = pm.voxelized(pitch=pitch)
        tv_grid = tm.voxelized(pitch=pitch)

        common_origin = bounds.min(0)
        global_shape = np.ceil((bounds.max(0) - common_origin) / pitch).astype(int) + 1

        def _embed(grid, shape):
            offset = np.round((np.array(grid.translation) - common_origin) / pitch).astype(int)
            mat = grid.matrix
            out = np.zeros(shape, dtype=bool)
            s = offset
            e = offset + np.array(mat.shape)
            out[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = mat
            return out

        pv = _embed(pv_grid, global_shape)
        tv = _embed(tv_grid, global_shape)
        inter = np.logical_and(pv, tv).sum()
        union = np.logical_or (pv, tv).sum()
        return torch.tensor(inter / (union + self.eps), dtype=torch.float32)

    # Batch evaluation

    def compute_all_metrics(self,
                            pred_deformations: Dict[str, torch.Tensor],
                            target_deformations: Dict[str, torch.Tensor],
                            template_meshes: Dict[str, Dict[str, torch.Tensor]],
                            affine_template_graph,
                            structure_masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:

        metrics = {}
        affine_verts_by_struct = self._extract_structure_vertices_from_graph(affine_template_graph)

        for name in pred_deformations.keys():
            if name not in target_deformations or name not in template_meshes:
                continue

            orig_tpl = template_meshes[name]['vertices'].to(self.device)
            aff_tpl  = affine_verts_by_struct[name].to(self.device)

            pred_verts   = aff_tpl  + pred_deformations[name]
            target_verts = orig_tpl + target_deformations[name]

            faces = template_meshes[name].get('faces')
            faces = faces.to(self.device) if faces is not None else None

            if structure_masks and name in structure_masks:
                m = structure_masks[name]
                if m.sum() == 0:
                    continue
                pred_verts, target_verts = pred_verts[m], target_verts[m]

            sm = {}
            cd = self.chamfer_distance(pred_verts[0], target_verts[0])
            sm['chamfer_distance'] = cd.item()

            for thr in [0.5, 1, 2]:  # mm thresholds in the pipeline
                f, p, r = self.f_score(pred_verts[0], target_verts[0], thr)
                sm[f'f_score_{thr}'] = f.item()
                sm[f'precision_{thr}'] = p.item()
                sm[f'recall_{thr}'] = r.item()

            if faces is not None and len(faces) > 0:
                iou = self.mesh_iou(pred_verts[0], target_verts[0], faces, faces)
                sm['iou'] = iou.item()

            metrics[name] = sm

        metrics['aggregate'] = self._compute_aggregate_metrics(metrics)
        return metrics

    def _extract_structure_vertices_from_graph(self, affine_graph):
        pos   = affine_graph.pos        # [B*V, 3]
        batch = affine_graph.batch
        B     = int(batch.max().item()) + 1
        Vtot  = pos.shape[0] // B
        pos   = pos.view(B, Vtot, 3)
        out = {}
        for name, info in self.structure_info.items():
            s, e = info['vertex_range']
            out[name] = pos[:, s:e, :]
        return out

    def _compute_aggregate_metrics(self, struct_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        agg, names = {}, set()
        for v in struct_metrics.values():
            if isinstance(v, dict): names.update(v.keys())
        for m in names:
            vals = [v[m] for k, v in struct_metrics.items() if k != 'aggregate' and isinstance(v, dict) and m in v]
            if vals:
                agg[f'mean_{m}'] = float(np.mean(vals))
                agg[f'std_{m}']  = float(np.std(vals))
        return agg
